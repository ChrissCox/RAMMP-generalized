#pragma once
// NEW RAMMP driver extension. This is not an existing upstream API.
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <time.h>
#include "kinova_lowlevel/transport.h"
#include "kinova_lowlevel/interface/arbiter.h"
#include "rclcpp/rclcpp.hpp"
#include "rammp_adl_interfaces/msg/driver_feedback.hpp"
#include "rammp_adl_interfaces/msg/driver_heartbeat.hpp"
#include "driver_build_id.h"

namespace rammp_driver {
inline uint64_t monotonic_ns() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) std::terminate();
  return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}
inline std::string read_identity(const char* path) {
  std::ifstream stream(path);
  std::string id;
  std::getline(stream, id);
  if (id.size() != 36) throw std::runtime_error("Linux boot/session identity unavailable");
  return id;
}

// Successful round-trip evidence is captured before the supervisor pump. Atomic
// scalar storage makes readers race-free even while the RT writer updates it.
// Re-publication does not advance sequence or either exchange timestamp.
class FeedbackProbe : public kinova::Transport {
 public:
  static_assert(std::atomic<double>::is_always_lock_free, "Feedback scalars must be lock-free");
  explicit FeedbackProbe(kinova::Transport& inner) : inner_(inner) {}
  void connect() override { inner_.connect(); }
  void set_servoing_low_level() override { inner_.set_servoing_low_level(); }
  void set_actuator_modes(const kinova::ActuatorModes& m) override { inner_.set_actuator_modes(m); }
  void safe_shutdown() override { inner_.safe_shutdown(); }
  void clear_faults() override { inner_.clear_faults(); }
  // Sticky for this driver session. Atomic admission does not wait for the RT
  // exchange or the arbiter mutex. A new process is required to clear it.
  void request_gripper_halt() noexcept { gripper_halt_requested_.store(true); }
  bool gripper_halt_requested() const noexcept { return gripper_halt_requested_.load(); }
  bool record_gripper_command() noexcept {
    gripper_command_admitted_.store(true);
    return !gripper_halt_requested_.load();
  }
  // Called only while the arbiter is unowned and prior revoke/halt delegation
  // has drained. The sink records admission before RT stamping, so false means
  // no old gripper command exists, including one not yet exchanged.
  bool prepare_unowned_gripper_grant() noexcept {
    if (gripper_command_admitted_.load()) return !gripper_halt_requested_.load();
    gripper_halt_requested_.store(false);
    return true;
  }
  void send(const kinova::JointCommand& c) override { inner_.send(guard_gripper(c)); }
  void receive(kinova::JointFeedback& fb) override {
    const auto start = monotonic_ns();
    inner_.receive(fb);
    save(fb, start, monotonic_ns());
  }
  void exchange(const kinova::JointCommand& c, kinova::JointFeedback& fb) override {
    const auto start = monotonic_ns();
    const auto guarded = guard_gripper(c);
    const bool held = gripper_halt_latched_ && guarded.gripper.active;
    inner_.exchange(guarded, fb);
    save(fb, start, monotonic_ns(), held);
  }
  bool snapshot(rammp_adl_interfaces::msg::DriverFeedback& msg) const {
    for (int attempt = 0; attempt < 8; ++attempt) {
      const auto before = seq_.load();
      if (!before || before % 2) continue;
      msg.exchange_sequence = before / 2;
      msg.exchange_start_monotonic_ns = start_.load();
      msg.exchange_end_monotonic_ns = end_.load();
      msg.transport_frame_id = frame_.load();
      for (size_t i = 0; i < 7; ++i) {
        msg.position_rad[i] = q_[i].load();
        msg.velocity_rad_s[i] = dq_[i].load();
        msg.effort_nm[i] = tau_[i].load();
      }
      msg.fault = fault_.load();
      msg.gripper_present = gripper_present_.load();
      msg.gripper_position_normalized = gripper_q_.load();
      msg.gripper_current_a = gripper_current_.load();
      msg.gripper_halt_supported = true;
      msg.gripper_halt_active = gripper_halt_ack_.load() != 0;
      msg.gripper_halt_generation = msg.gripper_halt_active ? 1 : 0;
      msg.gripper_halt_exchange_sequence = gripper_halt_ack_.load();
      msg.gripper_halt_position_normalized = gripper_halt_position_.load();
      if (seq_.load() == before) return true;
    }
    return false;
  }
 private:
  kinova::JointCommand guard_gripper(const kinova::JointCommand& command) {
    // RT writer only. Never use a default zero position to invent a hold.
    auto result = command;
    if (command.gripper.active &&
        (!std::isfinite(command.gripper.position) || command.gripper.position < 0 || command.gripper.position > 1 ||
         !std::isfinite(command.gripper.speed) || command.gripper.speed < 0 || command.gripper.speed > 1 ||
         !std::isfinite(command.gripper.force) || command.gripper.force < 0 || command.gripper.force > 1))
      request_gripper_halt();
    if (!gripper_halt_requested_.load() && command.gripper.active) {
      gripper_command_admitted_.store(true);
      last_gripper_ceiling_ = command.gripper.force;
      have_gripper_command_ = true;
    }
    if (gripper_halt_requested_.load()) {
      if (!gripper_halt_latched_ && have_gripper_command_ && gripper_present_.load() &&
          std::isfinite(gripper_q_.load()) && gripper_q_.load() >= 0 && gripper_q_.load() <= 1) {
        gripper_hold_q_ = gripper_q_.load();
        gripper_halt_latched_ = true;
      }
      if (gripper_halt_latched_) {
        result.gripper.active = true;
        result.gripper.position = gripper_hold_q_;
        result.gripper.speed = 0;
        result.gripper.force = last_gripper_ceiling_;
      } else {
        // No measured gripper position exists. Refuse new commands and expose
        // no acknowledgement; the caller must keep stop/handback unresolved.
        result.gripper.active = false;
      }
    }
    return result;
  }
  void save(const kinova::JointFeedback& fb, uint64_t start, uint64_t end, bool held = false) {
    seq_.fetch_add(1);
    start_.store(start); end_.store(end); frame_.store(fb.frame_id);
    for (size_t i = 0; i < 7; ++i) {
      q_[i].store(fb.q[i]); dq_[i].store(fb.qd[i]); tau_[i].store(fb.tau[i]);
    }
    fault_.store(fb.fault);
    gripper_present_.store(fb.gripper.present);
    gripper_q_.store(fb.gripper.position);
    gripper_current_.store(fb.gripper.current);
    if (held) {
      gripper_halt_ack_.store((seq_.load() + 1) / 2);
      gripper_halt_position_.store(gripper_hold_q_);
    }
    seq_.fetch_add(1);
  }
  kinova::Transport& inner_;
  std::atomic<uint64_t> seq_{0}, start_{0}, end_{0}, frame_{0};
  std::array<std::atomic<double>, 7> q_{}, dq_{}, tau_{};
  std::atomic<double> gripper_q_{0}, gripper_current_{0};
  std::atomic<bool> fault_{false}, gripper_present_{false};
  std::atomic<bool> gripper_halt_requested_{false};
  std::atomic<bool> gripper_command_admitted_{false};
  std::atomic<uint64_t> gripper_halt_ack_{0};
  std::atomic<double> gripper_halt_position_{0};
  // Only the transport's RT writer accesses these fields.
  bool gripper_halt_latched_{false}, have_gripper_command_{false};
  float gripper_hold_q_{0}, last_gripper_ceiling_{0};
};

// Under the same Arbiter token gate as the arm. Recording here (before the
// controller queues/stamps a target) makes arm-only ownership reuse distinguish
// an uncommanded gripper from a command not yet observed by the RT exchange.
class GuardedGripperSink : public kinova::interface::GripperSink {
 public:
  GuardedGripperSink(kinova::interface::GripperSink& inner, FeedbackProbe& probe)
      : inner_(inner), probe_(probe) {}
  void on_gripper_setpoint(const kinova::interface::GripperSetpoint& command) override {
    if (probe_.record_gripper_command()) inner_.on_gripper_setpoint(command);
  }
  kinova::interface::GripperState on_query_gripper() override { return inner_.on_query_gripper(); }
 private:
  kinova::interface::GripperSink& inner_;
  FeedbackProbe& probe_;
};

// ArbitrationServer is the sole owner-management caller. Serializing this
// wrapper's grants makes check-and-grant atomic without changing core's ABI.
// Motion remains gated by the same core Arbiter; no controller fallback exists.
class GuardedOwnership : public kinova::interface::ArbitrationSink {
 public:
  using Token = kinova::interface::Token;
  GuardedOwnership(kinova::interface::Arbiter& arb, double timeout_s, FeedbackProbe* feedback = nullptr)
      : arb_(arb), feedback_(feedback), timeout_s_(timeout_s),
        session_(read_identity("/proc/sys/kernel/random/uuid")),
        boot_(read_identity("/proc/sys/kernel/random/boot_id")) {
    if (!std::isfinite(timeout_s) || timeout_s <= 0 || timeout_s > 10)
      throw std::invalid_argument("heartbeat_timeout_s must be commissioned in (0,10]");
    timeout_ns_ = uint64_t(timeout_s * 1e9);
    thread_ = std::thread([this] { watch(); });
  }
  ~GuardedOwnership() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
  }
  kinova::interface::GrantResult grant(const std::string& owner_id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = arb_.status();
    if (owner_id.empty() || existing.owned || latched_.load() ||
        (feedback_ && !feedback_->prepare_unowned_gripper_grant()) || latched_.load())
      return {false, Token{}, existing.generation, "owner absent, already owned, or stop/session halt latched"};
    const auto grant = arb_.grant(owner_id);
    if (grant.accepted) {
      owner_ = owner_id; token_ = grant.token; generation_ = grant.generation;
      sequence_ = 0; sent_ = 0;
      deadline_.store(monotonic_ns() + timeout_ns_);
    }
    return grant;
  }
  void revoke() override {
    if (feedback_) feedback_->request_gripper_halt();
    std::lock_guard<std::mutex> lock(mutex_);
    // Leave watchdog armed until downstream revoke returns.
    arb_.revoke(); deadline_.store(0); token_ = Token{}; owner_.clear();
  }
  void estop() override {
    // The stop latch must not wait for the ownership mutex.
    latched_.store(true);
    if (feedback_) feedback_->request_gripper_halt();
    arb_.estop();
    std::lock_guard<std::mutex> lock(mutex_);
    deadline_.store(0); token_ = Token{}; owner_.clear();
  }
  void estop_clear() override {
    // Software reset cannot resume a task or restore ownership. Clearing a
    // watchdog fault requires a process restart and a fresh explicitly claimed
    // session; an arbitrary /estop false must not re-arm it.
    if (!latched_.load()) arb_.estop_clear();
  }
  kinova::interface::ArbitrationStatus status() const override { return arb_.status(); }
  bool heartbeat(const rammp_adl_interfaces::msg::DriverHeartbeat& msg) {
    const auto now = monotonic_ns();
    std::lock_guard<std::mutex> lock(mutex_);
    if (latched_.load() || !deadline_.load() || now >= deadline_.load() ||
        msg.driver_session_id != session_ || msg.host_boot_id != boot_ ||
        msg.owner_id != owner_ || msg.ownership_generation != generation_ ||
        msg.token != token_ || msg.sequence <= sequence_ ||
        msg.sent_monotonic_ns <= sent_ || msg.sent_monotonic_ns > now ||
        now - msg.sent_monotonic_ns >= timeout_ns_) return false;
    sequence_ = msg.sequence; sent_ = msg.sent_monotonic_ns;
    // Sender time, not reception time, prevents delayed traffic refreshing an
    // already old heartbeat. Deadline never gets extended by a duplicate.
    deadline_.store(sent_ + timeout_ns_);
    return true;
  }
  void decorate(rammp_adl_interfaces::msg::DriverFeedback& msg) {
    msg.driver_session_id = session_; msg.host_boot_id = boot_;
    msg.extension_build_id = RAMMP_DRIVER_GUARD_BUILD_ID;
    msg.heartbeat_required = true; msg.heartbeat_timeout_s = timeout_s_;
    msg.watchdog_latched = latched_.load();
    const auto status = arb_.status();
    msg.owned = status.owned; msg.owner_id = status.owner_id;
    msg.ownership_generation = status.generation;
    std::lock_guard<std::mutex> lock(mutex_);
    msg.last_heartbeat_sequence = sequence_;
  }
 private:
  void watch() {
    while (running_.load()) {
      const auto deadline = deadline_.load();
      if (deadline && monotonic_ns() >= deadline) {
        // Claim the expired deadline without waiting for ROS/ownership locks.
        // A heartbeat extending it wins the CAS; a heartbeat racing after this
        // latch cannot undo the driver's unconditional stop.
        auto expected = deadline;
        if (deadline_.compare_exchange_strong(expected, 0)) {
          latched_.store(true);
          if (feedback_) feedback_->request_gripper_halt();
          arb_.estop();
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  kinova::interface::Arbiter& arb_;
  FeedbackProbe* feedback_;
  const double timeout_s_;
  uint64_t timeout_ns_;
  const std::string session_, boot_;
  mutable std::mutex mutex_;
  std::string owner_;
  Token token_{};
  uint64_t generation_{0}, sequence_{0}, sent_{0};
  std::atomic<uint64_t> deadline_{0};
  std::atomic<bool> latched_{false}, running_{true};
  std::thread thread_;
};
}  // namespace rammp_driver
