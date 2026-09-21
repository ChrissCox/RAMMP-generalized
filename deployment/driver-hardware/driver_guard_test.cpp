// CPU/SimTransport tests only. No Kortex SDK or robot connection exists here.
#include "driver_guard.h"
#include "kinova_lowlevel/sim_transport.h"
#include <cassert>
#include <iostream>
using namespace kinova::interface;
using namespace std::chrono_literals;

struct GripperProbeTransport : kinova::Transport {
  kinova::JointFeedback feedback;
  kinova::JointCommand last;
  bool fail = false;
  GripperProbeTransport() { feedback.gripper.present = true; feedback.gripper.position = 0.4f; }
  void connect() override {}
  void set_servoing_low_level() override {}
  void set_actuator_modes(const kinova::ActuatorModes&) override {}
  void safe_shutdown() override {}
  void clear_faults() override {}
  void send(const kinova::JointCommand& c) override { last = c; }
  void receive(kinova::JointFeedback& fb) override { fb = feedback; }
  void exchange(const kinova::JointCommand& c, kinova::JointFeedback& fb) override {
    last = c;
    if (fail) throw std::runtime_error("explicit failed simulated exchange");
    fb = feedback;
  }
};

struct Sink : CommandSink, StreamSink, GripperSink {
  std::atomic<unsigned> halts{0};
  std::atomic<bool> block{false}, entered{false}, release{false};
  GoalResponse on_trajectory_goal(const TrajectoryGoal&) override { return GoalResponse::kAccept; }
  void on_trajectory_accepted(const GoalId&, const TrajectoryGoal&) override {}
  CancelResponse on_trajectory_cancel(const CancelRequest&) override { return CancelResponse::kAccept; }
  GainsResult on_set_gains(const GainsRequest&) override { return {}; }
  ArmState on_query_state() override { return {}; }
  void on_halt(HaltReason) override { halts.fetch_add(1); }
  StreamOpenResult on_stream_open(const StreamOpenRequest&) override {
    entered.store(true);
    while (block.load() && !release.load()) std::this_thread::sleep_for(1ms);
    return {};
  }
  void on_stream_close(const StreamCloseRequest&) override {}
  void on_setpoint_joint_position(const JointSetpoint&) override {}
  void on_setpoint_joint_velocity(const JointSetpoint&) override {}
  void on_setpoint_joint_torque(const JointSetpoint&) override {}
  void on_setpoint_pose(const PoseSetpoint&) override {}
  void on_setpoint_twist(const TwistSetpoint&) override {}
  StreamStatus on_query_stream() override { return {}; }
  void on_gripper_setpoint(const GripperSetpoint&) override {}
  GripperState on_query_gripper() override { return {}; }
};

template <typename Predicate> void wait_for(Predicate pred) {
  const auto until = std::chrono::steady_clock::now() + 1s;
  while (!pred() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(1ms);
  assert(pred());
}

rammp_adl_interfaces::msg::DriverHeartbeat heartbeat_for(
    rammp_driver::GuardedOwnership& owner, const GrantResult& grant) {
  rammp_adl_interfaces::msg::DriverFeedback evidence;
  owner.decorate(evidence);
  rammp_adl_interfaces::msg::DriverHeartbeat h;
  h.driver_session_id = evidence.driver_session_id;
  h.host_boot_id = evidence.host_boot_id;
  h.owner_id = evidence.owner_id;
  h.ownership_generation = grant.generation;
  h.token = grant.token;
  h.sequence = 1;
  h.sent_monotonic_ns = rammp_driver::monotonic_ns();
  return h;
}

int main() {
  unsigned passed = 0;
  {
    GripperProbeTransport inner;
    rammp_driver::FeedbackProbe probe(inner);
    Sink sink; rammp_driver::GuardedGripperSink guarded_gripper(sink, probe);
    Arbiter arb(sink, sink, guarded_gripper, ArbitrationMode::kEnforced);
    rammp_driver::GuardedOwnership owner(arb, 0.2, &probe);
    assert(owner.grant("first-arm-only-owner").accepted);
    owner.revoke();
    const auto second = owner.grant("second-arm-only-owner"); assert(second.accepted);
    // Admission is recorded even before the command reaches an RT exchange.
    GripperSetpoint command; command.token = second.token;
    arb.on_gripper_setpoint(command);
    owner.revoke();
    assert(!owner.grant("new-owner-with-stale-gripper-target").accepted);
    ++passed;
  }
  {
    GripperProbeTransport inner;
    rammp_driver::FeedbackProbe probe(inner);
    kinova::JointCommand command;
    command.gripper.active = true; command.gripper.position = 0.9f;
    command.gripper.speed = 0.25f; command.gripper.force = 0.3f;
    kinova::JointFeedback fb;
    probe.exchange(command, fb);
    probe.request_gripper_halt();
    rammp_adl_interfaces::msg::DriverFeedback evidence;
    assert(probe.snapshot(evidence) && !evidence.gripper_halt_active);
    inner.fail = true;
    try { probe.exchange(command, fb); assert(false); } catch (const std::runtime_error&) {}
    assert(probe.snapshot(evidence) && !evidence.gripper_halt_active);
    inner.fail = false;
    probe.exchange(command, fb);
    assert(inner.last.gripper.active && inner.last.gripper.position == 0.4f);
    assert(inner.last.gripper.speed == 0 && inner.last.gripper.force == 0.3f);
    assert(probe.snapshot(evidence) && evidence.gripper_halt_supported && evidence.gripper_halt_active);
    assert(evidence.gripper_halt_generation == 1 && evidence.gripper_halt_exchange_sequence == 2);
    // Later stale targets and changing feedback cannot chase/restore the old target.
    inner.feedback.gripper.position = 0.6f;
    probe.exchange(command, fb); probe.exchange(command, fb);
    assert(inner.last.gripper.position == 0.4f && inner.last.gripper.force == 0.3f);
    ++passed;
  }
  {
    GripperProbeTransport inner;
    rammp_driver::FeedbackProbe probe(inner);
    kinova::JointFeedback fb;
    probe.exchange(kinova::JointCommand{}, fb);
    probe.request_gripper_halt();
    probe.exchange(kinova::JointCommand{}, fb);
    rammp_adl_interfaces::msg::DriverFeedback evidence;
    assert(probe.snapshot(evidence));
    // Arm-only sessions must never invent a gripper current ceiling or command.
    assert(!inner.last.gripper.active && !evidence.gripper_halt_active);
    ++passed;
  }
  {
    GripperProbeTransport inner;
    rammp_driver::FeedbackProbe probe(inner);
    Sink sink; Arbiter arb(sink, sink, sink, ArbitrationMode::kEnforced);
    rammp_driver::GuardedOwnership owner(arb, 0.025, &probe);
    const auto grant = owner.grant("simulation-gripper-owner"); assert(grant.accepted);
    kinova::JointCommand command;
    command.gripper.active = true; command.gripper.position = 0.9f;
    command.gripper.speed = 0.2f; command.gripper.force = 0.3f;
    kinova::JointFeedback fb; probe.exchange(command, fb);
    wait_for([&] { return sink.halts.load() > 0; });
    probe.exchange(command, fb);
    rammp_adl_interfaces::msg::DriverFeedback evidence;
    assert(probe.snapshot(evidence) && evidence.gripper_halt_active);
    assert(inner.last.gripper.position == 0.4f && inner.last.gripper.speed == 0);
    assert(!owner.grant("new-owner-before-restart").accepted);
    ++passed;
  }
  {
    kinova::JointFeedback initial; initial.q[2] = 0.25; initial.qd[4] = 0.1;
    kinova::SimTransport transport(initial);
    rammp_driver::FeedbackProbe probe(transport);
    rammp_adl_interfaces::msg::DriverFeedback first, repeat, second;
    assert(!probe.snapshot(first));
    kinova::JointFeedback fb;
    probe.exchange(kinova::JointCommand{}, fb);
    assert(probe.snapshot(first) && first.exchange_sequence == 1);
    assert(first.position_rad[2] == 0.25 && first.velocity_rad_s[4] == 0.1);
    assert(first.exchange_end_monotonic_ns >= first.exchange_start_monotonic_ns);
    assert(probe.snapshot(repeat));
    assert(repeat.exchange_sequence == first.exchange_sequence);
    assert(repeat.exchange_end_monotonic_ns == first.exchange_end_monotonic_ns);
    probe.exchange(kinova::JointCommand{}, fb);
    assert(probe.snapshot(second) && second.exchange_sequence == 2);
    assert(second.transport_frame_id != first.transport_frame_id);
    ++passed;
  }
  {
    Sink sink; Arbiter arb(sink, sink, sink, ArbitrationMode::kEnforced);
    rammp_driver::GuardedOwnership owner(arb, 0.2);
    auto first = owner.grant("operator"); assert(first.accepted);
    assert(!owner.grant("intruder").accepted);
    assert(owner.status().owner_id == "operator" && sink.halts == 0);
    TrajectoryGoal goal; goal.token = first.token;
    assert(arb.on_trajectory_goal(goal) == GoalResponse::kAccept);
    owner.revoke(); assert(!owner.status().owned);
    ++passed;
  }
  {
    Sink sink; Arbiter arb(sink, sink, sink, ArbitrationMode::kEnforced);
    rammp_driver::GuardedOwnership owner(arb, 0.2);
    auto grant = owner.grant("operator");
    auto h = heartbeat_for(owner, grant);
    auto wrong = h; wrong.token[0] ^= 1; assert(!owner.heartbeat(wrong));
    wrong = h; wrong.driver_session_id = "old"; assert(!owner.heartbeat(wrong));
    wrong = h; wrong.host_boot_id = "other"; assert(!owner.heartbeat(wrong));
    wrong = h; wrong.ownership_generation++; assert(!owner.heartbeat(wrong));
    wrong = h; wrong.sent_monotonic_ns += 1000000000ULL; assert(!owner.heartbeat(wrong));
    wrong = h; wrong.sent_monotonic_ns -= 300000000ULL; assert(!owner.heartbeat(wrong));
    assert(owner.heartbeat(h));
    assert(!owner.heartbeat(h));
    ++h.sequence; h.sent_monotonic_ns = rammp_driver::monotonic_ns();
    assert(owner.heartbeat(h));
    owner.revoke();
    ++passed;
  }
  {
    Sink sink; Arbiter arb(sink, sink, sink, ArbitrationMode::kEnforced);
    rammp_driver::GuardedOwnership owner(arb, 0.04);
    auto grant = owner.grant("operator");
    auto h = heartbeat_for(owner, grant);
    assert(owner.heartbeat(h));
    wait_for([&] { return sink.halts.load() > 0; });
    assert(owner.status().estopped && !owner.status().owned);
    ++h.sequence; h.sent_monotonic_ns = rammp_driver::monotonic_ns();
    assert(!owner.heartbeat(h));
    owner.estop_clear(); assert(owner.status().estopped);
    assert(!owner.grant("operator").accepted);
    ++passed;
  }
  {
    Sink sink; Arbiter arb(sink, sink, sink, ArbitrationMode::kEnforced);
    rammp_driver::GuardedOwnership owner(arb, 0.04);
    const auto grant = owner.grant("operator");
    sink.block.store(true);
    StreamOpenRequest request; request.token = grant.token;
    std::thread blocked([&] { arb.on_stream_open(request); });
    wait_for([&] { return sink.entered.load(); });
    // This succeeds while the core arbiter's command mutex is still occupied.
    wait_for([&] { return sink.halts.load() > 0; });
    sink.release.store(true); blocked.join();
    assert(owner.status().estopped);
    ++passed;
  }
  std::cout << "{\"passed\":" << passed
            << ",\"hardware_started\":false,\"scope\":\"driver guard CPU and SimTransport\"}\n";
}
