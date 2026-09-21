#!/usr/bin/env python3
"""Prepare pinned driver sources plus the NEW RAMMP guard; never starts a driver."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PINS = {
    "kinova-gen3-driver": "5eb6582f015fc4dd8856c081c4cc14dc536be024",
    "kinova-gen3-ros2": "3337456a1e6062969934174091b3e0ce1bb55f9c",
    "rammp-interfaces-ros2": "38b326451ce289e0e3a473fee77a9be4cedeffa1",
    "RAMMP-CuRobo": "320872b709b276fc7283190d24edef7f8632bec9",
}


def run(*args: str, cwd: Path | None = None) -> bytes:
    return subprocess.check_output(args, cwd=cwd)


def replace(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("Pinned source patch context no longer matches uniquely")
    return source.replace(old, new)


def prepare(workspace: Path) -> dict:
    if workspace == ROOT or ROOT in workspace.parents:
        raise ValueError("Build workspace must be outside the project source tree")
    workspace.mkdir(parents=True, exist_ok=True)
    src = workspace / "src"
    src.mkdir(exist_ok=True)
    for name in ("kinova-gen3-driver", "kinova-gen3-ros2", "rammp-interfaces-ros2"):
        dest = src / name
        if not dest.exists():
            local = (Path("/home/abra/ros2_ws/src") if name == "rammp-interfaces-ros2"
                     else Path("/home/abra")) / name
            remote = str(local) if (local / ".git").is_dir() else f"https://github.com/rammp-org/{name}.git"
            run("git", "clone", "--no-checkout", remote, str(dest))
        try:
            run("git", "cat-file", "-e", PINS[name] + "^{commit}", cwd=dest)
        except subprocess.CalledProcessError:
            run("git", "fetch", f"https://github.com/rammp-org/{name}.git", PINS[name], cwd=dest)
        run("git", "checkout", "--detach", PINS[name], cwd=dest)
    planner = Path("/home/abra/RAMMP-CuRobo")
    archive = run("git", "archive", PINS["RAMMP-CuRobo"], "rammp_curobo_interfaces", cwd=planner)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("Unexpected interface archive entry")
            target = (src / member.name).resolve()
            if src.resolve() not in target.parents:
                raise ValueError("Interface archive escaped source tree")
        tar.extractall(src)
    shutil.copytree(ROOT / "interfaces", src / "rammp_adl_interfaces", dirs_exist_ok=True)
    ros = src / "kinova-gen3-ros2"
    pkg = ros / "kinova_gen3_ros2"
    def original(path: str) -> str:
        return run("git", "show", PINS["kinova-gen3-ros2"] + ":kinova_gen3_ros2/" + path, cwd=ros).decode()
    code = original("src/bringup_node.cpp")
    code = replace(code, '#include <atomic>', '#include <atomic>\n#include "driver_guard.h"')
    code = replace(code, 'GripperController grip(*base);', 'rammp_driver::FeedbackProbe feedback_probe(*base);\n  GripperController grip(feedback_probe);')
    code = replace(code, 'if (mode_str != "enforced" && mode_str != "disabled")', 'if (mode_str != "enforced")')
    code = replace(code, 'interface::Arbiter arb(sup, sup, sup, arb_mode);', '''rammp_driver::GuardedGripperSink guarded_gripper(sup, feedback_probe);
  interface::Arbiter arb(sup, sup, guarded_gripper, arb_mode);
  rammp_driver::GuardedOwnership guarded_owner(
      arb, node->declare_parameter("heartbeat_timeout_s", 0.0), &feedback_probe);
  auto evidence_pub = node->create_publisher<rammp_adl_interfaces::msg::DriverFeedback>(
      "/rammp/driver_feedback", rclcpp::QoS(1).reliable());
  auto heartbeat_group = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions heartbeat_options;
  heartbeat_options.callback_group = heartbeat_group;
  auto heartbeat_sub = node->create_subscription<rammp_adl_interfaces::msg::DriverHeartbeat>(
      "/rammp/driver_heartbeat", rclcpp::QoS(1).reliable(),
      [&guarded_owner](rammp_adl_interfaces::msg::DriverHeartbeat::SharedPtr msg) {
        guarded_owner.heartbeat(*msg);
      }, heartbeat_options);
  auto evidence_timer = node->create_wall_timer(std::chrono::milliseconds(10), [&] {
    rammp_adl_interfaces::msg::DriverFeedback msg;
    if (!feedback_probe.snapshot(msg)) return;
    guarded_owner.decorate(msg);
    msg.simulation = use_sim;
    msg.joint_names = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"};
    evidence_pub->publish(msg);
  });''')
    code = replace(code, 'node, arb, use_sim ? std::string("sim") : ip, estop_clear_max_age_s);', 'node, guarded_owner, use_sim ? std::string("sim") : ip, estop_clear_max_age_s);')
    (pkg / "src/bringup_node.cpp").write_text(code)
    # A rejected non-stealing acquire must not erase the current owner's release
    # token retained by the upstream ROS service wrapper.
    code = original("src/arbitration_server.cpp")
    code = replace(code, 'have_retained_ = r.accepted;\n    retained_token_ = r.accepted ? r.token : Token{};', 'if (r.accepted) { have_retained_ = true; retained_token_ = r.token; }')
    (pkg / "src/arbitration_server.cpp").write_text(code)
    code = original("CMakeLists.txt")
    code = replace(code, 'find_package(rclcpp REQUIRED)', 'find_package(rclcpp REQUIRED)\nfind_package(rammp_adl_interfaces REQUIRED)')
    code = replace(code, 'ament_target_dependencies(kinova_gen3_node rclcpp rclcpp_action rammp_arm_interfaces geometry_msgs)', 'ament_target_dependencies(kinova_gen3_node rclcpp rclcpp_action rammp_arm_interfaces geometry_msgs rammp_adl_interfaces)')
    code += '''\nadd_executable(rammp_driver_guard_test src/driver_guard_test.cpp)
ament_target_dependencies(rammp_driver_guard_test rclcpp rammp_adl_interfaces)
target_link_libraries(rammp_driver_guard_test kinova_lowlevel::kinova_lowlevel)
target_compile_options(rammp_driver_guard_test PRIVATE -UNDEBUG)
install(TARGETS rammp_driver_guard_test DESTINATION lib/${PROJECT_NAME})
'''
    (pkg / "CMakeLists.txt").write_text(code)
    code = original("package.xml")
    code = replace(code, '<depend>rclcpp</depend>', '<depend>rclcpp</depend>\n  <depend>rammp_adl_interfaces</depend>')
    (pkg / "package.xml").write_text(code)
    for name in ("driver_guard.h", "driver_guard_test.cpp"):
        shutil.copy2(HERE / name, pkg / "src" / name)
    manifest = {
        "pins": PINS, "workspace": str(workspace),
        "overlay_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                           for name in ("prepare.py", "driver_guard.h", "driver_guard_test.cpp")},
        "new_interfaces_sha256": {name: hashlib.sha256((ROOT / "interfaces/msg" / name).read_bytes()).hexdigest()
                                  for name in ("DriverFeedback.msg", "DriverHeartbeat.msg")},
        "upstream_interfaces_sha256": {
            name: hashlib.sha256((src / "rammp-interfaces-ros2" / name).read_bytes()).hexdigest()
            for name in ("rammp_arm_interfaces/action/ExecuteJointTrajectory.action",
                         "rammp_arm_interfaces/msg/GripperSetpoint.msg",
                         "rammp_arm_interfaces/msg/GripperState.msg",
                         "rammp_common_interfaces/msg/ControlStatus.msg",
                         "rammp_common_interfaces/msg/EStop.msg",
                         "rammp_common_interfaces/srv/AcquireControl.srv",
                         "rammp_common_interfaces/srv/ReleaseControl.srv")},
        "hardware_started": False,
    }
    identity = {key: value for key, value in manifest.items() if key not in {"workspace", "hardware_started"}}
    manifest["extension_build_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    with (pkg / "src/driver_build_id.h").open("w") as stream:
        stream.write('#define RAMMP_DRIVER_GUARD_BUILD_ID "' + manifest["extension_build_id"] + '"\n')
    (workspace / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/rammp-driver-hardware-build"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace.resolve()), indent=2))
