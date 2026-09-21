#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /opt/rammp/install/setup.bash
set -u
cd /opt/rammp

case "${1:-check}" in
  check)
    python3 tools/check_design.py
    python3 -m unittest discover -s tests -v
    colcon test --packages-select rammp_adl_interfaces rammp_adl_runtime --event-handlers console_direct+
    colcon test-result --verbose
    ;;
  launch)
    # Physical motion is false in the launch file and rejected by the node.
    exec ros2 launch rammp_adl_runtime simulation.launch.py enable_astra:=false
    ;;
  *)
    printf 'Supported commands: check, launch\n' >&2
    exit 2
    ;;
esac
