# Sourced by every Onyx tool: ROS, the interface overlays and the main checkout's venv.
# A research worktree has no .venv or artifacts/ (both untracked); it borrows the main checkout's.
MAIN="${RAMMP_MAIN_ROOT:-/home/abra/RAMMP-generalized}"
for shared in .venv artifacts; do
  [ -e "$shared" ] || ln -s "$MAIN/$shared" "$shared"
done
export PYTHONNOUSERSITE=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
set +u
source /opt/ros/humble/setup.bash
source /home/abra/rammp_deps_ws/install/setup.bash
source "$MAIN/artifacts/jetson/ros-install/setup.bash"
source "$MAIN/.venv/bin/activate"
set -u
unset CYCLONEDDS_URI ROS_LOCALHOST_ONLY ROS_DOMAIN_ID
