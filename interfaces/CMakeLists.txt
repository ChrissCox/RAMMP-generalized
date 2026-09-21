cmake_minimum_required(VERSION 3.16)
project(rammp_adl_interfaces VERSION 0.1.0)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Canonical IDL lives here; do not maintain a second set in the runtime package.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DriverFeedback.msg"
  "msg/DriverHeartbeat.msg"
  "msg/FactEffect.msg"
  "msg/ImageCrop.msg"
  "msg/ResolvedPose.msg"
  "msg/SafetyStatus.msg"
  "msg/WorldState.msg"
  "srv/CommitEffects.srv"
  "srv/GeneratePlan.srv"
  "srv/GetWorldSnapshot.srv"
  "srv/GroundTarget.srv"
  "srv/RequestStop.srv"
  "srv/ResolveBinding.srv"
  "srv/UserConfirm.srv"
  "srv/ValidatePlan.srv"
  "action/ExecuteSkill.action"
  "action/ExecuteTask.action"
  DEPENDENCIES action_msgs builtin_interfaces geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
