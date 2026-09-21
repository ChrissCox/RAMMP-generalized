# Proposed ROS 2 IDL

These are NEW rammp_adl_interfaces sources. They have not yet been packaged or compiled. See [contract semantics](../docs/design/12-interface-contracts.md) and [package plan](../docs/design/11-package-layout.md). Include builtin_interfaces, std_msgs, geometry_msgs and sensor_msgs when generating them with rosidl.

No file here redefines an upstream Kinova/cuRobo type. Strings containing JSON are strictly validated against the repository schemas or catalog-derived argument/output schemas before use.
