# NEW RAMMP driver extension, emitted by the instrumented transport only.
# Same-host CLOCK_MONOTONIC nanoseconds bracket a successful blocking exchange.
# These are NOT device acquisition timestamps. Commissioned sensor-age bounds
# are still required for physical admission. Publication does not refresh them.
string driver_session_id
string host_boot_id
# SHA256 of the pinned source/overlay/IDL manifest; not a ROS node name claim.
string extension_build_id
bool simulation
uint64 exchange_sequence
uint64 exchange_start_monotonic_ns
uint64 exchange_end_monotonic_ns
uint64 transport_frame_id
string[7] joint_names
float64[7] position_rad
float64[7] velocity_rad_s
float64[7] effort_nm
bool fault
bool gripper_present
float64 gripper_position_normalized
float64 gripper_current_a
# NEW latched gripper halt: the transport replaces the last target with measured
# aperture, zero speed and the previously commanded current ceiling. No opening.
# This is command-exchange acknowledgement, not proof of physical stopping.
# A halt cannot be cleared within this driver process; restart/recommission.
bool gripper_halt_supported
bool gripper_halt_active
uint64 gripper_halt_generation
uint64 gripper_halt_exchange_sequence
float64 gripper_halt_position_normalized
# Independent driver thread watches the transferred owner's heartbeat.
bool heartbeat_required
float64 heartbeat_timeout_s
bool watchdog_latched
bool owned
string owner_id
uint64 ownership_generation
uint64 last_heartbeat_sequence
