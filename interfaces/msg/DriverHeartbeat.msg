# NEW RAMMP driver extension. Owner token is a capability: never log it.
# Valid only on the same host/boot and current driver session. A late, replayed,
# wrong-owner or wrong-token heartbeat cannot refresh the watchdog deadline.
string driver_session_id
string host_boot_id
string owner_id
uint64 ownership_generation
uint8[16] token
uint64 sequence
uint64 sent_monotonic_ns
