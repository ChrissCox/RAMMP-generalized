"""Deterministic supervision, fresh assent and a mandatory execution gate.

This implementation intentionally has no physical arm command transport.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable


class SafetyError(RuntimeError):
    pass


def action_digest(node: dict, dependencies: dict) -> str:
    payload = json.dumps({"node": node, "dependencies": dependencies}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Confirmation:
    confirmation_id: str
    task_id: str
    epoch: int
    node_id: str
    digest: str
    expires_at: float


class ConfirmationStore:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._pending: dict[str, Confirmation] = {}

    def grant(self, *, task_id: str, epoch: int, node_id: str, digest: str,
              ttl_s: float, accepted: bool) -> Confirmation:
        if accepted is not True or not math.isfinite(ttl_s) or ttl_s <= 0 or not digest:
            raise SafetyError("Fresh explicit assent and a bounded lifetime are required")
        record = Confirmation(secrets.token_hex(16), task_id, epoch, node_id, digest, self.clock() + ttl_s)
        self._pending[record.confirmation_id] = record
        return record

    def consume(self, confirmation_id: str, *, task_id: str, epoch: int, node_id: str, digest: str) -> None:
        record = self._pending.pop(confirmation_id, None)
        if record is None or record.expires_at <= self.clock() or (record.task_id, record.epoch, record.node_id, record.digest) != (task_id, epoch, node_id, digest):
            raise SafetyError("Confirmation missing, stale, already used, or bound to a different action")

    def revoke_all(self) -> None:
        self._pending.clear()


async def _resolve(value):
    return await value if inspect.isawaitable(value) else value


class SafetySupervisor:
    SIMULATION_MODES = frozenset({"simulation_fixture", "mujoco_replay", "curobo_simulation"})

    def __init__(self, backend: Any, *, epoch_getter: Callable[[], int],
                 invalidate_epoch: Callable[[], int], trace: Any = None,
                 stop_timeout_s: float = 2.0, clock: Callable[[], float] = time.monotonic):
        if not math.isfinite(stop_timeout_s) or stop_timeout_s <= 0:
            raise ValueError("Invalid stop acknowledgement deadline")
        self.backend = backend
        self.epoch_getter = epoch_getter
        self.invalidate_epoch = invalidate_epoch
        self.trace = trace
        self.stop_timeout_s = stop_timeout_s
        self.clock = clock
        self.fault_latched = False
        self.reason = "STARTUP"
        self.held_verified = False
        self.stop_requested = asyncio.Event()
        self._stop_lock = asyncio.Lock()
        self._heartbeats: dict[str, tuple[float, float]] = {}

    def heartbeat(self, source: str, *, max_age_s: float, captured_at: float | None = None) -> None:
        captured_at = self.clock() if captured_at is None else captured_at
        if not source or not math.isfinite(max_age_s) or max_age_s <= 0 or not math.isfinite(captured_at) or captured_at > self.clock() + 1e-6:
            raise SafetyError("Invalid liveness evidence")
        self._heartbeats[source] = (captured_at, max_age_s)

    def check_liveness(self) -> None:
        now = self.clock()
        expired = [name for name, (stamp, age) in self._heartbeats.items() if now - stamp > age]
        if expired:
            raise SafetyError("Stale required evidence: " + ", ".join(sorted(expired)))

    def check_dispatch(self, epoch: int, *, skills=()) -> None:
        # Read-only observation can run through the same DAG. The exact concrete
        # adapter has no robot command port; neither a mode string nor a claimed
        # capability authorizes a physical motion handler through this gate.
        from .hardware_backend import HardwareObservationBackend
        observation_only = (type(self.backend) is HardwareObservationBackend
                            and self.backend.hardware_commands is False
                            and isinstance(skills, (tuple, list))
                            and bool(skills) and all(skill == "observe" for skill in skills))
        client = (getattr(self.backend, "physical_transport", None) == "sheppy_client"
                  and getattr(self.backend, "hardware_commands", False) is True)
        allowed = observation_only if type(self.backend) is HardwareObservationBackend else (
            getattr(self.backend, "mode", None) in self.SIMULATION_MODES or client)
        if not allowed:
            raise SafetyError("Physical command transport is disabled; select an explicit simulation backend")
        if self.fault_latched or self.stop_requested.is_set():
            raise SafetyError("Supervisor is stopped or fault latched")
        if epoch != self.epoch_getter():
            raise SafetyError("Obsolete execution epoch")
        self.check_liveness()

    async def verify_hold(self) -> None:
        held = await _resolve(self.backend.quiescent())
        self.held_verified = bool(held)
        if not self.held_verified:
            raise SafetyError("Backend has not verified held state")

    async def request_stop(self, reason: str, *, fault: bool = False) -> bool:
        self.stop_requested.set()
        async with self._stop_lock:
            self.reason = reason
            self.fault_latched = self.fault_latched or fault
            # Idempotent while already stopping; each fresh stop revokes once.
            if not getattr(self, "_epoch_revoked", False):
                self.invalidate_epoch()
                self._epoch_revoked = True
            self.held_verified = False
            if self.trace:
                self.trace.emit("stop_requested", reason=reason, epoch=self.epoch_getter())
            try:
                await asyncio.wait_for(_resolve(self.backend.stop()), self.stop_timeout_s)
                await self.verify_hold()
            except Exception as exc:
                self.fault_latched = True
                self.reason = "STOP_UNVERIFIED"
                if self.trace:
                    self.trace.emit("fault_latched", reason=self.reason, error_type=type(exc).__name__)
                return False
            if self.trace:
                self.trace.emit("hold_verified", reason=reason)
            return True

    async def reset(self) -> None:
        await self.verify_hold()
        if self.fault_latched:
            raise SafetyError("Latched fault requires explicit diagnostic resolution; reset cannot resume motion")
        self.stop_requested.clear()
        self._epoch_revoked = False
        self.reason = "HELD"

    async def watch(self, *, period_s: float = 0.02) -> None:
        if period_s <= 0:
            raise ValueError("Watchdog period must be positive")
        while not self.stop_requested.is_set():
            try:
                self.check_liveness()
            except SafetyError as exc:
                await self.request_stop(str(exc), fault=True)
                return
            await asyncio.sleep(period_s)
