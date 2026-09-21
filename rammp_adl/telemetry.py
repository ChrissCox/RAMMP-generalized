"""Structured local timing evidence; never records image bytes or credentials."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class TraceRecorder:
    clock: Callable[[], float] = time.monotonic
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        if any(key.lower() in {"api_key", "authorization", "image_bytes", "token"} for key in fields):
            raise ValueError("Sensitive material does not belong in timing traces")
        entry = {"time_s": self.clock(), "event": event, **fields}
        # Catch nonfinite values and nonserializable objects at their source.
        json.dumps(entry, allow_nan=False)
        self.events.append(entry)
        return entry

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in self.events), encoding="utf-8")

    def summary(self) -> dict[str, Any]:
        waits: dict[str, float] = {}
        running: dict[str, tuple[float, str]] = {}
        for event in self.events:
            if event["event"] == "wait_started":
                running[event["wait_id"]] = (event["time_s"], event["reason"])
            elif event["event"] == "wait_ended" and event["wait_id"] in running:
                start, reason = running.pop(event["wait_id"])
                waits[reason] = waits.get(reason, 0.0) + max(0.0, event["time_s"] - start)
        # Cloud planning precedes run_plan/task_started. Include that wait in
        # request latency, including when reading older traces without the
        # explicit task_request_started event.
        start = next((e["time_s"] for e in self.events
                      if e["event"] in {"task_request_started", "task_started"}
                      or (e["event"] == "wait_started" and e.get("reason") == "cloud_reasoning")), None)
        first = next((e["time_s"] for e in self.events if e["event"] == "motion_started"), None)
        return {
            "event_count": len(self.events),
            "elapsed_s": 0.0 if not self.events else self.events[-1]["time_s"] - self.events[0]["time_s"],
            "first_motion_latency_s": None if start is None or first is None else first - start,
            "first_motion_latency_scope": "local task request including cloud waits; excludes process startup",
            "wait_time_by_reason_s": waits,
            "open_waits": sorted(running),
            "timing_scope": "local_process; simulation timing is not Jetson or robot timing",
        }
