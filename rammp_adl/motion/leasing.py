"""Bounded non-preemptible planner leases with active-motion priority.

Catalog phase leasing is not enabled by this helper. The executor must grant an
explicit trusted phase authorization before speculative work may enter it.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import itertools
import math
import time

from .rolling import MotionError


async def drain_nonpreemptible(task):
    """Retain ownership until worker completion even after repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class PlannerLeaseBroker:
    def __init__(self, *, maximum_call_s, clock=time.monotonic):
        if not isinstance(maximum_call_s, (int, float)) or isinstance(maximum_call_s, bool) or not math.isfinite(maximum_call_s) or maximum_call_s <= 0:
            raise MotionError("lease requires positive measured call budget")
        self.maximum_call_s, self.clock = maximum_call_s, clock
        self._condition, self._sequence = asyncio.Condition(), itertools.count()
        self._waiters = []
        self.owner = None
        self.faulted = False
        self.events = []

    @asynccontextmanager
    async def lease(self, owner, *, active_motion, phase_authorized=False):
        if not active_motion and not phase_authorized:
            raise MotionError("speculation requires trusted catalog phase authorization")
        waiter = (0 if active_motion else 1, next(self._sequence), owner)
        async with self._condition:
            if self.faulted:
                raise MotionError("planner broker fault latched")
            self._waiters.append(waiter)
            try:
                await self._condition.wait_for(lambda: self.faulted or (self.owner is None and waiter == min(self._waiters)))
                if self.faulted:
                    raise MotionError("planner broker fault latched")
                self._waiters.remove(waiter)
                self.owner = owner
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._condition.notify_all()
                raise
        started = self.clock()
        try:
            yield
        finally:
            elapsed = self.clock()-started
            async with self._condition:
                # Never release an in-flight GPU call merely because a timeout
                # elapsed. This context must encompass its actual completion.
                self.owner = None
                self.events.append({"owner": owner, "elapsed_s": elapsed, "active_motion": active_motion})
                if elapsed > self.maximum_call_s:
                    self.faulted = True
                self._condition.notify_all()
            if elapsed > self.maximum_call_s:
                raise MotionError("planner exceeded bounded non-preemptible call budget")

    async def run(self, owner, operation, *, active_motion, phase_authorized=False):
        """Wait out cancelled non-preemptible work before releasing solver state."""
        async with self.lease(owner, active_motion=active_motion, phase_authorized=phase_authorized):
            task = asyncio.create_task(operation())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                try:
                    await drain_nonpreemptible(task)
                finally:
                    raise
