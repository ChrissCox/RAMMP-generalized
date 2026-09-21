"""Six explicit reusable handlers. Backends measure; the sole world writer commits.

Catalog declarations cannot create a handler, and no physical driver transport
is installed by this module. Feeding and user transfer are intentionally absent.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Callable


@dataclass
class ExecutionContext:
    task_id: str
    node_id: str
    attempt: int = 1
    execution_epoch: int = 1
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    snapshot: Any = None
    emit: Callable | None = None
    clock: Callable = time.monotonic
    validation_artifact: Any = None
    validation_id: str = ""

    async def feedback(self, **event):
        if self.emit:
            result = self.emit({"task_id": self.task_id, "node_id": self.node_id,
                                "execution_epoch": self.execution_epoch, **event})
            if inspect.isawaitable(result):
                await result


@dataclass
class SkillOutcome:
    status: str
    outputs: dict = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    proposed_effects: list[dict] = field(default_factory=list)
    backend_quiescent: bool = False
    failure_code: str | None = None
    detail: str = ""
    # Trusted local pose observations are committed atomically with their facts;
    # this field is never accepted from a plan or inferred from predicted motion.
    metric_poses: tuple = field(default_factory=tuple)


class BackendFailure(RuntimeError):
    def __init__(self, code, detail="", *, evidence=None, effects=None):
        super().__init__(detail or code)
        self.code, self.detail = code, detail
        self.evidence, self.effects = evidence or [], effects or []


class _Handler:
    skill_id = None

    def __init__(self, backend):
        if getattr(backend, "hardware_commands", True):
            # Two physical transports exist: the guarded gripper gateway, and
            # the client of sheppy's arm module. Anything else is refused.
            from .hardware_gripper import MeasuredGripperBackend
            gripper_route = type(self) is SetGripperHandler and type(backend) is MeasuredGripperBackend
            client_route = getattr(backend, "physical_transport", None) == "sheppy_client"
            if not (gripper_route or client_route):
                raise ValueError("this implementation has no physical robot transport")
        self.backend = backend

    async def execute(self, args, context):
        if context.cancel_event.is_set():
            return SkillOutcome("cancelled", backend_quiescent=await self.backend.skill_quiescent(self.skill_id), failure_code="cancelled")
        await context.feedback(event="skill_started", skill=self.skill_id)
        try:
            if hasattr(self.backend, "verify_artifact"):
                self.backend.verify_artifact(self.skill_id, args, context)
            result = await self.perform(args, context)
            if not isinstance(result, SkillOutcome):
                raise BackendFailure("safety_fault", "backend returned an invalid outcome")
            # This query concerns only this skill's commands. A concurrent
            # empty-gripper action can finish while the arm continues moving.
            result.backend_quiescent = await self.backend.skill_quiescent(self.skill_id)
            if result.status == "succeeded" and not result.backend_quiescent:
                raise BackendFailure("safety_fault", "backend completion is not quiescent")
            return result
        except asyncio.CancelledError:
            await self.backend.stop_skill(self.skill_id, "cancelled")
            return SkillOutcome("cancelled", backend_quiescent=await self.backend.skill_quiescent(self.skill_id), failure_code="cancelled")
        except BackendFailure as exc:
            await self.backend.stop_skill(self.skill_id, exc.code)
            return SkillOutcome("cancelled" if exc.code == "cancelled" else "failed",
                                evidence=exc.evidence, proposed_effects=exc.effects,
                                backend_quiescent=await self.backend.skill_quiescent(self.skill_id),
                                failure_code=exc.code, detail=exc.detail)


class ObserveHandler(_Handler):
    skill_id = "observe"

    async def perform(self, args, context):
        if not await self.backend.quiescent():
            raise BackendFailure("safety_fault", "observation/cloud wait requires verified global hold")
        return await self.backend.observe(args, context)


class MoveToPoseHandler(_Handler):
    skill_id = "move_to_pose"

    async def perform(self, args, context):
        return await self.backend.move_to_pose(args, context)


class SetGripperHandler(_Handler):
    skill_id = "set_gripper"

    async def perform(self, args, context):
        return await self.backend.set_gripper(args, context)


class GraspHandler(_Handler):
    skill_id = "grasp"

    async def perform(self, args, context):
        return await self.backend.grasp(args, context)


class ReleaseHandler(_Handler):
    skill_id = "release"

    async def perform(self, args, context):
        return await self.backend.release(args, context)


class FollowConstraintHandler(_Handler):
    skill_id = "follow_constraint"

    async def perform(self, args, context):
        return await self.backend.follow_constraint(args, context)


def build_handlers(backend):
    return {handler.skill_id: handler(backend) for handler in (
        ObserveHandler, MoveToPoseHandler, SetGripperHandler, GraspHandler,
        ReleaseHandler, FollowConstraintHandler,
    )}
