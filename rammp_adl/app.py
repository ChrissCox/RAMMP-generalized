"""Explicit composition roots. No physical robot transport is constructed here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import Catalog, SkillRegistry, checked_copy, strict_loads
from .executor import DagExecutor
from .handlers import build_handlers
from .telemetry import TraceRecorder
from .validation import PlanValidator
from .world import WorldModel


@dataclass
class Runtime:
    catalog: Catalog
    registry: SkillRegistry
    world: WorldModel
    validator: PlanValidator
    backend: object
    executor: DagExecutor
    trace: TraceRecorder


def fixture_runtime(context, *, root=None, time_scale=0.0, failures=None) -> Runtime:
    """Opt into synthetic task-state evidence for orchestration tests only.

    This does not attest cuRobo planning, geometric reachability or physical grasp
    success. The separate MuJoCo runner reports actual simulated joint dynamics.
    """
    from .simulation import FixtureBackend, FixtureGeometry
    catalog = Catalog(root)
    if isinstance(context, (Path, str)):
        context = strict_loads(Path(context).read_bytes())
    context = checked_copy(context)
    backend = FixtureBackend(context, time_scale=time_scale, failures=failures)
    handlers = build_handlers(backend)
    # Explicit emulation only; never use these IDs for a deployment registry.
    emulated = backend.simulation_capabilities(catalog)
    registry = SkillRegistry(catalog, handlers, emulated, mode="simulation")
    context["available_skills"] = list(registry.available_skills)
    world = WorldModel(context, catalog, trust_initial=True, max_evidence_age_s=120.0)
    geometry = FixtureGeometry(backend)
    validator = PlanValidator(catalog, registry, world, geometry)
    trace = TraceRecorder()
    executor = DagExecutor(catalog, registry, world, validator, backend, trace=trace)
    return Runtime(catalog, registry, world, validator, backend, executor, trace)


def astra_for(runtime: Runtime, *, transport=None):
    from .reasoning import AstraReasoner
    config = strict_loads((runtime.catalog.root / "config/reasoning.json").read_bytes())
    async def held():
        # A skill that asks the model a question owns the arm and waits for the
        # answer: the hold it must show is the arm measured still, not an idle
        # backend. Between skills the supervisor's held state is the gate.
        backend = runtime.backend
        if getattr(backend, "active", None) and callable(getattr(backend, "still_now", None)):
            return backend.still_now()
        await runtime.executor.safety.verify_hold()
        return True
    return AstraReasoner(runtime.catalog, config, transport=transport,
                         hold_assertion=held,
                         epoch_getter=lambda _task_id: runtime.world.snapshot().execution_epoch)


def sheppy_runtime(context, *, root=None, client, profiles=None, capabilities, commissioned,
                   observer=None, guard_factory=None, collision_guarded=False, aperture_map=None,
                   max_evidence_age_s=120., chain=None, constraints=None, constraint_store=None,
                   speed_scales=None, grasp_exclusion_m=None) -> Runtime:
    """Compose the runtime as a client of sheppy's arm module.

    The client is constructed and armed by the caller; this only binds the six
    handlers to it. ``capabilities`` is the operator's declaration and
    ``commissioned`` the operator's assertion: neither is inferred here, and
    with either withheld no physical skill registers. The world's robot facts
    are not trusted from JSON; bootstrap them from measurement afterwards.
    """
    from .sheppy_backend import SheppyArmBackend, SheppyGeometry
    catalog = Catalog(root)
    if isinstance(context, (Path, str)):
        context = strict_loads(Path(context).read_bytes())
    context = checked_copy(context)
    profiles = context["profiles"] if profiles is None else profiles
    # The backend needs the world and the world needs the registry's skill
    # list; bind the backend to a world built from the finished context.
    holder = {}
    backend = SheppyArmBackend(catalog, _LateWorld(holder), client=client, profiles=profiles,
                               observer=observer, guard_factory=guard_factory,
                               collision_guarded=collision_guarded, aperture_map=aperture_map,
                               chain=chain, constraints=constraints, constraint_store=constraint_store,
                               speed_scales=speed_scales,
                               **({} if grasp_exclusion_m is None else {"grasp_exclusion_m": grasp_exclusion_m}))
    registry = backend.registry(capabilities=capabilities, commissioned=commissioned, mode="hardware")
    context["available_skills"] = list(registry.available_skills)
    world = WorldModel(context, catalog, trust_initial=False, max_evidence_age_s=max_evidence_age_s)
    holder["world"] = world
    backend.world = world
    if backend.observation is not None:
        backend.observation.world = world
    validator = PlanValidator(catalog, registry, world, SheppyGeometry(backend))
    trace = TraceRecorder()
    executor = DagExecutor(catalog, registry, world, validator, backend, trace=trace)
    return Runtime(catalog, registry, world, validator, backend, executor, trace)


class _LateWorld:
    """Stands in for the world until the registry-dependent context is final."""

    def __init__(self, holder):
        self._holder = holder

    def __getattr__(self, name):
        world = self._holder.get("world")
        if world is None:
            raise RuntimeError("world is not constructed yet")
        return getattr(world, name)
