"""Measured set_gripper handler composition; physical catalog stays disabled.

Only calibrated empty-gripper aperture is implemented here. Grasp, release,
retention, geometry approval and hardware commissioning are not inferred. The
trusted geometry callback must admit the full aperture/stopping envelope, also
for parallel arm motion. Task recipes remain ordinary canonical DAGs.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from types import MappingProxyType
import uuid

from .contracts import ContractError, SkillRegistry, checked_copy, digest, validate_schema
from .handlers import BackendFailure, SetGripperHandler, SkillOutcome
from .motion.gripper import GripperTransport, GripperCommand, calibrated_command
from .motion.leasing import drain_nonpreemptible
from .validation import GeometryCheck


@dataclass(frozen=True)
class MeasuredGripperProfile:
    profile_id: str
    calibration_digest: str
    speed_fraction: float
    current_ceiling_fraction: float
    simulation: bool

    def __post_init__(self):
        if (not isinstance(self.profile_id, str) or not self.profile_id
                or not isinstance(self.calibration_digest, str) or len(self.calibration_digest) != 64
                or type(self.simulation) is not bool):
            raise ContractError('Explicit gripper profile/calibration identity required')
        for value in (self.speed_fraction, self.current_ceiling_fraction):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ContractError('Explicit bounded gripper speed/current ceiling required')


@dataclass(frozen=True, eq=False)
class _ApertureAdmission:
    node_id: str
    task_id: str
    epoch: int
    args_digest: str
    command: GripperCommand
    profile: MeasuredGripperProfile
    dependencies: object
    bounds_digest: str
    expires_at: float
    phase: str


class MeasuredGripperBackend:
    """One actual GripperTransport and a world-bound empty-aperture gate.

    envelope_check(node, snapshot, prediction, phase) is a trusted local geometry
    evaluator returning GeometryCheck. It must supply current dependencies and
    an explicit validated aperture envelope (including permitted concurrent arm
    motion). Its established facts are restricted to motion_profile_valid.
    command_check(context) proves the executor's exact GRIPPER reservation and
    execution identity; it is rechecked at every publish checkpoint.
    held_check verifies complete robot hold for task-level waits only.
    """
    fixture_capabilities = frozenset()
    physical_capabilities = frozenset()

    def __init__(self, catalog, world, *, transport, profiles, envelope_check, command_check, held_check):
        if (type(transport) is not GripperTransport or not all(callable(fn) for fn in
                (envelope_check, command_check, held_check))):
            raise ContractError('Concrete guarded gripper transport and local gates required')
        if transport._active or transport._faulted or transport.last_receipt is not None:
            raise ContractError('A measured backend must bind a fresh unused gripper gateway')
        records = tuple(profiles)
        if any(type(profile) is not MeasuredGripperProfile for profile in records) or len({p.profile_id for p in records}) != len(records):
            raise ContractError('Unique immutable measured gripper profiles required')
        self.catalog, self.world, self.transport = catalog, world, transport
        self.profiles = MappingProxyType({p.profile_id:p for p in records})
        self.envelope_check, self.command_check, self.held_check = envelope_check, command_check, held_check
        self.hardware_commands = transport.hardware_commands
        self.mode = 'physical_gripper' if self.hardware_commands else 'simulation_fixture'
        if not self.hardware_commands and not transport.feedback.simulation:
            raise ContractError('A physical feedback mode cannot be relabeled a simulation backend')
        self._issued = {}
        self._busy = False
        self._active_admission = self._active_context = self._cancel = None
        self._execution_identity = None
        self._work = None
        # One gateway owns the final gate; never chain an arbitrary preexisting
        # callback that can silently bypass this world's permit.
        transport.admission_check = self.admission_check

    def handlers(self):
        return {'set_gripper':SetGripperHandler(self)}

    def registry(self, *, capabilities, mode='hardware', commissioned=False):
        if mode == 'simulation' and self.hardware_commands:
            raise ContractError('A command-capable port cannot enter the simulation registry')
        return SkillRegistry(self.catalog,self.handlers(),capabilities,mode=mode,commissioned=commissioned)

    def _profile(self, args, snapshot):
        profile = self.profiles.get(args['profile_id'])
        records = {p['profile_id']:p for p in snapshot.context['profiles']}
        record = records.get(args['profile_id'])
        if (profile is None or record is None or record['safety_class'] != 'gripper'
                or record['simulation_only'] is not profile.simulation
                or profile.simulation is not self.transport.feedback.simulation
                or profile.calibration_digest != self.transport.feedback.calibration.digest):
            raise ContractError('Gripper profile/world/feedback/calibration binding mismatch')
        return profile

    def _dependencies(self, snapshot, profile_id):
        identities = snapshot.identities()
        names = ('execution_epoch','collision_revision','base_epoch','calibration_id',
                 'robot_config_id','attachment_id','grasp_state_id','profile:'+profile_id)
        return {name:identities[name] for name in names}

    def validate(self, node, snapshot, prediction, phase):
        if node['skill'] != 'set_gripper' or phase not in {'admission','dispatch'}:
            raise ContractError('Measured aperture geometry validates only set_gripper')
        validate_schema(node['args'],self.catalog.skills['set_gripper']['arguments'],'set_gripper args')
        args = node['args']
        profile = self._profile(args,snapshot)
        command = calibrated_command(self.transport.feedback.calibration,self.transport.feedback.bounds,
            aperture_m=args['aperture_m'],speed_fraction=profile.speed_fraction,
            current_ceiling_fraction=profile.current_ceiling_fraction)
        geometry = self.envelope_check(checked_copy(node),snapshot,prediction,phase)
        if (not isinstance(geometry,GeometryCheck) or geometry.status != 'validated'
                or not 0 < geometry.valid_for_s <= self.world.max_evidence_age_s):
            raise ContractError('Explicit current gripper aperture/stopping envelope approval required')
        allowed = {'predicate':'motion_profile_valid','args':{'profile_id':profile.profile_id},'validity':'true'}
        if any(fact != allowed for fact in geometry.established_facts):
            raise ContractError('Gripper geometry cannot invent emptiness, retention or aperture success')
        dependencies = self._dependencies(snapshot,profile.profile_id)
        for key,value in geometry.dependencies.items():
            if snapshot.identities().get(key) != value:
                raise ContractError('Gripper envelope dependency differs from current world')
            dependencies[key] = value
        expires_at = snapshot.captured_at+geometry.valid_for_s
        if self.world.clock() >= expires_at:
            raise ContractError('Gripper geometry expired while its envelope was being checked')
        artifact = _ApertureAdmission(node['id'],snapshot.context['task_id'],snapshot.execution_epoch,
            digest(args),command,profile,MappingProxyType(checked_copy(dependencies)),digest(self.transport.feedback.bounds.__dict__),
            expires_at,phase)
        if phase == 'dispatch':
            now = self.world.clock()
            self._issued = {key:value for key,value in self._issued.items() if value.expires_at > now}
            if len(self._issued) >= 256:
                raise ContractError('Unconsumed gripper dispatch bound exceeded')
            self._issued[id(artifact)] = artifact
        return GeometryCheck(end_state=geometry.end_state,dependencies=MappingProxyType(checked_copy(dependencies)),artifact=artifact,
                             established_facts=geometry.established_facts,valid_for_s=geometry.valid_for_s)

    def _current(self, artifact, context, command):
        if (type(artifact) is not _ApertureAdmission or artifact.phase != 'dispatch' or artifact.command != command
                or context.task_id != artifact.task_id or context.node_id != artifact.node_id
                or context.execution_epoch != artifact.epoch or context.attempt < 1
                or context.cancel_event.is_set() or self.world.clock() >= artifact.expires_at
                or self.command_check(context) is not True):
            raise BackendFailure('stale_state','Gripper command/epoch/resource admission revoked')
        if self._busy and self._execution_identity != (context.task_id,context.node_id,context.attempt,context.execution_epoch):
            raise BackendFailure('stale_state','Gripper execution attempt identity changed')
        snapshot = self.world.snapshot()
        try:
            current_profile = self._profile({'profile_id':artifact.profile.profile_id},snapshot)
        except ContractError as exc:
            raise BackendFailure('stale_state',str(exc)) from exc
        if (snapshot.context['task_id'] != artifact.task_id or snapshot.execution_epoch != artifact.epoch
                or any(snapshot.identities().get(key) != value for key,value in artifact.dependencies.items())
                or digest(self.transport.feedback.bounds.__dict__) != artifact.bounds_digest
                or current_profile != artifact.profile):
            raise BackendFailure('stale_state','Gripper world/profile/calibration changed')
        if snapshot.fact('gripper_empty',{'robot_id':'robot'}) != 'true':
            raise BackendFailure('safety_fault','Empty-gripper preshaping cannot change a retained/unknown grip')
        return snapshot

    def verify_artifact(self, skill_id, args, context):
        artifact = context.validation_artifact
        if (skill_id != 'set_gripper' or not context.validation_id or type(artifact) is not _ApertureAdmission
                or self._issued.get(id(artifact)) is not artifact or artifact.args_digest != digest(args)):
            raise BackendFailure('safety_fault','Exact issued measured gripper dispatch artifact required')
        self._current(artifact,context,artifact.command)

    def admission_check(self, artifact, command):
        try:
            if not self._busy or artifact is not self._active_admission or self._cancel.is_set():
                return False
            self._current(artifact,self._active_context,command)
            return True
        except (BackendFailure,ContractError,AttributeError):
            return False

    async def set_gripper(self, args, context):
        if self._busy:
            raise BackendFailure('safety_fault','Measured gripper backend already active')
        args = checked_copy(args)
        self.verify_artifact('set_gripper',args,context)
        artifact = self._issued.pop(id(context.validation_artifact))
        self._busy = True
        self._active_admission, self._active_context = artifact,context
        self._execution_identity = (context.task_id,context.node_id,context.attempt,context.execution_epoch)
        self._cancel = asyncio.Event()
        try:
            self._work = asyncio.create_task(self.transport.execute(artifact.command,artifact,cancel_event=self._cancel))
            try:
                receipt = await asyncio.shield(self._work)
            except asyncio.CancelledError:
                self._cancel.set()
                receipt = await drain_nonpreemptible(self._work)
            if self._cancel.is_set() or context.cancel_event.is_set():
                raise BackendFailure('cancelled','Gripper execution cancelled before measured effect publication')
            if receipt.status != 'succeeded':
                code = 'cancelled' if receipt.status == 'cancelled' else ('obstructed' if 'deadline' in receipt.reason else 'safety_fault')
                raise BackendFailure(code,receipt.reason)
            self._current(artifact,context,artifact.command)
            state = self.transport.feedback.read(ownership=self.transport.ownership)
            if (not receipt.measured_quiescent or not self.transport.measured_quiescent()
                    or abs(state.aperture_m-args['aperture_m'])+self.transport.feedback.calibration.error_m > self.transport.feedback.bounds.target_tolerance_m):
                raise BackendFailure('safety_fault','Current measured gripper aperture is not quiescent at the admitted target')
            evidence_id = 'gripper-'+uuid.uuid4().hex
            fact = {'predicate':'aperture_reached','args':{'aperture_m':args['aperture_m']},'validity':'true'}
            evidence = {'evidence_id':evidence_id,'source':self.mode,'observed_at':state.acquired_lower_s,
                'ttl_s':min(self.world.max_evidence_age_s,self.transport.feedback.bounds.state_max_age_s),
                'predicates':[fact],'dependencies':dict(artifact.dependencies),
                'data':{'simulation_only':self.transport.feedback.simulation,'calibration_digest':state.calibration_digest,
                        'measured_aperture_m':state.aperture_m,'current_a':state.current_a,
                        'exchange_sequence':state.sequence,'driver_session_id':state.source_id,
                        'retention_evaluated':False}}
            return SkillOutcome('succeeded',{'evidence_id':evidence_id},[evidence],[{**fact,'evidence_id':evidence_id}])
        finally:
            self._busy = False
            self._active_admission = self._active_context = None
            self._execution_identity = None

    async def skill_quiescent(self, skill_id):
        if skill_id != 'set_gripper' or self._busy: return False
        if self.transport.last_receipt is None:
            return await self.quiescent()
        return self.transport.measured_quiescent()

    async def quiescent(self):
        if self._busy: return False
        held = self.held_check()
        if asyncio.iscoroutine(held): held = await held
        return held is True and (self.transport.last_receipt is None or self.transport.measured_quiescent())

    async def stop_skill(self, skill_id, reason):
        if skill_id != 'set_gripper':
            raise BackendFailure('safety_fault','Measured gripper backend cannot stop another skill')
        if self._cancel is not None: self._cancel.set()
        if self._work is not None and not self._work.done():
            await drain_nonpreemptible(self._work)
        elif self.transport.last_receipt is not None:
            await self.transport.stop(reason)

    async def stop(self, reason='supervisor'):
        await self.stop_skill('set_gripper',reason)
