"""ROS 2 Humble transport for the local runtime.

Imports are deferred so the core works without ROS. With backend:=fixture this
node constructs the simulation backend and cannot publish robot commands. With
backend:=sheppy it is a client of sheppy's arm module: it starts nothing, and
sends trajectories and gripper setpoints only when arm_motion is true.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
import json
import os
from pathlib import Path
import math
import threading
import time

from .contracts import Catalog, ContractError, strict_loads
from .ros_bridge import RuntimeBridge


class AsyncWorker:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._closing = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self.thread = threading.Thread(target=self._run, name="rammp-runtime", daemon=True)
        self.thread.start()
        if not self._ready.wait(timeout=3.):
            raise RuntimeError("Runtime worker did not start")

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def submit(self, coroutine):
        with self._lifecycle_lock:
            if self._closing.is_set() or self._closed or not self.thread.is_alive():
                coroutine.close()
                raise RuntimeError("Runtime worker is closing or unavailable")
            return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def call(self, coroutine, timeout=95.0):
        future = self.submit(coroutine)
        try:
            return future.result(timeout)
        except FutureTimeout:
            future.cancel()
            raise RuntimeError("Local operation exceeded its bounded deadline") from None

    def close(self, timeout=5.0):
        if self._closed:
            return True
        with self._lifecycle_lock:
            self._closing.set()
        async def cancel_pending():
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        drained = False
        if self.thread.is_alive():
            future = asyncio.run_coroutine_threadsafe(cancel_pending(), self.loop)
            try:
                future.result(timeout=timeout)
                drained = True
            except (FutureTimeout, RuntimeError):
                future.cancel()
            finally:
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(timeout=timeout)
        if not self.thread.is_alive() and drained:
            self.loop.close()
            self._closed = True
        return drained and not self.thread.is_alive()


class LatestDepth:
    """Keep the newest wrist depth frame for the guard, without blocking it.

    The source enforces imagery locality itself: the deployment's domain-0
    streams are admitted by the deployment policy that lists that domain.
    """

    def __init__(self, *, rgb_topic, depth_topic, rgb_info_topic, depth_info_topic, policy, on_pair=None):
        from .perception.ros_rgbd import RosRgbdSource
        self.on_pair = on_pair
        self.source = RosRgbdSource(rgb_topic=rgb_topic, depth_topic=depth_topic, rgb_info_topic=rgb_info_topic,
                                    depth_info_topic=depth_info_topic, reliability="reliable", domain_id=0,
                                    locality_policy=policy)
        self._lock, self._latest, self._stop = threading.Lock(), None, threading.Event()
        self.thread = threading.Thread(target=self._run, name="rammp-wrist-depth", daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                pair = self.source.capture(500)
            except Exception:                               # noqa: BLE001 - a gap, not a crash
                continue
            metadata = pair.metadata
            frame = (pair.depth_m, tuple(metadata["depth_info"]["k"]),
                     float(metadata["received_at_monotonic_s"]["depth"]))
            with self._lock:
                self._latest = frame
            if self.on_pair is not None:
                try:
                    self.on_pair(pair)
                except Exception:                           # noqa: BLE001 - the consumer reports itself
                    pass

    def latest(self):
        with self._lock:
            return self._latest

    def close(self):
        self._stop.set()
        try:
            self.source.close()
        finally:
            self.thread.join(timeout=2.)


class MonotonicRosClockMapping:
    """Explicit local mapping; reject ROS time jumps/paused simulation time.

    MetricPose capture time is monotonic. Republishing it as ROS now would erase
    observation age. This mapping is valid only while the two local clocks
    advance together; device clocks still require their own calibrated adapter.
    """
    def __init__(self, monotonic_s, ros_nanoseconds, *, maximum_drift_s=0.25):
        self.monotonic_s, self.ros_nanoseconds = monotonic_s, ros_nanoseconds
        self.maximum_drift_s = maximum_drift_s

    def capture_and_expiry(self, captured_at, valid_for_s, *, monotonic_now, ros_now_ns):
        if not all(math.isfinite(v) for v in (captured_at, valid_for_s, monotonic_now)) or valid_for_s <= 0:
            raise ContractError("Invalid metric capture clock data")
        age = monotonic_now-captured_at
        if age < 0 or age >= valid_for_s:
            raise ContractError("Metric pose is future dated or expired")
        drift = (ros_now_ns-self.ros_nanoseconds)/1e9 - (monotonic_now-self.monotonic_s)
        if abs(drift) > self.maximum_drift_s:
            raise ContractError("ROS/source clock mapping invalid after clock jump or pause")
        captured_ns = self.ros_nanoseconds + round((captured_at-self.monotonic_s)*1e9)
        if captured_ns < 0:
            raise ContractError("Metric capture predates representable ROS time")
        return captured_ns, captured_ns+round(valid_for_s*1e9)


def create_node():
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.node import Node
    from rclpy.clock import Clock, ClockType
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rammp_adl_interfaces.action import ExecuteSkill, ExecuteTask
    from rammp_adl_interfaces.msg import SafetyStatus, WorldState
    from rammp_adl_interfaces.srv import (CommitEffects, GeneratePlan, GetWorldSnapshot,
        GroundTarget, RequestStop, ResolveBinding, UserConfirm, ValidatePlan)
    from .app import astra_for, fixture_runtime

    class AdlNode(Node):
        def __init__(self):
            super().__init__("rammp_adl_runtime")
            for name, default in (("project_root", ""), ("context_path", ""), ("enable_astra", False),
                                  ("time_scale", 0.02), ("task_timeout_s", 900.0), ("hardware_motion_enabled", False),
                                  ("backend", "fixture"), ("commissioned", False), ("arm_motion", False),
                                  ("capabilities", ""), ("imagery_policy_path", "config/imagery-locality.json"),
                                  ("sphere_bundle_dir", "artifacts/jetson/real-world-ready/assembly/bundle-2"),
                                  ("touch_nm", 3.0), ("keyframe_feed_hz", 5.0), ("scene", "grounded"),
                                  ("face_model_path", "artifacts/models/face_detection_yunet_2023mar.onnx"),
                                  ("keyframe_min_interval_s", 3.0), ("record_dir", "artifacts/bench"), ("scene_refresh", False), ("scene_refresh_interval_s", 30.0),
                                  ("model_progress_check", False), ("compliant_contact", False), ("jev_decisions", True),
                                  ("home_joints_path", "artifacts/bench/start-joints.json"),
                                  ("planner_world_dir", "/home/abra/.ros/rammp_box_opening/worlds"),
                                  ("skill_library_dir", "artifacts/skills"), ("local_perception", True),
                                  ("transit_speed_scale", 0.4), ("contact_speed_scale", 0.25), ("max_evidence_age_s", 600.0), ("max_viewpoints", 6), ("wrist_rgb_topic", "/wrist_camera/color/image_raw"),
                                  ("wrist_depth_topic", "/wrist_camera/aligned_depth_to_color/image_raw"),
                                  ("wrist_rgb_info_topic", "/wrist_camera/color/camera_info"),
                                  ("wrist_depth_info_topic", "/wrist_camera/aligned_depth_to_color/camera_info")):
                self.declare_parameter(name, default, descriptor=ParameterDescriptor(read_only=True))
            backend = self.get_parameter("backend").value
            if backend not in ("fixture", "sheppy"):
                raise RuntimeError("backend must be fixture or sheppy")
            if backend == "fixture" and self.get_parameter("hardware_motion_enabled").value:
                raise RuntimeError("The fixture backend deliberately has no physical robot command transport")
            catalog = Catalog(self.get_parameter("project_root").value or None)
            self.client = self.depth = self.scene = self._bootstrap_timer = None
            self._intake_phase, self._last_detect, self._last_scene_warning = None, 0., 0.
            self._last_refresh_at, self._status_misses, self._last_status_summary = 0., 0, None
            self._detect_hz = 5.
            self._refresh_timer = self._refresh_future = None
            if backend == "fixture":
                context_path = self.get_parameter("context_path").value or str(catalog.root / "examples/cabinet.context.json")
                self.runtime = fixture_runtime(context_path, root=catalog.root,
                                               time_scale=self.get_parameter("time_scale").value)
            else:
                self.runtime = self._sheppy_runtime(catalog)
            if self.get_parameter("enable_astra").value:
                try:
                    import openai  # noqa: F401 - warm the SDK import before any timer or loop needs it
                except Exception as exc:                    # noqa: BLE001 - reported; the first request will say so too
                    self.get_logger().warning(f"OpenAI SDK unavailable: {exc}")
            reasoner = astra_for(self.runtime) if self.get_parameter("enable_astra").value else None
            self.bridge = RuntimeBridge(self.runtime, reasoner)
            self._log_stops(self.runtime)
            if self.scene is not None:
                self.scene.reasoner = reasoner
            self._closing = threading.Event()
            self.callbacks = ReentrantCallbackGroup()
            # At most four long actions, one ordinary service, one status timer
            # and one stop service can occupy the eight ROS callback threads.
            self.service_callbacks = MutuallyExclusiveCallbackGroup()
            self.stop_callbacks = MutuallyExclusiveCallbackGroup()
            self.status_callbacks = MutuallyExclusiveCallbackGroup()
            self._goal_lock = threading.Lock()
            self._goal_count = 0
            self._safety_revision = 0
            self._status_stop = None
            self.task_timeout_s = float(self.get_parameter("task_timeout_s").value)
            if not math.isfinite(self.task_timeout_s) or not 1. <= self.task_timeout_s <= 3600.:
                raise RuntimeError("task_timeout_s must be finite and between 1 and 3600 seconds")
            self._clock_mapping = MonotonicRosClockMapping(self.runtime.world.clock(), self.get_clock().now().nanoseconds)
            durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.safety_publisher = self.create_publisher(SafetyStatus, "/rammp/safety/status", durable)
            self.world_publisher = self.create_publisher(WorldState, "/rammp/world/state", durable)
            self.adl_services = [self.create_service(kind, path, callback,
                callback_group=self.stop_callbacks if kind is RequestStop else self.service_callbacks)
                for kind, path, callback in (
                    (GetWorldSnapshot, "/rammp/world/snapshot", self.snapshot),
                    (ResolveBinding, "/rammp/world/resolve", self.resolve),
                    (CommitEffects, "/rammp/world/commit", self.commit),
                    (ValidatePlan, "/rammp/plan/validate", self.validate),
                    (GeneratePlan, "/rammp/reasoning/generate_plan", self.reason),
                    (GroundTarget, "/rammp/perception/ground", self.ground),
                    (RequestStop, "/rammp/supervisor/stop", self.stop),
                    (UserConfirm, "/rammp/hri/confirm", self.confirm))]
            self.skill_action = ActionServer(self, ExecuteSkill, "/rammp/execute_skill",
                execute_callback=self.execute_skill, goal_callback=self.accept_goal,
                cancel_callback=self.cancel_goal, callback_group=self.callbacks)
            self.task_action = ActionServer(self, ExecuteTask, "/rammp/execute_task",
                execute_callback=self.execute_task, goal_callback=self.accept_goal,
                cancel_callback=self.cancel_goal, callback_group=self.callbacks)
            self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self.timer = self.create_timer(.1, self.publish_state, callback_group=self.status_callbacks, clock=self._steady_clock)
            # No worker thread is started until parameter/entity construction
            # succeeds. Callbacks cannot run before main adds/spins this node.
            self.worker = AsyncWorker()
            if backend == "fixture":
                self.get_logger().info("Simulation runtime ready; all physical arm command transports are absent")
            else:
                self._bootstrap_timer = self.create_timer(.5, self._bootstrap_robot, callback_group=self.status_callbacks)
                if self.get_parameter("scene_refresh").value and self.scene is not None:
                    interval = max(1., float(self.get_parameter("keyframe_min_interval_s").value))
                    self._refresh_timer = self.create_timer(interval, self._refresh_tick, callback_group=self.status_callbacks)
                self.get_logger().info("sheppy client runtime ready; available skills: "
                                       + (", ".join(self.runtime.registry.available_skills) or "none")
                                       + f"; motion armed={self.client.motion_enabled}")

        async def _template_plan(self, runtime, task_text, goal, articulations):
            """A local plan for a common goal, its open choices put to Jev; None asks Astra as before."""
            from .contracts import ContractError
            from .plans import template_for_goal
            support_of = {item["handle"]: item["surface"]["entity_id"] for item in articulations}
            variants = {"release": True, "retract": True}
            decider = getattr(runtime.executor, "decider", None)
            if decider is not None and goal["predicate"] == "constraint_goal_verified":
                from .decisions import decide_plan_variants
                variants, _ = await decide_plan_variants(decider, task_text, "open the part and hold it at its goal",
                                                         threshold=getattr(runtime.executor, "goal_threshold", .8))
            plan = template_for_goal(goal, runtime.world.snapshot().context, runtime.catalog, support_of=support_of, **variants)
            if plan is None:
                return None
            try:
                await asyncio.to_thread(runtime.executor.validator.admit, plan)
            except ContractError as exc:
                self.get_logger().info(f"template plan for {goal['predicate']} not admitted ({str(exc)[:120]}); asking astra")
                return None
            self.get_logger().info(f"plan from the {goal['predicate']} template: {[node['id'] for node in plan['nodes']]}; "
                                   f"release {variants['release']}, retract {variants['retract']}")
            return plan

        def _jev_decider(self, runtime):
            """TypeSafe's Jev for fast typed decisions when the operator switched it on; (None, threshold) otherwise."""
            if not self.get_parameter("jev_decisions").value:
                return None, .85
            from .decisions import JevDecider, load_config
            try:
                config = load_config(runtime.catalog.root)
            except (OSError, ValueError) as exc:
                self.get_logger().warning(f"jev decisions requested but not configured ({exc}); astra decides alone")
                return None, .85
            decider = JevDecider(config, log=self.get_logger().info)
            if not decider.available:
                self.get_logger().warning(f"jev decisions requested but {config['api_key_env']} is not set; astra decides alone")
                return None, .85
            runtime.executor.goal_threshold = float(config["goal_min_confidence"])
            runtime.executor.recovery_threshold = float(config.get("recovery_min_confidence", .8))
            return decider, float(config["stop_min_confidence"])

        def _sheppy_runtime(self, catalog):
            """Compose the runtime as a client of sheppy's arm module; nothing is started."""
            from .motion.collision_guard import (CollisionGuard, EffortGuard, GuardSet, SphereModel,
                                                 load_spheres, mount_transform)
            from .motion.kinematics import UrdfChain
            from .motion.sheppy_arm import SheppyArmClient
            context_path = self.get_parameter("context_path").value or str(catalog.root / "config/sheppy-bench.context.json")
            capabilities = {c.strip() for c in self.get_parameter("capabilities").value.split(",") if c.strip()}
            self.client = SheppyArmClient(self)
            if self.get_parameter("arm_motion").value:
                self.client.arm()
            touch_nm = float(self.get_parameter("touch_nm").value)
            self._detect_hz = max(.5, float(self.get_parameter("keyframe_feed_hz").value))
            root = catalog.root
            bundle = Path(self.get_parameter("sphere_bundle_dir").value)
            bundle = bundle if bundle.is_absolute() else root/bundle
            policy_path = Path(self.get_parameter("imagery_policy_path").value)
            policy_path = policy_path if policy_path.is_absolute() else root/policy_path
            chain = None
            try:
                chain = UrdfChain.from_path(bundle/"arm-gripper-locked.urdf")
            except Exception as exc:                        # noqa: BLE001 - reported, features withheld
                self.get_logger().warning(f"no locked assembly model ({exc}); no collision guard, no wrist observation")
            collision, depth_reader = None, None
            if chain is not None:
                try:
                    d405 = load_spheres(bundle/"d405-collision-spheres.json")["wrist_d405_link"]
                    model = SphereModel(chain, load_spheres(bundle/"collision-spheres.json"),
                                        extra=[("end_effector_link", mount_transform(), d405[0], d405[1])])
                    collision = CollisionGuard(model)
                    self.depth = LatestDepth(
                        rgb_topic=self.get_parameter("wrist_rgb_topic").value,
                        depth_topic=self.get_parameter("wrist_depth_topic").value,
                        rgb_info_topic=self.get_parameter("wrist_rgb_info_topic").value,
                        depth_info_topic=self.get_parameter("wrist_depth_info_topic").value,
                        policy=strict_loads(policy_path.read_bytes()) if policy_path.is_file() else None,
                        on_pair=self._feed_scene)
                    depth_reader = self.depth.latest
                except Exception as exc:                    # noqa: BLE001 - reported, guard downgraded
                    self.get_logger().warning(f"live collision guard unavailable ({exc}); effort guard only")
                    collision, depth_reader, self.depth = None, None, None
            scene_kind = self.get_parameter("scene").value
            if scene_kind not in ("grounded", "none"):
                raise RuntimeError("scene must be grounded or none")
            if chain is not None and self.depth is not None and scene_kind != "none":
                try:
                    base_context = strict_loads(Path(context_path).read_bytes())
                    from .perception.grounded_scene import GroundedScene
                    from .perception.keyframes import FaceScreen, KeyframeSelector
                    face_path = Path(self.get_parameter("face_model_path").value)
                    face_path = face_path if face_path.is_absolute() else root/face_path
                    screen = None
                    try:
                        screen = FaceScreen(face_path)
                    except Exception as exc:            # noqa: BLE001 - reported, egress withheld
                        self.get_logger().warning(f"face screen unavailable ({exc}); no keyframe will leave the machine")
                    self.scene = GroundedScene(client=self.client, chain=chain, calibration_id=base_context["calibration_id"],
                                               face_screen=screen, pose_validity_s=600.,
                                               dump_dir=root/"artifacts/keyframes",
                                               selector=KeyframeSelector(min_interval_s=float(self.get_parameter("keyframe_min_interval_s").value)))
                    record_dir = self.get_parameter("record_dir").value
                    if record_dir:
                        from .perception.scene_record import RuntimeRecorder
                        record_root = Path(record_dir) if Path(record_dir).is_absolute() else root/record_dir
                        self.scene.recorder = RuntimeRecorder(record_root)
                    self.get_logger().info(f"{scene_kind} scene: "+json.dumps(self.scene.describe()))
                except Exception as exc:                    # noqa: BLE001 - reported, observe withheld
                    self.get_logger().warning(f"{scene_kind} scene unavailable ({exc}); observe is not wired")
                    self.scene = None
            elif chain is not None and scene_kind != "none":
                self.get_logger().warning("the scene needs the wrist camera; observe is not wired")

            def guard_factory(touch_nm=touch_nm, exclusions=(), tool_exclusion_m=0.):
                return GuardSet(effort=EffortGuard(touch_nm), collision=collision, depth_reader=depth_reader,
                                exclusions=exclusions, tool_exclusion_m=tool_exclusion_m)

            from .constraints import ConstraintStore
            self.constraint_store = ConstraintStore(root)
            transit = max(1., 1./max(1e-3, float(self.get_parameter("transit_speed_scale").value)))
            contact = max(1., 1./max(1e-3, float(self.get_parameter("contact_speed_scale").value)))
            self._sheppy_options = dict(root=root, client=self.client, capabilities=capabilities,
                                        commissioned=bool(self.get_parameter("commissioned").value),
                                        guard_factory=guard_factory, collision_guarded=collision is not None,
                                        observer=self.scene, chain=chain, constraint_store=self.constraint_store,
                                        speed_scales={"transit": transit, "contact": contact},
                                        max_evidence_age_s=float(self.get_parameter("max_evidence_age_s").value))
            self._base_context_path = context_path
            # Every task ends here: the bench's recorded start pose, unless the operator names another.
            # Discovery and grounding on the Jetson first (perception/local_discovery.py), warmed off the startup path.
            self._detector = None
            if self.get_parameter("local_perception").value:
                from .perception.local_discovery import OwlDetector
                self._detector = OwlDetector()

                def warm(detector=self._detector, log=self.get_logger()):
                    try:
                        from PIL import Image
                        began = time.monotonic()
                        detector.detect(Image.new("RGB", (64, 48)), ["door handle"])
                        log.info(f"local perception ready ({detector.name} on {detector.device}, {time.monotonic()-began:.1f} s)")
                    except Exception as exc:                # noqa: BLE001 - discovery then goes to Astra, as before
                        log.warning(f"local perception unavailable ({type(exc).__name__}: {str(exc)[:120]}); discovery uses Astra")
                threading.Thread(target=warm, daemon=True, name="local-perception-warmup").start()
            self._home_joints = None
            home_path = Path(self.get_parameter("home_joints_path").value) if self.get_parameter("home_joints_path").value else None
            if home_path is not None:
                home_path = home_path if home_path.is_absolute() else root/home_path
                try:
                    self._home_joints = [float(v) for v in strict_loads(home_path.read_bytes())["position_rad"]]
                    self.get_logger().info(f"tasks end at the home pose in {home_path}: {[round(v, 3) for v in self._home_joints]}")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.get_logger().warning(f"no home pose ({exc}); tasks end wherever they stop")
            self._closed_empty = None                       # the gripper's stop, measured before the first task
            self._held_after_task = None                    # what the last task left in the hand
            runtime = self._compose_runtime(context_path)
            for name, why in sorted(runtime.backend.declared_gaps.items()):
                self.get_logger().warning(f"declared capability {name} is a known gap: {why}")
            for skill, why in sorted(runtime.registry.unavailable.items()):
                self.get_logger().info(f"skill {skill} unavailable: {why}")
            return runtime

        def _compose_runtime(self, context, *, constraints=None):
            """One world per task; the scene's role observers follow the world."""
            from .app import sheppy_runtime
            runtime = sheppy_runtime(context, constraints=constraints, **self._sheppy_options)
            if self.scene is not None:
                self.scene.bind_world(runtime.world)
            runtime.executor.confirmation_callback = self._profile_confirmation(runtime)
            runtime.backend.log = self.get_logger().info
            runtime.backend.model_progress_check = bool(self.get_parameter("model_progress_check").value)
            runtime.backend.compliant_pull = bool(self.get_parameter("compliant_contact").value)
            runtime.backend.home_joints = getattr(self, "_home_joints", None)
            # Scenes written here reach the planner container by the same path (the manifest mounts it).
            world_dir = Path(self.get_parameter("planner_world_dir").value or ".")
            if self.get_parameter("planner_world_dir").value and world_dir.is_dir() and os.access(world_dir, os.W_OK):
                runtime.backend.planner_world_dir = str(world_dir)
            if getattr(self, "_closed_empty", None) is not None:
                runtime.backend.closed_empty_knuckle_rad = self._closed_empty
            runtime.executor.decider, runtime.executor.stop_threshold = self._jev_decider(runtime)
            if self.get_parameter("record_dir").value:
                record_dir = Path(self.get_parameter("record_dir").value)
                runtime.backend.record_root = record_dir if record_dir.is_absolute() else Path(runtime.catalog.root)/record_dir
            if self.scene is not None:
                runtime.executor.replan_images = self._replan_images
            return runtime

        def _profile_confirmation(self, runtime):
            """Per-node assent for physical profiles.

            The catalog asks for assent before profile-policy skills. On this
            bench the operator's arming (commissioned and arm_motion) is that
            assent; each grant is bound to the node's action digest and logged
            so the record shows exactly what was authorised and when.
            """
            logger = self.get_logger()
            armed = self.client is not None and self.client.motion_enabled

            def confirm(plan, node, digest):
                if not armed:
                    raise ContractError("motion is not armed; nothing is confirmed")
                grant = runtime.executor.confirmations.grant(task_id=plan["task_id"], epoch=plan["execution_epoch"],
                                                             node_id=node["id"], digest=digest, ttl_s=30., accepted=True)
                logger.info(f"confirmed {node['skill']} node {node['id']} under operator arming; digest {digest[:16]}")
                return grant.confirmation_id
            return confirm

        def _replan_images(self):
            """The latest screened keyframe, so a replan sees the scene as it is."""
            crop = self.scene.latest_crop()
            return [crop] if crop is not None else []

        def _feed_scene(self, pair):
            scene = self.scene
            if scene is None:
                return
            now = time.monotonic()
            if now-self._last_detect < 1./self._detect_hz:
                return
            self._last_detect = now
            try:
                scene.on_pair(pair)
            except Exception as exc:                        # noqa: BLE001 - a frame, not the node
                if now-self._last_scene_warning > 10.:
                    self._last_scene_warning = now
                    self.get_logger().warning(f"keyframe feed failed: {exc}")

        def _log_stops(self, runtime):
            """Name every stop request in the ROS log; the supervisor itself is silent."""
            safety = runtime.executor.safety
            original = safety.request_stop
            logger = self.get_logger()

            async def logged(reason, *, fault=False):
                logger.warning(f"stop requested: {reason} fault={fault}")
                return await original(reason, fault=fault)
            safety.request_stop = logged

        def _local_first(self, reasoner):
            """Discovery and grounding answered by the local detector when it can, Astra otherwise."""
            detector = getattr(self, "_detector", None)
            if detector is None or reasoner is None:
                return reasoner
            from .perception.local_discovery import LocalFirstReasoner
            return LocalFirstReasoner(reasoner, detector, log=self.get_logger().info)

        def _install_runtime(self, runtime, reasoner):
            self._log_stops(runtime)
            previous = self.bridge
            self.runtime = runtime
            self.bridge = RuntimeBridge(runtime, reasoner)
            if self.scene is not None:
                self.scene.reasoner = self._local_first(reasoner)
            old = getattr(previous, "reasoner", None)
            if old is not None and old is not reasoner:
                asyncio.ensure_future(old.close())

        def _refresh_tick(self):
            """Idle-time semantic refresh: one unsent keyframe with a moved camera or changed scene."""
            if self._closing.is_set() or self.scene is None or self.bridge.reasoner is None:
                return
            if self._refresh_future is not None and not self._refresh_future.done():
                return
            with self._goal_lock:
                busy = self._goal_count > 0 or self._intake_phase is not None
            if busy or self.scene.pending_refresh() is None:
                return
            now = time.monotonic()
            if now-self._last_refresh_at < float(self.get_parameter("scene_refresh_interval_s").value):
                return
            self._last_refresh_at = now
            self._refresh_future = self.worker.submit(self._refresh_scene())

        async def _refresh_scene(self):
            from .perception.grounded_scene import SceneError
            context = dict(self.runtime.world.snapshot().context)
            context["task_id"] = "scene-refresh-"+time.strftime("%Y%m%dT%H", time.gmtime())
            try:
                found = await self.scene.refresh(self.bridge.reasoner, context)
            except SceneError as exc:
                now = time.monotonic()
                if now-self._last_scene_warning > 30.:
                    self._last_scene_warning = now
                    self.get_logger().warning(f"scene refresh withheld: {exc}")
                return
            except Exception as exc:                        # noqa: BLE001 - reported, retried next tick
                self.get_logger().warning(f"scene refresh failed: {exc}")
                return
            if found is not None:
                self.get_logger().info(f"scene refresh ({found['reason']}, {found['status']}): {found['entities']}")

        async def _wait_for_refresh(self):
            """An idle keyframe refresh may hold the one cloud lane; a typed task waits for it, not against it."""
            pending = self._refresh_future
            if pending is None or pending.done():
                return
            self._intake_phase = "INTAKE_WAITING_FOR_REFRESH"
            try:
                await asyncio.wait_for(asyncio.wrap_future(pending), timeout=float(self.runtime.executor._reasoning_deadline_s))
            except Exception:                               # noqa: BLE001 - the refresh reports itself
                pass

        async def _articulate(self, task_id):
            """A constraint record for every discovered handle; a part without one simply cannot be followed."""
            from .perception.grounded_scene import SceneError
            articulations = []
            for handle in self.scene.handles():
                self._intake_phase = "INTAKE_PROPOSING_CONSTRAINT"
                try:
                    reasoner = self.bridge.reasoner
                    decider = getattr(self.runtime.executor, "decider", None)
                    if decider is not None:
                        from .decisions import JevConstraintReasoner
                        reasoner = JevConstraintReasoner(reasoner, decider, log=self.get_logger().info,
                                                         threshold=getattr(self.runtime.executor, "goal_threshold", .8))
                    found = await self.scene.articulate(reasoner, self.runtime.world.snapshot().context,
                                                        handle["entity_id"], store=self.constraint_store)
                except SceneError as exc:
                    self.get_logger().warning(f"no constraint for {handle['entity_id']}: {exc}")
                    continue
                articulations.append(found)
                shown = {k: found["record"].get(k) for k in ("constraint_id", "kind", "hinge_side", "opening", "door_width_m",
                                                             "contact_effort_nm", "parameters_version", "settled_locally",
                                                             "measured_door")}
                self.get_logger().info(f"task {task_id}: constraint {json.dumps(shown)}; proposal: {found['proposal'].get('rationale', '')[:160]}")
            return articulations

        def _carry_moved_parts(self, task_id, runtime, articulations, seen):
            """Handles not seen now whose part an earlier task left moved: their poses carried to where it was left.

            Memory alone is not believed: the depth where the handle was measured must no longer show it. The
            close view at the standoff then re-measures the part, and the grip measures how far it really is.
            """
            from .constraints import left_moved
            self.scene.carried.clear()
            carried = []
            for item in articulations:
                record, handle = item["record"], item["handle"]
                state = left_moved(record)
                if handle in seen or state is None:
                    continue
                place = self.scene.left_its_place(handle)
                if not place["gone"]:
                    self.get_logger().info(f"task {task_id}: the {record['label']} was left at {state['at']:.2f} {record['unit']}, "
                                           f"but {handle} is not seen gone from where it was measured ({place['detail']}); not carried")
                    continue
                runtime.backend.inherit_part_state(record["constraint_id"], state)
                self.scene.carried[handle] = {"constraint_id": record["constraint_id"], "at": state["at"], "unit": record["unit"],
                                              "left_by": state.get("task_id"), "place": place["detail"]}
                carried.append(handle)
                self.get_logger().info(f"task {task_id}: {handle} not seen; the {record['label']} was left at {state['at']:.2f} "
                                       f"{record['unit']} by {state.get('task_id')} and the depth where it was measured changed "
                                       f"({place['detail']}); its poses carried there")
            return carried

        async def _intake_and_run(self, task_text, skill=None):
            """Typed task, begun and ended with the arm at home (_prepare_for_task, _finish_at_home).

            With skill (a learned skill's source or library name), intake builds the task's world as for any
            task, and the skill runs in the jail against it instead of a plan (_run_learned).
            """
            outcome = await self._finish_at_home(await self._typed_task(task_text, skill=skill))
            self._held_after_task = getattr(self.runtime.backend, "holding_id", None)
            return outcome

        async def _prepare_for_task(self, task_id):
            """Before anything else: the arm at the exact home joints and the hand open (prepare_for_task)."""
            backend, safety = self.runtime.backend, self.runtime.executor.safety
            if getattr(backend, "home_joints", None) is None or not hasattr(backend, "prepare_for_task"):
                return None
            if safety.fault_latched:
                raise ContractError("a fault is latched; the arm is not moved")
            if safety.stop_requested.is_set():
                await safety.reset()
            self._intake_phase = "PREPARING"
            report = await backend.prepare_for_task(keep_grip=getattr(self, "_held_after_task", None) is not None,
                                                    measure_gripper=getattr(self, "_closed_empty", None) is None)
            if report.get("measured_stop_rad") is not None:
                self._closed_empty = report["measured_stop_rad"]
            self.get_logger().info(f"task {task_id}: ready at home ({'; '.join(report['done'])})")
            if report.get("moved"):
                from .perception.grounded_scene import SceneError
                try:                                        # discovery sees the scene from home, not from where the arm was
                    await self.scene.wait_for_keyframe(timeout_s=3., max_age_s=.5)
                except SceneError:
                    pass
            return report

        async def _finish_at_home(self, outcome):
            """The typed task's end: the arm home (ros_bridge.finish_at_home), shown as RETURNING_HOME."""
            from .ros_bridge import finish_at_home
            self._intake_phase = "RETURNING_HOME"
            try:
                return await finish_at_home(outcome, backend=self.runtime.backend, safety=self.runtime.executor.safety,
                                            closing=self._closing.is_set(), log=self.get_logger().info)
            finally:
                self._intake_phase = None

        def _skill_library(self):
            from .learned import SkillLibrary
            folder = Path(self.get_parameter("skill_library_dir").value)
            library = SkillLibrary(folder if folder.is_absolute() else Path(self.runtime.catalog.root)/folder)
            installed = library.install_seeds(Path(self.runtime.catalog.root)/"skills/learned_seeds")
            if installed:
                self.get_logger().info(f"skill library: seeds installed {installed}")
            return library

        async def _run_learned(self, runtime, task_id, goal, skill):
            """Run a learned skill against the task's world: each motion call an admitted, guarded plan.

            dry_run moves nothing (the validator admits the skill's steps as one chain). With a goal the task
            could be bound to, success is the goal measured true afterwards, and a library skill's record is
            updated with it; without one, the skill's own completion is all there is to report.
            """
            from .learned import run_skill
            from .learned.host import DryRunHost, ExecutorHost
            log = self.get_logger()
            library = self._skill_library()
            version = None
            if skill.get("library"):
                loaded = library.load(skill["library"], skill.get("version"))
                name, source, version = skill["library"], loaded["source"], loaded["version"]
            else:
                name, source = skill.get("name") or "skill", skill.get("source") or ""
            dry = bool(skill.get("dry_run"))
            self._skill_cancel = asyncio.Event()
            host = (DryRunHost if dry else ExecutorHost)(runtime, cancel=self._skill_cancel, log=log.info)
            self._intake_phase = "REHEARSING_SKILL" if dry else "RUNNING_SKILL"
            log.info(f"task {task_id}: {'rehearsing' if dry else 'running'} learned skill {name}"
                     + (f" v{version}" if version else "")+f" with {json.dumps(skill.get('args') or {})}")
            try:
                run = await run_skill(source, skill.get("args") or {}, host, name=name, library=library)
            finally:
                self._skill_cancel = None
            for line in run.logs:
                log.info(f"skill {name}: {line}")
            met = None if dry or goal is None else bool(runtime.world.goal_satisfied())
            if version is not None and met is not None:
                status = library.record_outcome(name, version, verified=met and run.status == "succeeded",
                                                evidence={"task_id": task_id, "task": skill.get("task_text", ""), "run": run.status})
                log.info(f"skill {name} v{version}: {'verified' if met else 'not verified'} by the goal; now {status}")
            ok = run.status == "succeeded" and met is not False
            reason = (run.error or ("the skill finished" if met is None else "the goal was measured met" if met else
                                    "the skill finished but the goal was not met"))[:512]
            return {"task_id": task_id, "status": "succeeded" if ok else "cancelled" if run.status == "aborted" and
                    "cancel" in run.error else "incomplete", "reason": reason, "nodes": [], "task_replans": 0,
                    "simulation_only": False, "goal": goal, "skill": {**run.to_dict(), "version": version, "dry_run": dry,
                                                                      "steps": getattr(host, "steps", [])}}

        async def _learn(self, runtime, task_id, task_text, goal, spec, before_keyframe=None):
            """Voyager's loop for this task: a model writes a skill, it is gated, rehearsed, run and verified; kept if so."""
            from .learned import run_skill
            from .learned.host import DryRunHost, ExecutorHost
            from .learned.loop import ModelWriter, learn_skill
            log = self.get_logger()
            config = strict_loads((Path(runtime.catalog.root)/"config/learning.json").read_bytes())
            library = self._skill_library()
            writer = ModelWriter(config)
            self._skill_cancel = asyncio.Event()
            cancel = self._skill_cancel

            async def rehearse(source):
                self._intake_phase = "REHEARSING_SKILL"
                return await run_skill(source, {}, DryRunHost(runtime, cancel=cancel), library=library)

            async def execute(source):
                self._intake_phase = "RUNNING_SKILL"
                return await run_skill(source, {}, ExecutorHost(runtime, cancel=cancel, log=log.info), library=library)

            async def verify(run):
                if goal is not None:
                    met = bool(runtime.world.goal_satisfied())
                    return met, f"the goal {json.dumps(goal['args'])} is {'measured met' if met else 'not met'}"
                return await self._judge_from_home(runtime, task_text, before_keyframe)

            async def choose(task, candidates):
                """Jev scores the rehearsed candidates at once; the writer's first when it is unsure."""
                decider = getattr(runtime.executor, "decider", None)
                if decider is None or not getattr(decider, "available", False):
                    return 0
                decision = await decider.choose(f"Request to a robot arm: {task}", question="skill",
                                                instructions="Which skill is most likely to do the request on the first try, safely?",
                                                options={str(i): f"{name}: {description}"[:300] for i, (name, description) in enumerate(candidates)})
                return int(decision.choice) if decision.choice is not None and decision.confidence >= .6 else 0
            self._intake_phase = "WRITING_SKILL"
            try:
                story = await learn_skill(task_text, writer=writer, library=library, scene=DryRunHost(runtime).scene(),
                                          rehearse=rehearse, execute=None if spec.get("rehearse_only") else execute,
                                          verify=verify, choose=choose, max_rounds=int(spec.get("rounds") or config["max_rounds"]),
                                          examples=int(config["examples"]), candidates=int(config.get("candidates", 3)), log=log.info)
            finally:
                self._skill_cancel = None
            story["writer_requests"] = writer.requests
            ok = story["status"] in ("learned", "rehearsed", "kept_unverified")
            last = story["rounds"][-1] if story["rounds"] else {}
            reason = (f"{story['status']}: {story.get('name')} after {len(story['rounds'])} rounds" if ok else
                      f"no skill worked in {len(story['rounds'])} rounds; last: {last.get('stage')}: {str(last.get('error'))[:300]}")
            return {"task_id": task_id, "status": "succeeded" if ok else "incomplete", "reason": reason[:512], "nodes": [],
                    "task_replans": 0, "simulation_only": False, "goal": goal,
                    "learning": {**story, "rounds": [{k: v for k, v in r.items() if k != "source"} for r in story["rounds"]]}}

        async def _propose(self, task_id, seen, count):
            """The curriculum: practice tasks for what the robot sees, aimed at what its library cannot do yet. Moves nothing."""
            from .learned.loop import ModelWriter, propose_tasks
            config = strict_loads((Path(self.runtime.catalog.root)/"config/learning.json").read_bytes())
            scene = [{"id": entity_id, **details} for entity_id, details in seen.items()]
            self._intake_phase = "PROPOSING_PRACTICE"
            proposals = await propose_tasks(ModelWriter(config), self._skill_library(), scene, count=count)
            for proposal in proposals:
                self.get_logger().info(f"practice: {proposal['task']} (why: {proposal['why'][:120]}; undo: {proposal['undo'][:80]})")
            return {"task_id": task_id, "status": "succeeded", "reason": f"{len(proposals)} practice tasks proposed", "nodes": [],
                    "task_replans": 0, "simulation_only": False, "goal": None, "proposals": proposals}

        async def _judge_from_home(self, runtime, task_text, before_keyframe):
            """The verifier RSIAgent keeps apart from the actor: the scene from home before and after, judged by a model
            that sees the request and the two views, never the skill's code. Done at 4 of 4 with confidence 0.7."""
            reasoner, scene = self.bridge.reasoner, self.scene
            if reasoner is None or scene is None:
                return None, "no measurable goal and no judge"
            try:
                await runtime.backend.return_home()
                after = await scene.wait_for_keyframe(timeout_s=3., max_age_s=.5)
                before = scene.keyframe_by_id(before_keyframe) if before_keyframe else None
                images = [await asyncio.to_thread(scene.crop_for, frame) for frame in (before, after) if frame is not None]
                judged = await reasoner.verify_progress(
                    runtime.world.snapshot().context, task_text=task_text, images=images,
                    rubric=["0: nothing the request asks for has happened", "1: started", "2: about half done",
                            "3: nearly done", "4: done as asked"],
                    question="The first image is before, the last after the robot acted. How much of the request is done?")
            except Exception as exc:                        # noqa: BLE001 - an unjudged run is reported as such
                return None, f"could not be judged: {type(exc).__name__}: {str(exc)[:120]}"
            if judged.status != "OK" or judged.proposal is None:
                return None, f"the judge declined: {judged.status}"
            score, confidence = judged.proposal["score"], judged.proposal["confidence"]
            return score == 4 and confidence >= .7, f"judged {score}/4 at {confidence:.2f} from the views before and after: {judged.detail[:160]}"

        async def _typed_task(self, task_text, skill=None):
            """Typed task: find the target, model what moves, build the task's world, settle the goal, run."""
            from .app import astra_for
            from .intake import (IntakeError, draft_context, new_task_id, normalize_task, search_until_visible, seed_articulation,
                                 seed_observations, seed_visibility)
            from .perception.grounded_scene import SceneError
            from .sheppy_backend import bootstrap_robot_facts
            if self.scene is None:
                raise ContractError("no scene is configured; a typed task has nothing to ground")
            if not self.get_parameter("enable_astra").value or self.bridge.reasoner is None:
                raise ContractError("Astra is not enabled; a typed task cannot be normalized")
            task_id, log = new_task_id(), self.get_logger()

            def declined(status, detail, visible=()):
                return {"task_id": task_id, "status": "incomplete", "reason": f"{status}: {detail}", "nodes": [],
                        "task_replans": 0, "simulation_only": False, "goal": None, "visible_entities": sorted(visible)}

            def searching(message):
                self._intake_phase = "INTAKE_SEARCHING"
                log.info(f"task {task_id}: {message}")
            try:
                try:
                    await self._prepare_for_task(task_id)
                except Exception as exc:                    # noqa: BLE001 - reported as the task's outcome
                    return declined("NOT_READY", f"the arm could not be made ready at home: {exc}")
                await self._wait_for_refresh()
                self._intake_phase = "INTAKE_DISCOVERING"
                try:
                    found = await search_until_visible(self.scene, self.runtime.backend, self._local_first(self.bridge.reasoner),
                                                       self.runtime.world.snapshot().context, task_text,
                                                       max_viewpoints=int(self.get_parameter("max_viewpoints").value), log=searching)
                except (IntakeError, SceneError) as exc:
                    return declined(exc.status, exc.detail)
                if found["viewpoints"]:
                    log.info(f"task {task_id}: found the target after looking {[m['hint'] for m in found['viewpoints']]}")
                seen = {e: {"kind": r["kind"], "attached_to": r["attached_to"], "grasp": (r["grasp"] or {}).get("strategy"),
                            "surface_width_m": None if not r.get("surface_geometry") else round(r["surface_geometry"]["width_m"], 3)}
                        for e, r in self.scene.entities.items() if e in found["entities"]}
                log.info(f"task {task_id}: discovery {found['status']} in keyframe {found['keyframe']} (saved under artifacts/keyframes): {json.dumps(seen)}")
                if skill is not None and skill.get("propose"):
                    return await self._propose(task_id, seen, int(skill.get("count") or 3))
                articulations = await self._articulate(task_id)
                descriptors = self.scene.descriptors()
                known = {d["entity_id"] for d in descriptors}
                descriptors += [a["surface"] for a in articulations if a["surface"]["entity_id"] not in known]
                base = strict_loads(Path(self._base_context_path).read_bytes())
                context = draft_context(base, descriptors, task_id=task_id, camera_id=self.scene.camera_id,
                                        constraints=[a["constraint"] for a in articulations])
                self._intake_phase = "INTAKE_BOOTSTRAP"
                runtime = self._compose_runtime(context, constraints={a["record"]["constraint_id"]: a["record"] for a in articulations})
                reasoner = astra_for(runtime)
                self.scene.reasoner = self._local_first(reasoner)
                try:
                    closed = getattr(self, "_closed_empty", None)
                    bootstrap = await bootstrap_robot_facts(runtime.world, self.client,
                                                            **({} if closed is None else {"closed_knuckle_rad": closed-.03}))
                    visibility = seed_visibility(runtime.world, self.scene, now=runtime.world.clock())
                    seed_articulation(runtime.world, articulations, now=runtime.world.clock())
                    carried = self._carry_moved_parts(task_id, runtime, articulations, found["entities"])
                    self._intake_phase = "INTAKE_OBSERVING"
                    seeded = await seed_observations(runtime, visibility["visible"]+carried)
                    log.info(f"task {task_id}: poses observed for {seeded['observed']}"
                             + (f"; skipped {seeded['skipped']}" if seeded["skipped"] else ""))
                    self._intake_phase = "INTAKE_NORMALIZING_GOAL"
                    try:
                        goal = await normalize_task(reasoner, runtime.catalog, runtime.world.snapshot().context,
                                                    task_text, available_skills=runtime.registry.available_skills,
                                                    decider=runtime.executor.decider, log=log.info,
                                                    threshold=getattr(runtime.executor, "goal_threshold", .8))
                    except IntakeError as exc:
                        if skill is None:
                            raise
                        goal = None                         # a learned skill may do what no goal predicate names
                        log.info(f"task {task_id}: no measurable goal ({exc.status}: {exc.detail[:120]}); the skill's result stands")
                    if goal is not None:
                        runtime.world.replace_goal(goal)
                except BaseException as exc:
                    # The task's runtime is abandoned: its reasoner closes and the scene answers to the standing one again.
                    await reasoner.close()
                    self.scene.reasoner = self.bridge.reasoner
                    self.scene.bind_world(self.runtime.world)
                    if isinstance(exc, IntakeError):
                        return declined(exc.status, exc.detail, found["entities"])
                    raise
                self._install_runtime(runtime, reasoner)
                log.info(f"task {task_id}: goal {json.dumps(goal)}; visible {visibility['visible']}")
                if skill is not None and skill.get("learn"):
                    outcome = await self._learn(runtime, task_id, task_text, goal, skill, before_keyframe=found["keyframe"])
                    return {"visible_entities": visibility["visible"], "bootstrap": bootstrap["facts"], **outcome}
                if skill is not None:
                    outcome = await self._run_learned(runtime, task_id, goal, {**skill, "task_text": task_text})
                    return {"visible_entities": visibility["visible"], "bootstrap": bootstrap["facts"], **outcome}
                first_plan = await self._template_plan(runtime, task_text, goal, articulations)
            finally:
                self._intake_phase = None
            result = await self.bridge.execute_task(task_id=task_id, task_text=task_text, first_plan=first_plan)
            return {"goal": goal, "visible_entities": visibility["visible"], "bootstrap": bootstrap["facts"],
                    **result.to_dict()}

        def _bootstrap_robot(self):
            """Assert held/empty from measurement once joint state is flowing."""
            from .sheppy_backend import bootstrap_robot_facts
            if self.client.live_joints() is None:
                return
            try:
                result = self.worker.call(bootstrap_robot_facts(self.runtime.world, self.client), timeout=5.)
                self.get_logger().info(f"robot facts bootstrapped: {result['facts']}")
            except Exception as exc:                        # noqa: BLE001 - retried by the timer
                self.get_logger().warning(f"robot bootstrap not yet possible: {exc}")
                return
            self._bootstrap_timer.cancel()

        def world_message(self):
            snapshot = self.runtime.world.snapshot()
            context = snapshot.context
            message = WorldState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "base_link"
            for field in ("snapshot_id", "revision", "collision_revision", "base_epoch", "calibration_id", "robot_config_id", "attachment_id", "grasp_state_id"):
                setattr(message, field, context[field])
            message.context_json = json.dumps(context, allow_nan=False)
            return message

        def publish_state(self):
            if self._closing.is_set():
                return
            executor = self.runtime.executor
            safety = executor.safety

            def status_for(quiescent):
                return {"execution_epoch": self.runtime.world.snapshot().execution_epoch,
                        "motion_allowed": not safety.fault_latched and not safety.stop_requested.is_set(),
                        "held_verified": bool(quiescent) and not executor._issued_motion,
                        "fault_latched": safety.fault_latched, "reason_code": safety.reason}

            async def measured_status():
                safety.check_liveness()
                return status_for(await executor._measured_quiescence())
            direct = getattr(executor.backend, "quiescent_now", None)
            try:
                if callable(direct):
                    # The client backend answers from measured joint state on
                    # this thread; the runtime loop's load is irrelevant to it.
                    safety.check_liveness()
                    status = status_for(direct())
                else:
                    status = self.worker.call(measured_status(), timeout=.2)
                self._status_misses = 0
            except Exception as exc:
                # A late answer from the loop is the loop being busy with
                # legitimate work; only a state that stays unreadable is a fault.
                self._status_misses += 1
                if self._status_misses in (1, 10):
                    self.get_logger().warning(f"status tick miss {self._status_misses}: {type(exc).__name__}: {exc}")
                status = {"execution_epoch": self.runtime.world.snapshot().execution_epoch,
                          "motion_allowed": False, "held_verified": False,
                          "fault_latched": self._status_misses >= 10, "reason_code": "STATE_UNAVAILABLE"}
                if (self._status_misses >= 10 and not self._closing.is_set()
                        and (self._status_stop is None or self._status_stop.done())):
                    try:
                        self._status_stop = self.worker.submit(self.runtime.executor._stop_safely("STATE_UNAVAILABLE", fault=True))
                    except RuntimeError:
                        pass  # Closing won the race; publish unavailable state.
            summary = (status["reason_code"], bool(status["fault_latched"]), bool(status["motion_allowed"]))
            if summary != self._last_status_summary:
                self._last_status_summary = summary
                self.get_logger().info(f"safety status: reason={summary[0]!r} fault_latched={summary[1]} motion_allowed={summary[2]}")
            message = SafetyStatus()
            message.header.stamp = self.get_clock().now().to_msg()
            message.execution_epoch = status["execution_epoch"]
            self._safety_revision += 1
            message.safety_revision = self._safety_revision
            for field in ("motion_allowed", "held_verified", "fault_latched", "reason_code"):
                setattr(message, field, status[field])
            self.safety_publisher.publish(message)
            self.world_publisher.publish(self.world_message())

        def snapshot(self, request, response):
            if request.task_id != self.runtime.world.snapshot().context["task_id"]:
                response.available, response.error_code = False, "UNKNOWN_TASK"
            else:
                response.available, response.snapshot = True, self.world_message()
            return response

        def resolve(self, request, response):
            snapshot = self.runtime.world.snapshot()
            if request.snapshot_id != snapshot.snapshot_id:
                response.status, response.detail = response.STALE, "Snapshot no longer current"
                return response
            pose = snapshot.metric_poses.get((request.entity_id, request.pose_role))
            if pose is None:
                response.status, response.detail = response.UNKNOWN, "No fresh locally fitted metric pose; symbolic fixtures are not metric measurements"
                return response
            context, resolved = snapshot.context, response.resolved
            try:
                capture_ns, expiry_ns = self._clock_mapping.capture_and_expiry(pose.captured_at, pose.valid_for_s,
                    monotonic_now=self.runtime.world.clock(), ros_now_ns=self.get_clock().now().nanoseconds)
            except ContractError as exc:
                response.status, response.detail = response.STALE, str(exc)
                return response
            resolved.entity_id, resolved.entity_revision = pose.entity_id, pose.entity_revision
            resolved.pose_role, resolved.orientation_valid = pose.pose_role, True
            from builtin_interfaces.msg import Time
            resolved.pose.header.stamp = Time(sec=int(capture_ns//1_000_000_000), nanosec=int(capture_ns%1_000_000_000))
            resolved.pose.header.frame_id = pose.frame_id
            for field, value in zip(("x", "y", "z"), pose.position_m):
                setattr(resolved.pose.pose.pose.position, field, float(value))
            for field, value in zip(("x", "y", "z", "w"), pose.orientation_xyzw):
                setattr(resolved.pose.pose.pose.orientation, field, float(value))
            resolved.pose.pose.covariance = [float(value) for value in pose.covariance]
            for field in ("snapshot_id", "calibration_id", "base_epoch", "robot_config_id", "attachment_id", "grasp_state_id"):
                setattr(resolved, field, context[field])
            resolved.valid_until = Time(sec=int(expiry_ns//1_000_000_000), nanosec=int(expiry_ns%1_000_000_000))
            response.status = response.RESOLVED
            return response

        def commit(self, request, response):
            try:
                if request.task_id != self.runtime.world.snapshot().context["task_id"]:
                    raise ContractError("Commit belongs to another task")
                # Remote JSON cannot mint evidence or register an in-flight operation.
                effects = [{"predicate": e.predicate, "args": strict_loads(e.bound_args_json),
                            "validity": {0: "unknown", 1: "false", 2: "true"}[e.validity],
                            "evidence_id": e.evidence_id} for e in request.effects]
                # Lookup only: the executor is the sole effect committer. Remote
                # callers can retrieve an exact existing receipt, never race or
                # repeat a local in-flight commit.
                receipt = self.runtime.world.committed_receipt(request.operation_id, effects, request.execution_epoch)
                response.committed, response.assigned_revision = True, receipt.revision
                response.receipt_id = receipt.receipt_id
            except Exception as exc:
                response.committed, response.error_code = False, str(exc)
            return response

        def validate(self, request, response):
            try:
                receipt = self.worker.call(self.bridge.validate(request.plan_json, snapshot_id=request.snapshot_id,
                    epoch=request.execution_epoch, node_id=request.dispatch_node_id))
                response.valid, response.validation_id = True, receipt.receipt_id
                response.checked_snapshot_id = receipt.snapshot_id
                response.execution_epoch = receipt.execution_epoch
            except Exception as exc:
                response.valid, response.error_codes, response.detail = False, ["INVALID_PLAN"], str(exc)
            return response

        def reason(self, request, response):
            response.request_id, response.execution_epoch = request.request_id, request.execution_epoch
            response.model_id = "gpt-6-astra"
            try:
                available = list(request.available_skill_ids)
                if len(available) != len(set(available)) or set(available) != set(self.runtime.registry.available_skills):
                    raise ContractError("Requested available skills do not match the current implemented registry")
                result = self.worker.call(self.bridge.generate_plan({"request_id": request.request_id,
                    "task_id": request.task_id, "epoch": request.execution_epoch,
                    "context": request.world_context_json, "catalog_hash": request.skill_library_hash,
                    "task_text": request.task_text, "image_ids": [image.image_id for image in request.images],
                    "feedback": strict_loads(request.feedback_json) if request.feedback_json else None}))
                response.status = getattr(response, result.status, response.INVALID_OUTPUT)
                response.detail = result.detail
                response.plan_json = json.dumps(result.plan, allow_nan=False) if result.plan else ""
            except Exception as exc:
                response.status, response.detail = response.UNAVAILABLE, str(exc)
            return response

        def ground(self, request, response):
            response.request_id, response.execution_epoch = request.request_id, request.execution_epoch
            try:
                result = self.worker.call(self.bridge.ground_target({
                    "request_id": request.request_id, "epoch": request.execution_epoch,
                    "entity_id": request.entity_id, "image_ids": [image.image_id for image in request.images],
                    "query": request.query}))
                response.status = getattr(response, result.status, response.INVALID_OUTPUT)
                response.candidates_json = json.dumps(result.candidates, allow_nan=False)
                response.detail = result.detail
            except Exception as exc:
                response.status, response.detail = response.UNAVAILABLE, str(exc)
            return response

        def stop(self, request, response):
            try:
                response.execution_epoch = self.worker.call(self.bridge.stop(request.reason_code), timeout=5.)
                response.accepted = True
            except Exception:
                response.accepted = False
            return response

        def confirm(self, request, response):
            # An incoming request is not the user's response. The deferred human
            # skills cannot run until an explicit local HRI provider is connected.
            response.accepted = False
            response.action_digest = request.action_digest
            response.responded_at = self.get_clock().now().to_msg()
            response.reason = "No local human-assent provider configured"
            return response

        def accept_goal(self, request):
            with self._goal_lock:
                if self._closing.is_set() or self._goal_count >= 4:
                    return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        def enter_goal(self):
            # Count actual callbacks, not goals whose acceptance response might
            # fail before ROS ever invokes execute_callback.
            with self._goal_lock:
                if self._closing.is_set() or self._goal_count >= 4:
                    return False
                self._goal_count += 1
                return True

        def cancel_goal(self, handle):
            if self._closing.is_set():
                return CancelResponse.REJECT
            running = getattr(self, "_skill_cancel", None)
            if running is not None:
                self.worker.loop.call_soon_threadsafe(running.set)
            future = self.worker.submit(self.bridge.stop("USER_CANCELLED"))
            future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            return CancelResponse.ACCEPT

        def execute_skill(self, handle):
            response = ExecuteSkill.Result()
            if not self.enter_goal():
                response.status, response.failure_code = response.REJECTED, "SERVER_BUSY_OR_CLOSING"
                handle.abort()
                return response
            try:
                request = handle.request
                if handle.is_cancel_requested:
                    response.status, response.failure_code = response.CANCELLED, "cancelled"
                    handle.canceled()
                    return response
                result = self.worker.call(self.bridge.execute_skill({"task_id": request.task_id,
                    "execution_epoch": request.execution_epoch, "node_id": request.node_id,
                    "attempt": request.attempt, "skill_id": request.skill_id,
                    "args": strict_loads(request.args_json), "validation_id": request.validation_id}), timeout=min(180., self.task_timeout_s))
                response.status = {"succeeded": response.SUCCESS, "failed": response.FAILED, "cancelled": response.CANCELLED}[result.status]
                response.failure_code, response.outputs_json = result.failure_code, json.dumps(result.outputs)
                response.evidence_id = result.outputs.get("evidence_id", "")
                response.backend_quiescent = result.backend_quiescent
                # Core already committed the effects; do not propose them twice.
                if handle.is_cancel_requested:
                    handle.canceled()
                elif result.status == "succeeded":
                    handle.succeed()
                else:
                    handle.abort()
            except Exception as exc:
                response.status, response.failure_code = response.REJECTED, str(exc)
                if not self._closing.is_set():
                    self.worker.submit(self.bridge.stop("EXECUTION_TRANSPORT_FAILED"))
                handle.abort()
            finally:
                with self._goal_lock:
                    self._goal_count -= 1
            return response

        def execute_task(self, handle):
            response = ExecuteTask.Result()
            if not self.enter_goal():
                response.status, response.result_json = "rejected", json.dumps({"detail": "SERVER_BUSY_OR_CLOSING"})
                handle.abort()
                return response
            future = None
            try:
                request = handle.request
                if handle.is_cancel_requested:
                    response.status, response.result_json = "cancelled", "{}"
                    handle.canceled()
                    return response
                if request.task_id:
                    future = self.worker.submit(self.bridge.execute_task(task_id=request.task_id, task_text=request.task_text,
                        plan=strict_loads(request.plan_json) if request.plan_json else None))
                else:
                    skill = None
                    if request.plan_json:
                        carried = strict_loads(request.plan_json)
                        if not isinstance(carried, dict) or not isinstance(carried.get("learned_skill"), dict):
                            raise ContractError("a typed task carries no plan; only {\"learned_skill\": {...}}")
                        skill = carried["learned_skill"]
                        if not (skill.get("source") or skill.get("library") or skill.get("learn") or skill.get("propose")):
                            raise ContractError("a learned skill needs its source, a library name, learn or propose")
                    self._intake_phase = "INTAKE_STARTING"     # feedback must not show the previous task's state
                    future = self.worker.submit(self._intake_and_run(request.task_text, skill=skill))
                deadline = time.monotonic()+self.task_timeout_s
                # Poll only the feedback transport. Motion dispatch remains event driven.
                while not future.done():
                    if self._closing.is_set() or time.monotonic() >= deadline:
                        future.cancel()
                        if not self._closing.is_set():
                            self.worker.call(self.bridge.stop("TASK_TRANSPORT_DEADLINE"), timeout=5.)
                        raise RuntimeError("Task transport closed or exceeded its bounded deadline")
                    feedback = ExecuteTask.Feedback()
                    feedback.phase = self._intake_phase or ("STOPPING" if self.runtime.executor.safety.stop_requested.is_set() else "RUNNING")
                    feedback.completed_nodes = list(self.runtime.world.snapshot().context["completed_nodes"])
                    handle.publish_feedback(feedback)
                    try:
                        future.result(timeout=.1)
                    except FutureTimeout:
                        pass
                outcome = future.result()
                payload = outcome if isinstance(outcome, dict) else outcome.to_dict()
                response.status, response.result_json = payload["status"], json.dumps(payload, allow_nan=False)
                if handle.is_cancel_requested:
                    handle.canceled()
                elif payload["status"] == "succeeded":
                    handle.succeed()
                else:
                    handle.abort()
            except Exception as exc:
                if future is not None and not future.done():
                    future.cancel()
                response.status, response.result_json = "rejected", json.dumps({"detail": str(exc)})
                handle.abort()
            finally:
                with self._goal_lock:
                    self._goal_count -= 1
            return response

        def close(self):
            self._closing.set()
            self.timer.cancel()
            for timer in (self._bootstrap_timer, self._refresh_timer):
                if timer is not None:
                    timer.cancel()
            if self.scene is not None:
                try:
                    self.scene.stop()
                except Exception:                           # noqa: BLE001 - teardown must finish
                    pass
            for resource in (self.depth, self.client):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:                       # noqa: BLE001 - teardown must finish
                        pass
            try:
                self.worker.call(self.bridge.stop("SHUTDOWN"), timeout=5.)
            except Exception:
                self.runtime.executor.safety.held_verified = False
                self.runtime.executor.safety.fault_latched = True
                self.get_logger().error("Shutdown stop did not verify hold")
            finally:
                try:
                    if not self.worker.close():
                        self.runtime.executor.safety.held_verified = False
                        self.get_logger().error("Runtime worker did not drain; shutdown is unverified")
                finally:
                    self.skill_action.destroy()
                    self.task_action.destroy()
                    self.destroy_node()

    return AdlNode()


def main(args=None):
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as exc:
        raise RuntimeError("ROS 2 Humble and compiled rammp_adl_interfaces are required; use the Python simulation CLI otherwise") from exc
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=8)
    try:
        node = create_node()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.close()
        finally:
            executor.shutdown(timeout_sec=5.)
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
