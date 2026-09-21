"""The sheppy composition root and the reviewed imagery policy."""
import asyncio
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rammp_adl.app import astra_for, sheppy_runtime
from rammp_adl.contracts import ContractError
from rammp_adl.perception.geometry import PerceptionError
from rammp_adl.perception.ros_locality import deployment_policy, require_local_imagery
from rammp_adl.perception.ros_rgbd import RosRgbdSource
from rammp_adl.sheppy_backend import PROVIDED_CAPABILITIES
from test_sheppy_backend import FakeClient


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT/"config/sheppy-bench.context.json"
POLICY = ROOT/"config/imagery-locality.json"


class RuntimeCompositionTests(unittest.TestCase):
    def test_uncommissioned_runtime_registers_nothing_and_trusts_no_robot_fact(self):
        runtime = sheppy_runtime(CONTEXT, root=ROOT, client=FakeClient(),
                                 capabilities=PROVIDED_CAPABILITIES, commissioned=False)
        self.assertEqual(runtime.registry.available_skills, ())
        snapshot = runtime.world.snapshot()
        self.assertEqual(snapshot.context["available_skills"], [])
        self.assertEqual(snapshot.fact("held_state", {"robot_id": "robot"}), "unknown")
        self.assertIs(runtime.backend.world, runtime.world)
        self.assertFalse(runtime.validator.geometry.simulation_only)

    def test_commissioned_runtime_registers_declared_skills_into_the_world(self):
        runtime = sheppy_runtime(CONTEXT, root=ROOT, client=FakeClient(),
                                 capabilities=PROVIDED_CAPABILITIES | {"calibrated_grasp"},
                                 commissioned=True)
        self.assertIn("grasp", runtime.registry.available_skills)
        self.assertEqual(set(runtime.world.snapshot().context["available_skills"]),
                         set(runtime.registry.available_skills))
        self.assertEqual(runtime.backend.declared_gaps, {"calibrated_grasp":
                         "grasp success is a knuckle stall, not a calibrated grip model"})

    def test_profiles_default_to_the_context_and_must_be_physical(self):
        with self.assertRaises(ContractError):
            sheppy_runtime(CONTEXT, root=ROOT, client=FakeClient(), capabilities=(), commissioned=False,
                           profiles=[{"profile_id": "sim", "safety_class": "transit", "simulation_only": True}])
        runtime = sheppy_runtime(CONTEXT, root=ROOT, client=FakeClient(), capabilities=(), commissioned=False)
        self.assertEqual(set(runtime.backend.profiles), {p["profile_id"] for p in json.loads(CONTEXT.read_text())["profiles"]})
        self.assertIn("bench_contact", runtime.backend.profiles)


class CloudHoldTests(unittest.TestCase):
    """The model may be asked a question between skills, and by a skill that owns a still arm; never by a moving one."""

    def guard(self, *, active=(), still=True):
        client = FakeClient()
        client.still = still
        runtime = sheppy_runtime(CONTEXT, root=ROOT, client=client, capabilities=PROVIDED_CAPABILITIES, commissioned=True)
        runtime.backend.active.update(active)
        reasoner = astra_for(runtime, transport=object())
        snapshot = runtime.world.snapshot()
        return asyncio.run(reasoner._guard(snapshot.context["task_id"], snapshot.execution_epoch))

    def test_an_idle_still_arm_is_held(self):
        self.assertIsNone(self.guard())

    def test_a_skill_that_owns_a_still_arm_may_ask(self):
        self.assertIsNone(self.guard(active={"move_to_pose"}))

    def test_a_moving_arm_is_refused_whoever_owns_it(self):
        self.assertIsNotNone(self.guard(still=False))
        self.assertIsNotNone(self.guard(active={"follow_constraint"}, still=False))


class StopVerificationTests(unittest.TestCase):
    """A stop while a skill waits on the model must verify the hold the arm really has."""

    def runtime(self):
        client = FakeClient()
        return client, sheppy_runtime(CONTEXT, root=ROOT, client=client, capabilities=PROVIDED_CAPABILITIES, commissioned=True)

    def test_a_still_arm_with_nothing_in_flight_verifies_even_while_a_skill_owns_it(self):
        client, runtime = self.runtime()
        runtime.backend.active.add("move_to_pose")
        self.assertTrue(asyncio.run(runtime.executor.safety.request_stop("SKILL_TIMEOUT")))
        self.assertFalse(runtime.executor.safety.fault_latched)

    def test_a_trajectory_in_flight_or_a_moving_arm_does_not_verify(self):
        for in_flight, still in ((True, True), (False, False)):
            client, runtime = self.runtime()
            client.in_flight, client.still = in_flight, still
            self.assertFalse(asyncio.run(runtime.executor.safety.request_stop("SKILL_TIMEOUT")))
            self.assertTrue(runtime.executor.safety.fault_latched)


class ImageryPolicyTests(unittest.TestCase):
    ENVIRON = {"ROS_LOCALHOST_ONLY": "0", "CYCLONEDDS_URI": "", "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}

    def test_the_shipped_policy_admits_the_deployment_domain_and_is_recorded(self):
        document = json.loads(POLICY.read_text())
        self.assertEqual(document["accepted_routable_domains"], [0])
        self.assertNotIn("reviewer", document)
        evidence = require_local_imagery(0, environ=self.ENVIRON, policy=document)
        self.assertEqual(evidence["mechanism"], "deployment_policy")
        self.assertFalse(evidence["confined_to_loopback"])
        self.assertEqual(len(evidence["policy"]["digest"]), 64)
        with self.assertRaisesRegex(PerceptionError, "not among"):
            require_local_imagery(87, environ=self.ENVIRON, policy=document)

    def test_confinement_still_wins_without_a_policy_and_malformed_policies_are_refused(self):
        environ = {"ROS_LOCALHOST_ONLY": "1", "CYCLONEDDS_URI": "", "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}
        self.assertTrue(require_local_imagery(0, environ=environ)["confined_to_loopback"])
        with self.assertRaises(PerceptionError):
            require_local_imagery(0, environ=self.ENVIRON)
        for broken in ({}, {"schema_version": 1, "accepted_routable_domains": [], "reason": "y"},
                       {"schema_version": 1, "accepted_routable_domains": [300], "reason": "y"},
                       {"schema_version": 1, "accepted_routable_domains": [0], "reason": ""}):
            with self.assertRaises(PerceptionError):
                deployment_policy(broken)

    def test_the_camera_source_threads_the_policy_through_its_guard(self):
        environ = {**self.ENVIRON, "ROS_DOMAIN_ID": "0"}
        with patch.dict("os.environ", environ), patch.dict("sys.modules", {"rclpy": None}):
            # The guard passes and construction proceeds to the ROS import,
            # which is absent here; without a policy the guard refuses first.
            with self.assertRaises((ImportError, TypeError, AttributeError)):
                RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri", depth_info_topic="/di",
                              domain_id=0, locality_policy=json.loads(POLICY.read_text()))
            with self.assertRaises(PerceptionError):
                RosRgbdSource(rgb_topic="/r", depth_topic="/d", rgb_info_topic="/ri", depth_info_topic="/di", domain_id=0)


if __name__ == "__main__":
    unittest.main()
