"""Continuous-path and local-certificate tests; no physical commissioning."""
from dataclasses import replace, asdict
import json
from pathlib import Path
import tempfile
import time
import unittest

from rammp_adl.motion.commissioning import (
    CommissioningSettings, CommissioningAdmission, CommissionedCellVerifier,
    _bernstein_bounds, load_settings, commissioning_status,
    profile_template, sha256,
    measured_transport_bounds,
)
from rammp_adl.motion.curobo import RAMMP_COMMIT
from rammp_adl.motion.driver_feedback import FeedbackBounds
from rammp_adl.motion.driver_transport import TransportBounds
from rammp_adl.motion.driver_state import ARM_JOINT_NAMES
from rammp_adl.motion.rolling import JointState, JointLimits, JointTrajectory, TrajectoryPoint, MotionError


def settings():
    transport = TransportBounds((.001,)*7, (.002,)*7, (.002,)*7, .001, .001,
        .1, .1, .005, 1., .05, .5, .1, .2, 10.)
    return CommissioningSettings("unit-test-only", True,
        FeedbackBounds(.0001, .01, .1, .1, .00001, .1, .1), transport, .1,
        JointLimits((-2.,)*7, (2.,)*7, (1.,)*7, (10.,)*7),
        (-.5,)*7, (.5,)*7, (.01,)*7, (.0011,)*7, (100.,)*7, (.1,)*7,
        1., 1., time.time()+600., "a"*64, "test-model", "test-world", (), (("fixture","numeric-test"),), ())


def trajectory(*, end=.01, start_velocity=0., duration=2., provenance=None):
    return JointTrajectory(ARM_JOINT_NAMES, (
        TrajectoryPoint(0., JointState((0.,)*7, (start_velocity,)*7, (0.,)*7)),
        TrajectoryPoint(duration, JointState((end,)*7, (0.,)*7, (0.,)*7))),
        provenance or "rammp_curobo:"+RAMMP_COMMIT+":UNIT_TEST_DOUBLE")


class CommissioningTests(unittest.TestCase):
    def test_existing_profile_cannot_enable_hardware(self):
        root = Path(__file__).resolve().parents[1]
        report = commissioning_status(root/"config/commissioning.json", "free_space")
        self.assertFalse(report["hardware_motion_enabled"])
        self.assertEqual(report["profile_status"], "unavailable")
        with self.assertRaisesRegex(MotionError, "disabled"):
            load_settings(root/"config/commissioning.json", "free_space")

    def test_certificate_bound_to_exact_trajectory_validator_and_single_use(self):
        first, second = CommissioningAdmission(settings()), CommissioningAdmission(settings())
        certificate = first.validate(trajectory())
        self.assertTrue(first.check(certificate, certificate.trajectory))
        with self.assertRaises(MotionError):
            first.validate(trajectory())
        with self.assertRaises(MotionError):
            second.check(certificate, certificate.trajectory)
        with self.assertRaises(MotionError):
            first.check(certificate, trajectory(end=.02))
        first.revoked = True
        with self.assertRaises(MotionError):
            first.check(certificate, certificate.trajectory)

    def test_stop_and_tracking_reserve_cannot_leave_cell(self):
        with self.assertRaisesRegex(MotionError, "stopping envelope"):
            CommissioningAdmission(settings()).validate(trajectory(end=.49))

    def test_continuous_overshoot_rejected_even_when_every_knot_is_inside(self):
        narrow = replace(settings(), cell_lower=(-.02,)*7, cell_upper=(.02,)*7,
                         stop_excursion=(.001,)*7, tracking_reserve=(.001,)*7)
        candidate = trajectory(end=0., start_velocity=.2, duration=1.)
        self.assertTrue(all(abs(p.state.position[0]) < .02 for p in candidate.points))
        with self.assertRaisesRegex(MotionError, "stopping envelope"):
            CommissionedCellVerifier(narrow)(candidate, dict(narrow.dependencies))

    def test_continuous_derivatives_and_attachment_speed_are_checked(self):
        with self.assertRaisesRegex(MotionError, "derivative"):
            CommissionedCellVerifier(settings())(trajectory(duration=.01), dict(settings().dependencies))
        slow = replace(settings(), max_tool_speed_m_s=.00001)
        with self.assertRaisesRegex(MotionError, "Cartesian speed"):
            CommissioningAdmission(slow).validate(trajectory())

    def test_changed_dependency_or_expired_cell_rejects_dispatch(self):
        s = settings()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"profile.json"
            path.write_text("first")
            stat = path.stat()
            saved = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            gate = CommissioningAdmission(replace(s, asset_stats=((path,saved),)))
            certificate = gate.validate(trajectory())
            path.write_text("second")
            with self.assertRaisesRegex(MotionError, "dependency changed"):
                gate.check(certificate, certificate.trajectory)
        with self.assertRaisesRegex(MotionError, "expired"):
            CommissioningAdmission(replace(s, evidence_expires_unix_s=1.)).validate(trajectory())

    def test_fixture_provenance_is_not_a_physical_planner(self):
        with self.assertRaisesRegex(MotionError, "cuRobo"):
            CommissioningAdmission(settings()).validate(trajectory(provenance="fixture"))

    def test_bernstein_hull_contains_interior_extremum(self):
        lower, upper = _bernstein_bounds([0.,4.,-4.])
        self.assertLessEqual(lower, 0.)
        self.assertGreaterEqual(upper, 1.)

    def test_complete_explicit_simulation_profile_cannot_be_promoted_by_mode_flag(self):
        root = Path(__file__).resolve().parents[1]
        template = profile_template(root/"config/commissioning.json")
        source = json.loads((root/"config/commissioning.json").read_text())
        profile = template["profile_fragment"]
        for key in source["required_profile_fields"]:
            profile[key] = 1.
        profile.update(allowed_contacts=[], max_joint_speed_rad_s=[1.]*7,
                       max_joint_acceleration_rad_s2=[10.]*7)
        s = settings()
        test = profile["free_space_test"]
        test.update(simulation_only=True, feedback=asdict(s.feedback), transport=asdict(s.transport),
            watchdog_timeout_s=s.watchdog_timeout_s, collision_free_cell_lower_rad=list(s.cell_lower),
            collision_free_cell_upper_rad=list(s.cell_upper), stopping_joint_excursion_rad=list(s.stop_excursion),
            joint_position_uncertainty_rad=[.0001]*7, joint_lower_rad=list(s.limits.lower), joint_upper_rad=list(s.limits.upper),
            max_joint_jerk_rad_s3=list(s.jerk), all_assembly_point_lever_bounds_m=list(s.maximum_link_lever_m),
            extension_build_id=s.extension_build_id, planner_loaded_robot_config_digest="sha256:"+"b"*64)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("robot_urdf", "planner_config", "planner_robot_config", "record"):
                (folder/name).write_text("explicit simulation fixture asset "+name)
            evidence = template["cell_evidence_template"]
            evidence.update(scope="simulation_transport_test", profile_id="test", reviewer="unit-test",
                method="synthetic loader fixture, no geometry validation", expires_unix_s=time.time()+600.,
                collision_free_cell_lower_rad=list(s.cell_lower), collision_free_cell_upper_rad=list(s.cell_upper),
                robot_urdf_digest=sha256(folder/"robot_urdf"), world_digest="sha256:"+"c"*64,
                records=[{"path":"record", "sha256":sha256(folder/"record")}])
            (folder/"cell_evidence").write_text(json.dumps(evidence))
            test["assets"] = {name:{"path":name,"sha256":sha256(folder/name)} for name in test["assets"]}
            source.update(hardware_motion_enabled=True, profiles={"test":profile})
            path = folder/"commissioning.json"
            path.write_text(json.dumps(source))
            loaded = load_settings(path, "test", simulation=True)
            self.assertTrue(loaded.simulation)
            with self.assertRaisesRegex(MotionError, "mode-matched"):
                load_settings(path, "test", simulation=False)
            (folder/"record").write_text("changed evidence")
            with self.assertRaisesRegex(MotionError, "changed"):
                loaded.unchanged()

    def test_completion_allowances_reserve_acquisition_uncertainty(self):
        s = settings()
        actual = measured_transport_bounds(s)
        self.assertAlmostEqual(actual.goal_position_rad[0], .0019)
        self.assertAlmostEqual(actual.start_position_rad[0], .0019)
        self.assertAlmostEqual(actual.stationary_velocity_rad_s, .00099)
        self.assertAlmostEqual(actual.stationary_position_span_rad, .0008)
        with self.assertRaises(MotionError):
            measured_transport_bounds(replace(s, tracking_reserve=(.002,)*7))


if __name__ == "__main__":
    unittest.main()
