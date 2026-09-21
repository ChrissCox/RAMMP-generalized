"""Local URDF forward kinematics, checked against MuJoCo and the real driver."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree

import numpy as np

from rammp_adl.motion.driver_state import ARM_JOINT_NAMES
from rammp_adl.motion.kinematics import (
    KinematicsError, UrdfChain, driver_joint_positions, frame_agreement,
    quaternion_matrix, rotation_angle)


ROOT = Path(__file__).resolve().parents[1]
DRIVER_MODEL = ROOT/"artifacts/jetson/passive-commissioning/driver-model.urdf"
RECORDING = ROOT/"artifacts/jetson/driver-feedback-2026-09-14/paired-window/driver.jsonl"
SYNTHETIC = """<robot name="two">
  <link name="root"/><link name="middle"/><link name="tip"/>
  <joint name="a" type="revolute">
    <parent link="root"/><child link="middle"/>
    <origin xyz="0 0 1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="b" type="fixed">
    <parent link="middle"/><child link="tip"/>
    <origin xyz="2 0 0" rpy="0 0 0"/>
  </joint>
</robot>"""


def driver_chain():
    return UrdfChain.from_path(DRIVER_MODEL)


class SyntheticChainTests(unittest.TestCase):
    def setUp(self):
        self.chain = UrdfChain(SYNTHETIC)

    def test_composes_origins_and_the_revolute_axis_exactly(self):
        pose = self.chain.base_from_link({"a": np.pi/2}, "tip")
        np.testing.assert_allclose(pose[:3, 3], [0., 2., 1.], atol=1e-12)
        np.testing.assert_allclose(pose[:3, :3] @ [1., 0., 0.], [0., 1., 0.], atol=1e-12)

    def test_results_are_proper_rigid_transforms(self):
        pose = self.chain.base_from_link({"a": .7}, "tip")
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(pose[:3, :3])), 1., places=12)
        np.testing.assert_allclose(pose[3], [0., 0., 0., 1.], atol=1e-15)

    def test_relative_base_selects_a_subchain(self):
        pose = self.chain.base_from_link({"a": .3}, "tip", base="middle")
        np.testing.assert_allclose(pose[:3, 3], [2., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-12)

    def test_declared_limits_and_violations(self):
        self.assertEqual(self.chain.declared_limits_rad, {"a": (-1., 1.)})
        self.assertEqual(self.chain.within_declared_limits({"a": .5}), [])
        self.assertEqual(self.chain.within_declared_limits({"a": 1.5}), ["a"])
        self.assertEqual(self.chain.chain("tip"), ("a", "b"))
        self.assertEqual(self.chain.actuated_chain("tip"), ("a",))

    def test_missing_unknown_and_nonfinite_inputs_are_refused(self):
        for call in (lambda: self.chain.base_from_link({}, "tip"),
                     lambda: self.chain.base_from_link({"a": float("nan")}, "tip"),
                     lambda: self.chain.base_from_link({"a": True}, "tip"),
                     lambda: self.chain.base_from_link({"a": 0.}, "absent"),
                     lambda: self.chain.base_from_link({"a": 0.}, "tip", base="absent"),
                     lambda: self.chain.base_from_link({"a": 0.}, "root", base="tip")):
            with self.assertRaises(KinematicsError):
                call()

    def test_malformed_models_are_refused(self):
        cases = ["<robot/>", "not xml",
                 '<robot><link name="a"/><link name="b"/><joint name="j" type="revolute">'
                 '<parent link="a"/><child link="b"/><limit lower="-1"/></joint></robot>',
                 '<robot><link name="a"/><link name="b"/><joint name="j" type="revolute">'
                 '<parent link="a"/><child link="b"/><limit lower="x" upper="1"/></joint></robot>',
                 '<robot><link name="a"/><link name="a"/></robot>',
                 '<robot><link name="a"/><link name="b"/><joint name="j" type="planar">'
                 '<parent link="a"/><child link="b"/></joint></robot>',
                 '<robot><link name="a"/><link name="b"/><joint name="j" type="revolute">'
                 '<parent link="a"/><child link="b"/><limit lower="1" upper="0"/></joint></robot>',
                 '<robot><link name="a"/><link name="b"/><link name="c"/>'
                 '<joint name="j" type="fixed"><parent link="a"/><child link="c"/></joint>'
                 '<joint name="k" type="fixed"><parent link="b"/><child link="c"/></joint></robot>']
        for text in cases:
            with self.assertRaises(KinematicsError):
                UrdfChain(text)
        with self.assertRaises(KinematicsError):
            UrdfChain("")

    def test_digest_identifies_the_source_bytes(self):
        self.assertEqual(len(self.chain.digest), 64)
        self.assertNotEqual(self.chain.digest, UrdfChain(SYNTHETIC.replace("2 0 0", "3 0 0")).digest)


class HelperTests(unittest.TestCase):
    def test_matrix_to_quaternion_round_trips_on_every_branch(self):
        from rammp_adl.motion.kinematics import quaternion_xyzw_from_matrix
        generator = np.random.default_rng(9)
        samples = [generator.normal(size=4) for _ in range(40)]
        # Force each Shepperd branch: rotations of pi about x, y and z.
        samples += [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]
        for raw in samples:
            q = np.asarray(raw, dtype=float)/np.linalg.norm(raw)
            matrix = quaternion_matrix(q.tolist())
            back = quaternion_xyzw_from_matrix(matrix)
            self.assertLess(rotation_angle(matrix, quaternion_matrix(back)), 1e-9)
            self.assertAlmostEqual(float(np.linalg.norm(back)), 1., places=12)

    def test_quaternion_and_rotation_angle(self):
        np.testing.assert_allclose(quaternion_matrix([0., 0., 0., 1.]), np.eye(3), atol=1e-15)
        half = quaternion_matrix([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
        self.assertAlmostEqual(rotation_angle(np.eye(3), half), np.pi/2, places=12)
        # An unnormalized quaternion is scaled, never silently accepted as-is.
        np.testing.assert_allclose(quaternion_matrix([0., 0., 0., 5.]), np.eye(3), atol=1e-12)
        for bad in ([0., 0., 0., 0.], [0., 0., 0., float("nan")], [0., 0., 0., True]):
            with self.assertRaises(KinematicsError):
                quaternion_matrix(bad)

    def test_driver_joint_names_are_mapped_onto_the_model(self):
        mapped = driver_joint_positions(("joint_1", "joint_2"), (.1, .2))
        self.assertEqual(mapped, {"gen3_joint_1": .1, "gen3_joint_2": .2})
        for call in (lambda: driver_joint_positions(("a",), (1., 2.)),
                     lambda: driver_joint_positions(("a", "a"), (1., 2.)),
                     lambda: driver_joint_positions(("a",), (float("inf"),)),
                     lambda: driver_joint_positions(("",), (1.,))):
            with self.assertRaises(KinematicsError):
                call()


@unittest.skipUnless(DRIVER_MODEL.exists(), "The inspected driver model is separate evidence")
class DriverModelTests(unittest.TestCase):
    def setUp(self):
        self.chain = driver_chain()

    def test_parses_the_inspected_chain(self):
        self.assertEqual(self.chain.root_link, "world")
        self.assertEqual(self.chain.actuated_joints, tuple("gen3_"+name for name in ARM_JOINT_NAMES))
        self.assertEqual(self.chain.actuated_chain("gen3_end_effector_link"),
                         self.chain.actuated_joints)

    def test_continuous_joints_declare_no_bound_and_none_is_invented(self):
        limits = self.chain.declared_limits_rad
        for index in (1, 3, 5, 7):
            self.assertIsNone(limits[f"gen3_joint_{index}"], index)
        for index, bound in ((2, 2.41), (4, 2.66), (6, 2.23)):
            self.assertEqual(limits[f"gen3_joint_{index}"], (-bound, bound))
        # A continuous joint cannot be flagged, so an envelope must bound it.
        wild = {name: 50. for name in self.chain.actuated_joints}
        self.assertEqual(self.chain.within_declared_limits(wild),
                         ["gen3_joint_2", "gen3_joint_4", "gen3_joint_6"])

    def test_world_and_base_link_are_coincident_in_this_model(self):
        zero = dict.fromkeys(self.chain.actuated_joints, 0.)
        np.testing.assert_allclose(self.chain.base_from_link(zero, "gen3_base_link"), np.eye(4), atol=1e-15)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is a separate simulation dependency")
    def test_forward_kinematics_matches_an_independent_mujoco_evaluation(self):
        import mujoco
        root = ElementTree.parse(DRIVER_MODEL).getroot()
        # FK depends only on joint origins and axes; drop geometry so the
        # comparison needs no meshes and stays a pure kinematic check.
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for element in link.findall(tag):
                    link.remove(element)
            if link.find("inertial") is None:
                inertial = ElementTree.SubElement(link, "inertial")
                ElementTree.SubElement(inertial, "mass", value="1")
                ElementTree.SubElement(inertial, "inertia", ixx="1", ixy="0", ixz="0",
                                       iyy="1", iyz="0", izz="1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"chain.urdf"
            ElementTree.ElementTree(root).write(path)
            model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        compared = 0
        generator = np.random.default_rng(3)
        for _ in range(25):
            angles = generator.uniform(-1.5, 1.5, size=7)
            data.qpos[:7] = angles
            mujoco.mj_forward(model, data)
            positions = dict(zip(self.chain.actuated_joints, angles.tolist()))
            for index, name in enumerate(bodies):
                if name not in self.chain.link_names:
                    continue
                pose = self.chain.base_from_link(positions, name)
                np.testing.assert_allclose(pose[:3, 3], data.xpos[index], atol=1e-9)
                self.assertLess(rotation_angle(pose[:3, :3], data.xmat[index].reshape(3, 3)), 1e-9)
                compared += 1
        self.assertGreaterEqual(compared, 25*7)

    @unittest.skipUnless(RECORDING.exists(), "Live driver publications are separate evidence")
    def test_recorded_publications_identify_the_reported_end_effector_frame(self):
        rows = [json.loads(line) for line in RECORDING.read_text().splitlines()]
        self.assertGreater(len(rows), 100)
        samples = [{"positions": driver_joint_positions(ARM_JOINT_NAMES, row["joints"]["position_rad"]),
                    "position_m": row["ee"]["position_m"],
                    "quaternion_xyzw": row["ee"]["quaternion_xyzw"]} for row in rows[::100]]
        report = frame_agreement(self.chain, samples, base="gen3_base_link")
        best = report["ranked"][0]
        self.assertIn(best["link"], ("gen3_end_effector_link", "gen3_robotiq_85_base_link"))
        self.assertLess(best["max_position_error_m"], 1e-9)
        self.assertLess(best["max_rotation_error_rad"], 1e-9)
        # The bracelet link differs by a real offset, so the match is a frame
        # identification rather than an artefact of a near-identity tail.
        bracelet = next(row for row in report["ranked"] if row["link"] == "gen3_bracelet_link")
        self.assertGreater(bracelet["max_position_error_m"], .05)
        self.assertEqual(report["model_digest"], self.chain.digest)
        self.assertIn("not evidence that the model matches the physical robot", report["semantics"])

    def test_frame_agreement_validates_its_inputs(self):
        zero = dict.fromkeys(self.chain.actuated_joints, 0.)
        good = {"positions": zero, "position_m": [0., 0., 0.], "quaternion_xyzw": [0., 0., 0., 1.]}
        with self.assertRaises(KinematicsError):
            frame_agreement(self.chain, [])
        with self.assertRaises(KinematicsError):
            frame_agreement(self.chain, [{"positions": zero}])
        with self.assertRaises(KinematicsError):
            frame_agreement(self.chain, [good], candidates=["absent"])
        ranked = frame_agreement(self.chain, [good], candidates=["gen3_base_link"])["ranked"]
        self.assertEqual(len(ranked), 1)


if __name__ == "__main__":
    unittest.main()
