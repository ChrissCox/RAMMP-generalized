"""Geometry preparation tests use synthetic meshes, never driver connections."""
import math
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from rammp_adl.motion.assembly import (
    AssemblyError, _clean_model, _resolve_mesh, cover_collision, freeze_gripper,
    full_gripper_envelope, origin_transform, prepare_bundle,
)


def gripper_fixture():
    root = ET.Element("robot", name="test")
    ET.SubElement(root, "link", name="base_link")
    parent = "base_link"
    for i in range(1, 8):
        child = f"arm_{i}"
        ET.SubElement(root, "link", name=child)
        j = ET.SubElement(root, "joint", name=f"joint_{i}", type="continuous")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
        parent = child
    base = ET.SubElement(root, "link", name="robotiq_85_base_link")
    j = ET.SubElement(root, "joint", name="gripper_base", type="fixed")
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=base.get("name"))
    names = ["left_knuckle", "right_knuckle", "left_inner_knuckle", "right_inner_knuckle", "left_finger_tip", "right_finger_tip"]
    for i, name in enumerate(names):
        child = "robotiq_85_" + name + "_link"
        ET.SubElement(root, "link", name=child)
        joint = ET.SubElement(root, "joint", name="robotiq_85_" + name + "_joint", type="revolute")
        ET.SubElement(joint, "parent", link=base.get("name"))
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz="0.1 0.2 0.3", rpy="0.3 -0.2 0.4")
        ET.SubElement(joint, "axis", xyz="0 -1 0")
        if i:
            ET.SubElement(joint, "mimic", joint="robotiq_85_left_knuckle_joint", multiplier=str((-1) ** i))
        else:
            ET.SubElement(joint, "limit", lower="0", upper="0.8")
    return root


class AssemblyTests(unittest.TestCase):
    def test_mimic_freezing_preserves_nonzero_origins_and_only_arm_dof(self):
        root = gripper_fixture()
        before = origin_transform(root.findall("joint")[8])
        frozen, angles = freeze_gripper(root, .6)
        self.assertEqual(len(angles), 6)
        self.assertEqual([j.get("name") for j in frozen.findall("joint") if j.get("type") != "fixed"], [f"joint_{i}" for i in range(1, 8)])
        np.testing.assert_allclose(origin_transform(frozen.findall("joint")[8])[:3, 3], before[:3, 3])
        # Direct expected rotation around the negative Y axis, independent of helper.
        c, s = math.cos(.6), math.sin(.6)
        expected = before[:3, :3] @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        np.testing.assert_allclose(origin_transform(frozen.findall("joint")[8])[:3, :3], expected, atol=1e-14)
        self.assertEqual(angles["robotiq_85_right_knuckle_joint"], -.6)
        self.assertIsNotNone(root.findall("joint")[8].find("axis"))
        self.assertIsNone(frozen.findall("joint")[8].find("axis"))

    def test_rejects_missing_chain_invalid_mimic_and_out_of_range(self):
        for angle in (float("nan"), True, -.01, .81):
            with self.assertRaises(AssemblyError):
                freeze_gripper(gripper_fixture(), angle)
        root = gripper_fixture()
        root.findall("joint")[-1].find("mimic").set("joint", "missing")
        with self.assertRaises(AssemblyError):
            freeze_gripper(root, .2)

    def test_cover_contains_box_volume_after_rotation(self):
        collision = ET.fromstring('<collision><origin xyz=".3 -.4 .2" rpy=".4 .5 -.6"/><geometry><box size=".08 .12 .17"/></geometry></collision>')
        spheres = cover_collision(collision, .035)
        transform = origin_transform(collision)
        rng = np.random.default_rng(243)
        local = rng.uniform([-.04, -.06, -.085], [.04, .06, .085], size=(3000, 3))
        points = local @ transform[:3, :3].T + transform[:3, 3]
        gap = np.array([np.linalg.norm(points - s["center"], axis=1) - s["radius"] for s in spheres])
        self.assertLessEqual(float(np.min(gap, axis=0).max()), 0.)

    def test_rejects_bad_sizes_scales_and_unbounded_cover(self):
        for xml in ('<box size="0 .1 .1"/>', '<sphere radius="-1"/>', '<cylinder radius="nan" length="1"/>', '<capsule radius="1"/>'):
            collision = ET.fromstring(f'<collision><geometry>{xml}</geometry></collision>')
            with self.assertRaises(AssemblyError):
                cover_collision(collision)
        with self.assertRaises(AssemblyError):
            cover_collision(ET.fromstring('<collision><geometry><box size="1 1 1"/></geometry></collision>'), .001)

    def test_full_travel_envelope_contains_unsampled_joint_rotations(self):
        root = gripper_fixture()
        local_center, local_radius = np.array([.05, -.08, .1]), .03
        spheres = {"robotiq_85_left_knuckle_link": [{"center": local_center.tolist(), "radius": local_radius}]}
        result = full_gripper_envelope(root, spheres)
        transform = origin_transform(root.findall("joint")[8])
        for angle in np.linspace(-7, 7, 501):
            c, s = math.cos(angle), math.sin(angle)
            rotated = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]) @ local_center
            center = transform[:3, :3] @ rotated + transform[:3, 3]
            self.assertLessEqual(np.linalg.norm(center) + local_radius, result["radius"] + 1e-14)

    def test_control_blocks_are_removed_and_other_plugins_refused(self):
        root = gripper_fixture()
        ET.SubElement(root, "ros2_control", name="hardware")
        ET.SubElement(root, "gazebo")
        clean = _clean_model(root)
        self.assertIsNone(clean.find("ros2_control"))
        self.assertIsNone(clean.find("gazebo"))
        ET.SubElement(root.find("link"), "plugin", filename="hardware.so")
        with self.assertRaises(AssemblyError):
            _clean_model(root)

    def test_local_only_assets_and_package_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "mesh.stl").write_text("fixture")
            self.assertEqual(_resolve_mesh("package://robot/mesh.stl", path, {"robot": path}), path / "mesh.stl")
            for filename in ("https://example.com/mesh", "package://robot/../secret", "package://missing/a.stl"):
                with self.assertRaises(AssemblyError):
                    _resolve_mesh(filename, path, {"robot": path})

    def test_bundle_pins_assets_keeps_camera_detached_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            arm = gripper_fixture()
            for link in arm.findall("link"):
                collision = ET.SubElement(link, "collision")
                ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=".01 .01 .01")
            ET.ElementTree(arm).write(path / "arm.urdf")
            # The fixture visual is only bundled, never used as collision data.
            (path / "d405.stl").write_bytes(b"local visual asset fixture")
            (path / "camera.urdf").write_text('<robot name="camera"><link name="d405_body"><visual><geometry><mesh filename="d405.stl"/></geometry></visual><collision><geometry><box size=".023 .042 .042"/></geometry></collision></link></robot>')
            result = prepare_bundle(arm_urdf=path / "arm.urdf", d405_urdf=path / "camera.urdf",
                                    output_dir=path / "bundle", knuckle_rad=.2)
            self.assertFalse(result["installed_assembly_complete"])
            self.assertFalse(result["hardware_motion_enabled"])
            self.assertIsNone(result["mount"]["parent_from_d405_bottom_screw"])
            locked = ET.parse(path / "bundle/arm-gripper-locked.urdf")
            self.assertIsNone(locked.find("link[@name='d405_body']"))
            for artifact in result["generated"] + result["assets"]:
                self.assertEqual(artifact["sha256"], "sha256:" + hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest())
            self.assertEqual(json.loads((path / "bundle/manifest.json").read_text()), result)
            with self.assertRaises(AssemblyError):
                prepare_bundle(arm_urdf=path / "arm.urdf", d405_urdf=path / "camera.urdf",
                               output_dir=path / "bundle", knuckle_rad=.2)


if __name__ == "__main__":
    unittest.main()
