"""Prepare local geometry assets for arm planning; never connect to a robot.

Input URDFs are expanded, trusted descriptions. The articulated gripper is kept
as source, and its six joints are frozen at an explicit model knuckle angle for
the seven-joint planner. The manufacturer's D405 component stays disconnected
pending an explicit mounting record. Neither nominal geometry nor
a complete model bundle is physical commissioning evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .driver_state import ARM_JOINT_NAMES, GRIPPER_JOINT_NAME


class AssemblyError(ValueError):
    pass


def _digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numbers(value, count):
    try:
        values = np.asarray(value.split() if isinstance(value, str) else value, dtype=float)
    except (TypeError, ValueError) as error:
        raise AssemblyError("Geometry requires finite numeric coordinates") from error
    if values.shape != (count,) or not np.isfinite(values).all():
        raise AssemblyError("Geometry requires finite numeric coordinates")
    return values


def _rotation(axis, angle):
    axis = _numbers(axis, 3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise AssemblyError("Joint axis must be nonzero")
    x, y, z = axis / norm
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def origin_transform(element):
    """URDF fixed-origin convention; this is forward geometry, not arm IK."""
    origin = element.find("origin")
    result = np.eye(4)
    if origin is not None:
        r, p, y = _numbers(origin.get("rpy", "0 0 0"), 3)
        result[:3, :3] = (_rotation([0, 0, 1], y) @ _rotation([0, 1, 0], p)
                          @ _rotation([1, 0, 0], r))
        result[:3, 3] = _numbers(origin.get("xyz", "0 0 0"), 3)
    return result


def _set_origin(element, transform):
    rotation = transform[:3, :3]
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
    if abs(math.cos(pitch)) > 1e-10:
        roll, yaw = math.atan2(rotation[2, 1], rotation[2, 2]), math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll, yaw = math.atan2(-rotation[1, 2], rotation[1, 1]), 0.
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    origin.set("xyz", " ".join(format(float(v), ".17g") for v in transform[:3, 3]))
    origin.set("rpy", " ".join(format(v, ".17g") for v in (roll, pitch, yaw)))


def _clean_model(root):
    root = copy.deepcopy(root)
    if root.tag != "robot" or root.findall(".//{*}include"):
        raise AssemblyError("Provide a fully expanded robot URDF")
    # Descriptions must never carry a second hardware-controller connection.
    for tag in ("ros2_control", "transmission", "gazebo"):
        for element in root.findall(tag):
            root.remove(element)
    if root.findall(".//plugin"):
        raise AssemblyError("Robot assets must not contain executable plugins")
    links = [v.get("name") for v in root.findall("link")]
    joints = [v.get("name") for v in root.findall("joint")]
    if len(set(links)) != len(links) or len(set(joints)) != len(joints):
        raise AssemblyError("URDF has duplicate link or joint names")
    parents = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None or parent.get("link") not in links or child.get("link") not in links:
            raise AssemblyError("URDF joint references an unavailable link")
        if child.get("link") in parents:
            raise AssemblyError("URDF link has multiple parents")
        parents[child.get("link")] = parent.get("link")
    if len(set(links) - parents.keys()) != 1:
        raise AssemblyError("Each source component must be one connected tree")
    for link in links:
        seen = set()
        while link in parents:
            if link in seen:
                raise AssemblyError("URDF joint tree contains a cycle")
            seen.add(link)
            link = parents[link]
    return root


def freeze_gripper(root, knuckle_rad):
    """Resolve mimic joints before freezing, retaining each body's mass/origin."""
    if type(knuckle_rad) not in (int, float) or not math.isfinite(knuckle_rad):
        raise AssemblyError("Specify a finite model gripper knuckle angle")
    result = _clean_model(root)
    joints = {joint.get("name"): joint for joint in result.findall("joint")}
    if any(name not in joints for name in ARM_JOINT_NAMES) or GRIPPER_JOINT_NAME not in joints:
        raise AssemblyError("Expected canonical joint_1..7 and Robotiq 2F-85 knuckle names")
    if any(joints[name].get("type") not in ("revolute", "continuous") or joints[name].find("mimic") is not None
           for name in ARM_JOINT_NAMES):
        raise AssemblyError("All seven arm joints must remain independently articulated")
    resolved = {GRIPPER_JOINT_NAME: float(knuckle_rad)}

    def value(name, active):
        if name in resolved:
            return resolved[name]
        if name in active or name not in joints:
            raise AssemblyError("Missing or cyclic gripper mimic dependency")
        mimic = joints[name].find("mimic")
        if mimic is None:
            raise AssemblyError("Unrecognized independent joint outside the seven arm joints")
        target = mimic.get("joint")
        resolved[name] = value(target, active | {name}) * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0"))
        return resolved[name]

    for name, joint in joints.items():
        if name in ARM_JOINT_NAMES or joint.get("type") == "fixed":
            continue
        if not name.startswith("robotiq_85_") or joint.get("type") not in ("revolute", "continuous"):
            raise AssemblyError("Only the Robotiq revolute mimic chain may be frozen")
        angle = value(name, set())
        if not math.isfinite(angle):
            raise AssemblyError("Invalid gripper mimic angle")
        limit = joint.find("limit")
        if limit is not None and not float(limit.get("lower", "-inf")) <= angle <= float(limit.get("upper", "inf")):
            raise AssemblyError("Gripper model angle violates declared joint limits")
        transform = origin_transform(joint)
        axis = joint.find("axis")
        transform[:3, :3] = transform[:3, :3] @ _rotation(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", angle)
        _set_origin(joint, transform)
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "mimic", "dynamics"):
            for child in joint.findall(tag):
                joint.remove(child)
    if len(resolved) != 6:
        raise AssemblyError("Expected all six Robotiq knuckle/mimic joints")
    return result, resolved


def _resolve_mesh(filename, source_dir, package_roots):
    if filename.startswith("package://"):
        package, separator, relative = filename[10:].partition("/")
        if not separator or package not in package_roots:
            raise AssemblyError(f"Missing local package mapping for {package}")
        root = Path(package_roots[package]).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise AssemblyError("Package mesh escapes its mapped root")
    elif filename.startswith("file://"):
        path = Path(filename[7:]).resolve()
    elif "://" in filename:
        raise AssemblyError("Only local mesh assets are accepted")
    else:
        path = (source_dir / filename).resolve()
    if not path.is_file():
        raise AssemblyError(f"Missing local mesh asset: {path}")
    return path


def bundle_meshes(root, source_dir, output_dir, package_roots):
    """Pin all referenced geometry bytes; preserve transforms and mesh units."""
    records = {}
    for mesh in root.findall(".//mesh"):
        path = _resolve_mesh(mesh.get("filename", ""), source_dir, package_roots)
        identity = _digest(path)
        target = output_dir / "meshes" / (identity[7:] + "-" + path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        mesh.set("filename", str(target.resolve()))
        records[str(target)] = {"source_path": str(path), "path": str(target), "sha256": identity}
        if path.suffix.lower() == ".dae":
            dae = ET.parse(path).getroot()
            if dae.findall(".//{*}library_images/{*}image"):
                raise AssemblyError("Textured DAE requires explicit texture bundling; refusing incomplete assets")
    return list(records.values())


def collision_bounds(collision):
    """Conservative local AABB of the complete mesh/primitive, never samples."""
    geometry = collision.find("geometry")
    if geometry is None or len(geometry) != 1:
        raise AssemblyError("Collision must contain one supported geometry")
    shape = geometry[0]
    if shape.tag == "mesh":
        try:
            import trimesh
        except ImportError as error:
            raise AssemblyError("Assembly meshes require the optional trimesh and pycollada packages") from error
        scene = trimesh.load(shape.get("filename"), force="scene", process=False)
        if scene.bounds is None:
            raise AssemblyError("Collision mesh contains no geometry")
        if scene.units not in (None, "m", "meter", "meters"):
            raise AssemblyError("Mesh unit conversion must be explicit before sphere generation")
        bounds = np.asarray(scene.bounds, dtype=float)
        scale = _numbers(shape.get("scale", "1 1 1"), 3)
        if np.any(scale <= 0):
            raise AssemblyError("Collision mesh scale must be positive")
        bounds *= scale
    elif shape.tag == "box":
        half = _numbers(shape.get("size", ""), 3) / 2
        bounds = np.array([-half, half])
    elif shape.tag == "sphere":
        radius = float(shape.get("radius", "nan"))
        bounds = np.array([[-radius] * 3, [radius] * 3])
    elif shape.tag == "cylinder":
        radius, length = float(shape.get("radius", "nan")), float(shape.get("length", "nan"))
        bounds = np.array([[-radius, -radius, -length / 2], [radius, radius, length / 2]])
    else:
        raise AssemblyError(f"Unsupported collision geometry: {shape.tag}")
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise AssemblyError("Collision geometry must have positive finite extent")
    return bounds


def cover_collision(collision, max_cell_m=0.06):
    """Tile its entire local AABB with circumscribed spheres, then transform.

    Each rectangular cell is contained in its circumscribed sphere; their union
    therefore contains the entire original collision geometry, including faces
    and interior. This is intentionally conservative, not a mesh fit or proof
    that nominal manufacturer geometry matches the installed robot.
    """
    if type(max_cell_m) not in (int, float) or not math.isfinite(max_cell_m) or max_cell_m <= 0:
        raise AssemblyError("Sphere cell size must be positive and finite")
    bounds = collision_bounds(collision)
    raw_counts = np.ceil((bounds[1] - bounds[0]) / max_cell_m)
    if not np.isfinite(raw_counts).all() or np.any(raw_counts > 10000):
        raise AssemblyError("Collision sphere cover exceeds bounded capacity")
    counts = raw_counts.astype(int)
    if math.prod(int(v) for v in counts) > 10000:
        raise AssemblyError("Collision sphere cover exceeds bounded capacity")
    size = (bounds[1] - bounds[0]) / counts
    radius = float(np.linalg.norm(size) / 2 + 1e-9)
    transform = origin_transform(collision)
    spheres = []
    for index in itertools.product(*(range(int(v)) for v in counts)):
        local = bounds[0] + (np.array(index) + .5) * size
        center = transform[:3, :3] @ local + transform[:3, 3]
        spheres.append({"center": center.tolist(), "radius": radius})
    return spheres


def full_gripper_envelope(root, bounds_by_link):
    """Continuous all-angle bound from triangle inequality, not pose sampling.

    Rotations cannot increase distance to a joint's origin. Summing each fixed
    origin translation length and each geometry radius covers every permitted
    combination of the gripper angles (and conservatively angles beyond them).
    """
    parent = {j.find("child").get("link"): j for j in root.findall("joint")}
    radius = 0.
    for link, spheres in bounds_by_link.items():
        if not link.startswith("robotiq_85_"):
            continue
        distance, current = 0., link
        while current != "robotiq_85_base_link":
            if current not in parent:
                raise AssemblyError("Gripper collision link is outside its base subtree")
            joint = parent[current]
            if joint.get("type") not in ("fixed", "revolute", "continuous"):
                raise AssemblyError("Full-angle bound does not support translating joints")
            distance += float(np.linalg.norm(origin_transform(joint)[:3, 3]))
            current = joint.find("parent").get("link")
        radius = max(radius, *(distance + float(np.linalg.norm(s["center"])) + s["radius"] for s in spheres))
    if radius <= 0:
        raise AssemblyError("Gripper collision geometry is missing")
    return {"frame": "robotiq_85_base_link", "center": [0., 0., 0.], "radius": radius,
            "method": "all-angle triangle-inequality bound over every gripper collision sphere",
            "scope": "nominal source geometry; conservative full travel; no physical aperture calibration"}


def prepare_bundle(*, arm_urdf, d405_urdf, output_dir, knuckle_rad, package_roots=None, max_cell_m=.06):
    """Prepare geometry-only components; unknown wrist mounting stays absent."""
    arm_urdf, d405_urdf = Path(arm_urdf).resolve(), Path(d405_urdf).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise AssemblyError("Assembly output must be a new directory")
    arm, camera = _clean_model(ET.parse(arm_urdf).getroot()), _clean_model(ET.parse(d405_urdf).getroot())
    frozen, angles = freeze_gripper(arm, knuckle_rad)
    if not any("d405" in (mesh.get("filename", "")).lower() for mesh in camera.findall(".//mesh")):
        raise AssemblyError("Camera component must reference the D405 manufacturer's geometry")
    if any(j.get("type") != "fixed" for j in camera.findall("joint")):
        raise AssemblyError("D405 component must contain only fixed internal joints")
    if any("d415" in (mesh.get("filename", "")).lower() for mesh in arm.findall(".//mesh")):
        raise AssemblyError("Arm source still includes the obsolete D415 assembly")
    output_dir.mkdir(parents=True)
    records = {}
    for model, source in ((arm, arm_urdf), (frozen, arm_urdf), (camera, d405_urdf)):
        for record in bundle_meshes(model, source.parent, output_dir, package_roots or {}):
            records[record["path"]] = record
    bounds = {}
    for link in arm.findall("link"):
        spheres = [s for c in link.findall("collision") for s in cover_collision(c, max_cell_m)]
        if spheres:
            bounds[link.get("name")] = spheres
        elif link.find("visual") is not None or link.find("inertial") is not None:
            raise AssemblyError(f"Physical link has no collision geometry: {link.get('name')}")
    camera_bounds = {}
    for link in camera.findall("link"):
        spheres = [s for c in link.findall("collision") for s in cover_collision(c, max_cell_m)]
        if spheres:
            camera_bounds[link.get("name")] = spheres
        elif link.find("visual") is not None or link.find("inertial") is not None:
            raise AssemblyError(f"Physical camera link has no collision geometry: {link.get('name')}")
    envelope = full_gripper_envelope(arm, bounds)
    for name, model in (("arm-gripper-articulated.urdf", arm), ("arm-gripper-locked.urdf", frozen), ("d405-component.urdf", camera)):
        ET.ElementTree(model).write(output_dir / name, encoding="utf-8", xml_declaration=True)
    _write_json(output_dir / "collision-spheres.json", {"collision_spheres": bounds})
    _write_json(output_dir / "d405-collision-spheres.json", {"collision_spheres": camera_bounds})
    _write_json(output_dir / "gripper-full-travel-envelope.json", envelope)
    manifest = {
        "format": "rammp-assembly-components-v1", "hardware_motion_enabled": False,
        "hardware_validated": False, "installed_assembly_complete": False,
        "scope": "nominal arm/gripper and detached manufacturer D405 geometry for local planning preparation",
        "source_urdfs": [{"path": str(path), "sha256": _digest(path)} for path in (arm_urdf, d405_urdf)],
        "assets": list(records.values()),
        "generated": [{"path": str(path), "sha256": _digest(path)} for path in sorted(output_dir.glob("*")) if path.is_file()],
        "arm_joint_names": list(ARM_JOINT_NAMES), "base_frame": "base_link", "ee_frame": "end_effector_link",
        "gripper": {"policy": "planner locked at explicit model knuckle angle; physical aperture unknown",
                    "model_knuckle_rad": float(knuckle_rad), "resolved_joint_angles_rad": angles,
                    "full_travel_envelope": envelope, "dynamic_actuation_available": False},
        "collision_geometry": {"covered_arm_gripper_links": sorted(bounds), "covered_d405_links": sorted(camera_bounds),
                               "sphere_count": sum(map(len, bounds.values())), "max_cell_m": max_cell_m,
                               "proof_scope": "circumscribed-cell cover of every source collision AABB; no fitted/sample-only coverage"},
        "mount": {"parent_link": "bracelet_link", "parent_from_d405_bottom_screw": None,
                  "calibration_id": None, "bracket_geometry": None, "cable_envelope": None},
        "required_before_installed_assembly_use": ["wrist-to-D405 mount transform and uncertainty",
            "bracket geometry and placement", "cable envelope", "physical model agreement and calibrated gripper mapping",
            "self-collision pair review for the final assembly", "measured dynamics/limits and stopping commissioning"],
        "notes": ["D405 internal optical frames are manufacturer's nominal values, not calibrated transforms",
                  "Manufacturer D405 inertial values are explicitly unreliable; this is not an approved dynamics model",
                  "Fixed gripper in a seven-DOF driver model is deliberate; planner and visualization assets have separate roles",
                  "No existing driver model was modified; no hardware control blocks or connections are created"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-urdf", type=Path, required=True)
    parser.add_argument("--d405-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-knuckle-rad", type=float, required=True)
    parser.add_argument("--max-cell-m", type=float, default=.06)
    parser.add_argument("--package", action="append", default=[], metavar="NAME=LOCAL_SHARE_PATH")
    args = parser.parse_args(argv)
    roots = {}
    for item in args.package:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in roots:
            parser.error("Each --package must supply a distinct NAME=LOCAL_SHARE_PATH")
        roots[name] = path
    manifest = prepare_bundle(arm_urdf=args.arm_urdf, d405_urdf=args.d405_urdf,
                              output_dir=args.output_dir, knuckle_rad=args.model_knuckle_rad,
                              package_roots=roots, max_cell_m=args.max_cell_m)
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"),
                      "installed_assembly_complete": manifest["installed_assembly_complete"],
                      "sphere_count": manifest["collision_geometry"]["sphere_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
