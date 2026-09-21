"""Local forward kinematics for the inspected driver URDF.

Frames, joint names and axes come from the model the driver was actually
launched with, so a pose computed here and a pose the driver publishes can be
compared instead of assumed equal. Nothing here adopts a frame identity the
driver has not demonstrated, and no transform is invented for geometry the
model does not contain.

Declared URDF limits are declared, never commissioned. In this model joints 1,
3, 5 and 7 are continuous and carry no bound at all, so any motion envelope
must supply its own bounds rather than read them from the model.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import math
import xml.etree.ElementTree as ElementTree

import numpy as np

from .assembly import _rotation, origin_transform
from .rolling import MotionError


DRIVER_JOINT_PREFIX = "gen3_"
REVOLUTE_TYPES = ("revolute", "continuous")
FIXED_TYPES = ("fixed",)
SUPPORTED_TYPES = REVOLUTE_TYPES + FIXED_TYPES
MAX_URDF_BYTES = 8*1024*1024
MAX_CHAIN_DEPTH = 64


class KinematicsError(MotionError):
    """Malformed model, unknown frame or an incomplete joint configuration."""


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise KinematicsError(f"{label} must be a finite number")
    return float(value)


def rotation_angle(first, second):
    """Geodesic angle between two rotation matrices, without a scipy dependency.

    Uses atan2 of the skew and trace parts rather than acos of the trace. An
    acos loses half the available precision near identity, which is exactly
    where frame-agreement evidence is read.
    """
    relative = first[:3, :3].T @ second[:3, :3]
    skew = np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                     relative[1, 0]-relative[0, 1]], dtype=float)
    return float(math.atan2(float(np.linalg.norm(skew))/2., (float(np.trace(relative))-1.)/2.))


def quaternion_matrix(quaternion_xyzw):
    x, y, z, w = (_finite(v, "quaternion element") for v in quaternion_xyzw)
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not norm > 1e-12:
        raise KinematicsError("Quaternion must be nonzero")
    x, y, z, w = (v/norm for v in (x, y, z, w))
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=float)


def quaternion_xyzw_from_matrix(rotation):
    """Unit quaternion for a rotation matrix; the branch with the largest trace term."""
    m = np.asarray(rotation, dtype=float)[:3, :3]
    trace = float(np.trace(m))
    if trace > 0.:
        s = math.sqrt(trace+1.)*2.
        w, x, y, z = .25*s, (m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s
    else:
        i = int(np.argmax(np.diagonal(m)))
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(1.+m[i, i]-m[j, j]-m[k, k])*2.
        q = [0., 0., 0.]
        q[i], q[j], q[k] = .25*s, (m[j, i]+m[i, j])/s, (m[k, i]+m[i, k])/s
        w, (x, y, z) = (m[k, j]-m[j, k])/s, q
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    return (x/norm, y/norm, z/norm, w/norm)


class UrdfChain:
    """Read-only kinematic tree; constructing it opens no driver and no device."""

    def __init__(self, text, *, source="<memory>"):
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if not 0 < len(data) <= MAX_URDF_BYTES:
            raise KinematicsError("URDF must be nonempty and under 8 MiB")
        try:
            root = ElementTree.fromstring(data.decode("utf-8"))
        except (ElementTree.ParseError, UnicodeDecodeError) as exc:
            raise KinematicsError(f"URDF could not be parsed: {exc}") from exc
        self.source, self.digest = source, hashlib.sha256(data).hexdigest()
        self.link_names = tuple(link.get("name") for link in root.findall("link"))
        if len(set(self.link_names)) != len(self.link_names) or not all(self.link_names):
            raise KinematicsError("URDF links must have distinct names")
        self._joints, children = OrderedDict(), set()
        for element in root.findall("joint"):
            name, kind = element.get("name"), element.get("type")
            parent, child = element.find("parent"), element.find("child")
            if not name or name in self._joints or kind not in SUPPORTED_TYPES or parent is None or child is None:
                raise KinematicsError(f"Unsupported or malformed URDF joint: {name}")
            parent_link, child_link = parent.get("link"), child.get("link")
            if (parent_link not in self.link_names or child_link not in self.link_names
                    or child_link in children):
                raise KinematicsError(f"URDF joint {name} does not connect two distinct known links")
            children.add(child_link)
            axis = element.find("axis")
            limit = element.find("limit")
            bounds = None
            if kind == "revolute" and limit is not None and (
                    limit.get("lower") is not None or limit.get("upper") is not None):
                if limit.get("lower") is None or limit.get("upper") is None:
                    raise KinematicsError(f"URDF joint {name} declares only one side of its range")
                try:
                    bounds = (_finite(float(limit.get("lower")), "lower limit"),
                              _finite(float(limit.get("upper")), "upper limit"))
                except ValueError as exc:
                    raise KinematicsError(f"URDF joint {name} has a non-numeric limit") from exc
                if bounds[0] >= bounds[1]:
                    raise KinematicsError(f"URDF joint {name} has an empty declared range")
            self._joints[name] = {
                "name": name, "type": kind, "parent": parent_link, "child": child_link,
                "origin": origin_transform(element),
                "axis": np.asarray([float(v) for v in (axis.get("xyz", "1 0 0") if axis is not None
                                                       else "1 0 0").split()], dtype=float),
                # A continuous joint is deliberately left without a declared range.
                "declared_limit_rad": bounds}
        roots = [name for name in self.link_names if name not in children]
        if len(roots) != 1:
            raise KinematicsError("URDF must describe exactly one kinematic root")
        self.root_link = roots[0]
        self._parent = {joint["child"]: joint for joint in self._joints.values()}
        self.actuated_joints = tuple(name for name, joint in self._joints.items()
                                     if joint["type"] in REVOLUTE_TYPES)

    @classmethod
    def from_path(cls, path):
        with open(path, "rb") as stream:
            return cls(stream.read(), source=str(path))

    @property
    def declared_limits_rad(self):
        """Declared model ranges; None means the model states no bound."""
        return {name: self._joints[name]["declared_limit_rad"] for name in self.actuated_joints}

    def joint_type(self, name):
        if name not in self._joints:
            raise KinematicsError(f"Unknown joint: {name}")
        return self._joints[name]["type"]

    def joint_origin(self, name):
        """The joint's fixed parent-relative origin; rotation never changes it."""
        if name not in self._joints:
            raise KinematicsError(f"Unknown joint: {name}")
        return self._joints[name]["origin"].copy()

    def chain(self, link):
        """Joint names from the root down to link, nearest the root first."""
        if link not in self.link_names:
            raise KinematicsError(f"Unknown link: {link}")
        joints, current = [], link
        while current in self._parent:
            joint = self._parent[current]
            joints.append(joint["name"])
            current = joint["parent"]
            if len(joints) > MAX_CHAIN_DEPTH:
                raise KinematicsError("URDF chain exceeds the supported depth")
        return tuple(reversed(joints))

    def actuated_chain(self, link):
        return tuple(name for name in self.chain(link) if self._joints[name]["type"] in REVOLUTE_TYPES)

    def base_from_link(self, positions, link, *, base=None):
        """Compose the model's fixed origins and revolute axes; no IK, no TF."""
        if base is not None and base not in self.link_names:
            raise KinematicsError(f"Unknown base link: {base}")
        transform = np.eye(4)
        started = base is None
        for name in self.chain(link):
            joint = self._joints[name]
            if not started:
                started = joint["parent"] == base
                if not started:
                    continue
            step = joint["origin"].copy()
            if joint["type"] in REVOLUTE_TYPES:
                if name not in positions:
                    raise KinematicsError(f"Missing joint position: {name}")
                step[:3, :3] = step[:3, :3] @ _rotation(joint["axis"], _finite(positions[name], name))
            transform = transform @ step
        if not started:
            raise KinematicsError(f"Link {link} is not a descendant of {base}")
        return transform

    def within_declared_limits(self, positions):
        """Report declared-range violations; continuous joints are never bounded here."""
        violations = []
        for name in self.actuated_joints:
            bounds = self._joints[name]["declared_limit_rad"]
            if name in positions and bounds is not None and not bounds[0] <= positions[name] <= bounds[1]:
                violations.append(name)
        return violations


def driver_joint_positions(names, positions, *, prefix=DRIVER_JOINT_PREFIX):
    """Map published /joint_states names onto the model's prefixed joint names."""
    if len(names) != len(positions):
        raise KinematicsError("Joint names and positions disagree in length")
    mapped = {}
    for name, value in zip(names, positions):
        if not isinstance(name, str) or not name:
            raise KinematicsError("Joint names must be nonempty strings")
        mapped[prefix+name] = _finite(value, name)
    if len(mapped) != len(names):
        raise KinematicsError("Published joint names must be distinct")
    return mapped


def frame_agreement(chain, samples, *, base=None, candidates=None):
    """Rank model links by how closely they reproduce published EE poses.

    This is diagnostic evidence about which frame a driver reports, not proof
    that the model matches the physical robot. Samples drawn from one posture
    constrain the frame choice at that posture only.
    """
    rows = []
    prepared = []
    for index, sample in enumerate(samples):
        try:
            positions = sample["positions"]
            position = np.asarray([_finite(v, "EE position") for v in sample["position_m"]], dtype=float)
            rotation = quaternion_matrix(sample["quaternion_xyzw"])
        except (KeyError, TypeError) as exc:
            raise KinematicsError(f"Sample {index} lacks positions, position_m and quaternion_xyzw") from exc
        prepared.append((positions, position, rotation))
    if not prepared:
        raise KinematicsError("At least one published EE sample is required")
    for link in (candidates if candidates is not None else chain.link_names):
        if link not in chain.link_names:
            raise KinematicsError(f"Unknown candidate link: {link}")
        try:
            errors = []
            for positions, position, rotation in prepared:
                pose = chain.base_from_link(positions, link, base=base)
                errors.append((float(np.linalg.norm(pose[:3, 3]-position)),
                               rotation_angle(pose[:3, :3], rotation)))
        except KinematicsError:
            continue
        rows.append({"link": link,
                     "max_position_error_m": max(e[0] for e in errors),
                     "max_rotation_error_rad": max(e[1] for e in errors),
                     "samples": len(errors)})
    rows.sort(key=lambda row: (row["max_position_error_m"], row["max_rotation_error_rad"]))
    return {"base": base or chain.root_link, "model_digest": chain.digest, "ranked": rows,
            "semantics": ("which model frame reproduces the driver's published pose; "
                          "not evidence that the model matches the physical robot, and "
                          "samples from one posture constrain only that posture")}
