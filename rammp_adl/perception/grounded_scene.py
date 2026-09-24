"""Untagged objects: keyframes to Astra for meaning, local depth for placement.

The scene keeps the entities Astra named in keyframes, each with the metric
geometry measured under its box. Discovery happens at intake and, while the
arm is idle, whenever a selected keyframe shows a moved camera or a changed
scene. Between keyframes an entity is carried by its base-frame position:
re-observation predicts where it must appear and checks the depth there.
"""
from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..contracts import digest
from ..handlers import BackendFailure
from ..hardware_backend import ObservationMeasurement
from ..motion.collision_guard import MEASURED_D405_MOUNT, mount_transform
from ..motion.sheppy_client import JOINTS
from ..world import MetricPose
from .geometry import PerceptionError, require_ids
from .keyframes import KeyframeSelector, StampReceiptClock, camera_moved, encode_keyframe
from .object_geometry import box_to_pixels, lift_point, object_geometry, points_in_region, pose_roles, support_plane, to_base

CAMERA_ROLE = "wrist"


class SceneError(Exception):
    def __init__(self, status, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def slug(label):
    text = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return text[:32] or "object"


class GroundedScene:
    def __init__(self, *, client, chain, calibration_id, camera_id="wrist_d405", clock_mapper=None,
                 face_screen=None, selector=None, pose_validity_s=60., max_timestamp_uncertainty_s=.15,
                 planning_frame="base_link", camera_parent=MEASURED_D405_MOUNT["parent_link"],
                 mount=MEASURED_D405_MOUNT, egress=None, geometry_options=None, match_distance_m=.15,
                 max_entities=16, keep_keyframes=4, dump_dir=None):
        require_ids(calibration_id, camera_id, planning_frame)
        self.client, self.chain, self.calibration_id, self.camera_id = client, chain, calibration_id, camera_id
        self.camera_role, self.planning_frame, self.camera_parent = CAMERA_ROLE, planning_frame, camera_parent
        self.ee_from_camera = mount_transform(mount)
        self.clock_mapper = clock_mapper or StampReceiptClock(max_latency_s=max_timestamp_uncertainty_s)
        self.face_screen, self.selector = face_screen, selector or KeyframeSelector()
        self.pose_validity_s, self.match_distance_m, self.max_entities = float(pose_validity_s), float(match_distance_m), int(max_entities)
        self.egress = dict(max_long_edge=640, max_bytes=200000, **(egress or {}))
        self.dump_dir = None if dump_dir is None else Path(dump_dir)
        self.recorder = None                            # called with each selected keyframe, for the offline bench
        self.geometry_options = dict(geometry_options or {})
        self.keyframes = deque(maxlen=keep_keyframes)
        self.latest = None
        self.sent = deque(maxlen=32)                    # capture IDs already shown to the model
        self.entities = {}
        self.reasoner = None
        self.world = None
        self._last_pair = None
        self._lock = threading.RLock()
        self._counter = {}
        self.clock = time.monotonic

    # -- capture side ---------------------------------------------------------
    def camera_pose(self):
        live = self.client.live_joints()
        if live is None:
            return None, None
        configuration = dict(zip(JOINTS, live["position_rad"]))
        return self.chain.base_from_link(configuration, self.camera_parent) @ self.ee_from_camera, tuple(live["position_rad"])

    def on_pair(self, pair, *, force=False):
        metadata = pair.metadata
        receipts = metadata.get("received_at_monotonic_s") or {}
        for key, stamp in metadata["source_stamps_ns"].items():
            if key in receipts:
                self.clock_mapper.note(stamp, receipts[key])
        with self._lock:
            self._last_pair = pair
            still_since = self.client.still_since_s()
            if still_since is None:
                return None
            pose, joints = self.camera_pose()
            if pose is None:
                return None
            keyframe = self.selector.consider(pair, base_from_camera=pose, joints=joints, still_since=still_since,
                                              clock_mapper=self.clock_mapper, now=self.clock(), force=force)
            if keyframe is not None:
                self.keyframes.append(keyframe)
                self.latest = keyframe
        if keyframe is not None and self.recorder is not None:
            self.recorder(keyframe)                     # outside the lock: a disk write never holds up the scene
        return keyframe

    def current_keyframe(self, *, max_age_s=10.):
        """The latest keyframe if the camera has not moved since and it is fresh."""
        with self._lock:
            keyframe = self.latest
            if keyframe is None:
                return None
            pose, _ = self.camera_pose()
            if pose is None or self.clock()-keyframe.oldest_capture_s > max_age_s:
                return None
            if camera_moved(keyframe.base_from_camera, pose, tolerance_m=self.selector.move_tolerance_m,
                            tolerance_rad=self.selector.move_tolerance_rad):
                return None
            return keyframe

    def keyframe_for_request(self, *, max_age_s=10.):
        """The current keyframe, or one forced from the newest capture; else why not."""
        keyframe = self.current_keyframe(max_age_s=max_age_s)
        if keyframe is not None:
            return keyframe
        with self._lock:
            pair = self._last_pair
            if pair is not None:
                keyframe = self.on_pair(pair, force=True)
        if keyframe is not None:
            return keyframe
        raise SceneError("NEED_OBSERVATION", self.refusal_reason())

    def refusal_reason(self):
        """Why no keyframe can be taken right now, from what the scene can see."""
        with self._lock:
            pair = self._last_pair
        if pair is None:
            return "no wrist camera frames have been received; check the wrist_camera container"
        receipts = pair.metadata.get("received_at_monotonic_s") or {}
        age = self.clock()-receipts["rgb"] if isinstance(receipts.get("rgb"), (int, float)) else None
        if age is not None and age > self.selector.max_capture_age_s:
            return f"the last wrist frame is {age:.1f} s old; the wrist camera stream has stalled"
        if self.client.still_since_s() is None:
            return "the arm is not verifiably still; hold it still with the scene in view"
        mapped, uncertainty = self.clock_mapper(pair.metadata["source_stamps_ns"]["rgb"]/10**9)
        if not all(math.isfinite(v) for v in (mapped, uncertainty)):
            return "the wrist frame's stamp cannot be bounded against the host clock; the camera clock is not the host's"
        return "no still, fresh wrist keyframe; hold the arm still with the scene in view"

    async def wait_for_keyframe(self, *, timeout_s=2., poll_s=.2, max_age_s=10.):
        """keyframe_for_request with a short wait for the next still capture."""
        deadline = self.clock()+timeout_s
        while True:
            try:
                return self.keyframe_for_request(max_age_s=max_age_s)
            except SceneError:
                if self.clock() >= deadline:
                    raise
            await asyncio.sleep(poll_s)

    async def surface_normal(self, *, timeout_s=1.5, min_range_m=.06, max_range_m=.6):
        """The dominant surface's normal (toward the camera) in base frame from a fresh still keyframe."""
        try:
            keyframe = await self.wait_for_keyframe(timeout_s=timeout_s)
        except SceneError:
            return None, None

        def measure():
            whole = to_base(points_in_region(keyframe, (0, 0, keyframe.width, keyframe.height), stride=4,
                                             min_range_m=min_range_m, max_range_m=max_range_m), keyframe.base_from_camera)
            plane = support_plane(whole, camera_position=keyframe.camera_position_base, min_points=100)
            return None if plane is None else [float(v) for v in plane["normal"]]
        return await asyncio.to_thread(measure), keyframe

    def pending_refresh(self):
        with self._lock:
            keyframe = self.latest
        return keyframe if keyframe is not None and keyframe.capture_id not in self.sent and keyframe.reason != "requested" else None

    def crop_for(self, keyframe):
        crop = encode_keyframe(keyframe, camera_id=self.camera_id, calibration_id=self.calibration_id,
                               screen=self.face_screen, **self.egress)
        self._dump(crop)
        return crop

    dump_keep = 40

    def _dump(self, crop):
        """Write the exact JPEG sent to the model, so what it saw can be inspected later."""
        if self.dump_dir is None:
            return
        try:
            folder = Path(self.dump_dir)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{crop.image_id}.jpg"
            if not path.exists():
                path.write_bytes(crop.jpeg_bytes)
            files = sorted(folder.glob("*.jpg"))
            for old_file in files[:-self.dump_keep]:
                old_file.unlink()
        except OSError:
            pass

    def latest_crop(self):
        """The current keyframe encoded for egress, or None when there is none or it is withheld."""
        keyframe = self.current_keyframe()
        if keyframe is None:
            return None
        try:
            return self.crop_for(keyframe)
        except PerceptionError:
            return None

    # -- geometry ---------------------------------------------------------------
    def _plane_around(self, keyframe, region, *, margin=1.):
        """The plane the boxed thing sits on or protrudes from: fitted around the box, else over the whole view."""
        x0, y0, x1, y1 = region
        dx, dy = max(40, int((x1-x0)*margin)), max(40, int((y1-y0)*margin))
        around = to_base(points_in_region(keyframe, (x0-dx, y0-dy, x1+dx, y1+dy), stride=4), keyframe.base_from_camera)
        plane = support_plane(around, camera_position=keyframe.camera_position_base)
        if plane is not None:
            return plane
        whole = to_base(points_in_region(keyframe, (0, 0, keyframe.width, keyframe.height), stride=8), keyframe.base_from_camera)
        return support_plane(whole, camera_position=keyframe.camera_position_base)

    def measure_box(self, keyframe, box_normalized, *, grasp_point_xy=None):
        region = box_to_pixels(box_normalized, keyframe.width, keyframe.height)
        plane = self._plane_around(keyframe, region)
        inside = to_base(points_in_region(keyframe, region, **self.geometry_options), keyframe.base_from_camera)
        geometry = object_geometry(inside, plane)
        grasp_point = lift_point(keyframe, grasp_point_xy) if grasp_point_xy is not None else None
        tool_x = self.chain.base_from_link(dict(zip(JOINTS, keyframe.joints_rad)), self.camera_parent)[:3, 0]
        roles = pose_roles(geometry, camera_position_base=keyframe.camera_position_base, grasp_point=grasp_point,
                           tool_x_axis=tool_x)
        return {"box_px": list(region), "geometry": geometry, "grasp": roles,
                "grasp_point_m": None if grasp_point is None else [float(v) for v in grasp_point]}

    def measure_surface(self, keyframe, box_normalized):
        """A flat part in the box: its plane, image-aligned in-plane axes, centre and extents."""
        region = box_to_pixels(box_normalized, keyframe.width, keyframe.height)
        inside = to_base(points_in_region(keyframe, region, stride=2), keyframe.base_from_camera)
        if len(inside) < 100:
            return None
        # The surface's own plane: the rest of the view may be another depth entirely.
        plane = support_plane(inside, camera_position=keyframe.camera_position_base, min_points=100)
        if plane is None:
            return None
        origin, normal = np.asarray(plane["origin_m"]), np.asarray(plane["normal"])
        on = inside[np.abs((inside-origin) @ normal) <= .015]
        if len(on) < 100:
            return None
        camera = keyframe.base_from_camera
        right = camera[:3, 0]-(camera[:3, 0] @ normal)*normal
        up = -camera[:3, 1]-(-camera[:3, 1] @ normal)*normal
        if np.linalg.norm(right) < 1e-6 or np.linalg.norm(up) < 1e-6:
            return None
        right, up = right/np.linalg.norm(right), up/np.linalg.norm(up)
        r, u = (on-origin) @ right, (on-origin) @ up
        centre = origin+right*float((r.min()+r.max())/2.)+up*float((u.min()+u.max())/2.)
        return {"normal": normal.tolist(), "right": right.tolist(), "up": up.tolist(), "centre_m": centre.tolist(),
                "width_m": float(r.max()-r.min()), "height_m": float(u.max()-u.min()), "points": int(len(on)),
                "rms_m": plane["rms_m"]}

    def door_for(self, handle):
        """The measured surface the handle sits on, and the handle's offsets from its edges."""
        with self._lock:
            surfaces = [dict(r) for r in self.entities.values() if r.get("surface_geometry") and r["entity_id"] != handle["entity_id"]]
        if handle["position_m"] is None:
            return None
        position = np.asarray(handle["position_m"])
        best, best_score = None, math.inf
        for surface in surfaces:
            geometry = surface["surface_geometry"]
            centre, right, up = (np.asarray(geometry[k]) for k in ("centre_m", "right", "up"))
            r, u = float((position-centre) @ right), float((position-centre) @ up)
            outside = max(0., abs(r)-geometry["width_m"]/2.)+max(0., abs(u)-geometry["height_m"]/2.)
            if outside < best_score:
                best, best_score = (surface, r, u), outside
        if best is None or best_score > .10:
            return self._door_from_plane(handle)
        surface, r, u = best
        geometry = surface["surface_geometry"]
        return {"entity_id": surface["entity_id"], "label": surface["label"], "width_m": geometry["width_m"],
                "height_m": geometry["height_m"], "normal": geometry["normal"], "right": geometry["right"], "up": geometry["up"],
                "handle_offsets_m": {"left": float(r+geometry["width_m"]/2.), "right": float(geometry["width_m"]/2.-r),
                                     "bottom": float(u+geometry["height_m"]/2.), "top": float(geometry["height_m"]/2.-u)}}

    def keyframe_by_id(self, capture_id):
        with self._lock:
            for keyframe in self.keyframes:
                if keyframe.capture_id == capture_id:
                    return keyframe
        return None

    def _door_from_plane(self, handle):
        """No discovered surface fits: the dominant plane of the handle's keyframe is the door."""
        keyframe = self.keyframe_by_id(handle.get("keyframe"))
        if keyframe is None or handle["position_m"] is None:
            return None
        whole = to_base(points_in_region(keyframe, (0, 0, keyframe.width, keyframe.height), stride=4), keyframe.base_from_camera)
        plane = support_plane(whole, camera_position=keyframe.camera_position_base)
        if plane is None:
            return None
        origin, normal = np.asarray(plane["origin_m"]), np.asarray(plane["normal"])
        on = whole[np.abs((whole-origin) @ normal) <= .015]
        if len(on) < 200:
            return None
        camera = keyframe.base_from_camera
        right = camera[:3, 0]-(camera[:3, 0] @ normal)*normal
        up = -camera[:3, 1]-(-camera[:3, 1] @ normal)*normal
        if np.linalg.norm(right) < 1e-6 or np.linalg.norm(up) < 1e-6:
            return None
        right, up = right/np.linalg.norm(right), up/np.linalg.norm(up)
        position = np.asarray(handle["position_m"])
        r, u = (on-origin) @ right, (on-origin) @ up
        hr, hu = float((position-origin) @ right), float((position-origin) @ up)
        width, height = float(r.max()-r.min()), float(u.max()-u.min())
        return {"entity_id": None, "label": handle["attached_to"] or "surface", "width_m": width, "height_m": height,
                "normal": normal.tolist(), "right": right.tolist(), "up": up.tolist(),
                "handle_offsets_m": {"left": float(hr-r.min()), "right": float(r.max()-hr),
                                     "bottom": float(hu-u.min()), "top": float(u.max()-hu)},
                "source": "dominant plane of the keyframe; may include the frame or a neighbouring door"}

    # -- discovery ----------------------------------------------------------------
    def _assign_id(self, label):
        base = slug(label)
        self._counter[base] = self._counter.get(base, 0)+1
        return f"{base}_{self._counter[base]}"

    def _absorb_locked(self, keyframe, candidates, source):
        """Measure every candidate first, then merge under the lock for as short a time as possible."""
        measured = [self._measure_candidate(keyframe, candidate) for candidate in candidates]
        with self._lock:
            return self._absorb(keyframe, measured, source=source)

    def _measure_candidate(self, keyframe, candidate):
        """Geometry for one candidate, a pure function of the keyframe: no lock, any thread."""
        candidate = {**candidate, "label": str(candidate["label"]),
                     "box_xyxy_normalized": [float(v) for v in candidate["box_xyxy_normalized"]],
                     "confidence": float(candidate["confidence"]),
                     "kind": str(candidate.get("kind", "free_object")), "attached_to": str(candidate.get("attached_to", "")),
                     "grasp_point_given": bool(candidate.get("grasp_point_given", False)),
                     "grasp_point_xy_normalized": [float(v) for v in candidate.get("grasp_point_xy_normalized", (0., 0.))]}
        surface_geometry = None
        region = list(box_to_pixels(candidate["box_xyxy_normalized"], keyframe.width, keyframe.height))
        if candidate["kind"] == "surface":
            # A surface is the plane itself, whatever protrudes from it
            # inside the box belongs to other entities; it is never grasped.
            try:
                surface_geometry = self.measure_surface(keyframe, candidate["box_xyxy_normalized"])
            except PerceptionError:
                surface_geometry = None
            measured = {"box_px": region, "geometry": None,
                        "grasp": {"strategy": "none", "roles": {}, "reason": "a surface is not grasped"}}
        else:
            try:
                measured = self.measure_box(keyframe, candidate["box_xyxy_normalized"],
                                            grasp_point_xy=candidate["grasp_point_xy_normalized"] if candidate["grasp_point_given"] else None)
            except PerceptionError as exc:
                measured = {"box_px": region, "geometry": None, "grasp": {"strategy": "none", "roles": {}, "reason": str(exc)}}
            if candidate["kind"] == "other":
                try:
                    surface_geometry = self.measure_surface(keyframe, candidate["box_xyxy_normalized"])
                except PerceptionError:
                    surface_geometry = None
        return candidate, measured, surface_geometry

    def _absorb(self, keyframe, measurements, *, source):
        """Merge measured candidates into the entity table; returns the ids seen. Caller holds the lock."""
        seen = []
        for candidate, measured, surface_geometry in measurements:
            if measured["geometry"] is not None:
                position = np.asarray(measured["geometry"]["centroid_m"])
            elif surface_geometry is not None:
                position = np.asarray(surface_geometry["centre_m"])
            else:
                position = None
            match, best = None, math.inf
            if position is not None:
                # The same place with the same kind is the same thing, however
                # the model words it this time; ids and first labels persist.
                for entity_id, record in self.entities.items():
                    if record["position_m"] is None or record["kind"] != candidate["kind"] or entity_id in seen:
                        continue
                    distance = float(np.linalg.norm(np.asarray(record["position_m"])-position))
                    if distance <= self.match_distance_m and distance < best:
                        match, best = entity_id, distance
            entity_id = candidate.get("entity_id") if source == "ground" else match
            if entity_id is None:
                if len(self.entities) >= self.max_entities:
                    continue
                entity_id = self._assign_id(candidate["label"])
            record = self.entities.setdefault(entity_id, {"entity_id": entity_id, "label": candidate["label"], "position_m": None,
                                                           "pose_roles": [], "geometry": None, "grasp": None, "keyframe": None,
                                                           "box": None, "confidence": 0., "seen_at": None, "grounding": None,
                                                           "kind": "free_object", "attached_to": "", "camera_pose": None, "aliases": []})
            if candidate["label"] != record["label"] and candidate["label"] not in record.setdefault("aliases", []):
                record["aliases"].append(candidate["label"])
            record.update(position_m=None if position is None else position.tolist(),
                          pose_roles=sorted(measured["grasp"]["roles"]) if measured["geometry"] is not None else [],
                          geometry=measured["geometry"], grasp=measured["grasp"], keyframe=keyframe.capture_id,
                          box=list(candidate["box_xyxy_normalized"]), confidence=float(candidate["confidence"]),
                          seen_at=keyframe.oldest_capture_s, grounding=digest(candidate),
                          kind=record["kind"] if record["position_m"] is not None or source == "ground" else candidate["kind"],
                          attached_to=record["attached_to"] or candidate["attached_to"],
                          camera_pose=keyframe.base_from_camera.tolist())
            if surface_geometry is not None:
                record["surface_geometry"] = surface_geometry
            seen.append(entity_id)
        return seen

    async def discover(self, reasoner, context, task_text, *, keyframe=None):
        keyframe = keyframe or await self.wait_for_keyframe()
        try:
            crop = await asyncio.to_thread(self.crop_for, keyframe)
        except PerceptionError as exc:
            raise SceneError("EGRESS_REFUSED", str(exc)) from exc
        result = await reasoner.discover_scene(context, [crop], task_text=task_text)
        self.sent.append(keyframe.capture_id)
        if result.status not in ("OK", "NO_DETECTION"):
            raise SceneError(result.status, result.detail)
        # Geometry is numpy-heavy; it runs off the runtime loop so the
        # supervisor's status tick keeps its budget.
        seen = await asyncio.to_thread(self._absorb_locked, keyframe, result.candidates, "discover")
        search = result.proposal or {"target_visible": True, "search_hint": "none", "search_note": ""}
        return {"keyframe": keyframe.capture_id, "reason": keyframe.reason, "entities": seen,
                "descriptors": self.descriptors(seen), "status": result.status, "search": search}

    async def refresh(self, reasoner, context):
        keyframe = self.pending_refresh()
        if keyframe is None:
            return None
        return await self.discover(reasoner, context, "", keyframe=keyframe)

    def handles(self):
        """Discovered parts that are attached to something and can be pulled or slid."""
        with self._lock:
            return [dict(record) for record in self.entities.values()
                    if record["kind"] == "handle" and record["geometry"] is not None and record["grasp"]["roles"]]

    def articulate(self, reasoner, context, entity_id, *, store, surface_entity_id=None, images=()):
        """Ask for a constraint for one handle and derive its metric record.

        Returns a coroutine result: the record, the symbolic constraint, and
        the surface entity descriptor the handle is attached to.
        """
        from ..constraints import (apply_parameters, context_constraint, history_summary, load_demonstration,
                                   metric_constraint, next_parameters, slug as constraint_slug)

        async def run():
            with self._lock:
                record = self.entities.get(entity_id)
                if record is None or record["kind"] != "handle" or record["geometry"] is None:
                    raise SceneError("NEED_OBSERVATION", f"{entity_id} is not a measured handle")
                handle = dict(record)
            label = handle["attached_to"] or handle["label"]
            door = self.door_for(handle)
            surface_id = surface_entity_id or (door["entity_id"] if door else f"{constraint_slug(label)}_surface")
            constraint_id = f"{constraint_slug(label)}_constraint"
            previous = store.load(label) if store is not None else None
            history = history_summary(previous) if previous else []
            crops = list(images)
            if not crops:
                keyframe = self.keyframe_by_id(handle["keyframe"]) or self.latest
                if keyframe is not None:
                    try:
                        crops = [await asyncio.to_thread(self.crop_for, keyframe)]
                    except PerceptionError:
                        crops = []
            if previous and store is not None:
                # The end frame of the last success is the best hint of where the part goes.
                demonstration = load_demonstration(store, previous, limit=2)
                if demonstration:
                    crops = (crops+[demonstration[-1]])[:2]
            result = await reasoner.propose_constraint(context, entity_id, label=label, geometry=handle["geometry"],
                                                       door=door, history=history, images=crops)
            if result.status == "OK" and result.proposal is not None:
                proposal = result.proposal
            elif door is not None and result.status in ("AMBIGUOUS", "UNSUPPORTED"):
                # The door is measured: the hinge is on the edge farther from
                # the handle, and a cabinet opens toward whoever pulls it.
                offsets = door["handle_offsets_m"]
                side = "left" if offsets["left"] > offsets["right"] else "right"
                proposal = {"status": "OK", "kind": "revolute", "hinge_side": side, "opening": "pull",
                            "door_width_m": float(door["width_m"]), "range": 1., "contact_effort_nm": 8.,
                            "rationale": f"local default: measured door {door['width_m']:.2f} m wide, handle nearer the "
                                         f"{'right' if side == 'left' else 'left'} edge; model said {result.status}: {result.detail[:120]}"}
            else:
                raise SceneError(result.status, result.detail)
            metric = metric_constraint(proposal, handle["geometry"], handle["camera_pose"], constraint_id=constraint_id,
                                       entity_id=entity_id, label=label, surface_entity_id=surface_id, door=door)
            if previous:
                metric["attempts"] = previous.get("attempts", [])
                metric["parameters_version"] = previous.get("parameters_version", 1)+1
                same = all(abs(float(previous.get(k, 0.))-float(metric.get(k, 0.))) < 1e-6 for k in ("door_width_m", "contact_effort_nm")) \
                    and previous.get("hinge_side") == metric.get("hinge_side") and previous.get("opening") == metric.get("opening")
                refinement = next_parameters(previous) if same else None
                if refinement:
                    metric = apply_parameters(metric, handle["geometry"], handle["camera_pose"], refinement, door=door)
                    metric["refined_locally"] = refinement
            if store is not None:
                store.save(metric)
            return {"record": metric, "constraint": context_constraint(metric),
                    "surface": {"entity_id": surface_id, "label": door["label"] if door else label, "pose_roles": []},
                    "handle": entity_id, "history": history, "door": door, "proposal": proposal}
        return run()

    def descriptors(self, entity_ids=None):
        with self._lock:
            ids = list(self.entities) if entity_ids is None else list(entity_ids)
            return [{"entity_id": entity_id, "label": self.entities[entity_id]["label"],
                     "pose_roles": list(self.entities[entity_id]["pose_roles"])} for entity_id in ids if entity_id in self.entities]

    def visible_entities(self, now_monotonic, *, max_age_s=30.):
        with self._lock:
            return {entity_id: {"visible": record["seen_at"] is not None and 0 <= now_monotonic-record["seen_at"] <= max_age_s,
                                "capture_id": record["keyframe"], "report_digest": record["grounding"]}
                    for entity_id, record in self.entities.items()}

    def describe(self):
        return {"camera_id": self.camera_id, "face_screen": self.face_screen is not None,
                "entities": {k: {"label": v["label"], "pose_roles": v["pose_roles"], "strategy": (v["grasp"] or {}).get("strategy")}
                             for k, v in self.entities.items()}}

    # -- the observer -------------------------------------------------------------
    def bind_world(self, world):
        self.world = world

    def stop(self):
        pass

    def _dependencies(self, snapshot, entity_id):
        identities = snapshot.identities()
        keys = ("execution_epoch", "calibration_id", "base_epoch", "entity:"+entity_id)
        try:
            return {name: identities[name] for name in keys}, identities
        except KeyError as exc:
            raise BackendFailure("stale_state", "observation target is not in this world") from exc

    async def __call__(self, args, context):
        entity_id, camera, purpose = args.get("entity_id"), args.get("camera"), args.get("purpose")
        if camera != self.camera_role:
            raise BackendFailure("no_detection", "only the wrist camera is grounded")
        if purpose not in ("state", "pose"):
            raise BackendFailure("geometry_invalid", "grounded observation establishes existence and pose only")
        with self._lock:
            record = self.entities.get(entity_id)
        if record is None:
            raise BackendFailure("no_detection", "the requested entity was never discovered")
        snapshot = context.snapshot
        if snapshot is None or context.execution_epoch != snapshot.execution_epoch or context.cancel_event.is_set():
            raise BackendFailure("cancelled", "observation has no active input snapshot")
        if snapshot.context["calibration_id"] != self.calibration_id:
            raise BackendFailure("geometry_invalid", "world calibration identity differs from the measured wrist mount")
        dependencies, identities = self._dependencies(snapshot, entity_id)
        keyframe = self.current_keyframe()
        if keyframe is None:
            raise BackendFailure("stale_state", "no still, fresh wrist keyframe to observe from")
        now = self.world.clock() if self.world is not None else self.clock()
        captured_at = keyframe.oldest_capture_s
        if not captured_at <= now < captured_at+self.pose_validity_s:
            raise BackendFailure("stale_state", "keyframe is outside the observation validity window")
        assertions = [{"predicate": "entity_exists", "args": {"entity_id": entity_id}, "validity": "true"},
                      {"predicate": "observation_valid", "args": {"entity_id": entity_id, "purpose": purpose}, "validity": "true"}]
        data = {"kind": "grounded_object", "label": record["label"], "keyframe": keyframe.capture_id,
                "camera_id": self.camera_id, "frames_uploaded": 0, "physical_contact_state_measured": False,
                "collision_geometry_complete": False}
        poses = ()
        if purpose == "state":
            check = self._carried(record, keyframe)
            if not check["consistent"]:
                raise BackendFailure("no_detection", check["detail"])
            data["carried"] = check
        else:
            reused = record["geometry"] is not None and (record["keyframe"] == keyframe.capture_id
                                                         or self._still_there(record, keyframe))
            if not reused:
                if self.reasoner is None:
                    raise BackendFailure("no_detection", "no reasoner is bound; the entity cannot be re-grounded")
                try:
                    crop = await asyncio.to_thread(self.crop_for, keyframe)
                except PerceptionError as exc:
                    raise BackendFailure("no_detection", str(exc)) from exc
                result = await self.reasoner.ground_target(snapshot.context, entity_id, [crop], query=record["label"])
                self.sent.append(keyframe.capture_id)
                data["frames_uploaded"] = 1
                if result.status != "OK":
                    raise BackendFailure("no_detection", f"grounding: {result.status} {result.detail}")
                chosen = await asyncio.to_thread(self._choose, record, keyframe, result.candidates)
                await asyncio.to_thread(self._absorb_locked, keyframe, [dict(chosen, entity_id=entity_id)], "ground")
                with self._lock:
                    record = self.entities[entity_id]
            grasp = record["grasp"] or {}
            if record["geometry"] is None or not grasp.get("roles"):
                raise BackendFailure("geometry_invalid", "the object's geometry gives no reachable grasp: "+str((grasp or {}).get("reason", "")))
            data.update(reused_keyframe=reused, geometry=record["geometry"], strategy=grasp["strategy"], box=record["box"])
        evidence_id = "grounded-observation-"+digest({"keyframe": keyframe.capture_id, "grounding": record["grounding"],
                                                       "entity": entity_id, "dependencies": dependencies,
                                                       "node": context.node_id, "attempt": context.attempt, "purpose": purpose})
        if purpose == "pose":
            revision = identities["entity:"+entity_id]+1
            poses = tuple(MetricPose(entity_id, role, tuple(pose["position_m"]), tuple(pose["orientation_xyzw"]),
                                     tuple(grasp["covariance"]), captured_at, self.planning_frame, revision,
                                     identities["calibration_id"], identities["base_epoch"], evidence_id, self.pose_validity_s)
                          for role, pose in grasp["roles"].items())
            assertions.extend({"predicate": "pose_valid", "args": {"entity_id": entity_id, "pose_role": role}, "validity": "true"}
                              for role in grasp["roles"])
        return ObservationMeasurement(entity_id, self.camera_role, purpose, evidence_id, captured_at, self.pose_validity_s,
                                      tuple(assertions), data, dependencies, metric_poses=poses)

    def _still_there(self, record, keyframe):
        """Measured from the same camera pose and the depth under its box unchanged: the measurement stands, no cloud call."""
        measured_in = self.keyframe_by_id(record["keyframe"])
        if measured_in is None or camera_moved(measured_in.base_from_camera, keyframe.base_from_camera,
                                               tolerance_m=self.selector.move_tolerance_m,
                                               tolerance_rad=self.selector.move_tolerance_rad):
            return False
        if record.get("box") is None:
            return False
        x0, y0, x1, y1 = box_to_pixels(record["box"], keyframe.width, keyframe.height)
        before, after = (np.asarray(k.depth_m, dtype=float)[y0:y1, x0:x1] for k in (measured_in, keyframe))
        both = np.isfinite(before) & np.isfinite(after) & (before > 0) & (after > 0)
        if before.shape != after.shape or int(both.sum()) < 20:
            return False
        return float((np.abs(before-after)[both] > .015).mean()) < .1

    def _carried(self, record, keyframe, *, tolerance_m=.05, patch=3):
        """Is the entity's remembered position still occupied in this keyframe?"""
        if record["position_m"] is None:
            return {"consistent": False, "detail": "the entity has no measured position to check"}
        projected = keyframe.project(record["position_m"])
        if projected is None:
            return {"consistent": False, "detail": "the entity's position is outside the current view"}
        u, v, expected = projected
        window = keyframe.depth_m[max(0, int(v)-patch):int(v)+patch+1, max(0, int(u)-patch):int(u)+patch+1]
        valid = window[np.isfinite(window) & (window > 0)]
        if not valid.size:
            return {"consistent": False, "detail": "no depth where the entity should be"}
        measured = float(np.median(valid))
        gap = measured-expected
        # Depth can be slightly nearer (the surface facing the camera) but not farther by more than the tolerance.
        consistent = -tolerance_m*2 <= gap <= tolerance_m
        return {"consistent": bool(consistent), "pixel": [u, v], "expected_depth_m": expected, "measured_depth_m": measured,
                "detail": "" if consistent else f"depth at the remembered position differs by {gap:.3f} m"}

    def _choose(self, record, keyframe, candidates):
        if not candidates:
            raise BackendFailure("no_detection", "grounding returned no candidate")
        if record["position_m"] is None:
            if len(candidates) == 1:
                return candidates[0]
            raise BackendFailure("no_detection", "ambiguous grounding with no remembered position")
        best, best_distance = None, math.inf
        for candidate in candidates:
            try:
                measured = self.measure_box(keyframe, candidate["box_xyxy_normalized"])
            except PerceptionError:
                continue
            distance = float(np.linalg.norm(np.asarray(measured["geometry"]["centroid_m"])-np.asarray(record["position_m"])))
            if distance < best_distance:
                best, best_distance = candidate, distance
        if best is None or best_distance > self.match_distance_m:
            raise BackendFailure("no_detection", "no grounding candidate matches the remembered position")
        return best
