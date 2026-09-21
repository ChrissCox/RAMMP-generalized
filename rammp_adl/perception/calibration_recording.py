"""Passive, local calibration evidence collection from existing ROS publishers.

This is a raw recording format, deliberately not extrinsics.fit_hand_eye input.
Receipt bracketing is a diagnostic association, never acquisition synchronization.
No commands, service clients, TF, clock calibration or selected IPPE branch.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

import numpy as np

from ..contracts import checked_copy, digest, strict_loads
from ..motion.driver_diagnostics import DriverStateRecording
from ..motion.driver_state import EE_MESSAGE_TYPES, RosDriverStateSource
from .fiducial import SingleMarkerObserver
from .geometry import PerceptionError
from .ros_locality import imagery_locality_report
from .ros_rgbd import RgbdSynchronizer, RosRgbdSource


def recording_config(document):
    """Explicit subscription mappings only; no executable hooks or motion flags."""
    config = checked_copy(document)
    if (not isinstance(config, dict) or set(config) != {"schema_version", "driver", "cameras"}
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise PerceptionError("Expected calibration recording config version 1")
    driver, cameras = config["driver"], config["cameras"]
    if (not isinstance(driver, dict) or set(driver) != {"domain_id", "joint_topic", "ee_topic", "ee_message_type"}
            or driver["ee_message_type"] not in EE_MESSAGE_TYPES):
        raise PerceptionError("Explicit inspected driver state interface is required")
    if not isinstance(cameras, dict) or not 1 <= len(cameras) <= 4:
        raise PerceptionError("Record one through four explicitly named cameras")
    endpoints = set()
    for name, settings in [("driver", driver), *cameras.items()]:
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
            raise PerceptionError("Camera names must be simple lowercase identifiers")
        if name in cameras and (name == "driver" or not isinstance(settings, dict)
                or set(settings) != {"domain_id", "reliability", *(k+"_topic" for k in RgbdSynchronizer.streams)}
                or settings["reliability"] not in ("reliable", "best_effort")):
            raise PerceptionError("Camera settings must contain domain, QoS and four explicit topics")
        domain = settings["domain_id"]
        if type(domain) is not int or not 0 <= domain <= 232:
            raise PerceptionError("ROS domains must be integers in 0..232")
        for key, topic in settings.items():
            if not key.endswith("_topic"):
                continue
            if (not isinstance(topic, str) or len(topic) > 256
                    or not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*", topic)
                    or (domain, topic) in endpoints):
                raise PerceptionError("Distinct absolute subscription topics are required")
            endpoints.add((domain, topic))
    return config


def associate_receipts(driver_pairs, camera_records, *, max_receipt_gap_s, source_integrity_ok=True):
    """Reference adjacent original publications without interpolation or retiming.

    The bound is on local callback receipts of both joint and EE publications.
    Cross-source header deltas are retained as numbers with unknown clock relation.
    """
    if (type(max_receipt_gap_s) not in (int, float) or not math.isfinite(max_receipt_gap_s)
            or not 0 < max_receipt_gap_s <= 1.):
        raise PerceptionError("Diagnostic receipt gap must be in (0, 1] seconds")
    stamps = [max(p.joints.received_at_monotonic_s, p.ee.received_at_monotonic_s) for p in driver_pairs]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        source_integrity_ok = False
    associations = []
    for record in camera_records:
        observation = record["observation"]
        receipt = record["metadata"]["received_at_monotonic_s"]["rgb"]
        row = {"capture_id": observation["capture_id"], "status": "no_receipt_bracket",
               "driver_indices": [], "nearest_driver_index": None,
               "maximum_receipt_gap_s": None, "camera_minus_driver_publication_ns": None,
               "clock_relation": "unknown; header deltas are not acquisition skew",
               "timestamp_uncertainty_s": None, "branch_resolution_evidence_id": None,
               "base_from_wrist": None, "calibration_sample_admitted": False}
        if not driver_pairs:
            row["status"] = "driver_unavailable"
        elif not source_integrity_ok:
            row["status"] = "source_integrity_failed"
        elif len(stamps) >= 2:
            after = bisect_left(stamps, receipt)
            # At an exact receipt match retain that sample plus its predecessor;
            # two distinct records are still required. Never extrapolate.
            if 0 < after < len(stamps):
                indices = [after-1, after]
                nearest = min(indices, key=lambda i: abs(receipt-stamps[i]))
                gap = max(abs(receipt-r) for i in indices for r in (
                    driver_pairs[i].joints.received_at_monotonic_s,
                    driver_pairs[i].ee.received_at_monotonic_s))
                row.update(driver_indices=indices, nearest_driver_index=nearest,
                           maximum_receipt_gap_s=gap,
                           camera_minus_driver_publication_ns=(record["metadata"]["source_stamps_ns"]["rgb"]
                               - driver_pairs[nearest].ee.source_stamp_ns),
                           status="receipt_bracketed" if gap <= max_receipt_gap_s else "receipt_gap_exceeded")
        if row["status"] == "receipt_bracketed" and observation["status"] != "provisional_pose_candidates":
            row["status"] = "marker_pose_unavailable"
        associations.append(row)
    return associations


def camera_integrity(records):
    """Latch changed identity, repeated frames or reversed clocks for the session."""
    identity, stamps, receipts = None, None, None
    captures = set()
    for record in records:
        meta, obs = record["metadata"], record["observation"]
        current = (digest(meta["rgb_info"]), digest(meta["depth_info"]), obs["marker_spec_digest"])
        if identity is not None and current != identity:
            return "camera intrinsics/frame or marker identity changed"
        if (obs["capture_id"] in captures or obs["source_stamps_ns"] != meta["source_stamps_ns"]
                or obs["rgb_info_digest"] != current[0] or obs["camera_frame"] != meta["rgb_info"]["frame_id"]):
            return "camera observation identity or provenance mismatch"
        if stamps is not None and any(meta["source_stamps_ns"][k] <= stamps[k] or
                meta["received_at_monotonic_s"][k] < receipts[k] for k in RgbdSynchronizer.streams):
            return "camera source or receipt clock repeated/reversed"
        identity, stamps, receipts = current, meta["source_stamps_ns"], meta["received_at_monotonic_s"]
        captures.add(obs["capture_id"])
    return None


def _publishers(node, topics):
    return {topic: [{"node": p.node_name, "namespace": p.node_namespace,
                     "message_type": p.topic_type, "endpoint_gid": list(p.endpoint_gid)}
                    for p in node.get_publishers_info_by_topic(topic)] for topic in topics}


def _publisher_match(publishers, expected):
    return all(len(publishers.get(topic, [])) == 1 and publishers[topic][0]["message_type"] == kind
               for topic, kind in expected.items())


def capture_calibration(*, config, output_dir, pose_label, duration_s=15.,
                        max_camera_samples=300, max_driver_pairs=10000, max_receipt_gap_s=.1,
                        marker_spec=None):
    """Record one operator-labelled pose window; never start or claim a driver.

    Domain IDs are per-context, not process-environment mutations. Detection runs
    independently for each warm camera while the main thread services robot state.
    Finite bounds are diagnostic storage/receipt settings, not commissioned limits.
    """
    config = recording_config(config)
    if (not isinstance(pose_label, str) or not 1 <= len(pose_label.strip()) <= 128
            or any(ord(c) < 32 for c in pose_label)):
        raise PerceptionError("A short operator pose label is required")
    if (type(duration_s) not in (int, float) or not math.isfinite(duration_s) or not .1 <= duration_s <= 60.
            or type(max_camera_samples) is not int or not 2 <= max_camera_samples <= 300):
        raise PerceptionError("Duration must be 0.1..60 seconds and camera cap 2..300")
    associate_receipts([], [], max_receipt_gap_s=max_receipt_gap_s)
    recording = DriverStateRecording(max_pairs=max_driver_pairs, max_receipt_age_s=1.)
    # Imagery locality is decided per domain: every camera domain must be
    # confined to loopback, while the driver domain may keep the routable
    # interface its publisher already binds. Read-only subscriptions there
    # carry no imagery.
    locality = imagery_locality_report(
        camera_domains=[settings["domain_id"] for settings in config["cameras"].values()],
        other_domains=[config["driver"]["domain_id"]])
    if not locality["ok"]:
        raise PerceptionError("Local calibration recording requires ROS_LOCALHOST_ONLY=1, or a CYCLONEDDS_URI "
                              "document confining every camera domain to loopback: "+locality["reason"])
    observers = {name: SingleMarkerObserver(marker_spec) for name in config["cameras"]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started, started_utc = time.monotonic(), datetime.now(timezone.utc).isoformat()
    session_id = "calibration-recording-"+uuid.uuid4().hex
    stop = threading.Event()
    cameras = {name: {"records": [], "last_pair": None, "source": None, "thread": None,
                       "error": None, "wait_timeouts": 0, "publishers_at_end": {}, "finished_at": None}
               for name in config["cameras"]}
    context = node = source = executor = None
    errors, publishers = [], {}
    termination = "duration_elapsed"
    rmw = None

    def collect(name):
        camera = cameras[name]
        try:
            while not stop.is_set() and len(camera["records"]) < max_camera_samples:
                try:
                    pair = camera["source"].capture(200)
                except PerceptionError:
                    if camera["source"].failure:
                        raise
                    camera["wait_timeouts"] += 1
                    continue
                observation = observers[name].observe(pair)
                record = {"metadata": pair.metadata, "observation": observation}
                # Force serialization now; errors cannot hide until after capture.
                json.dumps(record, allow_nan=False)
                if not stop.is_set():
                    camera["records"].append(record)
                    camera["last_pair"] = pair
        except Exception as exc:
            camera["error"] = str(exc)
        finally:
            camera["finished_at"] = time.monotonic()

    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        context = Context()
        rclpy.init(args=[], context=context, domain_id=config["driver"]["domain_id"])
        rmw = rclpy.get_rmw_implementation_identifier()
        if locality["config_dependent"] and rmw != locality["required_rmw_for_config"]:
            # A Cyclone DDS document confines nothing under another middleware.
            raise PerceptionError(f"Camera domain confinement was resolved from a CYCLONEDDS_URI document but "
                                  f"the running middleware is {rmw}; expected {locality['required_rmw_for_config']}")
        node = Node("rammp_calibration_"+uuid.uuid4().hex[:12], context=context,
                    enable_rosout=False, start_parameter_services=False, use_global_arguments=False)
        source = RosDriverStateSource(node, buffer=recording,
                    **{k: v for k, v in config["driver"].items() if k != "domain_id"})
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        for name, settings in config["cameras"].items():
            try:
                cameras[name]["source"] = RosRgbdSource(**settings)
                cameras[name]["thread"] = threading.Thread(target=collect, args=(name,),
                                                           name="calibration-"+name, daemon=True)
                cameras[name]["thread"].start()
            except Exception as exc:
                cameras[name]["error"] = str(exc)
        while time.monotonic()-started < duration_s and context.ok():
            if recording.full:
                termination = "driver_sample_cap_reached"
                break
            if all(len(c["records"]) >= max_camera_samples for c in cameras.values()):
                termination = "camera_sample_caps_reached"
                break
            executor.spin_once(timeout_sec=min(.05, max(0., duration_s-(time.monotonic()-started))))
        if not context.ok():
            termination = "context_shutdown"
    except KeyboardInterrupt:
        termination = "interrupted"
    except Exception as exc:
        termination = "source_error"
        errors.append(str(exc))
    finally:
        stop.set()
        ended = time.monotonic()
        # Assess continuing receipts at the recording cutoff, before teardown or
        # disk writes add unrelated latency. The driver executor is now stopped.
        driver_summary = recording.summary()
        for camera in cameras.values():
            if camera["thread"] is not None:
                camera["thread"].join(timeout=2.)
                if camera["thread"].is_alive():
                    errors.append("camera detection worker did not stop")
            if camera["source"] is not None:
                try:
                    camera["publishers_at_end"] = _publishers(camera["source"].node, camera["source"].topics.values())
                except Exception as exc:
                    errors.append("camera publisher discovery: "+str(exc))
                try:
                    camera["source"].close()
                except Exception as exc:
                    errors.append("camera cleanup: "+str(exc))
        if node is not None:
            try:
                publishers = _publishers(node, [config["driver"]["joint_topic"], config["driver"]["ee_topic"]])
            except Exception as exc:
                errors.append("driver publisher discovery: "+str(exc))
        # This executor ran only on this thread; no callbacks race the snapshot.
        for resource, close in ((source, "close"), (executor, "shutdown"), (node, "destroy_node"), (context, "try_shutdown")):
            if resource is not None:
                try:
                    getattr(resource, close)()
                except Exception as exc:
                    errors.append("driver subscriber cleanup: "+str(exc))

    pairs = tuple(recording.pairs)
    driver = driver_summary
    expected = {config["driver"]["joint_topic"]: "sensor_msgs/msg/JointState",
                config["driver"]["ee_topic"]: config["driver"]["ee_message_type"]}
    driver_publisher_ok = _publisher_match(publishers, expected)
    if driver_publisher_ok:
        driver_publisher_ok = len({(p[0]["node"], p[0]["namespace"]) for p in publishers.values()}) == 1
    frame_ids = {(p.joints.source_frame_id, p.ee.source_frame_id, p.joints.source_joint_names) for p in pairs}
    driver_ok = bool(pairs) and not recording.rejected and len(frame_ids) == 1 and driver_publisher_ok
    report = {"schema_version": 1, "scope": "passive_calibration_recording", "session_id": session_id,
              "pose_label": pose_label, "pose_label_semantics": "operator label; stationarity and distinct posture unverified",
              "started_utc": started_utc, "elapsed_s": ended-started,
              "requested_duration_s": duration_s, "termination": termination,
              "max_camera_samples": max_camera_samples, "max_driver_pairs": max_driver_pairs,
              "config": config, "config_digest": digest(config), "rmw_implementation": rmw,
              "transport_locality": locality,
              "diagnostic_max_receipt_gap_s": max_receipt_gap_s,
              "driver": {**driver, "publishers_at_end": publishers,
                         "single_matching_publisher_at_end": driver_publisher_ok,
                         "identity_consistent": len(frame_ids) == 1},
              "cameras": {}, "errors": errors, "frames_uploaded": 0,
              "motion_commanded_by_recorder": False, "hardware_motion_enabled": False,
              "robot_reference_calibrated": False, "calibration_samples_admitted": 0,
              "base_frame": None, "wrist_frame": None, "robot_model_digest": None,
              "base_epoch": None, "marker_placement_id": None, "timestamp_uncertainty_s": None,
              "limitations": [
                  "Receipt bracketing does not establish common acquisition time or source-clock mapping",
                  "Driver EE is model FK with unverified base/wrist frames and hardware acquisition age",
                  "All planar marker branches and nominal print scale remain provisional",
                  "One recorded window is not evidence of stationary hold or multiple diverse poses",
                  "Publisher graph at end is discovery provenance, not continuous source authentication",
                  "Camera domain confinement is a configuration claim about advertised DDS locators, "
                  "not a packet capture, and does not constrain a publisher another process started",
                  "Reviewed identities, clock bounds and resolved branches are required by the existing offline solver",
                  "No TF, world-model admission, commissioning limits or commands are produced"]}
    files = []

    def write_jsonl(path, rows):
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        files.append(path)

    write_jsonl(output/"driver.jsonl", ({"driver_index": i, **asdict(p)} for i, p in enumerate(pairs)))
    for name, camera in cameras.items():
        folder = output/name
        folder.mkdir()
        records = tuple(camera["records"])
        stats = camera["source"].statistics() if camera["source"] is not None else None
        identity_error = camera_integrity(records)
        settings = config["cameras"][name]
        camera_publisher_ok = _publisher_match(camera["publishers_at_end"], {
            settings[k+"_topic"]: "sensor_msgs/msg/CameraInfo" if k.endswith("info") else "sensor_msgs/msg/Image"
            for k in RgbdSynchronizer.streams})
        integrity_ok = (driver_ok and not identity_error and not camera["error"] and stats is not None
                        and not stats["rejected"] and camera_publisher_ok)
        associations = associate_receipts(pairs, records, max_receipt_gap_s=max_receipt_gap_s,
                                         source_integrity_ok=integrity_ok)
        write_jsonl(folder/"observations.jsonl", records)
        write_jsonl(folder/"associations.jsonl", associations)
        last = camera["last_pair"]
        if last is not None:
            path = folder/"last-capture.npz"
            with path.open("xb") as stream:
                np.savez(stream, rgb=last.rgb, depth_m=last.depth_m,
                         metadata_json=json.dumps({"capture_id": last.capture_id, **last.metadata}, allow_nan=False))
            files.append(path)
        report["cameras"][name] = {
            "samples_received": len(records), "statistics": stats,
            "sample_cap_reached": len(records) >= max_camera_samples,
            "observation_counts": dict(Counter(r["observation"]["status"] for r in records)),
            "association_counts": dict(Counter(r["status"] for r in associations)),
            "identity_error": identity_error, "worker_error": camera["error"],
            "wait_timeouts": camera["wait_timeouts"], "publishers_at_end": camera["publishers_at_end"],
            "single_matching_publisher_at_end": camera_publisher_ok,
            "last_rgb_receipt_age_s_at_camera_cutoff": (
                min(ended, camera["finished_at"])-records[-1]["metadata"]["received_at_monotonic_s"]["rgb"]
                if records and camera["finished_at"] is not None else None),
            "last_rgb_receipt_age_s": (ended-records[-1]["metadata"]["received_at_monotonic_s"]["rgb"]
                                       if records else None)}
    report["status"] = "recorded_unapproved" if (
        not errors and termination in ("duration_elapsed", "camera_sample_caps_reached") and driver_ok
        and driver["latest_pair_receipt_fresh"] and not recording.pairs_not_retained
        and all(c["association_counts"].get("receipt_bracketed", 0) >= 2
                and c["last_rgb_receipt_age_s_at_camera_cutoff"] is not None
                and 0 <= c["last_rgb_receipt_age_s_at_camera_cutoff"] <= 1.
                for c in report["cameras"].values())) else "incomplete"
    report["files_sha256"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with (output/"report.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def read_recording_config(path):
    path = Path(path)
    if path.stat().st_size > 32768:
        raise PerceptionError("Recording configuration exceeds 32 KiB")
    return recording_config(strict_loads(path.read_bytes()))
