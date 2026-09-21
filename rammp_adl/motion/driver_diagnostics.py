"""Bounded passive ROS publication recordings; no command or ownership path.

Reuse the diagnostic state converter/pairer. Delivery and observed variation are
not hardware acquisition age, accuracy, tracking error or stopping measurements.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid

from .driver_state import (
    ARM_JOINT_NAMES, EE_MESSAGE_TYPES, DriverStateBuffer, DriverStateError,
    RosDriverStateSource,
)


def _distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {"count": len(values), "min": values[0], "max": values[-1],
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values))-1]}


class DriverStateRecording(DriverStateBuffer):
    """Retain bounded, validated, same-publication pairs without callback I/O."""

    def __init__(self, *, max_pairs=10000, **kwargs):
        if type(max_pairs) is not int or not 2 <= max_pairs <= 100000:
            raise DriverStateError("recording max_pairs must be an integer from 2 through 100000")
        super().__init__(**kwargs)
        self.max_pairs = max_pairs
        self.pairs = []
        self.messages_received = {"joint": 0, "ee": 0}
        self.pairs_not_retained = 0

    def _ingest(self, stream, message):
        with self._lock:
            self.messages_received[stream] += 1
            paired = super()._ingest(stream, message)
            if paired is not None:
                if len(self.pairs) < self.max_pairs:
                    self.pairs.append(paired)
                else:
                    self.pairs_not_retained += 1
            return paired

    @property
    def full(self):
        return len(self.pairs) >= self.max_pairs

    def summary(self):
        with self._lock:
            pairs = tuple(self.pairs)
            try:
                latest = self.snapshot()
                receipt_age = self.clock()-min(latest.joints.received_at_monotonic_s, latest.ee.received_at_monotonic_s)
                receipt_fresh, receipt_error = True, ""
            except DriverStateError as exc:
                receipt_age = None
                receipt_fresh, receipt_error = False, str(exc)
            report = {
                "scope": "passive_driver_publications",
                "paired_samples": len(pairs), "messages_received": dict(self.messages_received),
                "rejected_messages": self.rejected, "pairing_queue_evictions": self.dropped,
                "pairs_not_retained": self.pairs_not_retained,
                "last_pairing_error": self.last_error,
                "latest_pair_receipt_fresh": receipt_fresh,
                "latest_pair_receipt_age_s": receipt_age,
                "receipt_check_error": receipt_error,
                "diagnostic_receipt_timeout_s": self.max_receipt_age_s,
                "joint_names": ARM_JOINT_NAMES,
                "motion_commanded_by_recorder": False, "motion_ready": False,
                "hardware_acquisition_age_s": None,
                "measured_stopping_latency_s": None, "measured_stopping_distance_rad": None,
                "measured_motion_tracking_error_rad": None,
                "commissioned_limits": None,
                "limitations": [
                    "Header stamps are publication times, not hardware acquisition times",
                    "Rates/gaps describe retained pairs; messages lost before receipt cannot be counted",
                    "Observed position variation is not accuracy, motion tracking error or a noise bound",
                    "Position spans are raw joint coordinates; continuous-joint wrapping can enlarge them",
                    "Effort is reported joint torque, not calibrated external contact force",
                    "EE pose is driver-model FK; gripper knuckle angle is not aperture",
                    "No motion, stopping, watchdog, ownership or fault response was exercised",
                ],
            }
            if not pairs:
                return report
            joints = [p.joints for p in pairs]
            receipt = [max(p.joints.received_at_monotonic_s, p.ee.received_at_monotonic_s) for p in pairs]
            gaps = [b-a for a, b in zip(receipt, receipt[1:])]
            # Subtract integer nanoseconds before float conversion.
            publication_gaps = [(b.source_stamp_ns-a.source_stamp_ns)/1e9 for a, b in zip(joints, joints[1:])]
            span = receipt[-1]-receipt[0]
            report.update(
                first_publication_stamp_ns=joints[0].source_stamp_ns,
                last_publication_stamp_ns=joints[-1].source_stamp_ns,
                paired_receipt_span_s=span,
                paired_delivery_hz=(len(pairs)-1)/span if span > 0 else None,
                paired_receipt_gap_s=_distribution(gaps),
                publication_gap_s=_distribution(publication_gaps),
                pair_receipt_skew_s=_distribution([abs(p.joints.received_at_monotonic_s-p.ee.received_at_monotonic_s) for p in pairs]),
                source_frame_ids={"joint": sorted({p.joints.source_frame_id for p in pairs}),
                                  "ee": sorted({p.ee.source_frame_id for p in pairs})},
                joints=[],
            )
            for i, name in enumerate(ARM_JOINT_NAMES):
                q = [j.position_rad[i] for j in joints]
                dq = [j.velocity_rad_s[i] for j in joints]
                effort = [j.effort_nm[i] for j in joints if j.effort_nm is not None]
                report["joints"].append({
                    "name": name, "first_position_rad": q[0], "last_position_rad": q[-1],
                    "position_span_rad": max(q)-min(q), "position_stddev_rad": statistics.pstdev(q),
                    "max_abs_reported_velocity_rad_s": max(abs(v) for v in dq),
                    "reported_effort_nm": _distribution(effort),
                })
            report["gripper_knuckle_position_rad"] = _distribution([
                j.gripper_knuckle_position_rad for j in joints if j.gripper_knuckle_position_rad is not None])
            report["last_observation"] = asdict(pairs[-1])
            return report


def capture_driver_state(*, output_dir, joint_topic, ee_topic, ee_message_type,
                         duration_s=30., max_pairs=10000):
    """Create a node with two sensor subscriptions and write only local files.

    ROS domain/RMW come from the caller's environment. No driver launch, action,
    service client, parameter client, command publisher or stop is constructed.
    """
    if (type(duration_s) not in (int, float) or not math.isfinite(duration_s)
            or not .1 <= duration_s <= 600.):
        raise DriverStateError("duration_s must be finite and between 0.1 and 600 seconds")
    if (ee_message_type not in EE_MESSAGE_TYPES or any(
            not isinstance(t, str) or not t.startswith("/") for t in (joint_topic, ee_topic))
            or joint_topic == ee_topic):
        raise DriverStateError("explicit distinct absolute state topics and a supported EeState type are required")
    recording = DriverStateRecording(max_pairs=max_pairs, max_receipt_age_s=1.)
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.node import Node

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    context = Context()
    node = source = executor = None
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    termination = "duration_elapsed"
    publishers = {}
    try:
        rclpy.init(args=[], context=context)
        node = Node("rammp_passive_record_"+uuid.uuid4().hex[:12], context=context,
                    enable_rosout=False, start_parameter_services=False,
                    use_global_arguments=False)
        source = RosDriverStateSource(node, buffer=recording, joint_topic=joint_topic,
                                      ee_topic=ee_topic, ee_message_type=ee_message_type)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        while context.ok() and time.monotonic()-started < duration_s and not recording.full:
            executor.spin_once(timeout_sec=min(.1, max(0., duration_s-(time.monotonic()-started))))
        if recording.full:
            termination = "sample_cap_reached"
        elif not context.ok():
            termination = "context_shutdown"
    except (KeyboardInterrupt, ExternalShutdownException):
        termination = "interrupted"
    finally:
        if node is not None:
            for topic in (joint_topic, ee_topic):
                publishers[topic] = [{"node": p.node_name, "namespace": p.node_namespace,
                                      "message_type": p.topic_type,
                                      "endpoint_gid": list(p.endpoint_gid)}
                                     for p in node.get_publishers_info_by_topic(topic)]
        if source is not None:
            source.close()
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)
    report = recording.summary()
    report.update(started_utc=started_utc, requested_duration_s=duration_s,
                  elapsed_s=time.monotonic()-started, termination=termination,
                  topics={"joint": joint_topic, "ee": ee_topic}, ee_message_type=ee_message_type,
                  ros_domain_id=os.environ.get("ROS_DOMAIN_ID", "0"),
                  rmw_implementation=rclpy.get_rmw_implementation_identifier(),
                  publishers_at_end=publishers, output_dir=str(output))
    expected = {joint_topic: "sensor_msgs/msg/JointState", ee_topic: ee_message_type}
    publisher_match = all(len(publishers[t]) == 1 and publishers[t][0]["message_type"] == expected[t] for t in expected)
    if publisher_match:
        a, b = publishers[joint_topic][0], publishers[ee_topic][0]
        publisher_match = (a["node"], a["namespace"]) == (b["node"], b["namespace"])
    report["single_matching_publisher_at_end"] = publisher_match
    # End-of-session discovery is provenance, never identity/freshness certification.
    report["status"] = ("captured" if len(recording.pairs) >= 2 and publisher_match
                        and report["latest_pair_receipt_fresh"] and not recording.rejected
                        and termination in ("duration_elapsed", "sample_cap_reached") else "incomplete")
    with (output/"samples.jsonl").open("x", encoding="utf-8") as stream:
        for pair in recording.pairs:
            stream.write(json.dumps(asdict(pair), allow_nan=False)+"\n")
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report
