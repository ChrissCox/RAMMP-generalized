#!/usr/bin/env python3
"""Score the door task's perception on recorded scenes, without the arm.

    python tools/bench_offline.py score [--no-network] [--report FILE]   one JSON line with "score"
    python tools/bench_offline.py label [--min-finds 2] [--report FILE]  operator: freeze labels from the current code
    python tools/bench_offline.py digest                                  the dataset's digest, for pinning

Every recorded scene (tools/bench_record.py, or any robot run: the node records
each keyframe) is replayed through the runtime's own discovery and geometry: the
same prompt, keyframe encoding and face screen, the same depth measurement and
grasp roles. Model answers are cached by the exact request, so an unchanged
prompt costs nothing and repeats exactly; a changed prompt asks Astra again.
Nothing here talks to ROS or moves anything.

The score is out of 100: handle recall 40, position accuracy 20, grasp validity 20,
visibility verdicts 10, no false handles 10, against labels the operator froze
with `label` and reviewed in its report. A score is evidence about perception on
this dataset, not about the robot.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import html
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = Path(os.environ.get("RAMMP_BENCH_DIR", ROOT/"artifacts/bench"))
TASK = os.environ.get("RAMMP_BENCH_TASK", "open the cabinet door in front of you")
BUNDLE = ROOT/"artifacts/jetson/real-world-ready/assembly/bundle-2/arm-gripper-locked.urdf"
FACE_MODEL = ROOT/"artifacts/models/face_detection_yunet_2023mar.onnx"
MATCH_M = .05                         # a detected handle within 5 cm of the label is the labelled handle
APPROACH_TOLERANCE_DEG = 20.
EPOCH = 1                             # the bench context's execution epoch; replay never advances it
WEIGHTS = {"handle_recall": 40, "position_accuracy": 20, "grasp_ok": 20, "visibility_agreement": 10, "no_false_handles": 10}


def labels_path(bench=None):
    return Path(bench or BENCH)/"scenes"/"labels.json"


# -- model answers, cached by the exact request -------------------------------------------
class CacheMiss(RuntimeError):
    pass


class CachingTransport:
    """The Responses transport, remembered: the same request gets the same answer, for free."""

    def __init__(self, directory, *, network=True, timeout_s=45.):
        self.directory, self.network, self.timeout_s, self.inner = Path(directory), network, timeout_s, None
        self.stats = {"requests": 0, "cached": 0, "asked": 0, "refused": 0, "input_tokens": 0, "output_tokens": 0, "model_latency_s": 0.}

    @staticmethod
    def key(request):
        return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    async def create(self, **request):
        self.stats["requests"] += 1
        path = self.directory/f"{self.key(request)}.json"
        if path.is_file():
            record = json.loads(path.read_text())
            self.stats["cached"] += 1
        else:
            if not self.network:
                self.stats["refused"] += 1
                raise CacheMiss("no cached answer for this request, and the network is off")
            if self.inner is None:
                from rammp_adl.reasoning import OpenAIResponsesTransport
                self.inner = OpenAIResponsesTransport(timeout_s=self.timeout_s)
            started = time.monotonic()
            response = await self.inner.create(**request)
            data = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
            record = {"response": data, "latency_s": round(time.monotonic()-started, 3),
                      "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record))
            temporary.replace(path)
            self.stats["asked"] += 1
        usage = (record.get("response") or {}).get("usage") or {}
        self.stats["input_tokens"] += int(usage.get("input_tokens") or 0)
        self.stats["output_tokens"] += int(usage.get("output_tokens") or 0)
        self.stats["model_latency_s"] = round(self.stats["model_latency_s"]+float(record.get("latency_s") or 0.), 3)
        return record["response"]

    async def close(self):
        if self.inner is not None:
            await self.inner.close()


# -- replaying a scene through the runtime's own code ----------------------------------------
class ReplayClient:
    """The arm as it stood when the scene was recorded: still, at the recorded joints."""

    def __init__(self, joints):
        self.joints = tuple(float(v) for v in joints)

    def live_joints(self, **_):
        return {"position_rad": self.joints, "knuckle_rad": None, "effort_nm": None, "velocity_rad_s": (0.,)*len(self.joints),
                "received_at_monotonic_s": time.monotonic()}

    def still_since_s(self):
        return time.monotonic()-60.


def unit(vector):
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    return array/norm if norm > 0 else array


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1., min(1., float(unit(a) @ unit(b))))))


class Replayer:
    def __init__(self, reasoner, *, bench=None, chain=None, screen=None):
        from rammp_adl.contracts import strict_loads
        from rammp_adl.intake import draft_context
        from rammp_adl.motion.kinematics import UrdfChain
        from rammp_adl.perception.keyframes import FaceScreen
        self.reasoner, self.bench = reasoner, Path(bench or BENCH)
        self.base = strict_loads((ROOT/"config/sheppy-bench.context.json").read_bytes())
        self.chain = chain or UrdfChain.from_path(BUNDLE)
        self.screen = screen if screen is not None else FaceScreen(FACE_MODEL)
        self.draft = lambda task_id: draft_context(self.base, [], task_id=task_id, camera_id="wrist_d405")

    async def replay(self, path):
        from rammp_adl.perception.grounded_scene import GroundedScene, SceneError
        from rammp_adl.perception.scene_record import load_scene
        keyframe, meta = load_scene(path)

        class ReplayScene(GroundedScene):
            def camera_pose(self):
                return keyframe.base_from_camera, keyframe.joints_rad

        scene = ReplayScene(client=ReplayClient(meta["joints_rad"]), chain=self.chain, calibration_id=self.base["calibration_id"],
                            face_screen=self.screen)
        out = {"scene": Path(path).name, "layout": meta.get("layout", "default"), "source": meta.get("source"),
               "recorded_utc": meta.get("recorded_utc"), "error": None, "handle": None, "entities": []}
        started = time.monotonic()
        try:
            task_id = "offline-"+hashlib.sha1(Path(path).name.encode()).hexdigest()[:12]
            found = await scene.discover(self.reasoner, self.draft(task_id), TASK, keyframe=keyframe)
        except SceneError as exc:
            out["error"] = f"{exc.status}: {exc.detail}"[:200]
            return out, keyframe
        search = found.get("search") or {}
        out.update(target_visible=bool(search.get("target_visible")), search_hint=search.get("search_hint"),
                   search_note=search.get("search_note", "")[:160])
        records = list(scene.entities.values())
        out["entities"] = [{"label": r["label"], "kind": r["kind"], "box": r["box"], "position_m": r["position_m"]} for r in records]
        out["handles_named"] = sum(1 for r in records if r["kind"] == "handle")
        handles = [r for r in records if r["kind"] == "handle" and r["position_m"] is not None]
        if handles:
            handle = max(handles, key=lambda r: r["confidence"])
            grasp = handle.get("grasp") or {}
            pose = (grasp.get("roles") or {}).get("grasp")
            from rammp_adl.motion.kinematics import quaternion_matrix
            rotation = quaternion_matrix(tuple(pose["orientation_xyzw"])) if pose else None
            door = scene.door_for(handle)
            out["handle"] = {"label": handle["label"], "position_m": [round(v, 4) for v in handle["position_m"]], "box": handle["box"],
                             "strategy": grasp.get("strategy"), "why_no_grasp": grasp.get("reason") or None,
                             "approach": None if rotation is None else [round(v, 4) for v in rotation[:3, 2]],
                             "closing": None if rotation is None else [round(v, 4) for v in rotation[:3, 0]],
                             "surface_normal": None if handle["geometry"] is None else [round(v, 4) for v in handle["geometry"]["up"]],
                             "door_width_m": None if door is None else round(door["width_m"], 3)}
        out["replay_s"] = round(time.monotonic()-started, 2)
        return out, keyframe


async def replay_all(scenes, reasoner, **options):
    replayer = Replayer(reasoner, **options)
    results, keyframes = [], {}
    for path in scenes:
        result, keyframe = await replayer.replay(path)
        results.append(result)
        keyframes[result["scene"]] = keyframe
    return results, keyframes


def astra(transport):
    from rammp_adl.contracts import Catalog
    from rammp_adl.reasoning import AstraReasoner
    return AstraReasoner(Catalog(ROOT), transport=transport, hold_assertion=lambda: True, epoch_getter=lambda _t: EPOCH)


# -- labels, frozen once and reviewed by the operator -----------------------------------------
def make_labels(results, keyframes, *, min_finds=2):
    """Per layout, the median of the handles the current code found is the handle; each scene says whether it can see it."""
    labels, consensus = {}, {}
    for layout in sorted({r["layout"] for r in results}):
        found = [r["handle"] for r in results if r["layout"] == layout and r.get("handle")]
        if len(found) >= min_finds:
            position = np.median(np.array([h["position_m"] for h in found]), axis=0)
            near = [h for h in found if np.linalg.norm(np.asarray(h["position_m"])-position) <= MATCH_M and h["surface_normal"]]
            normal = unit(np.median(np.array([h["surface_normal"] for h in near]), axis=0)) if near else None
            consensus[layout] = {"position_m": [round(float(v), 4) for v in position],
                                 "door_normal": None if normal is None else [round(float(v), 4) for v in normal], "from_scenes": len(found)}
    for r in results:
        c = consensus.get(r["layout"])
        if c is None:
            labels[r["scene"]] = {"handle_visible": None, "why": "too few detections in this layout to place the handle"}
            continue
        keyframe = keyframes[r["scene"]]
        projected = keyframe.project(c["position_m"])
        if projected is None:
            labels[r["scene"]] = {"handle_visible": False, "why": "outside the view", **c}
            continue
        u, v, expected = projected
        window = keyframe.depth_m[max(0, int(v)-3):int(v)+4, max(0, int(u)-3):int(u)+4]
        valid = window[np.isfinite(window)]
        if not valid.size:
            labels[r["scene"]] = {"handle_visible": None, "why": "no depth where the handle should be", **c}
            continue
        occluded = float(np.median(valid)) < expected-.05
        labels[r["scene"]] = {"handle_visible": not occluded, "why": "occluded" if occluded else "in view", **c}
    return labels


def score(results, labels):
    """The weighted score and its parts; unlabelled scenes count for nothing."""
    by_scene = {r["scene"]: r for r in results}
    visible = [s for s, l in labels.items() if l.get("handle_visible") is True and s in by_scene]
    hidden = [s for s, l in labels.items() if l.get("handle_visible") is False and s in by_scene]
    matched, verdicts = {}, []
    for s in visible:
        h = by_scene[s].get("handle")
        if h:
            distance = float(np.linalg.norm(np.asarray(h["position_m"])-np.asarray(labels[s]["position_m"])))
            if distance <= MATCH_M:
                matched[s] = distance
    for s in visible+hidden:
        r = by_scene[s]
        verdicts.append(r["error"] is None and bool(r.get("target_visible")) == (s in visible))
    grasp_ok = []
    for s in matched:
        h, normal = by_scene[s]["handle"], labels[s].get("door_normal")
        grasp_ok.append(bool(h["approach"] and normal and h["strategy"] == "top_down"
                             and angle_deg(h["approach"], -np.asarray(normal)) <= APPROACH_TOLERANCE_DEG))
    parts = {"handle_recall": len(matched)/len(visible) if visible else 0.,
             "position_accuracy": max(0., 1.-statistics.median(matched.values())/MATCH_M) if matched else 0.,
             "grasp_ok": sum(grasp_ok)/len(grasp_ok) if grasp_ok else 0.,
             "visibility_agreement": sum(verdicts)/len(verdicts) if verdicts else 0.,
             "no_false_handles": 1.-sum(1 for s in hidden if by_scene[s].get("handles_named"))/len(hidden) if hidden else 1.}
    total = sum(WEIGHTS[k]*v for k, v in parts.items()) if visible else 0.
    status = {}
    for s, r in by_scene.items():
        label = labels.get(s, {}).get("handle_visible")
        status[s] = ("error" if r["error"] else "unlabelled" if label is None else
                     ("found" if s in matched else "missed") if label else ("false" if r.get("handles_named") else "clear"))
    return {"score": round(total, 1), "parts": {k: round(v, 3) for k, v in parts.items()},
            "scenes": len(results), "labelled_visible": len(visible), "labelled_hidden": len(hidden),
            "errors": sum(1 for r in results if r["error"]),
            "median_position_error_m": round(statistics.median(matched.values()), 4) if matched else None,
            "status": status}


def digest(bench=None):
    """sha256 over every scene's files and the labels: two scores compare only on the same digest."""
    from rammp_adl.perception.scene_record import list_scenes
    bench = Path(bench or BENCH)
    h = hashlib.sha256()
    for path in list_scenes(bench):
        h.update(path.name.encode())
        for name in ("meta.json", "rgb.jpg", "depth.npz"):
            h.update(hashlib.sha256((path/name).read_bytes()).digest())
    if labels_path(bench).is_file():
        h.update(labels_path(bench).read_bytes())
    return h.hexdigest()


# -- the report ---------------------------------------------------------------------------------
REPORT_CSS = """
:root{--ground:#F2F4F3;--surface:#FFFFFF;--ink:#15201B;--muted:#5A6A63;--line:#D5DDD9;--accent:#2E6B64;--accent-soft:#DCEBE8;
--good:#2E7D4F;--warn:#A86A12;--bad:#B23B2A;--none:#7B8A84;--frame:#0E1412}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#101614;--surface:#18201D;--ink:#E4EBE8;--muted:#95A69F;
--line:#2A3530;--accent:#6CB5AA;--accent-soft:#1E3330;--good:#5DBB84;--warn:#E0A64A;--bad:#E07563;--none:#8C9A94;--frame:#060909;color-scheme:dark}}
:root[data-theme="dark"]{--ground:#101614;--surface:#18201D;--ink:#E4EBE8;--muted:#95A69F;--line:#2A3530;--accent:#6CB5AA;
--accent-soft:#1E3330;--good:#5DBB84;--warn:#E0A64A;--bad:#E07563;--none:#8C9A94;--frame:#060909;color-scheme:dark}
body{background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 56px;display:grid;gap:28px}
h1,h2{font-family:"Barlow Semi Condensed","Arial Narrow",sans-serif;font-weight:600;letter-spacing:.01em;text-wrap:balance;margin:0}
h1{font-size:2rem;line-height:1.1}h2{font-size:1.25rem}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.eyebrow{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{color:var(--muted);max-width:68ch;margin:6px 0 0}
.banner{border:1px dashed var(--warn);color:var(--warn);padding:10px 14px;border-radius:6px;max-width:80ch}
.top{display:grid;grid-template-columns:minmax(0,220px) minmax(0,1fr);gap:28px;align-items:start}
.total{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px 20px}
.total .big{font-family:"Barlow Semi Condensed","Arial Narrow",sans-serif;font-size:4rem;font-weight:600;line-height:1}
.total .of{color:var(--muted);font-size:1rem}
.facts{display:grid;gap:4px;margin-top:12px;font-size:.85rem;color:var(--muted)}
.parts{display:grid;gap:12px}
.part{display:grid;grid-template-columns:minmax(0,190px) minmax(0,1fr) 72px;gap:12px;align-items:center}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
.part .w{text-align:right;color:var(--muted);font-size:.85rem}
.map{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px;display:grid;gap:10px}
.map svg{width:100%;max-width:640px;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.8rem;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.scene{background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow:hidden;display:grid}
.shot{position:relative;background:var(--frame);aspect-ratio:16/9}
.shot img,.shot svg{position:absolute;inset:0;width:100%;height:100%}
.shot img{object-fit:fill}.shot .withheld{position:absolute;inset:0;display:grid;place-items:center;color:#9aa;font-size:.8rem}
.meta{padding:10px 12px;display:grid;gap:6px;font-size:.82rem}
.row{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.75rem;font-weight:600;border:1px solid currentColor}
.found{color:var(--good)}.clear{color:var(--good)}.missed{color:var(--bad)}.false{color:var(--bad)}.error{color:var(--warn)}.unlabelled{color:var(--none)}
.note{color:var(--muted)}
@media (max-width:640px){.top{grid-template-columns:1fr}.part{grid-template-columns:1fr 64px}.part .bar{grid-column:1/-1}}
"""
PART_NAMES = {"handle_recall": "Handle found", "position_accuracy": "Placed accurately", "grasp_ok": "Grasp straight in",
              "visibility_agreement": "In-view verdict right", "no_false_handles": "No handle invented"}
KIND_COLOUR = {"handle": "var(--accent)", "surface": "var(--muted)"}


def _frame_uri(keyframe, screen):
    from rammp_adl.perception.geometry import PerceptionError
    from rammp_adl.perception.keyframes import encode_keyframe
    try:
        crop = encode_keyframe(keyframe, camera_id="wrist_d405", calibration_id="report", screen=screen, max_long_edge=480, max_bytes=90000)
    except PerceptionError:
        return None
    return "data:image/jpeg;base64,"+base64.b64encode(crop.jpeg_bytes).decode("ascii")


def _point(keyframe, position):
    projected = keyframe.project(position) if position is not None else None
    return None if projected is None else (projected[0]/keyframe.width, projected[1]/keyframe.height)


def _map_svg(results, labels):
    """Top-down: where each scene put the handle, against the label, in centimetres from the label."""
    rows = []
    for r in results:
        label = labels.get(r["scene"], {})
        if r.get("handle") and label.get("position_m"):
            offset = (np.asarray(r["handle"]["position_m"])-np.asarray(label["position_m"]))*100.
            rows.append((float(offset[1]), float(offset[0]), r["scene"]))
    size, span = 320, 10.
    scale = lambda value: size/2-value/span*(size/2-24)
    marks = []
    for cm in (5, 10):
        radius = cm/span*(size/2-24)
        marks.append(f'<circle cx="{size/2}" cy="{size/2}" r="{radius:.1f}" fill="none" stroke="var(--line)" stroke-dasharray="3 3"/>'
                     f'<text x="{size/2+radius+3:.1f}" y="{size/2-4}" fill="var(--muted)" font-size="10" font-family="IBM Plex Mono,monospace">{cm} cm</text>')
    dots = [f'<circle cx="{scale(y):.1f}" cy="{scale(x):.1f}" r="4" fill="var(--accent)" fill-opacity=".8"><title>{html.escape(s)}: '
            f'{math.hypot(x, y):.1f} cm off</title></circle>' for y, x, s in rows if abs(x) <= span and abs(y) <= span]
    outside = sum(1 for y, x, _ in rows if abs(x) > span or abs(y) > span)
    return (f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="Handle positions relative to the label">'
            f'<rect width="{size}" height="{size}" fill="none"/>{"".join(marks)}'
            f'<line x1="{size/2-8}" y1="{size/2}" x2="{size/2+8}" y2="{size/2}" stroke="var(--ink)"/>'
            f'<line x1="{size/2}" y1="{size/2-8}" x2="{size/2}" y2="{size/2+8}" stroke="var(--ink)"/>'
            f'<text x="{size/2}" y="14" text-anchor="middle" fill="var(--muted)" font-size="10" font-family="IBM Plex Sans,sans-serif">'
            f'toward the door (+x)</text>{"".join(dots)}</svg>'), len(rows), outside


def write_report(path, results, labels, scored, keyframes, *, screen, title_note="", stats=None, dataset=None):
    tiles = []
    for r in results:
        keyframe, label = keyframes[r["scene"]], labels.get(r["scene"], {})
        state = scored["status"][r["scene"]]
        uri = _frame_uri(keyframe, screen)
        shapes = []
        for entity in r["entities"]:
            if entity["box"]:
                x0, y0, x1, y1 = entity["box"]
                colour = KIND_COLOUR.get(entity["kind"], "#C9D3CF")
                shapes.append(f'<rect x="{x0:.4f}" y="{y0:.4f}" width="{x1-x0:.4f}" height="{y1-y0:.4f}" fill="none" '
                              f'stroke="{colour}" stroke-width="2" vector-effect="non-scaling-stroke"/>')
        target = _point(keyframe, label.get("position_m"))
        if target:
            shapes.append(f'<circle cx="{target[0]:.4f}" cy="{target[1]:.4f}" r=".035" fill="none" stroke="#FFFFFF" '
                          f'stroke-width="2" vector-effect="non-scaling-stroke"/>')
        found = _point(keyframe, (r.get("handle") or {}).get("position_m"))
        if found:
            shapes.append(f'<circle cx="{found[0]:.4f}" cy="{found[1]:.4f}" r=".012" fill="#F5B83D"/>')
        shot = (f'<img src="{uri}" alt="Wrist view {html.escape(r["scene"])}">' if uri else
                '<div class="withheld">withheld by the face screen</div>')
        handle = r.get("handle") or {}
        lines = [f'<div class="row"><span class="chip {state}">{state}</span><span class="mono note">{html.escape(r["scene"][:22])}</span></div>']
        if r["error"]:
            lines.append(f'<div class="note">{html.escape(r["error"])}</div>')
        else:
            said = "sees the target" if r.get("target_visible") else f'says look {html.escape(str(r.get("search_hint")))}'
            lines.append(f'<div>Astra {said}; label: {html.escape(str(label.get("why", "none")))}</div>')
            if handle:
                grasp = handle.get("strategy") or "none"
                lines.append(f'<div class="mono">{html.escape(handle["label"])} · grasp {html.escape(grasp)}'
                             + (f' · door {handle["door_width_m"]:.2f} m' if handle.get("door_width_m") else "") + '</div>')
        tiles.append(f'<article class="scene"><div class="shot" style="aspect-ratio:{keyframe.width}/{keyframe.height}">{shot}<svg viewBox="0 0 1 1" preserveAspectRatio="none" '
                     f'aria-hidden="true">{"".join(shapes)}</svg></div><div class="meta">{"".join(lines)}</div></article>')
    parts = "".join(f'<div class="part"><div>{PART_NAMES[k]}</div><div class="bar"><i style="width:{v*100:.0f}%"></i></div>'
                    f'<div class="w mono">{v*100:.0f}% × {WEIGHTS[k]}</div></div>' for k, v in scored["parts"].items())
    map_svg, placed, outside = _map_svg(results, labels)
    stats = stats or {}
    facts = [f'{scored["scenes"]} scenes · {scored["labelled_visible"]} with the handle in view · {scored["labelled_hidden"]} without',
             f'median placement error {scored["median_position_error_m"]*100:.1f} cm' if scored["median_position_error_m"] is not None else "no handle placed",
             f'{stats.get("cached", 0)} cached answers · {stats.get("asked", 0)} new model calls · {scored["errors"]} errors']
    page = f"""<title>Door Bench Offline</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{REPORT_CSS}</style>
<div class="wrap">
<header><div class="eyebrow">RAMMP door task · offline perception bench</div><h1>Door Bench Offline</h1>
<p class="lede">Recorded wrist-camera scenes replayed through the runtime's own discovery and depth geometry, without the arm.
Task: <span class="mono">{html.escape(TASK)}</span>. Dataset <span class="mono">{html.escape((dataset or "")[:12])}</span>.</p></header>
{f'<div class="banner">{html.escape(title_note)}</div>' if title_note else ''}
<section class="top"><div class="total"><div class="eyebrow">Score</div><div class="big mono">{scored["score"]:.0f}<span class="of"> / 100</span></div>
<div class="facts">{"".join(f"<div>{html.escape(f)}</div>" for f in facts)}</div></div>
<div class="parts">{parts}</div></section>
<section class="map"><h2>Where the handle was placed</h2><p class="lede">Each dot is one scene's handle, seen from above, relative to the
labelled handle at the cross. Within the inner ring counts as the same handle.</p>{map_svg}
<div class="legend"><span>{placed} placed</span><span>{outside} beyond 10 cm, off the map</span></div></section>
<section style="display:grid;gap:14px"><h2>Scenes</h2><div class="legend"><span>boxes: what Astra named (teal: handle, grey: door or panel)</span>
<span>white ring: labelled handle</span><span>amber dot: where depth placed the found handle</span></div>
<div class="grid">{"".join(tiles)}</div></section>
</div>"""
    Path(path).write_text(page)
    return path


# -- commands ---------------------------------------------------------------------------------
def _scenes(bench):
    from rammp_adl.perception.scene_record import list_scenes
    return list_scenes(bench)


async def _run(bench, network, screen=None, reasoner=None):
    transport = None
    if reasoner is None:
        transport = CachingTransport(Path(bench)/"astra-cache", network=network)
        reasoner = astra(transport)
    try:
        results, keyframes = await replay_all(_scenes(bench), reasoner, bench=bench, screen=screen)
    finally:
        if transport is not None:
            await transport.close()
    return results, keyframes, (transport.stats if transport else {})


def command_score(args):
    bench = Path(args.bench)
    labels = json.loads(labels_path(bench).read_text())["labels"] if labels_path(bench).is_file() else {}
    started = time.monotonic()
    results, keyframes, stats = asyncio.run(_run(bench, not args.no_network))
    scored = score(results, labels)
    payload = {"tier": "offline_perception", "score": scored["score"] if labels else 0.,
               "reason": None if labels else "no labels: the operator runs `python tools/bench_offline.py label` and reviews its report",
               "parts": scored["parts"], "scenes": scored["scenes"], "labelled_visible": scored["labelled_visible"],
               "labelled_hidden": scored["labelled_hidden"], "errors": scored["errors"],
               "median_position_error_m": scored["median_position_error_m"], "model": stats,
               "duration_s": round(time.monotonic()-started, 1), "dataset": digest(bench),
               "failures": [{"scene": r["scene"], "status": scored["status"][r["scene"]], "error": r["error"],
                             "handle": (r.get("handle") or {}).get("position_m"), "target_visible": r.get("target_visible")}
                            for r in results if scored["status"][r["scene"]] in ("missed", "false", "error")][:20]}
    if args.report:
        from rammp_adl.perception.keyframes import FaceScreen
        write_report(args.report, results, labels, scored, keyframes, screen=FaceScreen(FACE_MODEL), stats=stats, dataset=payload["dataset"])
    print(json.dumps(payload), flush=True)


def command_label(args):
    bench = Path(args.bench)
    results, keyframes, stats = asyncio.run(_run(bench, True))
    labels = make_labels(results, keyframes, min_finds=args.min_finds)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    labels_path(bench).parent.mkdir(parents=True, exist_ok=True)
    labels_path(bench).write_text(json.dumps({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "from_code": head,
                                              "labels": labels}, indent=1)+"\n")
    scored = score(results, labels)
    if args.report:
        from rammp_adl.perception.keyframes import FaceScreen
        write_report(args.report, results, labels, scored, keyframes, screen=FaceScreen(FACE_MODEL), stats=stats, dataset=digest(bench),
                     title_note="Fresh labels, placed from this code's own detections. Check every white ring sits on the real handle "
                                "and correct labels.json where it does not, before research uses them.")
    counts = {state: list(scored["status"].values()).count(state) for state in set(scored["status"].values())}
    print(json.dumps({"labels": str(labels_path(bench)), "scenes": len(labels), "states": counts, "dataset": digest(bench)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", default=str(BENCH))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("score")
    run.add_argument("--no-network", action="store_true", help="answer only from the cache; a miss is an error")
    run.add_argument("--report")
    run.set_defaults(function=command_score)
    label = commands.add_parser("label")
    label.add_argument("--min-finds", type=int, default=2)
    label.add_argument("--report")
    label.set_defaults(function=command_label)
    commands.add_parser("digest").set_defaults(function=lambda args: print(digest(args.bench)))
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
