"""Discovery and grounding on the Jetson: an open-vocabulary detector first, Astra only when it does not find the target.

The detector (OWLv2, the one rammp_box_opening already runs here) looks at the same screened keyframe crop
Astra would get, for a household vocabulary and the task's own words, in one pass (about 0.5 s warm, fp16 on
the Orin's GPU, against 7-10 s and a paid request). Its boxes become discovery entities in Astra's own answer
shape: a label, a kind (handle, surface, free_object, other), what a handle is attached to (the door or drawer
box that holds it), a grasp point at the box's centre, and whether the task's target is among them. Everything
metric still comes from depth afterwards, as for Astra's boxes. When the target is not seen, or the detector
is unavailable, the request goes to Astra unchanged, which also gives the search hints.
"""
from __future__ import annotations

import asyncio
import io
import re
import threading

from ..reasoning import ReasoningResult

KINDS = {
    "handle": ("door handle", "drawer handle", "cabinet handle", "handle", "knob", "pull"),
    "surface": ("cabinet door", "door", "drawer", "lid", "flap", "oven door", "fridge door", "microwave door"),
    "free_object": ("cup", "mug", "bottle", "bowl", "can", "box", "spoon", "fork", "remote", "phone", "book", "sponge",
                    "towel", "plate", "glass", "jar", "fruit", "apple", "banana", "pen", "toy"),
    "other": ("button", "microwave", "switch", "table", "shelf", "sink", "container"),
}
THRESHOLDS = {"handle": .25, "surface": .30, "free_object": .20, "other": .30, "task": .15}
STOP = frozenset("""open close shut pick up grab take put place push pull press move turn get give hand bring hold lift
the a an in on of front you me to into onto from please and that this it its my your there here out off over under with for at
""".split())


GRASP_VERBS = frozenset("pick grab take hold lift carry bring give hand fetch".split())


def kind_of(phrase, *, default="free_object"):
    words = phrase.split()
    for kind in ("handle", "surface", "other", "free_object"):
        if phrase in KINDS[kind] or (words and words[-1] in {w.split()[-1] for w in KINDS[kind]}):
            return kind
    return default


def contained(a, b):
    """How much of the smaller box lies inside the other."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1)*max(0., y2-y1)
    smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return inter/smaller if smaller > 0 else 0.


def task_phrases(task_text):
    """The task's content words: runs of consecutive non-stop words, and each word alone."""
    words = re.findall(r"[a-z]+", task_text.lower())
    phrases, run = [], []
    for word in words+[""]:
        if word and word not in STOP:
            run.append(word)
            continue
        if run:
            phrases.append(" ".join(run))
            phrases += [w for w in run if len(run) > 1]
            run = []
    return list(dict.fromkeys(phrases))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1)*max(0., y2-y1)
    area = lambda box: max(0., box[2]-box[0])*max(0., box[3]-box[1])
    union = area(a)+area(b)-inter
    return inter/union if union > 0 else 0.


class OwlDetector:
    """OWLv2, loaded once and kept warm; detect() blocks (call it off the event loop)."""

    def __init__(self, name="google/owlv2-base-patch16-ensemble", *, device=None):
        self.name, self.device, self._model, self._processor = name, device, None, None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = Owlv2Processor.from_pretrained(self.name)
            model = Owlv2ForObjectDetection.from_pretrained(self.name).to(self.device).eval()
            self._model = model.half() if self.device == "cuda" else model

    def detect(self, image, queries, *, threshold=.1):
        """[(query index, score, [x1, y1, x2, y2] normalised to the image)] for a PIL image."""
        import torch
        self.load()
        with self._lock:
            inputs = self._processor(text=[[f"a photo of a {q}" for q in queries]], images=image, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.no_grad():
                output = self._model(**inputs)
            side = max(image.size)                       # OWLv2 pads to a square
            found = self._processor.post_process_object_detection(
                output, threshold=threshold, target_sizes=torch.tensor([[side, side]], device=self.device))[0]
        width, height = image.size
        return [(int(label), float(score), [max(0., box[0]/width), max(0., box[1]/height), min(1., box[2]/width), min(1., box[3]/height)])
                for score, label, box in zip(found["scores"].tolist(), found["labels"].tolist(), found["boxes"].tolist())]


def discover(detector, image, image_id, task_text, *, max_entities=12):
    """Discovery entities in Astra's answer shape, and whether the task's target is among them."""
    phrases = task_phrases(task_text)
    # A word the vocabulary does not know is a thing to pick up only when the task is about picking something up.
    unknown = "free_object" if GRASP_VERBS & set(re.findall(r"[a-z]+", task_text.lower())) else "other"
    vocabulary = [p for kind in KINDS.values() for p in kind]
    queries = list(dict.fromkeys(phrases+vocabulary))
    raw = detector.detect(image, queries, threshold=min(THRESHOLDS.values()))
    kept = []
    for index, score, box in sorted(raw, key=lambda d: -d[1]):
        phrase = queries[index]
        kind = kind_of(phrase, default=unknown)
        floor = THRESHOLDS["task"] if phrase in phrases and phrase not in vocabulary else THRESHOLDS[kind]
        area = (box[2]-box[0])*(box[3]-box[1])
        if score < floor or area <= 0 or area > .85:             # a box over the whole view says nothing
            continue
        # The same place named twice: a box mostly inside a better one of its own kind (a knob on the handle, a
        # second door around the door), or any box nearly the same as a better one.
        if any(iou(box, other["box_xyxy_normalized"]) > .6
               or (other["kind"] == kind and contained(box, other["box_xyxy_normalized"]) > .7) for other in kept):
            continue
        kept.append({"image_id": image_id, "label": phrase, "kind": kind, "attached_to": "", "grasp_point_given": True,
                     "grasp_point_xy_normalized": [(box[0]+box[2])/2., (box[1]+box[3])/2.],
                     "box_xyxy_normalized": [round(v, 4) for v in box], "confidence": round(score, 3)})
        if len(kept) >= max_entities:
            break
    # Within a kind, a detection at less than half the best one's score is the detector reaching, not a second thing.
    best = {}
    for entity in kept:
        best[entity["kind"]] = max(best.get(entity["kind"], 0.), entity["confidence"])
    kept = [e for e in kept if e["confidence"] >= .5*best[e["kind"]]]
    surfaces = [e for e in kept if e["kind"] == "surface"]
    for entity in kept:
        if entity["kind"] != "handle":
            continue
        gx, gy = entity["grasp_point_xy_normalized"]
        holders = [s for s in surfaces if s["box_xyxy_normalized"][0] <= gx <= s["box_xyxy_normalized"][2]
                   and s["box_xyxy_normalized"][1] <= gy <= s["box_xyxy_normalized"][3]]
        holder = max(holders, key=lambda s: s["confidence"])["label"] if holders else None
        entity["attached_to"] = holder or ("drawer" if "drawer" in entity["label"] else "door")
    # The target is the head of the task's first phrase ("door" of "cabinet door"); a detection naming it is the target.
    head = phrases[0].split()[-1] if phrases else None
    visible = head is None or any(head in e["label"].split() or (head in ("door", "drawer") and e["kind"] == "handle"
                                                                  and head in e["attached_to"].split()) for e in kept)
    return kept, visible


class LocalFirstReasoner:
    """Astra behind a local detector: discover_scene and ground_target are answered locally when they can be."""

    def __init__(self, reasoner, detector, *, log=None):
        self._reasoner, self._detector, self._log = reasoner, detector, log or (lambda message: None)
        self.local_answers = 0

    def __getattr__(self, name):
        return getattr(self._reasoner, name)

    @staticmethod
    def _image(crop):
        from PIL import Image
        return Image.open(io.BytesIO(crop.jpeg_bytes)).convert("RGB")

    async def discover_scene(self, context, images, *, task_text="", **kwargs):
        if images:
            try:
                crop = images[0]
                entities, visible = await asyncio.to_thread(discover, self._detector, self._image(crop), crop.image_id, task_text)
            except Exception as exc:                        # noqa: BLE001 - no local answer: Astra's, unchanged
                self._log(f"local discovery unavailable ({type(exc).__name__}: {str(exc)[:80]}); asking astra")
            else:
                if entities and visible:
                    self.local_answers += 1
                    seen = ", ".join("%s %.2f" % (e["label"], e["confidence"]) for e in entities)
                    self._log(f"local discovery: {seen}")
                    return ReasoningResult("OK", detail="local detector", model_id="owlv2", candidates=tuple(entities),
                                           proposal={"target_visible": True, "search_hint": "none", "search_note": ""})
                self._log(f"local discovery did not see the target of {task_text[:60]!r}; asking astra")
        return await self._reasoner.discover_scene(context, images, task_text=task_text, **kwargs)

    async def ground_target(self, context, entity_id, images, *, query="", **kwargs):
        entity = next((e for e in context.get("entities", ()) if e["entity_id"] == entity_id), None)
        if images and entity is not None:
            label = entity["label"]
            try:
                crop = images[0]
                found = await asyncio.to_thread(self._detector.detect, self._image(crop), [label], threshold=THRESHOLDS["task"])
            except Exception as exc:                        # noqa: BLE001 - no local answer: Astra's, unchanged
                self._log(f"local grounding unavailable ({type(exc).__name__}); asking astra")
            else:
                if found:
                    _, score, box = max(found, key=lambda d: d[1])
                    self.local_answers += 1
                    return ReasoningResult("OK", detail="local detector", model_id="owlv2", candidates=(
                        {"image_id": crop.image_id, "label": label, "entity_id": entity_id,
                         "box_xyxy_normalized": [round(v, 4) for v in box], "confidence": round(score, 3)},))
        return await self._reasoner.ground_target(context, entity_id, images, query=query, **kwargs)
