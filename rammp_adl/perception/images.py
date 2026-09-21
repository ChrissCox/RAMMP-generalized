"""Bounded local capture provenance and minimized image egress."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import math
import time
import uuid

from .geometry import CameraFrame, PerceptionError, positive_bounds


@dataclass(frozen=True)
class ImageCrop:
    image_id: str
    original_image_id: str
    camera_id: str
    source_frame: str
    calibration_id: str
    captured_at: float
    crop_xyxy: tuple[int, int, int, int]
    width: int
    height: int
    jpeg_bytes: bytes
    contains_face: bool | None = None
    face_redacted: bool = False

    def validate_for_egress(self, *, max_bytes, max_long_edge, allow_face):
        if not all(isinstance(value, str) and 0 < len(value) <= 256 for value in
                   (self.image_id, self.original_image_id, self.camera_id, self.source_frame, self.calibration_id)):
            raise PerceptionError("Crop provenance is missing or oversized")
        if not isinstance(self.jpeg_bytes, bytes) or not 0 < len(self.jpeg_bytes) <= max_bytes:
            raise PerceptionError("Crop byte budget exceeded")
        if not 0 < min(self.width, self.height) <= max(self.width, self.height) <= max_long_edge:
            raise PerceptionError("Crop dimensions exceed policy")
        if not allow_face and self.contains_face is not False:
            raise PerceptionError("Face content is present or unknown; local redaction/verification is required")
        if not math.isfinite(self.captured_at):
            raise PerceptionError("Crop capture time is invalid")
        x0, y0, x1, y1 = self.crop_xyxy
        if not 0 <= x0 < x1 or not 0 <= y0 < y1:
            raise PerceptionError("Crop mapping is invalid")
        from PIL import Image
        with Image.open(io.BytesIO(self.jpeg_bytes)) as image:
            if image.format != "JPEG" or image.size != (self.width, self.height):
                raise PerceptionError("Image encoding does not match registered crop")
            image.verify()

    def cloud_metadata(self):
        return {"image_id": self.image_id, "source_frame": self.source_frame,
                "camera_id": self.camera_id, "calibration_id": self.calibration_id,
                "face_redacted": self.face_redacted}

    def as_openai_input(self):
        return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(self.jpeg_bytes).decode("ascii"), "detail": "low"}

    def original_box(self, normalized_xyxy):
        if len(normalized_xyxy) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in normalized_xyxy):
            raise PerceptionError("Grounding box is not normalized")
        x0, y0, x1, y1 = normalized_xyxy
        if x0 >= x1 or y0 >= y1:
            raise PerceptionError("Grounding box is empty or reversed")
        left, top, right, bottom = self.crop_xyxy
        return (left+x0*(right-left), top+y0*(bottom-top), left+x1*(right-left), top+y1*(bottom-top))


class CaptureRegistry:
    def __init__(self, *, max_frames=8, max_crops=16, max_age_s=2.0, clock=time.monotonic):
        positive_bounds(max_age_s)
        if type(max_frames) is not int or not 1 <= max_frames <= 1024 or type(max_crops) is not int or not 1 <= max_crops <= 1024:
            raise PerceptionError("Capture cache must be bounded")
        self.max_frames, self.max_age_s, self.clock = max_frames, max_age_s, clock
        self.max_crops = max_crops
        self.frames = {}
        self.crops = {}

    def register(self, frame: CameraFrame):
        frame.require_fresh(self.clock(), self.max_age_s)
        if frame.image_id in self.frames:
            raise PerceptionError("Capture ID already registered")
        self.frames[frame.image_id] = frame
        while len(self.frames) > self.max_frames:
            old = next(iter(self.frames))
            del self.frames[old]
            self.crops = {key: crop for key, crop in self.crops.items() if crop.original_image_id != old}

    def create_crop(self, image_id, crop_xyxy, *, contains_face=None, redact_boxes=(), max_long_edge=640, max_bytes=200000):
        from PIL import Image, ImageDraw
        if type(max_long_edge) is not int or not 1 <= max_long_edge <= 4096 or type(max_bytes) is not int or not 1 <= max_bytes <= 2000000:
            raise PerceptionError("Invalid crop encoding budget")
        if contains_face is not None and type(contains_face) is not bool:
            raise PerceptionError("Face-content assessment must be boolean or unknown")
        frame = self.frames[image_id]
        frame.require_fresh(self.clock(), self.max_age_s)
        left, top, right, bottom = crop_xyxy
        height, width = frame.rgb.shape[:2]
        if any(type(v) is not int for v in crop_xyxy) or not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise PerceptionError("Crop lies outside original frame")
        picture = Image.fromarray(frame.rgb).crop((left, top, right, bottom))
        if redact_boxes:
            draw = ImageDraw.Draw(picture)
            for x0, y0, x1, y1 in redact_boxes:
                if not 0 <= x0 < x1 <= right-left or not 0 <= y0 < y1 <= bottom-top:
                    raise PerceptionError("Redaction lies outside crop")
                draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
            # Caller must still verify all face content is removed; masks alone
            # cannot certify that an unseen face outside those boxes is absent.
        picture.thumbnail((max_long_edge, max_long_edge))
        payload = b""
        for quality in (85, 70, 55, 40):
            output = io.BytesIO()
            picture.save(output, format="JPEG", quality=quality)
            payload = output.getvalue()
            if len(payload) <= max_bytes:
                break
        if len(payload) > max_bytes:
            raise PerceptionError("Crop cannot fit the configured byte budget")
        crop = ImageCrop("crop-"+uuid.uuid4().hex, frame.image_id, frame.camera_id, frame.frame_id,
                         frame.calibration_id, frame.captured_at, tuple(crop_xyxy), *picture.size,
                         payload, contains_face, bool(redact_boxes))
        self.crops[crop.image_id] = crop
        while len(self.crops) > self.max_crops:
            del self.crops[next(iter(self.crops))]
        return crop

    def resolve(self, crop_id):
        crop = self.crops[crop_id]
        frame = self.frames[crop.original_image_id]
        frame.require_fresh(self.clock(), self.max_age_s)
        return crop, frame
