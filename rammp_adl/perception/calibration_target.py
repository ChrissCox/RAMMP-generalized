"""NEW calibration preparation: printable target and local 2D observations only.

Nominal print dimensions never constitute physical measurements. No camera pose,
TF, clock mapping, metric admission or robot command is produced by this module.
OpenCV owns marker generation/detection; the PDF specifies physical print scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

import numpy as np

from ..contracts import Catalog, digest, strict_loads
from .geometry import PerceptionError, positive_bounds, require_ids


class CalibrationTarget:
    def __init__(self, spec=None):
        import cv2
        if spec is None:
            spec = strict_loads((Catalog().root/"config/calibration-target.json").read_bytes())
        keys = {"target_id", "dictionary", "squares_x", "squares_y", "square_length_m", "marker_length_m", "legacy_pattern"}
        if not isinstance(spec, dict) or set(spec) != keys:
            raise PerceptionError("Expected the closed local calibration target specification")
        require_ids(spec["target_id"])
        if spec["dictionary"] != "DICT_5X5_100" or type(spec["legacy_pattern"]) is not bool:
            raise PerceptionError("Unsupported target dictionary or pattern version")
        nx, ny = spec["squares_x"], spec["squares_y"]
        if any(type(n) is not int or not 3 <= n <= 12 for n in (nx, ny)):
            raise PerceptionError("Target square counts must be in 3..12")
        positive_bounds(spec["square_length_m"], spec["marker_length_m"])
        if not .005 <= spec["square_length_m"] <= .1 or not .4 <= spec["marker_length_m"]/spec["square_length_m"] <= .8:
            raise PerceptionError("Unsupported target feature sizes")
        self.spec_json = json.dumps(spec, sort_keys=True, allow_nan=False)
        self.spec_digest = digest(spec)
        self.cv2 = cv2
        self.board = cv2.aruco.CharucoBoard((nx, ny), spec["square_length_m"], spec["marker_length_m"],
                                           cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
        self.board.setLegacyPattern(spec["legacy_pattern"])
        self.detector = cv2.aruco.CharucoDetector(self.board)

    @property
    def spec(self):
        return json.loads(self.spec_json)

    def image(self):
        # Default 280-pixel squares give 210-pixel markers and exactly 30
        # pixels per dictionary cell including the one-cell black border.
        return self.board.generateImage((self.spec["squares_x"]*280, self.spec["squares_y"]*280), marginSize=0, borderBits=1)

    def detect(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not 0 < rgb.shape[0]*rgb.shape[1] <= 16000000:
            raise PerceptionError("Target detection requires a bounded uint8 RGB image")
        gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
        corners, ids, _, marker_ids = self.detector.detectBoard(gray)
        ids = [] if ids is None else ids.reshape(-1).tolist()
        pixels = [] if corners is None else corners.reshape(-1, 2).tolist()
        total = len(self.board.getChessboardCorners())
        if len(ids) != len(pixels) or len(set(ids)) != len(ids) or any(not 0 <= i < total for i in ids) or not np.isfinite(pixels).all():
            raise PerceptionError("Detector returned inconsistent corner observations")
        h, w = rgb.shape[:2]
        if any(not 0 <= x < w or not 0 <= y < h for x, y in pixels):
            raise PerceptionError("Detector returned corners outside the image")
        noncollinear = len(ids) >= 4 and not self.board.checkCharucoCornersCollinear(np.asarray(ids, dtype=np.int32))
        return {"status": "corners_observed" if ids else "target_not_observed",
                "target_id": self.spec["target_id"], "target_spec_digest": self.spec_digest,
                "opencv_version": self.cv2.__version__, "image_size_px": [w, h],
                "corner_ids": ids, "corners_xy_px": pixels, "corner_count": len(ids),
                "total_target_corners": total, "noncollinear_correspondences": bool(noncollinear),
                "marker_ids": [] if marker_ids is None else marker_ids.reshape(-1).tolist(),
                "physical_target_dimensions_verified": False, "extrinsics_calibrated": False,
                "metric_geometry_validated": False, "frames_uploaded": 0,
                "validation_scope": "Local 2D corner observations only; no pose, calibration or task facts"}


def _print_pdf(gray, spec, page_mm):
    """One-page PDF with an unscaled lossless fiducial and a 100 mm ruler."""
    unit = 72/25.4
    pw, ph = page_mm
    bw, bh = (spec["squares_x"]*spec["square_length_m"]*1000,
              spec["squares_y"]*spec["square_length_m"]*1000)
    if bw > pw-30 or bh > ph-50:
        raise PerceptionError("Target does not fit this page with print margins")
    x, y = (pw-bw)/2, (ph-bh)/2+5
    commands = [f"q {bw*unit:.8f} 0 0 {bh*unit:.8f} {x*unit:.8f} {y*unit:.8f} cm /Im0 Do Q"]
    title = "RAMMP calibration target - print Actual size / 100 percent"
    caption = f"Pattern: {bw:g} x {bh:g} mm. Squares: {spec['square_length_m']*1000:g} mm."
    for text, ty in ((title, ph-12), (caption, 14)):
        commands.append(f"BT /F1 10 Tf {x*unit:.8f} {ty*unit:.8f} Td ({text}) Tj ET")
    # Known physical length with end ticks, positioned clear of the target.
    rx, ry = pw/2-50, 23
    commands.append(f"0 G 0.6 w {rx*unit:.8f} {ry*unit:.8f} m {(rx+100)*unit:.8f} {ry*unit:.8f} l S")
    for tick in (rx, rx+100):
        commands.append(f"{tick*unit:.8f} {(ry-2)*unit:.8f} m {tick*unit:.8f} {(ry+2)*unit:.8f} l S")
    commands.append(f"BT /F1 9 Tf {(pw/2-7)*unit:.8f} {(ry+2)*unit:.8f} Td (100 mm) Tj ET")
    content = "\n".join(commands).encode("ascii")
    compressed = zlib.compress(gray.tobytes(), 9)
    def stream(header, data):
        return header+b" /Length "+str(len(data)).encode()+b" >>\nstream\n"+data+b"\nendstream"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw*unit:.8f} {ph*unit:.8f}] "
                "/Resources << /XObject << /Im0 4 0 R >> /Font << /F1 5 0 R >> >> /Contents 6 0 R >>").encode(),
               stream((f"<< /Type /XObject /Subtype /Image /Width {gray.shape[1]} /Height {gray.shape[0]} "
                       "/ColorSpace /DeviceGray /BitsPerComponent 8 /Interpolate false /Filter /FlateDecode").encode(), compressed),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", stream(b"<<", content)]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{i} 0 obj\n".encode()+obj+b"\nendobj\n")
    start = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    return bytes(output)


def generate_printables(output_dir):
    target = CalibrationTarget()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    gray = target.image()
    padded = np.pad(gray, 60, constant_values=255)
    check = target.detect(np.repeat(padded[..., None], 3, axis=2))
    if check["corner_count"] != check["total_target_corners"]:
        raise PerceptionError("Generated target failed its digital detection check")
    outputs = {}
    for name, page in (("a4", (297., 210.)), ("letter", (279.4, 215.9))):
        data = _print_pdf(gray, target.spec, page)
        path = destination/f"target-{name}.pdf"
        path.write_bytes(data)
        outputs[name] = {"file": path.name, "page_mm": list(page), "sha256": hashlib.sha256(data).hexdigest()}
    report = {"spec": target.spec, "spec_digest": target.spec_digest,
              "outputs": outputs, "digital_detection_check": check,
              "print_instruction": "Print landscape at Actual size / 100%; verify both pattern dimensions and the 100 mm ruler; mount flat and rigid.",
              "physical_target_dimensions_verified": False}
    (destination/"manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def read_local_capture(path):
    """Read an existing local capture without assigning it a new timestamp."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != {"rgb.npy", "depth_m.npy", "metadata_json.npy"}:
            raise PerceptionError("Expected the local RGB-D capture archive format")
        if sum(info.file_size for info in infos) > 192*1024*1024:
            raise PerceptionError("Capture archive exceeds bounded inspection size")
    with np.load(path, allow_pickle=False) as capture:
        meta = strict_loads(str(capture["metadata_json"]))
        rgb = capture["rgb"]
        info = meta["rgb_info"]
        if rgb.shape[:2] != (info["height"], info["width"]):
            raise PerceptionError("Capture dimensions disagree with CameraInfo")
    return rgb, meta


def inspect_capture(path):
    rgb, meta = read_local_capture(path)
    info = meta["rgb_info"]
    report = CalibrationTarget().detect(rgb)
    report.update(capture_file=str(path), capture_id=meta["capture_id"], source_stamps_ns=meta["source_stamps_ns"],
                  rgb_info_digest=digest(info), historical_capture=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output-dir", required=True)
    inspect = commands.add_parser("inspect-capture")
    inspect.add_argument("--capture", required=True)
    inspect.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            report = generate_printables(args.output_dir)
        else:
            report = inspect_capture(args.capture)
            with Path(args.output).open("x", encoding="utf-8") as output:
                output.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "unavailable_or_rejected", "detail": str(exc), "hardware_commands": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
