"""Local-only perception until an explicit minimized crop is passed to Astra."""
from .geometry import (CameraFrame, Intrinsics, CalibratedTransform, PerceptionError,
                       TrackStore, deproject, fit_plane, fit_prismatic, fit_revolute, fuse_points)
from .images import CaptureRegistry, ImageCrop
from .cameras import RealSenseSource, RosCameraBuffer, PerceptionLoop, probe_cameras, capture_preview
