"""Camera frame -> depth rays (the same 21 rays the policy was trained on).

1. Segment wall pixels (default: walls darker than the floor; HSV range or a
   custom function can be plugged in once real course photos are available).
2. Clean the mask with a small morphological opening (removes floor dots and noise).
3. Walk each ray's floor line in the image (RayLUT) and take the first wall pixel.
   Because the camera height/pitch are known, the image row gives the
   distance directly (ground-plane geometry); no depth sensor needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CameraSpec
from ..sim.camera import RayLUT, wall_mask_dark


@dataclass
class SegmentationConfig:
    method: str = "dark"                 # "dark" | "hsv"
    dark_threshold: int = 110            # gray < threshold -> wall
    hsv_lo: list = field(default_factory=lambda: [0, 0, 0])
    hsv_hi: list = field(default_factory=lambda: [180, 255, 90])
    open_kernel: int = 3                 # morphological opening size (0 = off)
    min_run_cm: float = 1.5


class Perception:
    def __init__(self, spec: CameraSpec, seg: SegmentationConfig | None = None):
        self.spec = spec
        self.seg = seg or SegmentationConfig()
        self.lut = RayLUT(spec)
        self.last_mask = None

    def wall_mask(self, rgb: np.ndarray) -> np.ndarray:
        s = self.seg
        if s.method == "hsv":
            import cv2
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array(s.hsv_lo, np.uint8), np.array(s.hsv_hi, np.uint8)) > 0
        else:
            mask = wall_mask_dark(rgb, s.dark_threshold)
        if s.open_kernel > 1:
            try:
                import cv2
                k = np.ones((s.open_kernel, s.open_kernel), np.uint8)
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
            except ImportError:
                pass
        return mask

    def rays(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[:2] != (self.spec.height, self.spec.width):
            import cv2
            rgb = cv2.resize(rgb, (self.spec.width, self.spec.height), interpolation=cv2.INTER_AREA)
        self.last_mask = self.wall_mask(rgb)
        return self.lut.ranges_from_mask(self.last_mask, self.seg.min_run_cm)
