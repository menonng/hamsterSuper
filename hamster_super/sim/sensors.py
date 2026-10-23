"""Simulated camera depth rays (identical interface to robot.perception)."""
from __future__ import annotations

import math

import numpy as np

from ..config import WALL_HEIGHT, CameraSpec, DomainRandomization
from .camera import RayLUT
from .geometry import raycast


class DepthRays:
    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.lut = RayLUT(spec)
        self.angles = spec.ray_angles()
        self.r_min = self.lut.r_min
        self.see_over = self.r_min * (1 - WALL_HEIGHT / spec.mount_height)
        self.pitch_err = 0.0     # actual tilt - configured tilt (rad)
        self.height_err = 0.0    # actual lens height - configured height (cm)

    def set_mount_error(self, pitch_err: float, height_err: float):
        """The camera tilt is set by hand once; perception then converts image
        rows to distance with the configured tilt/height, so any setup error
        bends every measured distance in the same way for the whole run."""
        self.pitch_err, self.height_err = float(pitch_err), float(height_err)

    def _mount_distortion(self, r: np.ndarray) -> np.ndarray:
        if self.pitch_err == 0.0 and self.height_err == 0.0:
            return r
        h = self.spec.mount_height
        depression = np.arctan2(h + self.height_err, r)        # true angle below horizontal
        seen = depression - self.pitch_err                      # what the configured model assumes
        with np.errstate(divide="ignore"):
            out = np.where(seen > 1e-3, h / np.tan(np.maximum(seen, 1e-3)), self.spec.max_range)
        return np.where(r >= self.spec.max_range, r, out)

    def origin(self, pose) -> np.ndarray:
        x, y, th = pose
        return np.array([x, y]) + self.spec.mount_forward * np.array([math.cos(th), math.sin(th)])

    def true_ranges(self, pose, boxes: np.ndarray) -> np.ndarray:
        """What the camera can actually see along each ray. A wall closer than
        about half the blind zone is *below* the line of sight to the first
        visible floor point, so the camera looks over it and sees whatever is
        behind it; the IR sensors have to cover that case."""
        th = pose[2] + self.angles
        dirs = np.stack([np.cos(th), np.sin(th)], axis=1)
        o = self.origin(pose)
        d, ex = raycast(o, dirs, boxes, self.spec.max_range, return_exit=True)
        # the camera only looks over a wall if even its top surface (0.5 cm deep)
        # is below the line of sight to the first visible floor point
        over = ex < self.see_over
        if over.any():
            o2 = o + dirs[over] * (ex[over, None] + 0.01)
            d[over] = np.minimum(raycast(o2, dirs[over], boxes, self.spec.max_range) + ex[over] + 0.01,
                                 self.spec.max_range)
        return d

    def measure(self, pose, boxes, dr: DomainRandomization, rng: np.random.Generator) -> np.ndarray:
        r = self._mount_distortion(self.true_ranges(pose, boxes))
        if dr.enabled:
            r = r * (1 + rng.normal(0, dr.ray_noise_std, r.shape))
            drop = rng.random(r.shape) < dr.ray_dropout
            r[drop] = self.spec.max_range
            spur = rng.random(r.shape) < dr.ray_spurious
            r[spur] = rng.uniform(self.r_min[spur], self.spec.max_range)
        return np.clip(r, self.r_min, self.spec.max_range)
