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
        over = d < self.see_over
        if over.any():
            o2 = o + dirs[over] * (ex[over, None] + 0.01)
            d[over] = np.minimum(raycast(o2, dirs[over], boxes, self.spec.max_range) + ex[over] + 0.01,
                                 self.spec.max_range)
        return d

    def measure(self, pose, boxes, dr: DomainRandomization, rng: np.random.Generator) -> np.ndarray:
        r = self.true_ranges(pose, boxes)
        if dr.enabled:
            r = r * (1 + rng.normal(0, dr.ray_noise_std, r.shape))
            drop = rng.random(r.shape) < dr.ray_dropout
            r[drop] = self.spec.max_range
            spur = rng.random(r.shape) < dr.ray_spurious
            r[spur] = rng.uniform(self.r_min[spur], self.spec.max_range)
        return np.clip(r, self.r_min, self.spec.max_range)
