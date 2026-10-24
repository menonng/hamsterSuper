"""Differential-drive Hamster dynamics with domain randomization."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..config import DomainRandomization, RobotSpec
from .geometry import raycast, rect_penetration


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SimRobot:
    def __init__(self, spec: RobotSpec, dr: DomainRandomization, rng: np.random.Generator):
        self.spec, self.dr, self.rng = spec, dr, rng
        self.reset((0.0, 0.0, 0.0))

    def reset(self, pose):
        self.x, self.y, self.th = map(float, pose)
        self.vl = self.vr = 0.0
        s, dr, rng = self.spec, self.dr, self.rng
        if dr.enabled:
            self.gain = rng.uniform(*dr.wheel_gain, size=2)
            self.vmax = rng.uniform(*dr.max_speed)
            self.tau = rng.uniform(*dr.motor_tau)
            self.delay = int(rng.integers(dr.action_delay_steps[0], dr.action_delay_steps[1] + 1))
            self.track = rng.uniform(*dr.track_width)
        else:
            self.gain = np.ones(2)
            self.vmax, self.tau, self.delay, self.track = s.max_speed, s.motor_tau, 0, s.track_width
        self.queue = deque([np.zeros(2)] * self.delay)

    @property
    def pose(self):
        return np.array([self.x, self.y, self.th])

    def step(self, cmd, dt: float, substeps: int, boxes: np.ndarray) -> bool:
        """cmd: (left, right) in [-1, 1]. Returns True if the robot touched a wall."""
        self.queue.append(np.clip(np.asarray(cmd, dtype=np.float64), -1, 1))
        cmd = self.queue.popleft()
        target = cmd * self.gain * self.vmax
        h = dt / substeps
        a = 1.0 - math.exp(-h / max(self.tau, 1e-3))
        collided = False
        hl, hw = self.spec.body_length / 2, self.spec.body_width / 2   # the footprint drawn in the 3D view
        for _ in range(substeps):
            self.vl += a * (target[0] - self.vl)
            self.vr += a * (target[1] - self.vr)
            vl, vr = self.vl, self.vr
            if self.dr.enabled and self.dr.slip_std > 0:
                vl *= 1 + self.rng.normal(0, self.dr.slip_std)
                vr *= 1 + self.rng.normal(0, self.dr.slip_std)
            v = 0.5 * (vl + vr)
            w = (vr - vl) / self.track
            nth = wrap(self.th + w * h + (self.rng.normal(0, self.dr.heading_drift_std) / substeps
                                          if self.dr.enabled else 0.0))
            nx = self.x + v * math.cos(nth) * h
            ny = self.y + v * math.sin(nth) * h
            # walls only block: a motion that would put the footprint into a wall does
            # not happen (no pushing out, no sliding along the wall). Try the full
            # motion, then turning only, then driving only; otherwise stay put.
            for cx, cy, ct in ((nx, ny, nth), (self.x, self.y, nth),
                               (self.x + v * math.cos(self.th) * h, self.y + v * math.sin(self.th) * h, self.th)):
                if rect_penetration(np.array([cx, cy]), ct, hl, hw, boxes)[0] <= 1e-9:
                    break
            else:
                cx, cy, ct = self.x, self.y, self.th
            if (cx, cy, ct) != (nx, ny, nth):
                collided = True
                # wheels slip against the wall: the motors lose speed
                self.vl *= 0.5
                self.vr *= 0.5
            self.x, self.y, self.th = float(cx), float(cy), float(ct)
        return collided

    # --------------------------------------------------------------- sensors
    def ir(self, boxes: np.ndarray) -> np.ndarray:
        """Normalized IR proximity in [0, 1] (1 = touching), (left, right)."""
        s = self.spec
        ang = self.th + np.array([s.ir_angle, -s.ir_angle])
        o = np.array([self.x, self.y]) + s.ir_offset * np.array([math.cos(self.th), math.sin(self.th)])
        d = raycast(o, np.stack([np.cos(ang), np.sin(ang)], axis=1), boxes, s.ir_range)
        out = 1.0 - d / s.ir_range
        if self.dr.enabled:
            out = out + self.rng.normal(0, self.dr.ir_noise_std, 2)
        return np.clip(out, 0, 1)
