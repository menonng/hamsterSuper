"""Pose estimation without wheel encoders.

The Hamster has no encoders, so odometry integrates the commanded wheel
speeds (dead reckoning). Drift is corrected by the fact that every wall lies
on a grid line x = 9i or y = 9j: wall hits seen by the camera are matched to
the nearest grid line and a small least-squares pose correction is applied
("grid snapping", an ICP against the lattice).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..config import GRID_SPACING, WALL_THICKNESS, RobotSpec

G = GRID_SPACING


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Odometry:
    def __init__(self, spec: RobotSpec, pose, delay_steps: int = 1):
        self.spec = spec
        self.pose = np.asarray(pose, dtype=float).copy()
        self.vl = self.vr = 0.0
        self.queue = deque([np.zeros(2)] * delay_steps)

    def step(self, cmd, dt: float, substeps: int = 4, blocked: bool = False):
        self.queue.append(np.asarray(cmd, dtype=float))
        cmd = self.queue.popleft()
        s = self.spec
        h = dt / substeps
        a = 1 - math.exp(-h / max(s.motor_tau, 1e-3))
        for _ in range(substeps):
            self.vl += a * (cmd[0] * s.max_speed - self.vl)
            self.vr += a * (cmd[1] * s.max_speed - self.vr)
            v = 0.0 if blocked else 0.5 * (self.vl + self.vr)
            w = (self.vr - self.vl) / s.track_width
            self.pose[2] = wrap(self.pose[2] + w * h)
            self.pose[0] += v * math.cos(self.pose[2]) * h
            self.pose[1] += v * math.sin(self.pose[2]) * h
        return self.pose


class GridSnapLocalizer:
    def __init__(self, tol: float = 2.0, alpha: float = 0.5, min_points: int = 4):
        self.tol, self.alpha, self.min_points = tol, alpha, min_points
        self.last_used = 0

    def correct(self, pose, cam_origin, angles, ranges, valid) -> np.ndarray:
        x, y, th = pose
        rows, res = [], []
        for a, r, ok in zip(angles, ranges, valid):
            if not ok:
                continue
            d = r + WALL_THICKNESS / 2
            px = cam_origin[0] + d * math.cos(th + a)
            py = cam_origin[1] + d * math.sin(th + a)
            rx = px - G * round(px / G)
            ry = py - G * round(py / G)
            if abs(rx) < 0.8 and abs(ry) < 0.8:
                continue   # at a dot: ambiguous
            if abs(rx) <= abs(ry) and abs(rx) < self.tol:
                rows.append((1.0, 0.0, -(py - y)))
                res.append(rx)
            elif abs(ry) < self.tol:
                rows.append((0.0, 1.0, (px - x)))
                res.append(ry)
        self.last_used = len(res)
        if len(res) < self.min_points:
            return np.asarray(pose, dtype=float)
        J = np.asarray(rows)
        r = np.asarray(res)
        # Robust weights (Huber) + damping prior
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.7)
        A = J.T @ (J * w[:, None]) + np.diag([0.5, 0.5, 20.0])
        b = -J.T @ (r * w)
        delta = np.linalg.solve(A, b)
        # A single axis seen only (e.g. a corridor) must not move the other axis
        if not np.any(J[:, 0]):
            delta[0] = 0.0
        if not np.any(J[:, 1]):
            delta[1] = 0.0
        out = np.asarray(pose, dtype=float) + self.alpha * delta
        out[2] = wrap(out[2])
        return out
