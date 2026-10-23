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
from ..maps.grid_map import edge_between

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
    def __init__(self, tol: float = 2.0, alpha: float = 0.5, min_points: int = 4, damp=(0.5, 20.0)):
        self.tol, self.alpha, self.min_points, self.damp = tol, alpha, min_points, damp
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
        A = J.T @ (J * w[:, None]) + np.diag([self.damp[0], self.damp[0], self.damp[1]])
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


class LatticeParticleFilter:
    """Monte Carlo localization against the 9 cm wall lattice.

    State per particle: x, y, heading, a speed scale (the real speed differs
    from the calibrated one by battery/floor, about +-12%) and a curvature
    bias from left/right wheel imbalance; both are learned online. Motion uses the
    odometry increment scaled per particle plus noise (covers slip, wheel
    imbalance and being pinned against a wall). Measurement: wall points seen
    by the camera / IR must lie on a grid line x = 9i or y = 9j.
    """

    def __init__(self, pose, n: int = 256, scale_range=(0.88, 1.12), sigma_hit: float = 0.8,
                 rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.n = n
        self.sigma = sigma_hit
        # columns: x, y, heading, speed scale, curvature bias (rad per cm; wheel imbalance)
        self.p = np.zeros((n, 5))
        self.p[:, :3] = np.asarray(pose, dtype=float)
        self.p[:, 0:2] += self.rng.normal(0, 0.3, (n, 2))
        self.p[:, 2] += self.rng.normal(0, 0.03, n)
        self.p[:, 3] = self.rng.uniform(*scale_range, n)
        self.p[:, 4] = self.rng.normal(0, 0.012, n)
        self.w = np.full(n, 1.0 / n)
        self.last_used = 0

    def predict(self, dist: float, dtheta: float, walls=None):
        """dist / dtheta: odometry increments at nominal calibration.
        walls: edges known to be walls; a particle that would drive through one
        is (almost) impossible - the real robot cannot pass a rigid wall."""
        n, p, rng = self.n, self.p, self.rng
        before = np.floor(p[:, :2] / G).astype(int) if walls else None
        sc = p[:, 3]
        d = dist * sc * (1 + rng.normal(0, 0.08, n)) + rng.normal(0, 0.05 + 0.1 * abs(dist), n)
        # being pinned against a wall: some particles do not move at all
        stuck = rng.random(n) < 0.05
        d[stuck] = 0.0
        # rotation shares the unknown wheel-speed scale; fast spins + latency need wide noise
        dth = (dtheta * sc * (1 + rng.normal(0, 0.2, n)) + p[:, 4] * d
               + rng.normal(0, 0.01 + 0.02 * abs(dist) + 0.25 * abs(dtheta), n))
        th = p[:, 2] + 0.5 * dth
        p[:, 0] += d * np.cos(th)
        p[:, 1] += d * np.sin(th)
        p[:, 2] = th + 0.5 * dth
        p[:, 3] = np.clip(p[:, 3] + rng.normal(0, 0.003, n), 0.5, 1.6)
        p[:, 4] = np.clip(p[:, 4] + rng.normal(0, 0.0005, n), -0.05, 0.05)
        if before is not None:
            after = np.floor(p[:, :2] / G).astype(int)
            moved = np.flatnonzero(np.any(after != before, axis=1))
            if moved.size:
                bad = [i for i in moved if _crosses(tuple(before[i]), tuple(after[i]), walls)]
                if bad:
                    self.w[bad] *= 0.02
                    self.w /= self.w.sum()

    def correct_heading(self, wall_angle: float, gain: float = 0.7, gate: float = math.radians(12),
                        noise: float = 0.02) -> bool:
        """Absolute heading fix from a measured wall orientation (robot frame).
        Walls are axis-aligned, so heading = -wall_angle (mod 90 deg). Particles
        move toward the nearest consistent heading; implausibly large
        corrections are treated as outliers and ignored."""
        th = self.p[:, 2]
        delta = (-wall_angle - th + math.pi / 4) % (math.pi / 2) - math.pi / 4
        mean = math.atan2(np.sum(self.w * np.sin(delta)), np.sum(self.w * np.cos(delta)))
        if abs(mean) > gate:
            return False
        self.p[:, 2] = th + gain * delta + self.rng.normal(0, noise, self.n)
        return True

    def update(self, pts_local: np.ndarray, belief=None):
        """pts_local: (M, 2) wall points in the robot frame (x fwd, y left)."""
        self.last_used = len(pts_local)
        if len(pts_local) < 3:
            return
        p = self.p
        c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
        wx = p[:, 0:1] + c[:, None] * pts_local[None, :, 0] - s[:, None] * pts_local[None, :, 1]
        wy = p[:, 1:2] + s[:, None] * pts_local[None, :, 0] + c[:, None] * pts_local[None, :, 1]
        gx, gy = wx / G, wy / G
        dx = np.abs(gx - np.round(gx)) * G
        dy = np.abs(gy - np.round(gy)) * G
        d = np.minimum(dx, dy)
        lik = np.exp(-0.5 * (d / self.sigma) ** 2)
        if belief is not None:
            # points must also agree with what is already mapped: on a known
            # wall (x1), unknown edge (x0.5) or an edge seen to be free (x0.1)
            x0, y0, V, H = belief.edge_arrays()
            vert = dx < dy
            ii = np.where(vert, np.round(gx), np.floor(gx)).astype(int) - x0
            jj = np.where(vert, np.floor(gy), np.round(gy)).astype(int) - y0
            ii = np.clip(ii, 0, V.shape[0] - 1)
            jj = np.clip(jj, 0, V.shape[1] - 1)
            st = np.where(vert, V[ii, jj], H[ii, jj])
            lik = lik * np.choose(st + 1, [0.1, 0.5, 1.0])
        ll = np.log(0.15 + lik).sum(axis=1)
        ll -= ll.max()
        self.w *= np.exp(ll)
        tot = self.w.sum()
        if not np.isfinite(tot) or tot <= 0:
            self.w[:] = 1.0 / self.n
        else:
            self.w /= tot
        if 1.0 / np.sum(self.w ** 2) < self.n / 2:
            self._resample()

    def _resample(self):
        n = self.n
        pos = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), n - 1)
        self.p = self.p[idx]
        self.w = np.full(n, 1.0 / n)

    @property
    def pose(self) -> np.ndarray:
        w = self.w
        th = math.atan2(np.sum(w * np.sin(self.p[:, 2])), np.sum(w * np.cos(self.p[:, 2])))
        return np.array([np.sum(w * self.p[:, 0]), np.sum(w * self.p[:, 1]), th])

    @property
    def spread(self) -> float:
        """Position uncertainty (cm): weighted std of the particle cloud."""
        w, p = self.w, self.p
        mx, my = np.sum(w * p[:, 0]), np.sum(w * p[:, 1])
        return float(math.sqrt(np.sum(w * ((p[:, 0] - mx) ** 2 + (p[:, 1] - my) ** 2))))

    @property
    def scale(self) -> float:
        return float(np.sum(self.w * self.p[:, 3]))


def _crosses(a, b, walls) -> bool:
    """Moving from cell a to cell b passes a wall (diagonal moves: both ways blocked)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) + abs(dy) == 1:
        return edge_between(a, b) in walls
    if abs(dx) == 1 and abs(dy) == 1:
        m1, m2 = (b[0], a[1]), (a[0], b[1])
        return (edge_between(a, m1) in walls or edge_between(m1, b) in walls) and \
            (edge_between(a, m2) in walls or edge_between(m2, b) in walls)
    return False


_CAND = np.radians(np.arange(-45.0, 45.0, 1.0))
_ROT = np.stack([np.cos(_CAND), np.sin(_CAND)], axis=1)


def wall_orientation(pts: np.ndarray, sigma: float = 0.4, min_sep: float = 3.0, max_sep: float = 12.0,
                     max_skip: int = 5, min_pairs: int = 4, min_ratio: float = 2.5):
    """Dominant wall direction (mod 90 deg) in the robot frame.

    Walls are axis-aligned, so in the correctly rotated frame two points on
    the same wall share their x (or y) coordinate no matter how far apart they
    are along the wall. Score every candidate angle by how many well-separated
    point pairs line up; the best angle wins. Returns radians or None.
    """
    if len(pts) < 4:
        return None
    i, j = np.triu_indices(len(pts), 1)
    d = pts[j] - pts[i]
    sep = np.hypot(d[:, 0], d[:, 1])
    # neighbouring rays, moderately apart: most likely the same wall
    keep = (sep > min_sep) & (sep < max_sep) & (j - i <= max_skip)
    if keep.sum() < min_pairs:
        return None
    d = d[keep]
    c, s = _ROT[:, 0:1], _ROT[:, 1:2]
    dx = c * d[None, :, 0] + s * d[None, :, 1]        # (A, P)
    dy = -s * d[None, :, 0] + c * d[None, :, 1]
    k2 = 0.5 / sigma ** 2
    score = (np.exp(-k2 * dx ** 2) + np.exp(-k2 * dy ** 2)).sum(axis=1)
    k = int(np.argmax(score))
    if score[k] < min_pairs or score[k] < min_ratio * np.median(score):
        return None
    l, r = score[k - 1], score[(k + 1) % len(score)]
    den = l - 2 * score[k] + r
    off = 0.5 * (l - r) / den if den < 0 else 0.0
    return float(_CAND[k] + np.radians(off))
