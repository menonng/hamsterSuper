"""Online wall map built from depth rays.

Walls can only lie on the 9 cm dot lattice, so each ray endpoint is snapped
to the nearest lattice edge (+1 evidence) and every lattice edge the ray
passes through gets negative evidence. This makes the map robust to noisy
rays and to false detections.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import GRID_SPACING, WALL_THICKNESS
from ..maps.grid_map import Edge, GridMap, make_edge

G = GRID_SPACING


class BeliefMap:
    def __init__(self, bounds, hit_gain=1.0, miss_gain=0.4, threshold=1.5,
                 snap_tol=1.5, ambiguity=1.5, map_range=25.0, prior: GridMap | None = None):
        self.bounds = tuple(bounds)
        self.scores: dict[Edge, float] = {}
        self.hit_gain, self.miss_gain, self.threshold, self.snap_tol = hit_gain, miss_gain, threshold, snap_tol
        self.ambiguity = ambiguity
        self.map_range = map_range   # far hits are too sensitive to heading error
        self.fixed: set = set()
        if prior is not None:   # known map: walls are certain
            self.fixed = set(prior.walls)
        self._cache = None

    @property
    def walls(self) -> set:
        if self._cache is None:
            self._cache = self.fixed | {e for e, s in self.scores.items() if s >= self.threshold}
        return self._cache

    def _add(self, e: Edge, v: float):
        s = self.scores.get(e, 0.0) + v
        self.scores[e] = -4.0 if s < -4.0 else (8.0 if s > 8.0 else s)
        self._cache = None

    def update(self, origin, heading: float, angles, ranges, max_range: float, r_min):
        ox, oy = float(origin[0]), float(origin[1])
        for a, r, rmin in zip(angles, ranges, r_min):
            dx, dy = math.cos(heading + a), math.sin(heading + a)
            hit = (r < self.map_range) and (r > rmin + 0.3)
            free_until = min(r - 1.0, self.map_range)
            if r > rmin + 0.3:
                self._free_ray(ox, oy, dx, dy, free_until)
            if hit:
                d = r + WALL_THICKNESS / 2
                e = self.snap(ox + dx * d, oy + dy * d)
                if e is not None:
                    self._add(e, self.hit_gain)

    def update_ir(self, pose, ir, spec):
        """IR proximity hits: covers the camera's blind zone right in front."""
        x, y, th = pose
        ox, oy = x + spec.ir_offset * math.cos(th), y + spec.ir_offset * math.sin(th)
        for val, sign in zip(ir, (1, -1)):
            if val < 0.15:
                continue
            a = th + sign * spec.ir_angle
            d = (1.0 - val) * spec.ir_range + WALL_THICKNESS / 2
            e = self.snap(ox + d * math.cos(a), oy + d * math.sin(a))
            if e is not None:
                self._add(e, self.hit_gain)

    def snap(self, px: float, py: float) -> Edge | None:
        gx, gy = px / G, py / G
        dvx = abs(gx - round(gx)) * G     # distance to nearest vertical line
        dhy = abs(gy - round(gy)) * G     # distance to nearest horizontal line
        if min(dvx, dhy) > self.snap_tol:
            return None
        if abs(dvx - dhy) < self.ambiguity:   # near a dot: which edge is it? skip
            return None
        if dvx < dhy:
            i, j = int(round(gx)), int(math.floor(gy))
            return ((i, j), (i, j + 1))
        i, j = int(math.floor(gx)), int(round(gy))
        return ((i, j), (i + 1, j))

    def _free_ray(self, ox, oy, dx, dy, length):
        if length <= 0:
            return
        ex, ey = ox + dx * length, oy + dy * length
        for (o, d, e, other_o, other_d, vertical) in ((ox, dx, ex, oy, dy, True), (oy, dy, ey, ox, dx, False)):
            if abs(d) < 1e-9:
                continue
            lo, hi = sorted((o, e))
            for k in range(math.ceil(lo / G), math.floor(hi / G) + 1):
                t = (k * G - o) / d
                if t <= 0 or t >= length:
                    continue
                q = other_o + other_d * t
                frac = q / G - math.floor(q / G)
                if frac < 0.12 or frac > 0.88:   # too close to a dot
                    continue
                m = int(math.floor(q / G))
                edge = ((k, m), (k, m + 1)) if vertical else ((m, k), (m + 1, k))
                self._add(edge, -self.miss_gain)

    def as_gridmap(self, start, goal) -> GridMap:
        return GridMap(set(self.walls), start, goal, None, self.bounds)
