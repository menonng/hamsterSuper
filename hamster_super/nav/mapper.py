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
        self.free_threshold = -1.0
        self.seen: set = set()          # cells whose floor the camera has looked at
        self.decay = 1.0                # < 1: forget old evidence (local map for racing)
        self.clear_gain = 0.3           # extra clearing per unit of wall evidence when seen through
        # cells with something off the lattice in them (a swinging wall, a sliding
        # block): blocked for the planner for a few seconds after the last sighting
        self.blocked: dict = {}
        self.block_steps = 30
        self.block_range = 15.0
        self.t = 0
        self._free_cache, self._free_version = None, -1
        self.version = 0
        self._arrays, self._arrays_version = None, -1

    @property
    def free_edges(self) -> set:
        """Edges the camera has seen through (observed open)."""
        if self._free_cache is None or self._free_version != self.version:
            self._free_cache = {e for e, sc in self.scores.items() if sc <= self.free_threshold}
            self._free_version = self.version
        return self._free_cache

    @property
    def walls(self) -> set:
        if self._cache is None:
            w = self.fixed | {e for e, s in self.scores.items() if s >= self.threshold}
            for (i, j) in self.blocked:      # something off the lattice stands in this cell
                w |= {make_edge((i, j), (i + 1, j)), make_edge((i, j + 1), (i + 1, j + 1)),
                      make_edge((i, j), (i, j + 1)), make_edge((i + 1, j), (i + 1, j + 1))}
            self._cache = w
        return self._cache

    def strong_walls(self, min_score: float = 4.0) -> set:
        """Walls with plenty of evidence (fixed ones included)."""
        return self.fixed | {e for e, sc in self.scores.items() if sc >= min_score}

    def _state(self, s: float) -> int:
        return 1 if s >= self.threshold else (-1 if s <= self.free_threshold else 0)

    def _add(self, e: Edge, v: float):
        old = self.scores.get(e, 0.0)
        s = old + v
        s = -4.0 if s < -4.0 else (8.0 if s > 8.0 else s)
        self.scores[e] = s
        if self._state(s) != self._state(old):
            self._cache = None
            self.version += 1

    def edge_arrays(self):
        """Dense edge-state grids for fast lookup: (x0, y0, V, H) with
        V[i - x0, j - y0] = state of vertical edge (i, j)-(i, j+1) and
        H[i - x0, j - y0] = state of horizontal edge (i, j)-(i+1, j);
        state 1 = wall, 0 = unknown, -1 = observed free."""
        if self._arrays is not None and self._arrays_version == self.version:
            return self._arrays
        x0, y0, x1, y1 = self.bounds
        V = np.zeros((x1 - x0 + 3, y1 - y0 + 3), np.int8)
        H = np.zeros_like(V)
        for e, sc in self.scores.items():
            st = self._state(sc)
            self._put(V, H, e, st, x0, y0)
        for e in self.fixed:
            self._put(V, H, e, 1, x0, y0)
        self._arrays, self._arrays_version = (x0, y0, V, H), self.version
        return self._arrays

    @staticmethod
    def _put(V, H, e, st, x0, y0):
        (a, b) = e
        arr = V if a[0] == b[0] else H
        i, j = a[0] - x0, a[1] - y0
        if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
            arr[i, j] = st

    def forget(self):
        """Shrink all evidence a little (decay < 1). Walls the camera keeps seeing
        stay; walls drawn from an old, drifted pose estimate fade away instead of
        misleading the robot for the rest of the run."""
        if self.decay >= 1.0:
            return
        for e, sc in list(self.scores.items()):
            ns = sc * self.decay
            if abs(ns) < 0.05:
                del self.scores[e]
                ns = 0.0
            else:
                self.scores[e] = ns
            if self._state(ns) != self._state(sc):
                self._cache = None
                self.version += 1

    def update(self, origin, heading: float, angles, ranges, max_range: float, r_min):
        self.forget()
        self.t += 1
        ox, oy = float(origin[0]), float(origin[1])
        own = (int(math.floor(ox / G)), int(math.floor(oy / G)))
        inner = {}
        for a, r, rmin in zip(angles, ranges, r_min):
            dx, dy = math.cos(heading + a), math.sin(heading + a)
            hit = (r < self.map_range) and (r > rmin + 0.3)
            free_until = min(r - 1.0, self.map_range)
            if r > rmin + 0.3:
                # nothing closer than the blind zone is observed (the camera may even look
                # over a wall there), so only the visible part of the ray clears edges
                self._free_ray(ox, oy, dx, dy, free_until, start=rmin)
                for t in np.arange(rmin, max(rmin, free_until), 2.0):
                    self.seen.add((int(math.floor((ox + dx * t) / G)), int(math.floor((oy + dy * t) / G))))
            if hit:
                d = r + WALL_THICKNESS / 2
                e = self.snap(ox + dx * d, oy + dy * d)
                if e is not None:
                    self._add(e, self.hit_gain)
                elif r < self.block_range:      # near only: far ranges bend with a tilt error
                    px, py = ox + dx * (r + 0.3), oy + dy * (r + 0.3)
                    fx, fy = px / G - math.floor(px / G), py / G - math.floor(py / G)
                    if 0.2 < fx < 0.8 and 0.2 < fy < 0.8:       # well inside a cell, not on a wall line
                        c = (int(math.floor(px / G)), int(math.floor(py / G)))
                        inner[c] = inner.get(c, 0) + 1
        changed = False
        for c, k in inner.items():
            x0, y0, x1, y1 = self.bounds
            if k >= 2 and c != own and x0 <= c[0] <= x1 and y0 <= c[1] <= y1:   # two rays agree: not noise
                changed |= c not in self.blocked
                self.blocked[c] = self.t + self.block_steps
        for c in [c for c, until in self.blocked.items() if until < self.t or c == own]:
            del self.blocked[c]
            changed = True
        if changed:
            self._cache = None
            self.version += 1

    def update_ir(self, pose, ir, spec):
        """IR proximity hits: covers the camera's blind zone right in front."""
        x, y, th = pose
        if len(ir) == 2 and min(ir) > 0.5:
            # both front sensors see a wall very close: it is the wall straight ahead,
            # even when the hit point is too near a dot to snap (and the camera is blind there)
            k = round(th / (math.pi / 2))
            if abs(th - k * math.pi / 2) < 0.45:
                d = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
                c = (int(math.floor(x / G)), int(math.floor(y / G)))
                fx = (c[0] + 0.5 + 0.5 * d[0]) * G - x
                fy = (c[1] + 0.5 + 0.5 * d[1]) * G - y
                if abs(fx * d[0] + fy * d[1]) < spec.ir_offset + spec.ir_range * 0.6:
                    self._add(make_edge(*self._edge_pts(c, d)), self.hit_gain)
        ox, oy = x + spec.ir_offset * math.cos(th), y + spec.ir_offset * math.sin(th)
        for val, sign in zip(ir, (1, -1)):
            if val < 0.15:
                continue
            a = th + sign * spec.ir_angle
            d = (1.0 - val) * spec.ir_range + WALL_THICKNESS / 2
            e = self.snap(ox + d * math.cos(a), oy + d * math.sin(a))
            if e is not None:
                self._add(e, self.hit_gain)

    @staticmethod
    def _edge_pts(c, d):
        """The two lattice dots of cell c's side in direction d."""
        i, j = c
        if d == (1, 0):
            return (i + 1, j), (i + 1, j + 1)
        if d == (-1, 0):
            return (i, j), (i, j + 1)
        if d == (0, 1):
            return (i, j + 1), (i + 1, j + 1)
        return (i, j), (i + 1, j)

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

    def _free_ray(self, ox, oy, dx, dy, length, start=0.0):
        if length <= start:
            return
        ex, ey = ox + dx * length, oy + dy * length
        for (o, d, e, other_o, other_d, vertical) in ((ox, dx, ex, oy, dy, True), (oy, dy, ey, ox, dx, False)):
            if abs(d) < 1e-9:
                continue
            lo, hi = sorted((o, e))
            for k in range(math.ceil(lo / G), math.floor(hi / G) + 1):
                t = (k * G - o) / d
                if t <= start or t >= length:
                    continue
                q = other_o + other_d * t
                frac = q / G - math.floor(q / G)
                if frac < 0.12 or frac > 0.88:   # too close to a dot
                    continue
                m = int(math.floor(q / G))
                edge = ((k, m), (k, m + 1)) if vertical else ((m, k), (m + 1, k))
                # seeing through a mapped wall removes it quickly (walls can move):
                # the stronger the old evidence, the bigger the step
                self._add(edge, -(self.miss_gain + self.clear_gain * max(self.scores.get(edge, 0.0), 0.0)))

    def add_border(self):
        """Races use the whole arena: its outer boundary is a known wall. Fixed
        walls anchor the localizer globally (the inner lattice repeats every 9 cm)."""
        x0, y0, x1, y1 = self.bounds
        for i in range(x0, x1 + 1):
            self.fixed.add(make_edge((i, y0), (i + 1, y0)))
            self.fixed.add(make_edge((i, y1 + 1), (i + 1, y1 + 1)))
        for j in range(y0, y1 + 1):
            self.fixed.add(make_edge((x0, j), (x0, j + 1)))
            self.fixed.add(make_edge((x1 + 1, j), (x1 + 1, j + 1)))
        self._cache = None
        self.version += 1

    def as_gridmap(self, start, goal) -> GridMap:
        return GridMap(set(self.walls), start, goal, None, self.bounds)
