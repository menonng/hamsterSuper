"""Cell-level global planner (BFS) + continuous progress potential."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..config import GRID_SPACING
from ..maps.grid_map import NEIGHBORS, cell_center, cell_of, edge_between

G = GRID_SPACING


class Planner:
    def __init__(self, bounds, goal):
        self.bounds = tuple(bounds)
        self.goal = tuple(goal)
        self.walls: set = set()
        self.dist: dict = {}
        self._bfs_walls: set | None = None
        self._bfs_dist: dict = {}

    def in_bounds(self, c):
        x0, y0, x1, y1 = self.bounds
        return x0 <= c[0] <= x1 and y0 <= c[1] <= y1

    def neighbors(self, c, walls):
        for dx, dy in NEIGHBORS:
            n = (c[0] + dx, c[1] + dy)
            if self.in_bounds(n) and edge_between(c, n) not in walls:
                yield n

    def update(self, walls: set, current_cell=None):
        """Recompute the distance field. If the robot's cell becomes unreachable
        (e.g. a false wall closed the only path) fall back to an empty map."""
        if walls != self._bfs_walls:
            self._bfs_walls = set(walls)
            self._bfs_dist = self._bfs(walls)
        self.walls, self.dist = walls, self._bfs_dist
        if current_cell is not None and self.in_bounds(current_cell) and current_cell not in self.dist:
            self.walls = set()
            self.dist = self._bfs(self.walls)
        return self.dist

    def _bfs(self, walls):
        dist = {self.goal: 0}
        q = deque([self.goal])
        while q:
            c = q.popleft()
            for n in self.neighbors(c, walls):
                if n not in dist:
                    dist[n] = dist[c] + 1
                    q.append(n)
        return dist

    # ---------------------------------------------------------------- queries
    def next_cells(self, c, k=2):
        out = []
        prev_dir = None
        for _ in range(k):
            if c not in self.dist or self.dist[c] == 0:
                break
            best = None
            for n in self.neighbors(c, self.walls):
                if self.dist.get(n, 1 << 30) == self.dist[c] - 1:
                    d = (n[0] - c[0], n[1] - c[1])
                    score = 0 if d == prev_dir else 1
                    if best is None or score < best[0]:
                        best = (score, n, d)
            if best is None:
                break
            _, c, prev_dir = best
            out.append(c)
        return out

    def waypoints(self, pos):
        """Two upcoming waypoints (world cm): next cell centers on the path."""
        c = cell_of(pos)
        goal_c = cell_center(self.goal)
        if c not in self.dist:
            return goal_c, goal_c
        cells = self.next_cells(c, 2)
        pts = [cell_center(n) for n in cells]
        while len(pts) < 2:
            pts.append(goal_c)
        return pts[0], pts[1]

    def potential(self, pos) -> float:
        """Approximate geodesic distance (cm) from pos to the goal center."""
        c = cell_of(pos)
        p = np.asarray(pos[:2], dtype=float)
        if c == self.goal:
            return float(np.linalg.norm(p - cell_center(c)))
        if c not in self.dist:
            return float(np.linalg.norm(p - cell_center(self.goal))) + 9 * G
        best = math.inf
        center = cell_center(c)
        for n in self.neighbors(c, self.walls):
            dn = self.dist.get(n)
            if dn is None or dn >= self.dist[c]:
                continue
            mid = 0.5 * (center + cell_center(n))
            best = min(best, float(np.linalg.norm(p - mid)) + 0.5 * G + G * dn)
        return best if best < math.inf else float(np.linalg.norm(p - cell_center(self.goal))) + 9 * G
