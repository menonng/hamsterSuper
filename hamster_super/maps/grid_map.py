"""Grid map: floor dots every 9 cm, walls connect two adjacent dots.

Coordinates
-----------
* Lattice point (i, j) sits at world position (9*i, 9*j) cm. +x is east, +y is north.
* Cell (i, j) is the 9x9 square [9i, 9i+9] x [9j, 9j+9].
* A wall is an unordered pair of 4-adjacent lattice points, stored as a sorted tuple.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import GRID_SPACING, WALL_THICKNESS

Point = tuple[int, int]
Edge = tuple[Point, Point]
Cell = tuple[int, int]

DIRS = {"E": 0.0, "N": math.pi / 2, "W": math.pi, "S": -math.pi / 2}
NEIGHBORS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def make_edge(a: Point, b: Point) -> Edge:
    a, b = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
        raise ValueError(f"wall must connect two adjacent dots, got {a}-{b}")
    return (a, b) if a <= b else (b, a)


def edge_between(c: Cell, n: Cell) -> Edge:
    """The wall edge separating two 4-adjacent cells."""
    (i, j), (k, l) = c, n
    if k == i + 1:
        return ((k, j), (k, j + 1))
    if k == i - 1:
        return ((i, j), (i, j + 1))
    if l == j + 1:
        return ((i, l), (i + 1, l))
    if l == j - 1:
        return ((i, j), (i + 1, j))
    raise ValueError(f"cells {c} and {n} are not adjacent")


def cell_center(c: Cell) -> np.ndarray:
    return (np.asarray(c, dtype=np.float64) + 0.5) * GRID_SPACING


def cell_of(pos) -> Cell:
    return (int(math.floor(pos[0] / GRID_SPACING)), int(math.floor(pos[1] / GRID_SPACING)))


@dataclass
class GridMap:
    walls: set = field(default_factory=set)
    start: Cell = (0, 0)
    goal: Cell = (0, 0)
    start_heading: float | None = None   # radians, None = choose a free direction
    # Cell range (inclusive) used for planning; None = derived from walls/start/goal.
    bounds: tuple[int, int, int, int] | None = None
    name: str = ""

    # ------------------------------------------------------------------ edits
    def add_wall(self, a: Point, b: Point):
        self.walls.add(make_edge(a, b))

    def remove_wall(self, a: Point, b: Point):
        self.walls.discard(make_edge(a, b))

    def has_wall(self, a: Point, b: Point) -> bool:
        return make_edge(a, b) in self.walls

    def blocked(self, c: Cell, n: Cell) -> bool:
        return edge_between(c, n) in self.walls

    # ------------------------------------------------------------- geometry
    def cell_bounds(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) inclusive cell range for planning."""
        if self.bounds is not None:
            return self.bounds
        pts = [p for e in self.walls for p in e]
        xs = [p[0] for p in pts] + [self.start[0], self.start[0] + 1, self.goal[0], self.goal[0] + 1]
        ys = [p[1] for p in pts] + [self.start[1], self.start[1] + 1, self.goal[1], self.goal[1] + 1]
        # dots span [min, max]; cells inside are [min, max-1]
        return (min(xs), min(ys), max(xs) - 1, max(ys) - 1)

    def in_bounds(self, c: Cell) -> bool:
        x0, y0, x1, y1 = self.cell_bounds()
        return x0 <= c[0] <= x1 and y0 <= c[1] <= y1

    def wall_boxes(self) -> np.ndarray:
        """Axis-aligned boxes (xmin, ymin, xmax, ymax) in cm, one per wall."""
        if not self.walls:
            return np.zeros((0, 4))
        h = WALL_THICKNESS / 2
        out = []
        for (a, b) in sorted(self.walls):
            x1, y1 = a[0] * GRID_SPACING, a[1] * GRID_SPACING
            x2, y2 = b[0] * GRID_SPACING, b[1] * GRID_SPACING
            if y1 == y2:   # horizontal
                out.append((min(x1, x2), y1 - h, max(x1, x2), y1 + h))
            else:          # vertical
                out.append((x1 - h, min(y1, y2), x1 + h, max(y1, y2)))
        return np.asarray(out, dtype=np.float64)

    # ------------------------------------------------------------- planning
    def open_neighbors(self, c: Cell):
        for dx, dy in NEIGHBORS:
            n = (c[0] + dx, c[1] + dy)
            if self.in_bounds(n) and not self.blocked(c, n):
                yield n

    def distance_field(self, target: Cell | None = None) -> dict:
        """BFS cell distance to the target (default: goal)."""
        target = self.goal if target is None else target
        dist = {target: 0}
        q = deque([target])
        while q:
            c = q.popleft()
            for n in self.open_neighbors(c):
                if n not in dist:
                    dist[n] = dist[c] + 1
                    q.append(n)
        return dist

    def shortest_path(self, src: Cell | None = None, dst: Cell | None = None):
        src = self.start if src is None else src
        dist = self.distance_field(dst)
        if src not in dist:
            return None
        path = [src]
        c = src
        while dist[c] > 0:
            c = min((n for n in self.open_neighbors(c) if dist.get(n, 1 << 30) == dist[c] - 1))
            path.append(c)
        return path

    def free_headings(self, c: Cell) -> list[float]:
        out = []
        for name, (dx, dy) in zip("ENWS", NEIGHBORS):
            n = (c[0] + dx, c[1] + dy)
            if self.in_bounds(n) and not self.blocked(c, n):
                out.append(DIRS[name])
        return out

    def copy(self) -> "GridMap":
        return GridMap(set(self.walls), self.start, self.goal, self.start_heading, self.bounds, self.name)
