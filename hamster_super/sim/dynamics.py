"""Course parts that move during a run.

* RotatingWall: a 9 cm wall on a pivot dot in the middle of a 2-cell-wide
  section. Every few seconds it swings 90 degrees (taking about a second)
  between lying along the lane (the section splits into two 1-cell lanes)
  and lying across one half (the section narrows to one cell).
* SlidingBlock: a block that moves sideways across a 2-cell-wide section,
  from one half-lane to the other, so the passable half changes over time.

Both rest on lattice positions and are off-lattice only while moving. They
never push the robot: a move that would touch the robot waits. Neither one
ever closes the course (one half always stays open), so the course and its
finish stay the same.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import GRID_SPACING, WALL_THICKNESS
from .geometry import rect_penetration

G = GRID_SPACING
T = WALL_THICKNESS


def _segment_boxes(p, ang: float, length: float = G) -> np.ndarray:
    """AABBs covering a thin wall from p at angle ang (exact when axis-aligned)."""
    c, s = math.cos(ang), math.sin(ang)
    if abs(s) < 1e-6 or abs(c) < 1e-6:
        q = (p[0] + c * length, p[1] + s * length)
        x0, x1 = sorted((p[0], q[0]))
        y0, y1 = sorted((p[1], q[1]))
        return np.array([[x0 - T / 2, y0 - T / 2, x1 + T / 2, y1 + T / 2]])
    n = 10
    h = 0.5 * (length / n) * max(abs(c), abs(s)) + T / 2
    out = []
    for k in range(n):
        t = (k + 0.5) / n * length
        x, y = p[0] + c * t, p[1] + s * t
        out.append([x - h, y - h, x + h, y + h])
    return np.array(out)


class _Element:
    def __init__(self, rng):
        self.rng = rng

    def _blocked(self, boxes, robot_pose, half) -> bool:
        if robot_pose is None:
            return False
        return rect_penetration(np.asarray(robot_pose[:2]), float(robot_pose[2]), half[0] + 0.1, half[1] + 0.1,
                                boxes)[0] > 0


class RotatingWall(_Element):
    def __init__(self, pivot, ang_a: float, ang_b: float, rng, rest=(3.0, 10.0), swing_time: float = 1.0):
        super().__init__(rng)
        self.pivot = np.asarray(pivot, float)
        self.ang_a, self.ang_b = ang_a, ang_b
        self.rest, self.swing_time = rest, swing_time
        self.ang = ang_a if rng.random() < 0.5 else ang_b
        self.target = self.ang
        self.wait = float(rng.uniform(*rest))

    def boxes(self) -> np.ndarray:
        return _segment_boxes(self.pivot, self.ang)

    def step(self, dt: float, robot_pose=None, half=(2.0, 1.75)):
        if self.ang == self.target:
            self.wait -= dt
            if self.wait <= 0:
                self.target = self.ang_b if self.ang == self.ang_a else self.ang_a
            return
        rate = (math.pi / 2) / self.swing_time
        d = self.target - self.ang
        new = self.target if abs(d) <= rate * dt else self.ang + math.copysign(rate * dt, d)
        if self._blocked(_segment_boxes(self.pivot, new), robot_pose, half):
            return                                   # the robot is in the way: wait
        self.ang = new
        if self.ang == self.target:
            self.wait = float(self.rng.uniform(*self.rest))


class SlidingBlock(_Element):
    def __init__(self, a, b, rng, size: float = 6.0, speed: float = 3.0, rest=(3.0, 8.0)):
        super().__init__(rng)
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.size, self.speed, self.rest = size, speed, rest
        self.pos = (self.a if rng.random() < 0.5 else self.b).copy()
        self.target = self.pos.copy()
        self.wait = float(rng.uniform(*rest))

    def boxes(self) -> np.ndarray:
        h = self.size / 2
        return np.array([[self.pos[0] - h, self.pos[1] - h, self.pos[0] + h, self.pos[1] + h]])

    def step(self, dt: float, robot_pose=None, half=(2.0, 1.75)):
        if np.allclose(self.pos, self.target):
            self.wait -= dt
            if self.wait <= 0:
                self.target = (self.b if np.allclose(self.pos, self.a) else self.a).copy()
            return
        d = self.target - self.pos
        dist = float(np.linalg.norm(d))
        new = self.target.copy() if dist <= self.speed * dt else self.pos + d / dist * self.speed * dt
        h = self.size / 2
        if self._blocked(np.array([[new[0] - h, new[1] - h, new[0] + h, new[1] + h]]), robot_pose, half):
            return
        self.pos = new
        if np.allclose(self.pos, self.target):
            self.wait = float(self.rng.uniform(*self.rest))


class CourseDynamics:
    """All moving parts of one run."""

    def __init__(self, elements=()):
        self.elements = list(elements)

    def __bool__(self):
        return bool(self.elements)

    def step(self, dt: float, robot_pose=None, half=(2.0, 1.75)):
        for e in self.elements:
            e.step(dt, robot_pose, half)

    def boxes(self) -> np.ndarray:
        if not self.elements:
            return np.zeros((0, 4))
        return np.concatenate([e.boxes() for e in self.elements])


def make_dynamics(m, rng, n_max: int = 3) -> CourseDynamics:
    """Place rotating walls / sliding blocks in the map's wide blocks (2x2 blocks
    where the lane is 2 cells wide, with their direction of travel)."""
    wide = list(getattr(m, "wide_blocks", None) or [])
    keep_out = {tuple(m.start), tuple(m.goal)}
    wide = [(b, d) for b, d in wide if d is not None
            and not any((2 * b[0] + i, 2 * b[1] + j) in keep_out for i in (0, 1) for j in (0, 1))]
    if not wide:
        return CourseDynamics()
    rng.shuffle(wide)
    els = []
    for (bx, by), d in wide[:int(rng.integers(1, n_max + 1))]:
        d = (int(d[0]), int(d[1]))
        side = (-d[1], d[0]) if rng.random() < 0.5 else (d[1], -d[0])
        if rng.random() < 0.6:
            # pivot on the block centre dot; along the lane <-> across one half-lane
            pivot = ((2 * bx + 1) * G, (2 * by + 1) * G)
            els.append(RotatingWall(pivot, math.atan2(d[1], d[0]), math.atan2(side[1], side[0]), rng))
        else:
            # block in the first cell row of the block (seen in the direction of travel)
            cells = [(2 * bx + i, 2 * by + j) for i in (0, 1) for j in (0, 1)]
            back = min(cells, key=lambda c: c[0] * d[0] + c[1] * d[1])
            row = [c for c in cells if c[0] * d[0] + c[1] * d[1] == back[0] * d[0] + back[1] * d[1]]
            a, b = [((c[0] + 0.5) * G, (c[1] + 0.5) * G) for c in row]
            els.append(SlidingBlock(a, b, rng))
    return CourseDynamics(els)
