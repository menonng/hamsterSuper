"""Racing line through the planned cells, with an inside/outside call per corner.

Why the inside line: a differential-drive robot is limited by its outer
wheel, so driving a stretch takes about (path length + track/2 * total
turning) / top speed. The total turning is the same for every line through
the same corners, which makes the *shortest* line (hugging the inside of
each corner) the fastest; unlike a car, slowing down for a tight radius
costs nothing extra. What it does cost is margin: the inside line passes
close to wall corners, so the clearance is widened when the pose estimate is
uncertain, which pushes the line toward the centre or the outside.

The line is the shortest path through "portals": the openings between
consecutive cells, shrunk by the clearance at both ends, plus a diagonal
portal across every turning cell so the line cannot clip the inside corner.
It is found with the funnel (string-pulling) algorithm.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import GRID_SPACING, WALL_THICKNESS
from ..maps.grid_map import cell_center

G = GRID_SPACING
HALF = G / 2


def _cross(a, b, c) -> float:
    """z of (b - a) x (c - a): > 0 when c is left of a->b."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _ordered(p, q, mid, direction):
    """Return (left, right) of a portal as seen when travelling along `direction`."""
    side = direction[0] * (p[1] - mid[1]) - direction[1] * (p[0] - mid[0])
    return (p, q) if side > 0 else (q, p)


def build_portals(cells, clearance: float):
    """Portals (left, right, kind, cell) along a cell path. kind: 'edge' | 'corner'."""
    portals = []
    lim = max(0.0, HALF - clearance)          # half width left for the line
    for i in range(len(cells) - 1):
        c, n = np.array(cells[i]), np.array(cells[i + 1])
        d = (n - c).astype(float)
        mid = (cell_center(cells[i]) + cell_center(cells[i + 1])) / 2
        perp = np.array([-d[1], d[0]])
        portals.append((mid + perp * lim, mid - perp * lim, "edge", cells[i + 1]))
        # a turn happens in cell n when the next step changes direction
        if i + 2 < len(cells):
            d2 = (np.array(cells[i + 2]) - n).astype(float)
            if not np.allclose(d, d2):
                centre = cell_center(cells[i + 1])
                turn_left = d[0] * d2[1] - d[1] * d2[0] > 0
                # inside corner of the turn = cell centre + half a cell back along d2... toward the inside
                inside = centre + (d2 - d) * HALF            # the lattice point on the inside of the turn
                outside = centre - (d2 - d) * HALF
                u = (outside - inside) / np.linalg.norm(outside - inside)
                diag = np.linalg.norm(outside - inside)
                a = inside + u * clearance
                b = inside + u * max(clearance, diag - clearance)
                travel = (d + d2) / 2
                left, right = _ordered(a, b, centre, travel)
                portals.append((left, right, "corner", cells[i + 1], bool(turn_left), inside, outside))
    return portals


def funnel(start, goal, portals):
    """Simple stupid funnel algorithm (string pulling) through (left, right) portals."""
    pts = [(np.asarray(start, float), np.asarray(start, float))]
    pts += [(np.asarray(p[0], float), np.asarray(p[1], float)) for p in portals]
    pts.append((np.asarray(goal, float), np.asarray(goal, float)))
    path = [pts[0][0]]
    apex, left, right = pts[0][0], pts[0][0], pts[0][1]
    apex_i = left_i = right_i = 0
    i = 1
    n = len(pts)
    guard = 0
    while i < n and guard < 10 * n:
        guard += 1
        l, r = pts[i]
        # tighten the right side
        if _cross(apex, right, r) >= 0:                        # r is left of (or on) the right edge
            if np.allclose(apex, right) or _cross(apex, left, r) < 0:
                right, right_i = r, i
            else:                                              # right crossed over left: left becomes a corner
                path.append(left)
                apex, apex_i = left, left_i
                left = right = apex
                left_i = right_i = apex_i
                i = apex_i + 1
                continue
        # tighten the left side
        if _cross(apex, left, l) <= 0:                         # l is right of (or on) the left edge
            if np.allclose(apex, left) or _cross(apex, right, l) > 0:
                left, left_i = l, i
            else:
                path.append(right)
                apex, apex_i = right, right_i
                left = right = apex
                left_i = right_i = apex_i
                i = apex_i + 1
                continue
        i += 1
    if not np.allclose(path[-1], pts[-1][0]):
        path.append(pts[-1][0])
    return np.array(path)


def point_at(line: np.ndarray, s: float) -> np.ndarray:
    """Point at arc length s along a polyline (clamped to its end)."""
    for a, b in zip(line[:-1], line[1:]):
        seg = float(np.linalg.norm(b - a))
        if s <= seg and seg > 1e-9:
            return a + (b - a) * (s / seg)
        s -= seg
    return line[-1].copy()


class RacingLine:
    """Racing line over the planner's next cells, recomputed every step."""

    def __init__(self, robot_radius: float = 2.3, margin: float = 0.3, uncertainty_gain: float = 1.5,
                 horizon: int = 10, lookahead=(4.0, 10.0)):
        self.base = WALL_THICKNESS / 2 + robot_radius + margin
        self.min_clearance = WALL_THICKNESS / 2 + robot_radius + 0.05     # never closer than touching
        self.k = uncertainty_gain
        self.horizon = horizon
        self.lookahead = lookahead
        self.line = None
        self.corners: list = []
        self.clearance = self.base

    def update(self, planner, pos, sigma: float = 0.0, bias: float = 0.0):
        """pos: estimated (x, y); sigma: position uncertainty (cm); bias in [-1, 1]:
        the policy's own inside (-) / outside (+) choice, +-1.2 cm of clearance."""
        from ..maps.grid_map import cell_of
        c = cell_of(pos)
        goal = cell_center(planner.goal)
        if c not in planner.dist:
            self.line, self.corners = np.array([pos[:2], goal]), []
            return self.line
        cells = [c] + planner.next_cells(c, self.horizon)
        self.clearance = float(np.clip(self.base + self.k * sigma + 1.2 * float(np.clip(bias, -1, 1)),
                                       self.min_clearance, HALF - 0.05))
        portals = build_portals(cells, self.clearance)
        end = goal if cells[-1] == planner.goal else cell_center(cells[-1])
        self.line = funnel(np.asarray(pos[:2], float), end, portals)
        self.corners = self._classify(portals)
        return self.line

    def _classify(self, portals):
        """For each turning cell: does the line take it 'in', 'mid' or 'out'?"""
        out = []
        for p in portals:
            if p[2] != "corner":
                continue
            _, _, _, cell, turn_left, inside, outside = p
            # where the line crosses the cell diagonal: 0 = inside wall corner, 1 = outside corner
            t = self._crossing(inside, outside)
            label = "in" if t < 0.4 else ("out" if t > 0.6 else "mid")
            out.append({"cell": [int(cell[0]), int(cell[1])], "label": label, "t": round(float(t), 2),
                        "turn": "left" if turn_left else "right"})
        return out

    def _crossing(self, a, b) -> float:
        line = self.line
        best = 0.5
        for p, q in zip(line[:-1], line[1:]):
            d1, d2 = _cross(p, q, a), _cross(p, q, b)
            if d1 * d2 <= 0 and abs(d1 - d2) > 1e-9:
                t = d1 / (d1 - d2)
                # make sure the crossing lies on segment p-q too
                x = a + (b - a) * t
                if -1e-6 <= np.dot(x - p, q - p) <= np.dot(q - p, q - p) + 1e-6:
                    return float(np.clip(t, 0, 1))
        return best

    def waypoints(self):
        return point_at(self.line, self.lookahead[0]), point_at(self.line, self.lookahead[1])
