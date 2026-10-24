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

Inside/outside is a choice everywhere, not only in corners: the policy's
line action picks the lateral position across every portal - straights
included: +1 = the left side of the lane, -1 = the right side, 0 = the plain
shortest line. The policy sees the walls ahead, so it learns which side is
the inside of the coming turn (in a race the map ahead is unknown, so the
line itself often cannot tell). In wide (2-cell) sections the portals span
the whole open width, so the side of the section is chosen too. The IN /
MID / OUT labels say where the line runs relative to the next known turn.
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


def _open(a, b, walls, bounds) -> bool:
    from ..maps.grid_map import edge_between
    x0, y0, x1, y1 = bounds
    if not (x0 <= b[0] <= x1 and y0 <= b[1] <= y1):
        return False
    return edge_between(a, b) not in walls


def build_portals(cells, clearance: float, walls=None, bounds=None, default_turn=None):
    """Portals along a cell path: dicts with left/right end points, kind
    ('edge' | 'corner') and the cell. With walls/bounds, an edge portal is
    widened over a neighbouring lane cell when the section is open there
    (wide sections), so the line may use the full width."""
    portals = []
    for i in range(len(cells) - 1):
        c, n = tuple(cells[i]), tuple(cells[i + 1])
        d = np.array([n[0] - c[0], n[1] - c[1]], float)
        mid = (cell_center(c) + cell_center(n)) / 2
        perp = np.array([-d[1], d[0]])                  # left of travel
        hi, lo = HALF, -HALF
        if walls is not None and bounds is not None:
            for sgn in (1, -1):
                o = (int(perp[0] * sgn), int(perp[1] * sgn))
                cs, ns = (c[0] + o[0], c[1] + o[1]), (n[0] + o[0], n[1] + o[1])
                if _open(c, cs, walls, bounds) and _open(n, ns, walls, bounds) and _open(cs, ns, walls, bounds):
                    if sgn > 0:
                        hi += G
                    else:
                        lo -= G
        a, b = max(0.0, hi - clearance), min(0.0, lo + clearance)
        portals.append({"left": mid + perp * a, "right": mid + perp * b, "kind": "edge", "cell": n,
                        "wall_left": mid + perp * hi, "wall_right": mid + perp * lo})
        # a turn happens in cell n when the next step changes direction
        if i + 2 < len(cells):
            d2 = np.array([cells[i + 2][0] - n[0], cells[i + 2][1] - n[1]], float)
            if not np.allclose(d, d2):
                centre = cell_center(n)
                turn_left = d[0] * d2[1] - d[1] * d2[0] > 0
                inside = centre + (d2 - d) * HALF            # the lattice point on the inside of the turn
                outside = centre - (d2 - d) * HALF
                u = (outside - inside) / np.linalg.norm(outside - inside)
                diag = np.linalg.norm(outside - inside)
                pa = inside + u * clearance
                # the outside corner is a concave wall corner: keep the clearance from
                # both of its walls, i.e. sqrt(2) x clearance along the diagonal
                pb = inside + u * max(clearance, diag - math.sqrt(2) * clearance)
                left, right = _ordered(pa, pb, centre, (d + d2) / 2)
                portals.append({"left": left, "right": right, "kind": "corner", "cell": n,
                                "turn_left": bool(turn_left), "inside": inside, "outside": outside})
    # every portal learns which way the next turn goes (straights included)
    nxt = None
    for p in reversed(portals):
        if p["kind"] == "corner":
            nxt = p["turn_left"]
        p["next_left"] = nxt
    prev = None
    for p in portals:          # past the last turn in view: keep the side of the previous one
        if p["kind"] == "corner":
            prev = p["turn_left"]
        if p["next_left"] is None:
            p["next_left"] = prev if prev is not None else default_turn
    return portals


def apply_bias(portals, bias: float, min_width: float = 0.4):
    """Narrow every portal toward its left end (bias > 0) or right end
    (bias < 0), as seen in the direction of travel. +-1 pins the line to that
    side of the lane / section."""
    bias = float(np.clip(bias, -1, 1))
    if abs(bias) < 1e-6:
        return portals
    out = []
    for p in portals:
        near, far = (p["left"], p["right"]) if bias > 0 else (p["right"], p["left"])
        w = float(np.linalg.norm(far - near))
        if w < 1e-6:
            out.append(p)
            continue
        u = (far - near) / w
        keep = near + u * max(min_width, w * (1 - abs(bias)))
        out.append({**p, "left": near, "right": keep} if bias > 0 else {**p, "left": keep, "right": near})
    return out


def funnel(start, goal, portals):
    """Simple stupid funnel algorithm (string pulling) through (left, right) portals."""
    pts = [(np.asarray(start, float), np.asarray(start, float))]
    pts += [(np.asarray(p["left"], float), np.asarray(p["right"], float)) for p in portals]
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


def _too_close(line: np.ndarray, walls, clearance: float) -> bool:
    """Does the polyline pass a wall closer than `clearance` (centre line, cm)?"""
    if line is None or len(line) < 2:
        return False
    lo, hi = line.min(axis=0) - G, line.max(axis=0) + G
    segs = []
    for (a, b) in walls:
        ax, ay, bx, by = a[0] * G, a[1] * G, b[0] * G, b[1] * G
        if max(ax, bx) < lo[0] or min(ax, bx) > hi[0] or max(ay, by) < lo[1] or min(ay, by) > hi[1]:
            continue
        segs.append((ax, ay, bx, by))
    if not segs:
        return False
    W = np.array(segs)                                     # wall centre segments
    P, Q = line[:-1], line[1:]
    # sample the line densely (0.5 cm) and measure the distance to every wall segment
    pts = [P[i] + (Q[i] - P[i]) * t for i in range(len(P))
           for t in np.linspace(0, 1, max(2, int(np.linalg.norm(Q[i] - P[i]) / 0.5) + 1))]
    X = np.array(pts)
    A, B = W[:, :2], W[:, 2:]
    AB = B - A
    t = np.clip(np.einsum("pk,wk->pw", X, AB) - np.einsum("wk,wk->w", A, AB), 0, None) / \
        np.maximum(np.einsum("wk,wk->w", AB, AB), 1e-9)
    t = np.clip(t, 0, 1)
    C = A[None] + t[..., None] * AB[None]
    d = np.linalg.norm(X[:, None] - C, axis=-1)
    return float(d.min()) < clearance


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
        self.bias = 0.0
        self.last_turn = None

    def update(self, planner, pos, sigma: float = 0.0, bias: float = 0.0):
        """pos: estimated (x, y); sigma: position uncertainty (cm); bias in [-1, 1]:
        the policy's lateral choice (+ left, - right of the direction of travel),
        applied across every portal: corners, straights and wide sections."""
        from ..maps.grid_map import cell_of
        c = cell_of(pos)
        goal = cell_center(planner.goal)
        if c not in planner.dist:
            self.line, self.corners = np.array([pos[:2], goal]), []
            return self.line
        cells = [c] + planner.next_cells(c, self.horizon)
        self.clearance = float(np.clip(self.base + self.k * sigma, self.min_clearance, HALF - 0.05))
        self.bias = float(np.clip(bias, -1, 1))
        walls = getattr(planner, "walls", None)
        end = goal if cells[-1] == planner.goal else cell_center(cells[-1])
        start = np.asarray(pos[:2], float)
        portals = build_portals(cells, self.clearance, walls, planner.bounds, default_turn=self.last_turn)
        turns = [q["turn_left"] for q in portals if q["kind"] == "corner"]
        if turns:
            self.last_turn = turns[0]      # labels in a short (exploring) line refer to the last turn known
        self.line = funnel(start, end, apply_bias(portals, self.bias))
        if walls and _too_close(self.line, walls, self.clearance - 0.3):
            # cutting across a wide section brushed a wall end: lane-wide portals only
            portals = build_portals(cells, self.clearance, default_turn=self.last_turn)
            self.line = funnel(start, end, apply_bias(portals, self.bias))
        self.corners = self._classify(portals)
        return self.line

    def _classify(self, portals):
        """Where the line runs across each portal, relative to the next turn:
        'in' / 'mid' / 'out', for corners and for straight cells alike."""
        out, done = [], set()
        for p in portals:
            if p["kind"] != "corner":
                continue
            t = self._crossing(p["inside"], p["outside"])
            label = "in" if t < 0.4 else ("out" if t > 0.6 else "mid")
            cell = (int(p["cell"][0]), int(p["cell"][1]))
            done.add(cell)
            out.append({"cell": list(cell), "label": label, "t": round(float(t), 2), "kind": "corner",
                        "turn": "left" if p["turn_left"] else "right"})
        for p in portals:
            cell = (int(p["cell"][0]), int(p["cell"][1]))
            if p["kind"] != "edge" or cell in done or p["next_left"] is None:
                continue
            done.add(cell)
            ins, outs = (p["wall_left"], p["wall_right"]) if p["next_left"] else (p["wall_right"], p["wall_left"])
            t = self._crossing(ins, outs)
            label = "in" if t < 0.4 else ("out" if t > 0.6 else "mid")
            out.append({"cell": list(cell), "label": label, "t": round(float(t), 2), "kind": "straight",
                        "turn": "left" if p["next_left"] else "right"})
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
