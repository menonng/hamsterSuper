"""Race navigation without a destination.

mode "lap" (the race): a closed circuit. Leave the start cell in the
heading direction, follow the lane while mapping it, and the lap is done
when the robot drives back into the start cell from the other side.

mode "roundtrip": a single-path course with a dead end: drive to its dead
end, then back to the start.

mode "race" (default for real races): the robot does not know the course
type. It follows the course; entering the start cell again from behind
means a lap is done, reaching a dead end that is not the start means the
end of a one-way course. Cells the camera has already looked at count as
explored, so wide sections are crossed instead of swept.

There is no destination given. On the way out the planner's goal is the
nearest cell not visited yet (extended straight ahead into unknown space so
the racing line stays long and the robot keeps its speed). A visited cell,
other than the start, with known walls on three sides is the dead end: the
robot turns around and the goal becomes the start cell, reached on the map it
has built (the whole corridor is known by then, so the return is fast).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..maps.grid_map import NEIGHBORS, cell_center, edge_between

OUT, BACK, DONE = "out", "back", "done"


class RoundTrip:
    def __init__(self, start_cell, bounds, extend: int = 4, dead_end_dist: float = 3.0,
                 mode: str = "roundtrip", start_heading: float | None = None):
        self.mode = mode
        self.start = tuple(start_cell)
        # only recently visited cells count as visited (a drifted pose must not make
        # the robot think the lane ahead was already driven); 0 = remember forever
        self.recent_window = 0
        self.t = 0
        self.last_visit = {self.start: 0}
        self.cells_driven = 0                      # cell changes since the start
        self.min_lap_cells = 4                     # no finish (lap or dead end) before this many cells
        # races use every cell of the arena: a finish is only believed after the robot
        # has driven through most of it (a drifted pose cannot fake an early finish)
        self.min_coverage = 0
        self.prev_cell = self.start
        self.ahead = None
        if start_heading is not None:
            d = (round(math.cos(start_heading)), round(math.sin(start_heading)))
            self.ahead = (self.start[0] + d[0], self.start[1] + d[1])
        self.bounds = tuple(bounds)
        self.phase = OUT
        self.visited = {self.start}
        self.extend = extend
        self.dead_end_dist = dead_end_dist
        self.turn_cell = None
        self.front_confirm = 9.0                   # cm: end wall must be seen this close

    def in_bounds(self, c):
        x0, y0, x1, y1 = self.bounds
        return x0 <= c[0] <= x1 and y0 <= c[1] <= y1

    def open_neighbors(self, c, walls):
        for dx, dy in NEIGHBORS:
            n = (c[0] + dx, c[1] + dy)
            if self.in_bounds(n) and edge_between(c, n) not in walls:
                yield n

    def is_dead_end(self, c, walls) -> bool:
        if c == self.start:
            return False
        closed = sum(1 for dx, dy in NEIGHBORS
                     if not self.in_bounds((c[0] + dx, c[1] + dy))
                     or edge_between(c, (c[0] + dx, c[1] + dy)) in walls)
        return closed >= 3

    def goal(self, cell, pos, heading, walls, seen=None, free=None, front=None, strong=None):
        """Update the phase and return the cell the planner should head for.
        front: distance (cm) to the nearest wall straight ahead measured right now
        (camera / IR); a dead end only counts when the sensors confirm it."""
        cell = tuple(cell)
        self.t += 1
        self.front = front
        self.strong = strong        # walls seen many times (weak ones may be false detections)
        self.seen = seen if seen is not None else set()
        self.free = free if free is not None else set()
        if self.mode in ("lap", "race"):
            return self._lap_goal(cell, heading, walls, pos)
        if self.in_bounds(cell):
            self.visited.add(cell)
            self.last_visit[cell] = self.t
        if self.phase == OUT:
            near_centre = np.linalg.norm(np.asarray(pos[:2]) - cell_center(cell)) < self.dead_end_dist
            if near_centre and self.is_dead_end(cell, walls):
                self.phase, self.turn_cell = BACK, cell
            else:
                return self._explore_target(cell, heading, walls)
        if self.phase == BACK and cell == self.start and \
                np.linalg.norm(np.asarray(pos[:2]) - cell_center(cell)) < 3.5:
            self.phase = DONE
        return self.start

    def recent(self, c) -> bool:
        if c not in self.last_visit:
            return False
        return self.recent_window <= 0 or self.t - self.last_visit[c] <= self.recent_window

    def _lap_goal(self, cell, heading, walls, pos=None):
        came_from = self.prev_cell
        if cell != self.prev_cell:
            self.prev_cell = cell
            self.cells_driven += 1
        if self.phase == DONE:
            return self.start
        covered = self.covered(walls)
        if cell == self.start and self.cells_driven >= self.min_lap_cells and covered \
                and came_from not in (self.start, self.ahead):
            self.phase = DONE                      # entered the start cell from behind: lap complete
            return self.start
        if self.in_bounds(cell):
            self.visited.add(cell)
            self.last_visit[cell] = self.t
        if self.mode == "race" and cell != self.start and self.cells_driven >= self.min_lap_cells and covered \
                and pos is not None \
                and self.is_dead_end(cell, walls) \
                and (getattr(self, "front", None) is None or self.front < self.front_confirm) \
                and np.linalg.norm(np.asarray(pos[:2]) - cell_center(cell)) < self.dead_end_dist:
            self.phase, self.turn_cell = DONE, cell   # end of a one-way course: finished
            return cell
        # closing the lap: the start is the next cell straight ahead and nothing blocks it
        if self.cells_driven >= self.min_lap_cells and len(self.visited) >= 4 and cell != self.start:
            dx, dy = self.start[0] - cell[0], self.start[1] - cell[1]
            if abs(dx) + abs(dy) == 1 and edge_between(cell, self.start) not in walls \
                    and dx * math.cos(heading) + dy * math.sin(heading) > 0.5:
                return self.start
        # the course runs forward from the start: never explore through the start's back side
        behind = self._behind()
        explore_walls = walls | {edge_between(self.start, behind)} if behind is not None else walls
        target = self._explore_target(cell, heading, explore_walls)
        if self.phase == BACK and not covered and self.strong is not None:
            # the course is not done yet, so a weakly seen wall must be closing the
            # way on: look past the weak walls (driving closer re-checks them)
            self.phase = OUT
            strong = self.strong | ({edge_between(self.start, behind)} if behind is not None else set())
            target = self._explore_target(cell, heading, strong)
        if self.phase == BACK:                     # nothing unvisited left: head for the start
            self.phase = OUT
            return self.start
        return target

    def covered(self, walls) -> bool:
        """Has the robot been through (nearly) the whole arena? Cells driven into
        count, and cells of a wide section it drove past (an open 2x2 square)."""
        if len(self.visited) < 4:
            return False
        if self.min_coverage <= 0 or len(self.visited) >= self.min_coverage:
            return True
        need = self.min_coverage - len(self.visited)
        extra = 0
        for c in self.seen:
            if c not in self.visited and self.in_bounds(c) and self._passed_by(c, walls):
                extra += 1
                if extra >= need:
                    return True
        return False

    def _passed_by(self, c, walls) -> bool:
        """In a wide section the robot drives past cells without entering them.
        Such a cell counts as explored when the camera has seen it and it shares
        an open 2x2 square (all four inner edges observed free) with a visited
        cell. In a one-cell lane there is no open square, so the cells ahead
        still have to be driven into; a dead end always has to be entered."""
        if c not in getattr(self, "seen", ()) or self.is_dead_end(c, walls):
            return False
        free = getattr(self, "free", ())
        for ox in (c[0] - 1, c[0]):
            for oy in (c[1] - 1, c[1]):
                sq = [(ox, oy), (ox + 1, oy), (ox + 1, oy + 1), (ox, oy + 1)]
                inner = [edge_between(a, b) for a, b in zip(sq, sq[1:] + sq[:1])]
                if all(e in free for e in inner) and any(self.recent(x) for x in sq if x != c):
                    return True
        return False

    def _behind(self):
        if self.ahead is None:
            return None
        return (2 * self.start[0] - self.ahead[0], 2 * self.start[1] - self.ahead[1])

    def _explore_target(self, cell, heading, walls):
        """Nearest unvisited cell (ties: straight ahead first), then extended straight on."""
        fwd = (round(math.cos(heading)), round(math.sin(heading)))
        dist = {cell: 0}
        first = {cell: None}
        q = deque([cell])
        best = None
        while q:
            c = q.popleft()
            explored = self.recent(c) or self._passed_by(c, walls)
            if not explored:
                # prefer targets reached by first moving forward rather than turning back
                f = first[c]
                penalty = 0 if f == fwd else (2 if f is not None and f == (-fwd[0], -fwd[1]) else 1)
                score = dist[c] + penalty
                if best is None or score < best[0]:
                    best = (score, c)
                continue
            for n in self.open_neighbors(c, walls):
                if n not in dist:
                    dist[n] = dist[c] + 1
                    first[n] = (n[0] - c[0], n[1] - c[1]) if c == cell else first[c]
                    q.append(n)
        if best is None:          # everything reachable is visited: go home
            self.phase = BACK
            self.target = None
            return self.start
        # commit to the current target while it is still unexplored and about as
        # good: re-deciding every step from the heading made the robot swing
        # between two targets (and spin in place, e.g. right at the start)
        cur = getattr(self, "target", None)
        if cur is not None and cur != best[1] and cur in dist and not self.recent(cur) \
                and not self._passed_by(cur, walls) and dist[cur] <= best[0] + 2:
            best = (dist[cur], cur)
        self.target = best[1]
        target = best[1]
        # keep going into the lane ahead so the line (and the speed) carries on:
        # straight on while that is open, and around a turn the camera has
        # already seen (straight is a known wall, exactly one side is open) -
        # so the racing line can take the corner instead of stopping in it
        prev = target
        step = None
        for n in self.open_neighbors(target, walls):
            if dist.get(n) is not None and dist[n] < dist[target]:
                step = (target[0] - n[0], target[1] - n[1])
        for _ in range(self.extend if step else 0):
            n = (prev[0] + step[0], prev[1] + step[1])
            if not self.in_bounds(n) or edge_between(prev, n) in walls:
                sides = [(-step[1], step[0]), (step[1], -step[0])]
                opts = [d for d in sides if self.in_bounds((prev[0] + d[0], prev[1] + d[1]))
                        and edge_between(prev, (prev[0] + d[0], prev[1] + d[1])) not in walls]
                if len(opts) != 1:
                    break
                step = opts[0]
                n = (prev[0] + step[0], prev[1] + step[1])
            if self.recent(n):
                break
            prev = n
        return prev
