"""Race navigation without a destination.

mode "lap" (the race): a closed circuit. Leave the start cell in the
heading direction, follow the lane while mapping it, and the lap is done
when the robot drives back into the start cell from the other side.

mode "roundtrip": a single-path course with a dead end: drive to its dead
end, then back to the start.

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
    def __init__(self, start_cell, bounds, extend: int = 3, dead_end_dist: float = 3.0,
                 mode: str = "roundtrip", start_heading: float | None = None):
        self.mode = mode
        self.start = tuple(start_cell)
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

    def goal(self, cell, pos, heading, walls):
        """Update the phase and return the cell the planner should head for."""
        cell = tuple(cell)
        if self.mode == "lap":
            return self._lap_goal(cell, heading, walls)
        if self.in_bounds(cell):
            self.visited.add(cell)
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

    def _lap_goal(self, cell, heading, walls):
        came_from = self.prev_cell
        if cell != self.prev_cell:
            self.prev_cell = cell
        if self.phase == DONE:
            return self.start
        if cell == self.start and len(self.visited) >= 4 and came_from not in (self.start, self.ahead):
            self.phase = DONE                      # entered the start cell from behind: lap complete
            return self.start
        if self.in_bounds(cell):
            self.visited.add(cell)
        # closing the lap: the start is the next cell straight ahead and nothing blocks it
        if len(self.visited) >= 4 and cell != self.start:
            dx, dy = self.start[0] - cell[0], self.start[1] - cell[1]
            if abs(dx) + abs(dy) == 1 and edge_between(cell, self.start) not in walls \
                    and dx * math.cos(heading) + dy * math.sin(heading) > 0.5:
                return self.start
        target = self._explore_target(cell, heading, walls)
        if self.phase == BACK:                     # nothing unvisited left: head for the start
            self.phase = OUT
            return self.start
        return target

    def _explore_target(self, cell, heading, walls):
        """Nearest unvisited cell (ties: straight ahead first), then extended straight on."""
        fwd = (round(math.cos(heading)), round(math.sin(heading)))
        dist = {cell: 0}
        first = {cell: None}
        q = deque([cell])
        best = None
        while q:
            c = q.popleft()
            if c not in self.visited:
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
            return self.start
        target = best[1]
        # keep going straight into unknown space so the line (and the speed) carries on
        prev = target
        step = None
        for n in self.open_neighbors(target, walls):
            if dist.get(n) is not None and dist[n] < dist[target]:
                step = (target[0] - n[0], target[1] - n[1])
        for _ in range(self.extend if step else 0):
            n = (prev[0] + step[0], prev[1] + step[1])
            if not self.in_bounds(n) or edge_between(prev, n) in walls or n in self.visited:
                break
            prev = n
        return prev
