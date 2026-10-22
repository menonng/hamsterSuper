"""Random race-map generator with a difficulty curriculum."""
from __future__ import annotations

import math

import numpy as np

from .grid_map import NEIGHBORS, GridMap, edge_between, make_edge

# (min cells, max cells) per side for each curriculum level
LEVELS = [(2, 3), (3, 4), (3, 6), (4, 8), (5, 10), (6, 12)]
STYLES = ("perfect", "braid", "rooms", "sparse")


def _boundary(m: GridMap, w: int, h: int):
    for i in range(w):
        m.walls.add(make_edge((i, 0), (i + 1, 0)))
        m.walls.add(make_edge((i, h), (i + 1, h)))
    for j in range(h):
        m.walls.add(make_edge((0, j), (0, j + 1)))
        m.walls.add(make_edge((w, j), (w, j + 1)))


def _all_interior_edges(w: int, h: int):
    for i in range(w):
        for j in range(h):
            if i + 1 < w:
                yield (i, j), (i + 1, j)
            if j + 1 < h:
                yield (i, j), (i, j + 1)


def _perfect_maze(m: GridMap, w: int, h: int, rng: np.random.Generator):
    for c, n in _all_interior_edges(w, h):
        m.walls.add(edge_between(c, n))
    stack = [(int(rng.integers(w)), int(rng.integers(h)))]
    seen = {stack[0]}
    while stack:
        c = stack[-1]
        opts = [(c[0] + dx, c[1] + dy) for dx, dy in NEIGHBORS]
        opts = [n for n in opts if 0 <= n[0] < w and 0 <= n[1] < h and n not in seen]
        if not opts:
            stack.pop()
            continue
        n = opts[int(rng.integers(len(opts)))]
        m.walls.discard(edge_between(c, n))
        seen.add(n)
        stack.append(n)


def generate(rng: np.random.Generator, level: int = 3, style: str | None = None) -> GridMap:
    level = int(np.clip(level, 0, len(LEVELS) - 1))
    lo, hi = LEVELS[level]
    w = int(rng.integers(lo, hi + 1))
    h = int(rng.integers(lo, hi + 1))
    if level == 0:
        h = 1 if rng.random() < 0.5 else h       # straight corridor warm-up
    style = style or STYLES[int(rng.integers(len(STYLES)))]
    for _ in range(50):
        m = GridMap(name=f"gen-L{level}-{style}-{w}x{h}")
        _boundary(m, w, h)
        if style == "perfect":
            _perfect_maze(m, w, h, rng)
        elif style == "braid":
            _perfect_maze(m, w, h, rng)
            interior = [e for e in m.walls if not _on_boundary(e, w, h)]
            k = int(len(interior) * rng.uniform(0.1, 0.35))
            for idx in rng.permutation(len(interior))[:k]:
                m.walls.discard(interior[idx])
        elif style == "rooms":
            # long straight wall runs with gaps
            for _ in range(int(rng.integers(1, max(2, (w + h) // 2) + 1))):
                horiz = rng.random() < 0.5
                if horiz and h > 1:
                    y = int(rng.integers(1, h))
                    gap = int(rng.integers(w))
                    for i in range(w):
                        if i != gap and rng.random() < 0.9:
                            m.walls.add(make_edge((i, y), (i + 1, y)))
                elif w > 1:
                    x = int(rng.integers(1, w))
                    gap = int(rng.integers(h))
                    for j in range(h):
                        if j != gap and rng.random() < 0.9:
                            m.walls.add(make_edge((x, j), (x, j + 1)))
        else:  # sparse: random isolated walls between any two adjacent dots
            n_walls = int(w * h * rng.uniform(0.2, 0.6))
            for _ in range(n_walls):
                if rng.random() < 0.5 and h > 1:
                    i, j = int(rng.integers(0, w)), int(rng.integers(1, h))
                    m.walls.add(make_edge((i, j), (i + 1, j)))
                elif w > 1:
                    i, j = int(rng.integers(1, w)), int(rng.integers(0, h))
                    m.walls.add(make_edge((i, j), (i, j + 1)))
        m.bounds = (0, 0, w - 1, h - 1)
        if _place_start_goal(m, w, h, rng, level):
            return m
    # fall back to an empty arena
    m = GridMap(name=f"gen-L{level}-empty-{w}x{h}")
    _boundary(m, w, h)
    m.bounds = (0, 0, w - 1, h - 1)
    _place_start_goal(m, w, h, rng, level, min_frac=0.0)
    return m


def _on_boundary(e, w, h):
    (a, b) = e
    return (a[0] == b[0] and a[0] in (0, w)) or (a[1] == b[1] and a[1] in (0, h))


def _place_start_goal(m: GridMap, w, h, rng, level, min_frac=0.5) -> bool:
    cells = [(i, j) for i in range(w) for j in range(h)]
    start = cells[int(rng.integers(len(cells)))]
    dist = m.distance_field(start)
    if len(dist) < 2:
        return False
    far = max(dist.values())
    cand = [c for c, d in dist.items() if d >= max(1, math.ceil(far * min_frac))]
    if not cand:
        return False
    m.start = start
    m.goal = cand[int(rng.integers(len(cand)))]
    heads = m.free_headings(start)
    m.start_heading = None if not heads else float(heads[int(rng.integers(len(heads)))])
    return True
