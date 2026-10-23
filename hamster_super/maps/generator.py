"""Random race-map generator with a difficulty curriculum."""
from __future__ import annotations

import math

import numpy as np

from .grid_map import NEIGHBORS, GridMap, edge_between, make_edge

# (min cells, max cells) per side for each curriculum level
LEVELS = [(2, 3), (3, 4), (3, 6), (4, 8), (5, 10), (6, 12), (8, 14), (10, 16), (12, 20)]
RACE_LEVEL = 5          # race courses: 14 x 14 cells (COURSE_SIZES[5])
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


def generate_track(rng: np.random.Generator, level: int = 3) -> GridMap:
    """Race course: one corridor without branches, closed at both ends.
    Built as the longest path of a random perfect maze, then walled in."""
    level = int(np.clip(level, 0, len(LEVELS) - 1))
    lo, hi = LEVELS[level]
    w = int(rng.integers(lo, hi + 1))
    h = int(rng.integers(lo, hi + 1))
    if level == 0 and rng.random() < 0.5:
        w, h = int(rng.integers(2, 4)), 1
    tree = GridMap()
    _perfect_maze(tree, w, h, rng)
    tree.bounds = (0, 0, w - 1, h - 1)
    # longest path in the spanning tree: farthest from a random cell, then farthest from that
    a = (int(rng.integers(w)), int(rng.integers(h)))
    d = tree.distance_field(a)
    b = max(d, key=d.get)
    d = tree.distance_field(b)
    e = max(d, key=d.get)
    path = tree.shortest_path(b, e)
    if rng.random() < 0.5:
        path = path[::-1]
    on = set(path)
    m = GridMap(name=f"track-L{level}-{w}x{h}-{len(path)}", kind="roundtrip")
    for i, c in enumerate(path):
        keep = {path[j] for j in (i - 1, i + 1) if 0 <= j < len(path)}
        for dx, dy in NEIGHBORS:
            n = (c[0] + dx, c[1] + dy)
            if n not in keep:
                m.walls.add(edge_between(c, n))
    m.start, m.goal = path[0], path[-1]
    if len(path) > 1:
        d0 = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        m.start_heading = {(1, 0): 0.0, (0, 1): math.pi / 2, (-1, 0): math.pi, (0, -1): -math.pi / 2}[d0]
    m.bounds = (0, 0, w - 1, h - 1)
    del on
    return m


def _loop_cells(nodes, tree_edges):
    """Hamiltonian cycle around a tree of 2x2 cell blocks.
    Every block starts as a 4-cell ring; each tree edge merges the two rings
    into one (a 2-opt swap), so the result is a single loop through all cells."""
    adj: dict = {}

    def link(a, b):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def unlink(a, b):
        adj[a].discard(b)
        adj[b].discard(a)

    def block(i, j):   # BL, BR, TR, TL
        return (2 * i, 2 * j), (2 * i + 1, 2 * j), (2 * i + 1, 2 * j + 1), (2 * i, 2 * j + 1)

    for i, j in nodes:
        bl, br, tr, tl = block(i, j)
        link(bl, br), link(br, tr), link(tr, tl), link(tl, bl)
    for a, b in tree_edges:
        (ai, aj), (bi, bj) = sorted((a, b))
        abl, abr, atr, atl = block(ai, aj)
        bbl, bbr, btr, btl = block(bi, bj)
        if bi == ai + 1:      # B right of A
            unlink(abr, atr), unlink(bbl, btl)
            link(abr, bbl), link(atr, btl)
        else:                 # B above A
            unlink(atl, atr), unlink(bbl, bbr)
            link(atl, bbl), link(atr, bbr)
    return adj


def generate_loop(rng: np.random.Generator, level: int = 3) -> GridMap:
    """Closed circuit (like an F1 / kart track): one lane, no branches.
    A random tree of 2x2 blocks (biased to grow straight, for long straights)
    is traced around; the loop gets hairpins and lanes running side by side."""
    level = int(np.clip(level, 0, len(LEVELS) - 1))
    lo, hi = LEVELS[level]
    w = int(rng.integers(lo, hi + 1))
    h = int(rng.integers(lo, hi + 1))
    m_, n_ = max(1, w // 2), max(1, h // 2)
    k = max(1, int(round(m_ * n_ * rng.uniform(0.55, 1.0))))
    root = (int(rng.integers(m_)), int(rng.integers(n_)))
    nodes, edges, came = {root}, [], {root: None}
    order = [root]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for _ in range(20 * k):
        if len(nodes) >= k:
            break
        # grow mostly from recent nodes, and mostly straight on
        a = order[-1 - int(min(len(order) - 1, rng.exponential(2.0)))]
        opts = [d for d in dirs if 0 <= a[0] + d[0] < m_ and 0 <= a[1] + d[1] < n_
                and (a[0] + d[0], a[1] + d[1]) not in nodes]
        if not opts:
            continue
        d = came[a] if came[a] in opts and rng.random() < 0.65 else opts[int(rng.integers(len(opts)))]
        b = (a[0] + d[0], a[1] + d[1])
        nodes.add(b)
        edges.append((a, b))
        came[b] = d
        order.append(b)
    adj = _loop_cells(nodes, edges)
    m = GridMap(name=f"circuit-L{level}-{2 * m_}x{2 * n_}-{len(adj)}", kind="lap")
    for c, nb in adj.items():
        for dx, dy in NEIGHBORS:
            n = (c[0] + dx, c[1] + dy)
            if n not in nb:
                m.walls.add(edge_between(c, n))
    cells = sorted(adj)
    m.start = cells[int(rng.integers(len(cells)))]
    ahead = sorted(adj[m.start])[int(rng.integers(2))]
    d0 = (ahead[0] - m.start[0], ahead[1] - m.start[1])
    m.start_heading = {(1, 0): 0.0, (0, 1): math.pi / 2, (-1, 0): math.pi, (0, -1): -math.pi / 2}[d0]
    behind = next(c for c in adj[m.start] if c != ahead)
    m.goal = behind               # the last cell of the lap (crossing it into the start = lap done)
    m.bounds = (0, 0, 2 * m_ - 1, 2 * n_ - 1)
    return m


def lap_order(m: GridMap) -> list:
    """Cells of a lap course in driving order, starting at the start cell."""
    d = (round(math.cos(m.start_heading)), round(math.sin(m.start_heading)))
    prev, cur = m.start, (m.start[0] + d[0], m.start[1] + d[1])
    order = [m.start]
    while cur != m.start and len(order) < 100000:
        order.append(cur)
        nxt = [n for n in m.open_neighbors(cur) if n != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
    return order


COURSE_SIZES = [4, 6, 8, 10, 12, 14]     # course side (cells) per curriculum level; races are 14 x 14
COURSE_LEVELS = len(COURSE_SIZES)


def _spanning_tree(m_, n_, rng, straight=0.65):
    """Random spanning tree over every block, grown mostly straight (long straights)."""
    root = (int(rng.integers(m_)), int(rng.integers(n_)))
    nodes, edges, came, order = {root}, [], {root: None}, [root]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    total = m_ * n_
    while len(nodes) < total:
        a = order[-1 - int(min(len(order) - 1, rng.exponential(2.0)))]
        opts = [d for d in dirs if 0 <= a[0] + d[0] < m_ and 0 <= a[1] + d[1] < n_
                and (a[0] + d[0], a[1] + d[1]) not in nodes]
        if not opts:
            # fall back to any node that can still grow
            growable = [x for x in order if any(0 <= x[0] + d[0] < m_ and 0 <= x[1] + d[1] < n_
                                                and (x[0] + d[0], x[1] + d[1]) not in nodes for d in dirs)]
            a = growable[int(rng.integers(len(growable)))]
            opts = [d for d in dirs if 0 <= a[0] + d[0] < m_ and 0 <= a[1] + d[1] < n_
                    and (a[0] + d[0], a[1] + d[1]) not in nodes]
        d = came[a] if came[a] in opts and rng.random() < straight else opts[int(rng.integers(len(opts)))]
        b = (a[0] + d[0], a[1] + d[1])
        nodes.add(b)
        edges.append((a, b))
        came[b] = d
        order.append(b)
    return nodes, edges


def generate_course(rng: np.random.Generator, level: int = COURSE_LEVELS - 1,
                    open_course: bool | None = None, wide_prob: float | None = None) -> GridMap:
    """Race course that uses *every* cell of an N x N area (N = 14 for races).

    * A spanning tree over all (N/2)^2 blocks of 2x2 cells is traced around:
      one lane through all N^2 cells.
    * Some leaf blocks (hairpins) are opened into 2x2 areas: wide sections
      where the inside/outside line is a real choice.
    * open_course: cut the loop at one point -> start and finish are the two
      ends (one-way course); otherwise it is a circuit raced for one lap.
    """
    level = int(np.clip(level, 0, COURSE_LEVELS - 1))
    size = COURSE_SIZES[level]
    m_ = n_ = size // 2
    # every map draws its own character: twisty (low) ... long straights (high)
    straight = float(rng.uniform(0.2, 0.9))
    nodes, edges = _spanning_tree(m_, n_, rng, straight=straight)
    adj = _loop_cells(nodes, edges)
    # loop order (driving direction chosen later)
    first = min(adj)
    order, prev, cur = [first], None, first
    while True:
        nxt = [c for c in sorted(adj[cur]) if c != prev]
        prev, cur = cur, nxt[0]
        if cur == first:
            break
        order.append(cur)
    if rng.random() < 0.5:
        order = order[:1] + order[1:][::-1]
    n_cells = len(order)
    open_course = bool(rng.random() < 0.5) if open_course is None else open_course
    # walls: everything that is not a lane connection
    walls = set()
    for c, nb in adj.items():
        for dx, dy in NEIGHBORS:
            n = (c[0] + dx, c[1] + dy)
            if n not in nb:
                walls.add(edge_between(c, n))
    # wide sections: open the inside of some leaf blocks (U-turns)
    degree = {b: 0 for b in nodes}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    wide_prob = rng.uniform(0.2, 0.7) if wide_prob is None else wide_prob
    block_of = lambda c: (c[0] // 2, c[1] // 2)
    shift = int(rng.integers(n_cells))
    order = order[shift:] + order[:shift]
    if open_course:
        cut = (order[-1], order[0])            # wall between the finish and the start
    wide = []
    for b in nodes:
        if degree[b] == 1 and rng.random() < wide_prob:
            if open_course and (block_of(cut[0]) == b or block_of(cut[1]) == b):
                continue
            i, j = b
            cells = [(2 * i, 2 * j), (2 * i + 1, 2 * j), (2 * i + 1, 2 * j + 1), (2 * i, 2 * j + 1)]
            for a_, b_ in zip(cells, cells[1:] + cells[:1]):
                walls.discard(edge_between(a_, b_))
            wide.append(b)
    if open_course:
        walls.add(edge_between(*cut))
    kind = "oneway" if open_course else "lap"
    m = GridMap(walls=walls, name=f"course-{size}x{size}-{kind}-wide{len(wide)}-s{straight:.2f}", kind=kind)
    m.course = order
    m.start = order[0]
    d0 = (order[1][0] - order[0][0], order[1][1] - order[0][1])
    m.start_heading = {(1, 0): 0.0, (0, 1): math.pi / 2, (-1, 0): math.pi, (0, -1): -math.pi / 2}[d0]
    m.goal = order[-1]                          # finish cell (one-way) / last cell before the line (lap)
    m.bounds = (0, 0, size - 1, size - 1)
    return m


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
