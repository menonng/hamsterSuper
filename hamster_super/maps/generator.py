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
                    open_course: bool | None = None, wide_prob: float | None = None,
                    wide_straight_prob: float = 0.5) -> GridMap:
    """Race course that uses *every* cell of an N x N area (N = 14 for races).

    * A spanning tree over all (N/2)^2 blocks of 2x2 cells is traced around:
      one lane through all N^2 cells.
    * Some leaf blocks (hairpins) are opened into 2x2 areas: wide sections
      where the inside/outside line is a real choice.
    * open_course: cut the loop at one point -> start and finish are the two
      ends (one-way course); otherwise it is a circuit raced for one lap.
    * wide_straight_prob: chance of a course with 2-cell-wide straights instead
      (generate_wide_course).
    """
    level = int(np.clip(level, 0, COURSE_LEVELS - 1))
    if wide_straight_prob > 0 and rng.random() < wide_straight_prob:
        # half the courses: 2-cell-wide straights (falls back when no route fits)
        m = generate_wide_course(rng, level, open_course=open_course)
        if m is not None:
            return m
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
    # one-way: the wall between the finish and the start; lap: the start/finish line.
    # Neither block may be opened up, or the lap gets a shortcut across the line
    cut = (order[-1], order[0])
    wide = []
    for b in nodes:
        if degree[b] == 1 and rng.random() < wide_prob:
            if block_of(cut[0]) == b or block_of(cut[1]) == b:
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


def _block_route(m_: int, n_: int, rng, closed: bool, straight: float = 0.6, tries: int = 200):
    """Random Hamiltonian path (or cycle) over an m_ x n_ grid of blocks:
    randomized depth-first search with Warnsdorff ordering and a bias to keep
    going straight (long straights). None if none was found."""
    N = m_ * n_
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    inb = lambda b: 0 <= b[0] < m_ and 0 <= b[1] < n_
    for _ in range(tries):
        start = (int(rng.integers(m_)), int(rng.integers(n_))) if not closed else (0, 0)
        path, seen = [start], {start}
        budget = [20000]

        def free_deg(b):
            return sum(1 for d in dirs if inb((b[0] + d[0], b[1] + d[1])) and (b[0] + d[0], b[1] + d[1]) not in seen)

        def dfs():
            budget[0] -= 1
            if budget[0] < 0:
                return False
            if len(path) == N:
                if not closed:
                    return True
                a, b = path[-1], path[0]
                return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            cur = path[-1]
            prev_d = (cur[0] - path[-2][0], cur[1] - path[-2][1]) if len(path) > 1 else None
            cand = [(cur[0] + d[0], cur[1] + d[1]) for d in dirs]
            cand = [c for c in cand if inb(c) and c not in seen]
            keys = []
            for c in cand:
                d = (c[0] - cur[0], c[1] - cur[1])
                k = free_deg(c) + rng.random() * 1.5
                if d == prev_d and rng.random() < straight:
                    k -= 2.0
                keys.append(k)
            for _, c in sorted(zip(keys, cand)):
                path.append(c)
                seen.add(c)
                if dfs():
                    return True
                path.pop()
                seen.discard(c)
            return False

        if dfs():
            return path
    return None


def _narrow_path(entry, exit_dir):
    """1-cell lane through the 4 cells of a block (local coords 0/1), starting at
    `entry` and ending on the block side `exit_dir` points to."""
    e = entry
    for nb in ((1 - e[0], e[1]), (e[0], 1 - e[1])):
        path = [e, nb, (1 - e[0], 1 - e[1]), None]
        # the 4th cell is the one not used yet
        used = {e, nb, (1 - e[0], 1 - e[1])}
        last = [c for c in ((0, 0), (1, 0), (0, 1), (1, 1)) if c not in used][0]
        path[3] = last
        # side test: exit_dir (dx, dy) -> the last cell must lie on that side
        if (exit_dir[0] == 1 and last[0] == 1) or (exit_dir[0] == -1 and last[0] == 0) or \
                (exit_dir[1] == 1 and last[1] == 1) or (exit_dir[1] == -1 and last[1] == 0):
            return path
    return None


def generate_wide_course(rng: np.random.Generator, level: int = COURSE_LEVELS - 1,
                         open_course: bool | None = None, wide_prob: float | None = None) -> GridMap | None:
    """Race course with 2-cell-wide straights. The arena is split into 2x2
    blocks and a random route visits every block once. A block is either
    *wide* (the lane is 2 cells wide there: all 4 cells open) or *narrow*
    (a 1-cell lane winds through its 4 cells). Whole straight runs of blocks
    are made wide together, so the course has wide straights that narrow down
    again. Every cell of the N x N area is used. Returns None if no route
    (a closed route needs an even number of blocks)."""
    level = int(np.clip(level, 0, COURSE_LEVELS - 1))
    size = COURSE_SIZES[level]
    M = size // 2
    open_course = bool(rng.random() < 0.5) if open_course is None else open_course
    if not open_course and (M * M) % 2:
        return None
    straight = float(rng.uniform(0.3, 0.9))
    route = _block_route(M, M, rng, closed=not open_course, straight=straight)
    if route is None:
        return None
    nb = len(route)
    wide_prob = rng.uniform(0.3, 0.7) if wide_prob is None else wide_prob
    # straight runs: consecutive blocks entered and left in the same direction
    d_in = [None] + [(route[k][0] - route[k - 1][0], route[k][1] - route[k - 1][1]) for k in range(1, nb)]
    if not open_course:
        d_in[0] = (route[0][0] - route[-1][0], route[0][1] - route[-1][1])
    d_out = [(route[k + 1][0] - route[k][0], route[k + 1][1] - route[k][1]) for k in range(nb - 1)] + [None]
    if not open_course:
        d_out[-1] = (route[0][0] - route[-1][0], route[0][1] - route[-1][1])
    wide = [False] * nb
    k = 0
    while k < nb:
        j = k
        while j + 1 < nb and d_out[j] is not None and d_out[j] == d_out[j + 1]:
            j += 1
        run = list(range(k, j + 2 if j + 1 < nb and d_out[j] is not None and j > k else j + 1))
        if len(run) >= 2 and rng.random() < wide_prob:
            for r in run:
                wide[r] = True
        k = j + 1
    for r in range(nb):                       # a few wide corners too
        if rng.random() < 0.15 * wide_prob:
            wide[r] = True
    wide[0] = False                           # the start is in a 1-cell lane
    if open_course:
        wide[-1] = False                      # ... and a one-way finish is a dead end
    else:
        wide[-1] = True                       # the closing block must be able to meet the start
    walls = set()
    for i in range(size):
        for j in range(size):
            for dx, dy in NEIGHBORS:
                walls.add(edge_between((i, j), (i + dx, j + dy)))
    cell = lambda b, l: (2 * b[0] + l[0], 2 * b[1] + l[1])
    order = []
    # entry cell of the first block: a corner whose lane can leave toward the next block
    entry = None
    for e in ((0, 0), (1, 0), (0, 1), (1, 1)):
        if _narrow_path(e, d_out[0]) is not None:
            if d_in[0] is None or (d_in[0][0] == 1 and e[0] == 0) or (d_in[0][0] == -1 and e[0] == 1) or \
                    (d_in[0][1] == 1 and e[1] == 0) or (d_in[0][1] == -1 and e[1] == 1) or open_course:
                entry = e
                break
    if entry is None:
        return None
    first_entry = entry
    for k, b in enumerate(route):
        out = d_out[k]
        if not wide[k]:
            if out is None:                   # one-way finish block: any lane through the 4 cells
                lp = _narrow_path(entry, (0, 1)) or _narrow_path(entry, (0, -1)) or \
                    _narrow_path(entry, (1, 0)) or _narrow_path(entry, (-1, 0))
            else:
                lp = _narrow_path(entry, out)
            if lp is None:
                return None
            cells = [cell(b, l) for l in lp]
            for a, c in zip(cells, cells[1:]):
                walls.discard(edge_between(a, c))
            order += cells
            if out is None:
                break
            ex = lp[-1]
            nxt_entry = ((ex[0] + out[0]) % 2, (ex[1] + out[1]) % 2)     # the cell across the boundary
            walls.discard(edge_between(cell(b, ex), cell(route[(k + 1) % nb], nxt_entry)))
            entry = nxt_entry
        else:
            quad = [(0, 0), (1, 0), (1, 1), (0, 1)]
            for a, c in zip(quad, quad[1:] + quad[:1]):
                walls.discard(edge_between(cell(b, a), cell(b, c)))
            # sweep order (for lap progress): entry side first
            o = out or (0, 0)
            order += sorted((cell(b, l) for l in quad), key=lambda c: (c[0] - 2 * b[0]) * o[0] + (c[1] - 2 * b[1]) * o[1])
            if out is None:
                break
            nxt = route[(k + 1) % nb]
            # the two cells of this block on the side facing the next block
            side = [l for l in quad if (out[0] == 1 and l[0] == 1) or (out[0] == -1 and l[0] == 0)
                    or (out[1] == 1 and l[1] == 1) or (out[1] == -1 and l[1] == 0)]
            nwide = wide[(k + 1) % nb] and not (k + 1 == nb)
            if k + 1 == nb:                   # closing a loop: only into the start cell
                ins = [l for l in side if ((l[0] + out[0]) % 2, (l[1] + out[1]) % 2) == first_entry]
                if not ins:
                    return None
                walls.discard(edge_between(cell(b, ins[0]), cell(nxt, first_entry)))
            elif nwide:                       # wide into wide: the whole 2-cell width is open
                for l in side:
                    walls.discard(edge_between(cell(b, l), cell(nxt, ((l[0] + out[0]) % 2, (l[1] + out[1]) % 2))))
            else:                             # wide into narrow: one of the two cells
                l = side[int(rng.integers(2))]
                entry = ((l[0] + out[0]) % 2, (l[1] + out[1]) % 2)
                if _narrow_path(entry, d_out[k + 1] or (0, 1)) is None and d_out[k + 1] is not None:
                    l = side[0] if l == side[1] else side[1]
                    entry = ((l[0] + out[0]) % 2, (l[1] + out[1]) % 2)
                walls.discard(edge_between(cell(b, l), cell(nxt, entry)))
    if not open_course and len(order) != size * size:
        return None
    nwide = sum(wide)
    kind = "oneway" if open_course else "lap"
    m = GridMap(walls=walls, name=f"course-{size}x{size}-{kind}-wide-straights{nwide}-s{straight:.2f}", kind=kind)
    m.course = order
    # wide blocks inside a wide straight (wide before and after, same direction):
    # moving course parts go there, where one half of the width always stays open
    m.wide_blocks = [(route[k], d_out[k]) for k in range(1, nb - 1)
                     if wide[k - 1] and wide[k] and wide[k + 1] and d_in[k] == d_out[k] == d_out[k - 1]]
    m.start = order[0]
    d0 = (order[1][0] - order[0][0], order[1][1] - order[0][1])
    m.start_heading = {(1, 0): 0.0, (0, 1): math.pi / 2, (-1, 0): math.pi, (0, -1): -math.pi / 2}[d0]
    if open_course:
        m.goal = order[-1]
    else:
        # last cell before the line: the closing block's cell next to the start
        m.goal = [c for c in order[-4:] if abs(c[0] - m.start[0]) + abs(c[1] - m.start[1]) == 1][0]
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
