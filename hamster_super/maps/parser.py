"""Map parser: ASCII art and JSON.

ASCII format (top row = north). Every '+' is a floor dot; '---' between two
dots is a horizontal wall, '|' between two dot columns is a vertical wall.
'S' marks the start cell (optionally followed by a heading letter E/N/W/S,
e.g. 'SE'), 'G' marks the goal cell. Without a 'G' the map is a round-trip
course: drive to its dead end and come back to S. Lines starting with '#' are comments.

    +---+---+---+
    | SE    |   |
    +   +---+   +
    |         G |
    +---+---+---+

Each cell column is 4 characters wide ('+---'), each cell row is 2 lines.

JSON format:
    {"walls": [[[x1, y1], [x2, y2]], ...],   # lattice point pairs
     "start": [cx, cy], "goal": [cx, cy], "start_heading": "E"}
"""
from __future__ import annotations

import json
from pathlib import Path

from .grid_map import DIRS, GridMap, make_edge


def parse_ascii(text: str, name: str = "") -> GridMap:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines or "+" not in lines[0]:
        raise ValueError("ASCII map must start with a dot row ('+---+')")
    # Align so that the first '+' of the first row is column 0
    off = min(ln.index("+") for ln in lines if "+" in ln)
    lines = [ln[off:] for ln in lines]
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width) for ln in lines]
    n_rows = len(lines)
    if n_rows % 2 == 0:
        raise ValueError("ASCII map needs an odd number of lines (dot rows and cell rows alternate)")
    n_cells_y = (n_rows - 1) // 2
    n_cells_x = (width - 1) // 4
    m = GridMap(name=name)
    start = goal = None
    for r, ln in enumerate(lines):
        if r % 2 == 0:  # dot row -> horizontal walls
            jy = n_cells_y - r // 2
            for i in range(n_cells_x):
                seg = ln[4 * i + 1: 4 * i + 4]
                if "-" in seg or "=" in seg:
                    m.walls.add(make_edge((i, jy), (i + 1, jy)))
        else:           # cell row -> vertical walls + markers
            cy = n_cells_y - 1 - r // 2
            for i in range(n_cells_x + 1):
                if 4 * i < len(ln) and ln[4 * i] in "|#":
                    m.walls.add(make_edge((i, cy), (i, cy + 1)))
            for i in range(n_cells_x):
                body = ln[4 * i + 1: 4 * i + 4]
                if "S" in body:
                    start = (i, cy)
                    k = body.index("S")
                    tail = body[k + 1:].strip()
                    if tail and tail[0] in DIRS:
                        m.start_heading = DIRS[tail[0]]
                if "G" in body:
                    goal = (i, cy)
    if start is None:
        raise ValueError("ASCII map needs an 'S' (start) cell")
    m.start = start
    m.bounds = (0, 0, n_cells_x - 1, n_cells_y - 1)
    if goal is None:
        _race_kind(m)

    else:
        m.goal = goal
    return m


def to_ascii(m: GridMap) -> str:
    x0, y0, x1, y1 = m.cell_bounds()
    rows = []
    for jy in range(y1 + 1, y0 - 1, -1):
        s = "+"
        for i in range(x0, x1 + 1):
            s += ("---" if make_edge((i, jy), (i + 1, jy)) in m.walls else "   ") + "+"
        rows.append(s)
        if jy == y0:
            break
        cy = jy - 1
        s = ""
        for i in range(x0, x1 + 2):
            s += "|" if make_edge((i, cy), (i, cy + 1)) in m.walls else " "
            if i <= x1:
                body = "   "
                if (i, cy) == m.start:
                    hd = ""
                    if m.start_heading is not None:
                        hd = min(DIRS, key=lambda k: abs(_wrap(DIRS[k] - m.start_heading)))
                    body = (" S" + hd).ljust(3)
                elif (i, cy) == m.goal and m.kind == "goal":
                    body = " G "
                s += body
        rows.append(s)
    return "\n".join(rows) + "\n"


def _wrap(a):
    import math
    return (a + math.pi) % (2 * math.pi) - math.pi


def _race_kind(m: GridMap):
    """No goal given: a closed circuit (start has two open sides) is a lap race,
    otherwise a round trip to the dead end. `goal` keeps the true lap end / dead end."""
    import math
    if len(list(m.open_neighbors(m.start))) == 2:
        m.kind = "lap"
        if m.start_heading is None:
            n = sorted(m.open_neighbors(m.start))[0]
            m.start_heading = math.atan2(n[1] - m.start[1], n[0] - m.start[0])
        d = (round(math.cos(m.start_heading)), round(math.sin(m.start_heading)))
        ahead = (m.start[0] + d[0], m.start[1] + d[1])
        m.goal = next(c for c in m.open_neighbors(m.start) if c != ahead)
    else:
        m.kind = "roundtrip"
        dist = m.distance_field(m.start)
        m.goal = max(dist, key=dist.get)


def parse_json(data: dict | str, name: str = "") -> GridMap:
    if isinstance(data, str):
        data = json.loads(data)
    m = GridMap(name=name or data.get("name", ""))
    for a, b in data["walls"]:
        m.walls.add(make_edge(tuple(a), tuple(b)))
    m.start = tuple(data["start"])
    if data.get("goal") is None:
        _race_kind(m)
    else:
        m.goal = tuple(data["goal"])
    hd = data.get("start_heading")
    if isinstance(hd, str):
        m.start_heading = DIRS[hd.upper()]
    elif hd is not None:
        m.start_heading = float(hd)
    if "bounds" in data:
        m.bounds = tuple(data["bounds"])
    return m


def to_json(m: GridMap) -> str:
    d = {"name": m.name,
         "walls": [[list(a), list(b)] for a, b in sorted(m.walls)],
         "start": list(m.start), "goal": list(m.goal) if m.kind == "goal" else None}
    if m.start_heading is not None:
        d["start_heading"] = m.start_heading
    if m.bounds is not None:
        d["bounds"] = list(m.bounds)
    return json.dumps(d, indent=1)


def load_map(path) -> GridMap:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return parse_json(text, name=p.stem)
    return parse_ascii(text, name=p.stem)
