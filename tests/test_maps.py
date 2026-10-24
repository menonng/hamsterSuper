import numpy as np
import pytest

from hamster_super.maps.generator import LEVELS, generate
from hamster_super.maps.grid_map import GridMap, edge_between, make_edge
from hamster_super.maps.parser import load_map, parse_ascii, parse_json, to_ascii, to_json


def test_wall_must_join_adjacent_dots():
    make_edge((0, 0), (1, 0))
    with pytest.raises(ValueError):
        make_edge((0, 0), (1, 1))      # diagonal: 12.7 cm, not a 9 cm wall
    with pytest.raises(ValueError):
        make_edge((0, 0), (2, 0))


def test_wall_box_dimensions():
    m = GridMap()
    m.add_wall((1, 1), (2, 1))
    b = m.wall_boxes()[0]
    assert b[2] - b[0] == pytest.approx(9.0)    # length
    assert b[3] - b[1] == pytest.approx(0.5)    # thickness


def test_edge_between_cells():
    assert edge_between((0, 0), (1, 0)) == make_edge((1, 0), (1, 1))
    assert edge_between((0, 0), (0, 1)) == make_edge((0, 1), (1, 1))
    assert edge_between((1, 1), (0, 1)) == make_edge((1, 1), (1, 2))


def test_ascii_roundtrip_and_examples():
    for f in ["maps/example_01_s_curve.txt", "maps/example_02_maze.txt", "maps/example_03_open_arena.txt"]:
        m = load_map(f)
        assert m.shortest_path() is not None, f
        m2 = parse_ascii(to_ascii(m))
        assert m2.walls == m.walls and m2.start == m.start and m2.goal == m.goal
        m3 = parse_json(to_json(m))
        assert m3.walls == m.walls


@pytest.mark.parametrize("level", range(len(LEVELS)))
def test_generated_maps_are_solvable(level):
    rng = np.random.default_rng(level)
    for _ in range(20):
        m = generate(rng, level)
        assert m.shortest_path() is not None
        assert m.start != m.goal


def test_wide_straight_courses():
    """Courses with 2-cell-wide straights: every cell used, the finish (one-way)
    or the whole lap reachable, and real 2-wide straight runs present."""
    from hamster_super.maps.generator import generate_wide_course
    from hamster_super.maps.grid_map import edge_between
    made = 0
    for k in range(20):
        for open_course in (True, False):
            m = generate_wide_course(np.random.default_rng(k), 4, open_course=open_course)
            if m is None:
                continue
            made += 1
            n = m.bounds[2] + 1
            assert len(set(m.course)) == n * n
            assert m.shortest_path()
            # a 2-wide straight: two side-by-side cells open to each other and to the
            # next pair along the run, for at least 2 steps
            def wide_at(i, j):
                return all(edge_between(a, b) not in m.walls for a, b in
                           [((i, j), (i, j + 1)), ((i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j)),
                            ((i, j + 1), (i + 1, j + 1)), ((i + 1, j), (i + 2, j)), ((i + 1, j + 1), (i + 2, j + 1))])
            def wide_v(i, j):
                return all(edge_between(a, b) not in m.walls for a, b in
                           [((i, j), (i + 1, j)), ((i, j + 1), (i + 1, j + 1)), ((i, j), (i, j + 1)),
                            ((i + 1, j), (i + 1, j + 1)), ((i, j + 1), (i, j + 2)), ((i + 1, j + 1), (i + 1, j + 2))])
            assert any(wide_at(i, j) or wide_v(i, j) for i in range(n - 2) for j in range(n - 2)), m.name
    assert made >= 20
