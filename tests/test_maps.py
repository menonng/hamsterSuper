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
