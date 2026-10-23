import math

import numpy as np

from hamster_super.config import EnvConfig
from hamster_super.maps.parser import load_map
from hamster_super.nav.localizer import GridSnapLocalizer
from hamster_super.nav.mapper import BeliefMap
from hamster_super.nav.planner import Planner
from hamster_super.nav.stack import cam_origin
from hamster_super.sim.sensors import DepthRays


def test_mapper_recovers_visible_walls():
    cfg = EnvConfig()
    m = load_map("maps/example_01_s_curve.txt")
    boxes = m.wall_boxes()
    rays = DepthRays(cfg.camera)
    b = BeliefMap(m.cell_bounds())
    for th in np.linspace(-math.pi, math.pi, 16, endpoint=False):
        pose = np.array([4.5, 22.5, th])
        r = rays.true_ranges(pose, boxes)
        for _ in range(3):
            b.update(cam_origin(cfg, pose), th, rays.angles, r, cfg.camera.max_range, rays.r_min)
    assert b.walls, "no walls mapped"
    assert b.walls <= m.walls, f"false walls: {b.walls - m.walls}"


def test_planner_potential_decreases_along_path():
    m = load_map("maps/example_02_maze.txt")
    p = Planner(m.cell_bounds(), m.goal)
    p.update(m.walls)
    path = m.shortest_path()
    from hamster_super.maps.grid_map import cell_center
    phis = [p.potential(cell_center(c)) for c in path]
    assert all(a > b for a, b in zip(phis, phis[1:]))
    assert phis[-1] == 0.0


def test_grid_snap_corrects_offset():
    cfg = EnvConfig()
    m = load_map("maps/example_02_maze.txt")
    boxes = m.wall_boxes()
    rays = DepthRays(cfg.camera)
    loc = GridSnapLocalizer(alpha=1.0)
    true = np.array([13.5, 58.5, 0.3])
    r = rays.true_ranges(true, boxes)
    est = true + np.array([1.0, -0.8, 0.04])
    valid = (r < cfg.camera.max_range - 0.5) & (r > rays.r_min + 0.3)
    for _ in range(5):
        est = loc.correct(est, cam_origin(cfg, est), rays.angles, r, valid)
    assert np.linalg.norm(est[:2] - true[:2]) < np.linalg.norm([1.0, -0.8])
    assert abs(est[2] - true[2]) < 0.04


def test_wall_orientation_recovers_heading():
    from hamster_super.nav.localizer import wall_orientation
    from hamster_super.nav.stack import NavStack
    cfg = EnvConfig()
    m = load_map("maps/example_02_maze.txt")
    rays = DepthRays(cfg.camera)
    st = NavStack(cfg, m.cell_bounds(), (13.5, 58.5, 0.0), m.goal, rays.r_min)
    pose = np.array([13.5, 58.5, 0.3])
    r = rays.true_ranges(pose, m.wall_boxes())
    pts = st.wall_points(r, np.zeros(2))
    wa = wall_orientation(pts[:st._n_cam_pts])
    err = ((-wa - pose[2]) + math.pi / 4) % (math.pi / 2) - math.pi / 4
    assert abs(err) < math.radians(3)


def test_particle_filter_tracks_unknown_speed():
    """Robot is 20% faster than odometry thinks; the filter must keep up."""
    from hamster_super.nav.localizer import LatticeParticleFilter
    cfg = EnvConfig()
    m = load_map("maps/example_01_s_curve.txt")   # top corridor runs east from (0,2)
    boxes = m.wall_boxes()
    rays = DepthRays(cfg.camera)
    from hamster_super.nav.stack import NavStack
    st = NavStack(cfg, m.cell_bounds(), (4.5, 22.5, 0.0), m.goal, rays.r_min)
    pf = LatticeParticleFilter((4.5, 22.5, 0.0), rng=np.random.default_rng(0))
    true = np.array([4.5, 22.5, 0.0])
    for _ in range(12):
        true[0] += 1.2                           # actual motion
        pf.predict(1.0, 0.0)                     # odometry says 1.0 cm
        pf.update(st.wall_points(rays.true_ranges(true, boxes), np.zeros(2)), st.belief)
    assert abs(pf.pose[1] - true[1]) < 1.0       # lateral locked by the side walls
    assert abs(pf.pose[2]) < math.radians(5)


def _min_wall_distance(line, boxes):
    worst = np.inf
    for a, b in zip(line[:-1], line[1:]):
        for t in np.linspace(0, 1, 60):
            p = a + (b - a) * t
            cx = np.clip(p[0], boxes[:, 0], boxes[:, 2])
            cy = np.clip(p[1], boxes[:, 1], boxes[:, 3])
            worst = min(worst, float(np.min(np.hypot(p[0] - cx, p[1] - cy))))
    return worst


def test_racing_line_cuts_corners_but_clears_walls():
    from hamster_super.maps.grid_map import cell_center
    from hamster_super.nav.racing_line import RacingLine
    for f in ["maps/example_01_s_curve.txt", "maps/example_02_maze.txt", "maps/example_03_open_arena.txt"]:
        m = load_map(f)
        p = Planner(m.cell_bounds(), m.goal)
        p.update(m.walls)
        rl = RacingLine(horizon=60)
        line = rl.update(p, cell_center(m.start), sigma=0.0)
        length = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
        centre = 9.0 * (len(m.shortest_path()) - 1)
        assert length < centre, f
        assert _min_wall_distance(line, m.wall_boxes()) > 2.3, f      # robot radius
        assert all(c["label"] in ("in", "mid", "out") for c in rl.corners)


def test_racing_line_backs_off_when_uncertain():
    from hamster_super.maps.grid_map import cell_center
    from hamster_super.nav.racing_line import RacingLine
    m = load_map("maps/example_01_s_curve.txt")
    p = Planner(m.cell_bounds(), m.goal)
    p.update(m.walls)
    sure, unsure = RacingLine(horizon=60), RacingLine(horizon=60)
    sure.update(p, cell_center(m.start), sigma=0.0)
    unsure.update(p, cell_center(m.start), sigma=1.0)
    assert _min_wall_distance(unsure.line, m.wall_boxes()) > _min_wall_distance(sure.line, m.wall_boxes()) + 1.0
    assert all(b["t"] > a["t"] for a, b in zip(sure.corners, unsure.corners))


def test_roundtrip_finds_dead_end_and_returns():
    """Ideal-pose round trip with a simple waypoint follower on a generated track."""
    from hamster_super.config import DomainRandomization
    from hamster_super.env.hamster_env import HamsterMazeEnv
    cfg = EnvConfig(course_prob=0.0, lap_prob=0.0, roundtrip_prob=1.0, dr=DomainRandomization(enabled=False))
    env = HamsterMazeEnv(cfg, level=2, seed=1)
    k = cfg.camera.n_rays + 2
    wins = 0
    for _ in range(4):
        obs, _ = env.reset(options={"level": 2})
        done = False
        while not done:
            ang = math.atan2(obs[k + 1], obs[k])
            fwd = max(0.0, math.cos(ang)) ** 3 * 0.9
            turn = float(np.clip(1.5 * ang, -0.7, 0.7))
            obs, _, te, tr, info = env.step((fwd - turn, fwd + turn))
            done = te or tr
        wins += info["success"] and info["turned"]
    assert wins == 4


def test_track_generator_is_a_single_path():
    from hamster_super.maps.generator import generate_track
    rng = np.random.default_rng(0)
    for level in range(6):
        m = generate_track(rng, level)
        assert m.kind == "roundtrip"
        path = m.shortest_path()
        for c in path:       # every corridor cell has at most two open sides
            assert len(list(m.open_neighbors(c))) <= 2
        assert len(list(m.open_neighbors(m.start))) == 1
        assert len(list(m.open_neighbors(m.goal))) == 1


def test_circuit_generator_is_one_closed_lane():
    from hamster_super.maps.generator import generate_loop, lap_order
    rng = np.random.default_rng(0)
    for level in range(9):
        m = generate_loop(rng, level)
        assert m.kind == "lap"
        order = lap_order(m)
        assert order[-1] == m.goal                         # the lap closes back onto the start
        for c in order:                                    # one lane: exactly two open sides per cell
            assert len(list(m.open_neighbors(c))) == 2
        assert len(set(order)) == len(order)


def test_one_lap_with_waypoint_follower():
    from hamster_super.config import DomainRandomization
    from hamster_super.env.hamster_env import HamsterMazeEnv
    cfg = EnvConfig(course_prob=0.0, lap_prob=1.0, dr=DomainRandomization(enabled=False))
    env = HamsterMazeEnv(cfg, level=3, seed=2)
    k = cfg.camera.n_rays + 2
    for _ in range(3):
        obs, _ = env.reset(options={"level": 3})
        done = False
        while not done:
            ang = math.atan2(obs[k + 1], obs[k])
            fwd = max(0.0, math.cos(ang)) ** 3 * 0.9
            turn = float(np.clip(1.5 * ang, -0.7, 0.7))
            obs, _, te, tr, info = env.step((fwd - turn, fwd + turn))
            done = te or tr
        assert info["success"] and info["lap_frac"] >= 0.9
