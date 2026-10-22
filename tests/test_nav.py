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
