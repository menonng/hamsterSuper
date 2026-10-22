import math

import numpy as np

from hamster_super.config import DomainRandomization, EnvConfig, RobotSpec
from hamster_super.env.hamster_env import HamsterMazeEnv
from hamster_super.maps.parser import load_map
from hamster_super.robot.perception import Perception
from hamster_super.sim.camera import Renderer
from hamster_super.sim.geometry import circle_penetration, raycast
from hamster_super.sim.robot import SimRobot
from hamster_super.sim.sensors import DepthRays


def test_raycast_hits_box_face():
    boxes = np.array([[10.0, -5.0, 10.5, 5.0]])
    d = raycast(np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), boxes, 50.0)
    assert d[0] == 10.0 and d[1] == 50.0


def test_robot_cannot_pass_through_wall():
    boxes = np.array([[10.0, -5.0, 10.5, 5.0]])
    r = SimRobot(RobotSpec(), DomainRandomization(enabled=False), np.random.default_rng(0))
    r.reset((0.0, 0.0, 0.0))
    hit = False
    for _ in range(30):
        hit |= r.step((1.0, 1.0), 0.1, 4, boxes)
    assert hit
    assert r.x < 10.0 - RobotSpec().collision_radius + 1e-6
    assert circle_penetration(r.pose[:2], RobotSpec().collision_radius - 1e-3, boxes)[0] == 0


def test_straight_drive_speed():
    r = SimRobot(RobotSpec(), DomainRandomization(enabled=False), np.random.default_rng(0))
    r.reset((0.0, 0.0, 0.0))
    for _ in range(20):
        r.step((1.0, 1.0), 0.1, 4, np.zeros((0, 4)))
    assert abs(r.x - RobotSpec().max_speed * 2.0) < 2.0   # ~2 s at max speed minus spin-up


def test_perception_matches_sim_rays():
    cfg = EnvConfig()
    m = load_map("maps/example_02_maze.txt")
    boxes = m.wall_boxes()
    ren, rays, per = Renderer(cfg.camera), DepthRays(cfg.camera), Perception(cfg.camera)
    rng = np.random.default_rng(3)
    errs = []
    while len(errs) < 40:
        pose = (rng.uniform(3, 87), rng.uniform(3, 69), rng.uniform(-math.pi, math.pi))
        if circle_penetration(np.array(pose[:2]), 2.3, boxes)[0] > 0:
            continue
        est = per.rays(ren.render(pose, boxes, rng))
        tru = np.clip(rays.true_ranges(pose, boxes), rays.r_min, cfg.camera.max_range)
        errs.append(np.abs(est - tru))
    e = np.concatenate(errs)
    assert np.median(e) < 0.5
    assert (e > 2.0).mean() < 0.05


def test_env_api():
    env = HamsterMazeEnv(level=2, seed=0)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    for _ in range(50):
        obs, r, te, tr, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if te or tr:
            assert "success" in info
            env.reset()


def test_expert_reaches_goal_known_map():
    """A simple waypoint-following controller must solve a map using the env's
    planner observations; guards the observation/waypoint geometry."""
    cfg = EnvConfig(known_map=True, dr=DomainRandomization(enabled=False))
    env = HamsterMazeEnv(cfg, seed=0)
    obs, _ = env.reset(options={"map": load_map("maps/example_01_s_curve.txt")})
    k = cfg.camera.n_rays + 2
    for _ in range(600):
        wx, wy = obs[k] , obs[k + 1]
        ang = math.atan2(wy, wx)
        fwd = max(0.0, math.cos(ang)) * 0.6
        turn = np.clip(1.5 * ang, -0.6, 0.6)
        obs, r, te, tr, info = env.step((fwd - turn, fwd + turn))
        if te or tr:
            break
    assert info.get("success"), info
