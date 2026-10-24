import math

import numpy as np

from hamster_super.config import DomainRandomization, EnvConfig, RobotSpec
from hamster_super.env.hamster_env import HamsterMazeEnv
from hamster_super.maps.parser import load_map
from hamster_super.robot.perception import Perception
from hamster_super.sim.camera import Renderer
from hamster_super.sim.geometry import circle_penetration, raycast, rect_penetration
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
    spec = RobotSpec()
    assert r.x < 10.0 - spec.body_length / 2 + 1e-6
    assert rect_penetration(r.pose[:2], r.th, spec.body_length / 2 - 1e-3, spec.body_width / 2 - 1e-3, boxes)[0] == 0


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


def test_pitch_calibration_recovers_tilt():
    """Render a wall at a known distance with a tilted camera and recover the tilt."""
    import dataclasses
    from hamster_super.robot.calibrate import pitch_from_base_row, wall_base_row
    from hamster_super.sim.camera import wall_mask_bright
    cfg = EnvConfig()
    true_pitch = math.radians(37.0)
    spec = dataclasses.replace(cfg.camera, pitch=true_pitch)
    boxes = np.array([[30.0, -20.0, 30.5, 20.0]])          # wall face at x = 30
    pose = (30.0 - 10.0 - spec.mount_forward, 0.0, 0.0)     # lens 10 cm from the face
    img = Renderer(spec).render(pose, boxes)
    v = wall_base_row(wall_mask_bright(img), range(spec.width // 2 - 6, spec.width // 2 + 7))
    est = pitch_from_base_row(v, cfg.camera, 10.0)
    assert abs(est - true_pitch) < math.radians(1.0)


def test_perception_ignores_room_background():
    """Random rooms (people, white furniture) around the course must not create walls."""
    from hamster_super.sim.backgrounds import BackgroundBank
    cfg = EnvConfig()
    m = load_map("maps/example_02_maze.txt")
    boxes = m.wall_boxes()
    arena = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    ren, rays, per = Renderer(cfg.camera), DepthRays(cfg.camera), Perception(cfg.camera)
    bank = BackgroundBank(folder="/nonexistent")          # procedural rooms only
    rng = np.random.default_rng(4)
    errs = []
    while len(errs) < 40:
        pose = (rng.uniform(3, 87), rng.uniform(3, 69), rng.uniform(-math.pi, math.pi))
        if circle_penetration(np.array(pose[:2]), 2.3, boxes)[0] > 0:
            continue
        img = ren.render(pose, boxes, rng, background=bank.sample(rng), arena=arena)
        tru = np.clip(rays.true_ranges(pose, boxes), rays.r_min, cfg.camera.max_range)
        errs.append(np.abs(per.rays(img) - tru))
    e = np.concatenate(errs)
    assert np.median(e) < 0.5
    assert (e > 2.0).mean() < 0.06


def test_footprint_never_inside_walls():
    """The collision shape is the robot's drawn footprint (rectangle), also while
    turning against walls: no part of the model may end up inside a wall."""
    m = load_map("maps/example_02_maze.txt")
    boxes = m.wall_boxes()
    spec = RobotSpec()
    rng = np.random.default_rng(1)
    r = SimRobot(spec, DomainRandomization(), rng)
    r.reset((4.5, 4.5, 0.3))
    worst = 0.0
    for _ in range(1500):
        r.step(rng.uniform(-1, 1, 2) if rng.random() < 0.3 else (1.0, 0.8), 0.1, 4, boxes)
        worst = max(worst, rect_penetration(r.pose[:2], r.th, spec.body_length / 2, spec.body_width / 2, boxes)[0])
    assert worst < 0.05
