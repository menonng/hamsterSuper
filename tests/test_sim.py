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


def test_walls_block_without_pushing():
    """Driving at an angle into a wall: the robot stops at the wall instead of
    being pushed out sideways or sliding along it."""
    boxes = np.array([[10.0, -20.0, 10.5, 20.0]])
    r = SimRobot(RobotSpec(), DomainRandomization(enabled=False), np.random.default_rng(0))
    r.reset((0.0, 0.0, math.radians(40)))
    for _ in range(40):
        r.step((1.0, 1.0), 0.1, 4, boxes)
    y_at_wall = r.y
    for _ in range(20):
        hit = r.step((1.0, 1.0), 0.1, 4, boxes)
    assert hit
    assert abs(r.y - y_at_wall) < 0.05          # no sliding along the wall
    assert abs(r.th - math.radians(40)) < 1e-6  # not turned by the wall


def test_rewards_favour_backing_off_a_wall():
    """Grinding against a wall costs more and more; reversing right after
    touching a wall is not penalized."""
    from hamster_super.config import DomainRandomization
    env = HamsterMazeEnv(EnvConfig(dr=DomainRandomization(enabled=False)), level=2, seed=0)
    env.reset()
    th = env.robot.th
    # drive straight until blocked, then keep pushing
    rewards = []
    for _ in range(200):
        _, r, te, tr, _ = env.step((1.0, 1.0, 0.0))
        if env.blocked_run >= 1:
            rewards.append(r)
        if len(rewards) >= 6 or te or tr:
            break
    if len(rewards) >= 6:
        assert rewards[5] < rewards[1]                       # escalating
        _, r_back, _, _, _ = env.step((-1.0, -1.0, 0.0))
        assert env.t - env.last_hit_t <= env.cfg.reverse_free_steps


def test_moving_course_parts_never_close_the_course_or_hit_the_robot():
    from collections import deque
    from hamster_super.maps.generator import generate_wide_course
    from hamster_super.maps.grid_map import NEIGHBORS, edge_between
    from hamster_super.sim.dynamics import RotatingWall, SlidingBlock, make_dynamics
    G = 9.0
    for k in range(6):
        m = None
        s = k
        while m is None:
            m = generate_wide_course(np.random.default_rng(s), 5, open_course=True)
            s += 100
        dyn = make_dynamics(m, np.random.default_rng(k))
        assert dyn.elements
        for t in range(400):                       # 40 s of motion, robot far away
            dyn.step(0.1)
            if t % 20:
                continue
            walls, closed = set(m.walls), set()
            for e in dyn.elements:
                if isinstance(e, RotatingWall):
                    c, s_ = math.cos(e.ang), math.sin(e.ang)
                    if abs(c) < 1e-6 or abs(s_) < 1e-6:          # at rest on the lattice
                        a = (round(e.pivot[0] / G), round(e.pivot[1] / G))
                        b = (a[0] + round(c), a[1] + round(s_))
                        walls.add(tuple(sorted((a, b))))
                    else:                                       # swinging: its quadrant cell is blocked
                        closed.add((int((e.pivot[0] + 4.5 * c) // G), int((e.pivot[1] + 4.5 * s_) // G)))
                else:
                    closed.add((int(e.pos[0] // G), int(e.pos[1] // G)))
            seen, q = {m.start}, deque([m.start])
            while q:
                c = q.popleft()
                for dx, dy in NEIGHBORS:
                    n = (c[0] + dx, c[1] + dy)
                    if n not in seen and n not in closed and 0 <= n[0] < 14 and 0 <= n[1] < 14 \
                            and edge_between(c, n) not in walls:
                        seen.add(n)
                        q.append(n)
            assert m.goal in seen, (m.name, t)


def test_robot_never_inside_moving_parts():
    from hamster_super.config import DomainRandomization
    cfg = EnvConfig(dynamic_prob=1.0, dr=DomainRandomization(enabled=False))
    env = HamsterMazeEnv(cfg, level=5, seed=3)
    rng = np.random.default_rng(0)
    spec = RobotSpec()
    worst = 0.0
    for _ in range(3):
        env.reset(options={"level": 5})
        for _ in range(400):
            _, _, te, tr, _ = env.step(rng.uniform(-1, 1, 3))
            worst = max(worst, rect_penetration(env.robot.pose[:2], env.robot.th, spec.body_length / 2,
                                                spec.body_width / 2, env.boxes)[0])
            if te or tr:
                break
    assert worst < 0.05


def test_swinging_wall_waits_for_the_robot():
    from hamster_super.sim.dynamics import RotatingWall
    w = RotatingWall((18.0, 18.0), 0.0, -math.pi / 2, np.random.default_rng(0), rest=(0.1, 0.1))
    w.ang = w.target = 0.0
    robot = (22.5, 13.5, 0.0)            # in the quadrant the wall sweeps through
    spec = RobotSpec()
    for _ in range(50):
        w.step(0.1, robot, (spec.body_length / 2, spec.body_width / 2))
        assert rect_penetration(np.array(robot[:2]), 0.0, spec.body_length / 2, spec.body_width / 2, w.boxes())[0] == 0
    assert w.ang > -math.pi / 2 + 0.1    # it could not finish the swing
