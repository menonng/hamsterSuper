"""Gymnasium environment: Hamster maze race from start cell to goal cell."""
from __future__ import annotations

import dataclasses
import math

import gymnasium as gym
import numpy as np

from ..config import GRID_SPACING, EnvConfig
from ..maps.generator import COURSE_LEVELS, generate, generate_course, generate_loop, generate_track, lap_order
from ..maps.grid_map import GridMap, cell_center, cell_of, edge_between
from ..nav.mapper import BeliefMap
from ..nav.planner import Planner
from ..nav.racing_line import RacingLine
from ..nav.roundtrip import DONE, RoundTrip
from ..nav.stack import NavStack, build_obs, cam_origin, front_distance, obs_dim
from ..sim.dynamics import CourseDynamics, make_dynamics
from ..sim.robot import SimRobot, wrap
from ..sim.sensors import DepthRays

G = GRID_SPACING


class HamsterMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: EnvConfig | None = None, maps: list[GridMap] | None = None,
                 level: int = 0, seed: int | None = None, full_stack: bool = False):
        self.cfg = cfg or EnvConfig()
        self.maps = maps            # fixed map pool; None = random generated maps
        self.level = level
        self.full_stack = full_stack
        self.rng = np.random.default_rng(seed)
        self.robot = SimRobot(self.cfg.robot, self.cfg.dr, self.rng)
        self.rays = DepthRays(self.cfg.camera)
        n = obs_dim(self.cfg)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        # left wheel, right wheel, racing-line side (+1 left ... -1 right of the lane, everywhere)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.map: GridMap | None = None
        self.cur_level = level

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.robot.rng = self.rng
        options = options or {}
        if "map" in options:
            self.map = options["map"]
            self.cur_level = -1
        elif self.maps:
            self.map = self.maps[int(self.rng.integers(len(self.maps)))]
            self.cur_level = -1
        else:
            self.cur_level = options.get("level", self._sample_level())
            u = self.rng.random()
            if u < self.cfg.course_prob:
                self.map = generate_course(self.rng, min(self.cur_level, COURSE_LEVELS - 1))
            elif (u := u - self.cfg.course_prob) < self.cfg.lap_prob:
                self.map = generate_loop(self.rng, self.cur_level)
            elif u < self.cfg.lap_prob + self.cfg.roundtrip_prob:
                self.map = generate_track(self.rng, self.cur_level)
            else:
                self.map = generate(self.rng, self.cur_level)
        m = self.map
        self.static_boxes = m.wall_boxes()
        dyn = options.get("dynamic")
        if dyn is None:
            dyn = self.rng.random() < self.cfg.dynamic_prob
        self.dynamics = make_dynamics(m, self.rng) if dyn else CourseDynamics()
        self._update_boxes()
        self.bounds = m.cell_bounds()
        heading = m.start_heading
        if heading is None:
            heads = m.free_headings(m.start) or [0.0]
            heading = heads[int(self.rng.integers(len(heads)))]
        jitter = self.cfg.dr.enabled
        pos = cell_center(m.start) + (self.rng.uniform(-1, 1, 2) if jitter else 0)
        heading = heading + (self.rng.uniform(-0.25, 0.25) if jitter else 0)
        self.robot.reset((pos[0], pos[1], wrap(heading)))
        if self.cfg.dr.enabled:
            self.rays.set_mount_error(self.rng.uniform(-1, 1) * self.cfg.dr.cam_pitch_err,
                                      self.rng.uniform(-1, 1) * self.cfg.dr.cam_height_err)
        else:
            self.rays.set_mount_error(0.0, 0.0)
        path = m.shortest_path()
        self.is_rt = m.kind == "roundtrip"
        self.is_lap = m.kind == "lap"
        self.is_oneway = m.kind == "oneway"
        one_way = len(path) - 1 if path else 0
        if self.is_lap or self.is_oneway:
            self.lap = list(m.course) if m.course else lap_order(m)
            one_way = len(self.lap)
            # lap progress is counted in 2x2 blocks: wide sections need not touch every cell
            unit = (lambda c: (c[0] // 2, c[1] // 2)) if m.course else (lambda c: c)
            self.lap_unit = unit
            self.lap_units = {unit(c) for c in self.lap}
        self.opt_cells = 2 * one_way if self.is_rt else one_way       # cells driven by an ideal run
        self.max_steps = self.cfg.steps_per_cell * max(1, self.opt_cells) + (200 if (self.is_rt or self.is_lap) else 100)
        # rewards use the true map; the robot itself only knows what it has seen
        self.true_phase = "out"
        self.true_to_goal = Planner(self.bounds, m.goal)
        if self.is_lap:
            # cut the loop between the last cell and the start: the lap becomes a
            # path start -> ... -> last cell, then one more cell into the start
            self.true_to_goal.update(set(m.walls) | {edge_between(m.start, m.goal)})
            self.true_visited = {self.lap_unit(m.start)}
        else:
            self.true_to_goal.update(m.walls)
        if self.is_rt:
            self.true_to_start = Planner(self.bounds, m.start)
            self.true_to_start.update(m.walls)
            self.leg = 9.0 * one_way
        self.t = 0
        self.prev_action = np.zeros(3)
        self.prev_rays = None
        self.blocked_run = 0
        self.last_hit_t = -10 ** 6
        self.collisions = 0
        self.pose_err = np.zeros(3)
        self._last_th = self.robot.th
        self.belief = BeliefMap(self.bounds, prior=m if self.cfg.known_map else None)
        race = self.is_rt or self.is_lap or self.is_oneway
        # the robot is not told whether it is a lap or a one-way course ("race" mode)
        race_mode = "roundtrip" if self.is_rt else "race"
        self.planner = Planner(self.bounds, m.start if race else m.goal)
        self.rt = (RoundTrip(m.start, self.bounds, mode=race_mode, start_heading=m.start_heading)
                   if race else None)
        if self.rt is not None and race_mode == "race":
            self.belief.decay = self.cfg.race_map_decay
            if self.cfg.race_border:
                self.belief.add_border()
            self.rt.recent_window = self.cfg.race_recent_steps
            x0, y0, x1, y1 = self.bounds
            self.rt.min_lap_cells = max(4, int(0.6 * (x1 - x0 + 1) * (y1 - y0 + 1)))   # courses fill the arena
            self.rt.min_coverage = self.cfg.race_min_coverage
        self.line = RacingLine(robot_radius=self.cfg.robot.collision_radius)
        self.use_stack = self.full_stack or (self.rng.random() < self.cfg.full_stack_prob)
        if self.use_stack:
            self.stack = NavStack(self.cfg, self.bounds, self.robot.pose, None if race else m.goal,
                                  self.rays.r_min, race=race_mode,
                                  prior=m if self.cfg.known_map else None, rng=self.rng)
            self.stack.set_odometry_spec(self._calibrated_spec())
        obs = self._observe()
        self.prev_pose = self.robot.pose
        return obs, {}

    def _update_boxes(self):
        self.dyn_boxes = self.dynamics.boxes()
        self.boxes = np.concatenate([self.static_boxes, self.dyn_boxes]) if len(self.dyn_boxes) else self.static_boxes

    def _calibrated_spec(self):
        """Odometry parameters as a user would measure them with the
        calibration tool: the true robot's values plus a small error."""
        r, e = self.robot, self.cfg.dr.calib_error if self.cfg.dr.enabled else 0.0
        jit = lambda: 1 + self.rng.normal(0, e)
        return dataclasses.replace(self.cfg.robot, max_speed=r.vmax * float(np.mean(r.gain)) * jit(),
                                   track_width=r.track * jit(), motor_tau=r.tau)

    def _sample_level(self) -> int:
        """Curriculum: mostly the current max level, sometimes easier ones."""
        if self.level <= 0 or self.rng.random() < 0.7:
            return self.level
        return int(self.rng.integers(0, self.level))

    # -------------------------------------------------------------- observing
    def _est_pose(self):
        dr = self.cfg.dr
        if not dr.enabled:
            return self.robot.pose
        # bounded random-walk localization error
        self.pose_err += self.rng.normal(0, [0.3, 0.3, 0.01])
        self.pose_err[:2] = np.clip(self.pose_err[:2], -dr.pose_err_pos, dr.pose_err_pos)
        self.pose_err[2] = np.clip(self.pose_err[2], -dr.pose_err_yaw, dr.pose_err_yaw)
        return self.robot.pose + self.pose_err

    def _observe(self):
        cfg = self.cfg
        pose = self.robot.pose
        self.last_rays = rays = self.rays.measure(pose, self.boxes, cfg.dr, self.rng)
        ir = self.robot.ir(self.boxes)
        if self.use_stack:
            obs = self.stack.step(rays, ir, self.prev_action)
            self.belief, self.planner = self.stack.belief, self.stack.planner
            self.est = self.stack.pose.copy()
            self.wp = self.stack.wp
            self.line = self.stack.line
            self.rt = self.stack.roundtrip
            return obs
        est = self._est_pose()
        self.est = est
        spin = abs(float((self.robot.th - self._last_th + math.pi) % (2 * math.pi) - math.pi))
        self._last_th = self.robot.th
        if spin < 0.12:   # same rule as NavStack: no mapping while spinning fast
            self.belief.update(cam_origin(cfg, est), est[2], self.rays.angles, rays,
                               cfg.camera.max_range, self.rays.r_min)
            self.belief.update_ir(est, ir, cfg.robot)
        if self.rt is not None:
            self.planner.set_goal(self.rt.goal(cell_of(est), est, est[2], self.belief.walls,
                                               seen=self.belief.seen, free=self.belief.free_edges,
                                               front=front_distance(rays, ir), strong=self.belief.strong_walls()))
        self.planner.update(self.belief.walls, cell_of(est), free=self.belief.free_edges)
        # the localization error the planner would report (in the full stack: particle spread)
        self.line.update(self.planner, est, sigma=float(np.linalg.norm(self.pose_err[:2])) * 0.5 + 0.2,
                         bias=float(self.prev_action[2]))
        self.wp = self.line.waypoints()
        obs = build_obs(cfg, rays, ir, est, self.wp[0], self.wp[1], self.prev_action[:2], self.prev_rays)
        self.prev_rays = np.asarray(rays, float).copy()
        return obs

    # ------------------------------------------------------------------- step
    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if a.shape[0] < 3:
            a = np.concatenate([a, np.zeros(3 - a.shape[0])])
        if self.dynamics:      # moving course parts (they wait instead of pushing the robot)
            r = cfg.robot
            self.dynamics.step(cfg.dt, self.robot.pose, (r.body_length / 2, r.body_width / 2))
            self._update_boxes()
        hit = self.robot.step(a[:2], cfg.dt, cfg.substeps, self.boxes)
        self.t += 1
        self.collisions += int(hit)
        da = (a - self.prev_action)[:2]
        self.prev_action = a
        obs = self._observe()
        pose = self.robot.pose
        phi_prev, phi = self._potential(self.prev_pose), self._potential(pose)
        self.prev_pose = pose
        progress = float(np.clip(phi_prev - phi, -5, 5))
        reward = cfg.r_progress * progress + cfg.r_time + cfg.r_smooth * float(da @ da)
        # pushing on against a wall costs more the longer it goes on, and backing
        # off a wall is free: the policy learns to reverse out instead of grinding
        self.blocked_run = self.blocked_run + 1 if hit else 0
        if hit:
            self.last_hit_t = self.t
            reward += cfg.r_collision * (1 + cfg.r_collision_growth * min(self.blocked_run - 1, 8))
        back = -min(0.0, float(a[0] + a[1]) / 2)
        if self.t - self.last_hit_t > cfg.reverse_free_steps:
            reward += cfg.r_reverse * back
        terminated = truncated = False
        info = {}
        m = self.map
        at = lambda c: cell_of(pose) == tuple(c) and float(np.linalg.norm(pose[:2] - cell_center(c))) < 3.5
        c = cell_of(pose)
        if self.is_rt and self.true_phase == "out" and c == tuple(m.goal):
            self.true_phase = "back"                      # reached the dead end
            reward += 0.5 * cfg.r_goal
        lap_done = False
        if self.is_lap:
            if c != tuple(m.start):
                self.true_visited.add(self.lap_unit(c))
            # back in the start cell after (nearly) the whole circuit
            lap_done = c == tuple(m.start) and len(self.true_visited) >= 0.9 * len(self.lap_units)
            # the finish line is the start cell's edge: a robot that calls the lap done
            # within one cell of it (localization tolerance) has finished
            if self.rt is not None and self.rt.phase == DONE and len(self.true_visited) >= 0.9 * len(self.lap_units) \
                    and abs(c[0] - m.start[0]) + abs(c[1] - m.start[1]) <= 1:
                lap_done = True
        if self.is_oneway and c == tuple(m.goal):
            lap_done = True                               # reached the end of the one-way course
        if (not (self.is_rt or self.is_lap or self.is_oneway) and at(m.goal)) or lap_done or \
                (self.is_rt and self.true_phase == "back" and at(m.start)):
            reward += cfg.r_goal
            terminated = True
            info["success"] = True
        elif not self._inside(pose):
            reward += cfg.r_out
            terminated = True
            info["success"] = False
        elif (self.is_rt or self.is_lap or self.is_oneway) and self.rt is not None and self.rt.phase == DONE:
            terminated = True                             # robot thinks it is done but it is not
            info["success"] = False
        elif self.t >= self.max_steps:
            truncated = True
            info["success"] = False
        if terminated or truncated:
            info.update(steps=self.t, time_s=self.t * cfg.dt, collisions=self.collisions,
                        opt_cells=self.opt_cells, level=self.cur_level, roundtrip=self.is_rt,
                        lap=self.is_lap, turned=self.true_phase == "back",
                        lap_frac=(len(self.true_visited) / len(self.lap_units)) if self.is_lap else None,
                        oneway=self.is_oneway)
        return obs, reward, terminated, truncated, info

    def _potential(self, pose) -> float:
        """True remaining distance (cm) of the whole task from `pose`."""
        if self.is_lap:
            if cell_of(pose) == tuple(self.map.start) and len(self.true_visited) > 0.5 * len(self.lap):
                return 0.0                                # crossed the finish line
            return self.true_to_goal.potential(pose) + 9.0
        if not self.is_rt:
            return self.true_to_goal.potential(pose)
        if self.true_phase == "out":
            return self.true_to_goal.potential(pose) + self.leg
        return self.true_to_start.potential(pose)

    def _inside(self, pose):
        x0, y0, x1, y1 = self.bounds
        return x0 * G - 4.5 <= pose[0] <= (x1 + 1) * G + 4.5 and y0 * G - 4.5 <= pose[1] <= (y1 + 1) * G + 4.5

    # ----------------------------------------------------------------- render
    def render_topdown(self, scale: int = 6) -> np.ndarray:
        """Simple top-down RGB image (for debugging/videos)."""
        x0, y0, x1, y1 = self.bounds
        W = int((x1 - x0 + 1) * G * scale) + 1
        H = int((y1 - y0 + 1) * G * scale) + 1
        img = np.full((H, W, 3), 245, np.uint8)

        def px(p):
            return int((p[0] - x0 * G) * scale), int(H - 1 - (p[1] - y0 * G) * scale)
        for b in self.boxes:
            u0, v1 = px((b[0], b[1]))
            u1, v0 = px((b[2], b[3]))
            img[max(v0, 0):v1 + 1, max(u0, 0):u1 + 1] = (51, 85, 187)
        for c, col in ((self.map.goal, (255, 0, 69)), (self.map.start, (255, 204, 17))):
            u, v = px(cell_center(c))
            img[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5] = col
        u, v = px(self.robot.pose)
        img[max(v - 6, 0):v + 7, max(u - 6, 0):u + 7] = (255, 126, 0)
        return img
