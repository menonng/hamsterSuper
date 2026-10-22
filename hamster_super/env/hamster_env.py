"""Gymnasium environment: Hamster maze race from start cell to goal cell."""
from __future__ import annotations

import dataclasses
import math

import gymnasium as gym
import numpy as np

from ..config import GRID_SPACING, EnvConfig
from ..maps.generator import generate
from ..maps.grid_map import GridMap, cell_center, cell_of
from ..nav.mapper import BeliefMap
from ..nav.planner import Planner
from ..nav.stack import NavStack, build_obs, cam_origin, obs_dim
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
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
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
            self.map = generate(self.rng, self.cur_level)
        m = self.map
        self.boxes = m.wall_boxes()
        self.bounds = m.cell_bounds()
        heading = m.start_heading
        if heading is None:
            heads = m.free_headings(m.start) or [0.0]
            heading = heads[int(self.rng.integers(len(heads)))]
        jitter = self.cfg.dr.enabled
        pos = cell_center(m.start) + (self.rng.uniform(-1, 1, 2) if jitter else 0)
        heading = heading + (self.rng.uniform(-0.25, 0.25) if jitter else 0)
        self.robot.reset((pos[0], pos[1], wrap(heading)))
        path = m.shortest_path()
        self.opt_cells = len(path) - 1 if path else 0
        self.max_steps = self.cfg.steps_per_cell * max(1, self.opt_cells) + 100
        self.t = 0
        self.prev_action = np.zeros(2)
        self.collisions = 0
        self.pose_err = np.zeros(3)
        self.belief = BeliefMap(self.bounds, prior=m if self.cfg.known_map else None)
        self.planner = Planner(self.bounds, m.goal)
        self.use_stack = self.full_stack or (self.rng.random() < self.cfg.full_stack_prob)
        if self.use_stack:
            self.stack = NavStack(self.cfg, self.bounds, self.robot.pose, m.goal, self.rays.r_min,
                                  prior=m if self.cfg.known_map else None, rng=self.rng)
            self.stack.odom.spec = self._calibrated_spec()
        obs = self._observe()
        self.prev_pose = self.robot.pose
        return obs, {}

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
            return obs
        est = self._est_pose()
        self.est = est
        self.belief.update(cam_origin(cfg, est), est[2], self.rays.angles, rays,
                           cfg.camera.max_range, self.rays.r_min)
        self.belief.update_ir(est, ir, cfg.robot)
        self.planner.update(self.belief.walls, cell_of(est))
        self.wp = self.planner.waypoints(est)
        return build_obs(cfg, rays, ir, est, self.wp[0], self.wp[1], self.prev_action)

    # ------------------------------------------------------------------- step
    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        hit = self.robot.step(a, cfg.dt, cfg.substeps, self.boxes)
        self.t += 1
        self.collisions += int(hit)
        da = a - self.prev_action
        self.prev_action = a
        obs = self._observe()
        # progress on the (possibly updated) belief map: both potentials use the new map
        pose = self.robot.pose
        phi_prev = self.planner.potential(self.prev_pose)
        phi = self.planner.potential(pose)
        self.prev_pose = pose
        progress = float(np.clip(phi_prev - phi, -5, 5))
        reward = cfg.r_progress * progress + cfg.r_time + cfg.r_smooth * float(da @ da)
        if hit:
            reward += cfg.r_collision
        back = -min(0.0, float(a[0] + a[1]) / 2)
        reward += cfg.r_reverse * back
        terminated = truncated = False
        info = {}
        goal_d = float(np.linalg.norm(pose[:2] - cell_center(self.map.goal)))
        if cell_of(pose) == self.map.goal and goal_d < 3.5:
            reward += cfg.r_goal
            terminated = True
            info["success"] = True
        elif not self._inside(pose):
            reward += cfg.r_out
            terminated = True
            info["success"] = False
        elif self.t >= self.max_steps:
            truncated = True
            info["success"] = False
        if terminated or truncated:
            info.update(steps=self.t, time_s=self.t * cfg.dt, collisions=self.collisions,
                        opt_cells=self.opt_cells, level=self.cur_level)
        return obs, reward, terminated, truncated, info

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
