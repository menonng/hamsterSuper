"""Observation builder and the full navigation stack used on the real robot.

During RL training the environment uses the same mapper/planner but feeds
them the true pose plus a bounded random error (fast). On the real robot
(and in `--full-stack` evaluation) `NavStack` runs dead-reckoning odometry
+ grid-snap localization instead.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import GRID_SPACING, EnvConfig
from ..maps.grid_map import GridMap, cell_of
from .localizer import GridSnapLocalizer, Odometry
from .mapper import BeliefMap
from .planner import Planner

WP_CLIP = 3.0


def obs_dim(cfg: EnvConfig) -> int:
    return cfg.camera.n_rays + 2 + 4 + 2


def to_local(pose, pt) -> np.ndarray:
    x, y, th = pose
    dx, dy = pt[0] - x, pt[1] - y
    c, s = math.cos(th), math.sin(th)
    return np.array([c * dx + s * dy, -s * dx + c * dy])


def build_obs(cfg: EnvConfig, rays, ir, pose_est, wp1, wp2, prev_action) -> np.ndarray:
    w1 = np.clip(to_local(pose_est, wp1) / GRID_SPACING, -WP_CLIP, WP_CLIP)
    w2 = np.clip(to_local(pose_est, wp2) / GRID_SPACING, -WP_CLIP, WP_CLIP)
    return np.concatenate([np.asarray(rays) / cfg.camera.max_range, ir, w1, w2, prev_action]).astype(np.float32)


def cam_origin(cfg: EnvConfig, pose) -> np.ndarray:
    x, y, th = pose
    return np.array([x, y]) + cfg.camera.mount_forward * np.array([math.cos(th), math.sin(th)])


class NavStack:
    def __init__(self, cfg: EnvConfig, bounds, start_pose, goal, r_min, prior: GridMap | None = None,
                 localize: bool = True, delay_steps: int = 1):
        self.cfg = cfg
        self.angles = cfg.camera.ray_angles()
        self.r_min = np.asarray(r_min)
        self.belief = BeliefMap(bounds, prior=prior)
        self.planner = Planner(bounds, goal)
        self.odom = Odometry(cfg.robot, start_pose, delay_steps=delay_steps)
        self.loc = GridSnapLocalizer() if localize else None
        self.prev_action = np.zeros(2)
        self.wp = (None, None)

    @property
    def pose(self):
        return self.odom.pose

    def step(self, rays, ir, last_action) -> np.ndarray:
        cfg = self.cfg
        self.odom.step(last_action, cfg.dt, cfg.substeps)
        mr = cfg.camera.max_range
        valid = (rays < mr - 0.5) & (rays > self.r_min + 0.3) & (rays < 25.0)
        if self.loc is not None:
            self.odom.pose[:] = self.loc.correct(self.odom.pose, cam_origin(cfg, self.odom.pose),
                                                 self.angles, rays, valid)
        pose = self.odom.pose
        self.belief.update(cam_origin(cfg, pose), pose[2], self.angles, rays, mr, self.r_min)
        self.planner.update(self.belief.walls, cell_of(pose))
        self.wp = self.planner.waypoints(pose)
        self.prev_action = np.asarray(last_action, dtype=float)
        return build_obs(cfg, rays, ir, pose, self.wp[0], self.wp[1], self.prev_action)

    def reached_goal(self, tol: float = 3.5) -> bool:
        from ..maps.grid_map import cell_center
        return float(np.linalg.norm(self.pose[:2] - cell_center(self.planner.goal))) < tol
