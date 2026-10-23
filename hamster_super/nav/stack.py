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
from .localizer import LatticeParticleFilter, Odometry, wall_orientation
from .mapper import BeliefMap
from .planner import Planner
from .racing_line import RacingLine
from .roundtrip import DONE, RoundTrip

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


def front_distance(rays, ir, r_min=None) -> float:
    """Nearest wall straight ahead (cm) from the middle camera rays and the IR."""
    rays = np.asarray(rays)
    k = len(rays) // 2
    d = float(np.min(rays[k - 1:k + 2]))
    if len(ir) and float(np.max(ir)) > 0.2:
        d = min(d, 3.0)
    return d


class NavStack:
    def __init__(self, cfg: EnvConfig, bounds, start_pose, goal, r_min, prior: GridMap | None = None,
                 localize: bool = True, delay_steps: int = 1, rng=None, race: str = "race"):
        self.cfg = cfg
        self.angles = cfg.camera.ray_angles()
        self.r_min = np.asarray(r_min)
        self.belief = BeliefMap(bounds, prior=prior)
        # goal=None: race without a destination - "lap" (one lap of a closed circuit)
        # or "roundtrip" (to the dead end of a single-path course and back)
        self.roundtrip = (RoundTrip(cell_of(start_pose), bounds, mode=race, start_heading=float(start_pose[2]))
                          if goal is None else None)
        self.planner = Planner(bounds, goal if goal is not None else cell_of(start_pose))
        if self.roundtrip is not None and race == "race":
            # racing: a local map (old, drift-corrupted walls fade) and recent-visit memory
            self.belief.decay = cfg.race_map_decay
            if cfg.race_border:
                self.belief.add_border()
            self.roundtrip.recent_window = cfg.race_recent_steps
            x0, y0, x1, y1 = bounds
            self.roundtrip.min_lap_cells = max(4, int(0.6 * (x1 - x0 + 1) * (y1 - y0 + 1)))   # courses fill the arena
            self.roundtrip.min_coverage = int(cfg.race_min_coverage * (x1 - x0 + 1) * (y1 - y0 + 1))
        self.odom = Odometry(cfg.robot, start_pose, delay_steps=delay_steps)
        self.pf = LatticeParticleFilter(start_pose, rng=rng) if localize else None
        self.prev_action = np.zeros(2)
        self.wp = (None, None)
        self.loc_range = 35.0
        self.map_max_spin = 0.12             # rad per step
        self.line = RacingLine(robot_radius=cfg.robot.collision_radius)

    @property
    def pose(self):
        return self.odom.pose

    def step(self, rays, ir, last_action) -> np.ndarray:
        cfg = self.cfg
        before = self.odom.pose.copy()
        self.odom.step(last_action, cfg.dt, cfg.substeps)
        mr = cfg.camera.max_range
        if self.pf is not None:
            d = before[:2] - self.odom.pose[:2]
            dist = float(np.hypot(*d)) * (1 if np.dot(-d, [math.cos(before[2]), math.sin(before[2])]) >= 0 else -1)
            dth = float((self.odom.pose[2] - before[2] + math.pi) % (2 * math.pi) - math.pi)
            self.pf.predict(dist, dth)
            pts = self.wall_points(rays, ir)
            wa = wall_orientation(pts[:self._n_cam_pts])
            # while spinning fast, latency makes the predicted heading unreliable
            # and the mod-90 fix could snap to the wrong quadrant
            if wa is not None and abs(dth) < 0.15:
                self.pf.correct_heading(wa)
            self.pf.update(pts, self.belief)
            self.odom.pose[:] = self.pf.pose
        pose = self.odom.pose
        # while spinning fast the heading is least certain (latency): skip mapping
        # rather than draw walls in the wrong place
        spin = abs(float((pose[2] - before[2] + math.pi) % (2 * math.pi) - math.pi))
        if spin < self.map_max_spin:
            self.belief.update(cam_origin(cfg, pose), pose[2], self.angles, rays, mr, self.r_min)
            self.belief.update_ir(pose, ir, cfg.robot)
        if self.roundtrip is not None:
            self.planner.set_goal(self.roundtrip.goal(cell_of(pose), pose, pose[2], self.belief.walls,
                                                      seen=self.belief.seen, free=self.belief.free_edges,
                                                      front=front_distance(rays, ir)))
        self.planner.update(self.belief.walls, cell_of(pose))
        bias = float(last_action[2]) if len(last_action) > 2 else 0.0      # learned inside/outside line choice
        self.line.update(self.planner, pose, sigma=self.pf.spread if self.pf is not None else 0.0, bias=bias)
        self.wp = self.line.waypoints()
        self.prev_action = np.asarray(last_action, dtype=float)
        return build_obs(cfg, rays, ir, pose, self.wp[0], self.wp[1], self.prev_action[:2])

    def wall_points(self, rays, ir) -> np.ndarray:
        """Wall-centerline points in the robot frame from camera rays + IR."""
        cfg = self.cfg
        ok = (rays < self.loc_range) & (rays > self.r_min + 0.3)
        d = rays[ok] + 0.25
        a = self.angles[ok]
        f = cfg.camera.mount_forward
        pts = [np.stack([f + d * np.cos(a), d * np.sin(a)], axis=1)]
        self._n_cam_pts = len(d)
        rs = cfg.robot
        for val, sign in zip(ir, (1, -1)):
            if val > 0.15:
                dd = (1 - val) * rs.ir_range + 0.25
                pts.append(np.array([[rs.ir_offset + dd * math.cos(sign * rs.ir_angle),
                                      dd * math.sin(sign * rs.ir_angle)]]))
        return np.concatenate(pts)

    def reached_goal(self, tol: float = 3.5) -> bool:
        from ..maps.grid_map import cell_center
        if self.roundtrip is not None:
            return self.roundtrip.phase == DONE
        return float(np.linalg.norm(self.pose[:2] - cell_center(self.planner.goal))) < tol
