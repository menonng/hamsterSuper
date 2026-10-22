"""Real-time control loop: sensors -> perception -> nav stack -> policy -> wheels."""
from __future__ import annotations

import threading
import time

import numpy as np

from ..config import EnvConfig
from ..maps.grid_map import GridMap, cell_center, cell_of
from ..nav.stack import NavStack
from ..sim.sensors import DepthRays
from .perception import Perception


class Runtime:
    def __init__(self, cfg: EnvConfig, hw, policy, start_pose, goal, bounds, prior: GridMap | None = None,
                 perception: Perception | None = None, realtime: bool = True, stale_frame_s: float = 0.5):
        self.cfg, self.hw, self.policy = cfg, hw, policy
        self.perception = perception or Perception(cfg.camera)
        self.r_min = DepthRays(cfg.camera).r_min
        self.start_pose = np.asarray(start_pose, dtype=float)
        self.bounds, self.prior = tuple(bounds), prior
        self.realtime, self.stale_frame_s = realtime, stale_frame_s
        self.lock = threading.Lock()
        self.running = False
        self.status = "idle"
        self._thread = None
        self.reset(goal)

    def reset(self, goal=None, start_pose=None):
        if goal is not None:
            self.goal = tuple(goal)
        if start_pose is not None:
            self.start_pose = np.asarray(start_pose, dtype=float)
        self.stack = NavStack(self.cfg, self.bounds, self.start_pose, self.goal, self.r_min, prior=self.prior)
        self.last_action = np.zeros(2)
        self.rays = np.full(self.cfg.camera.n_rays, self.cfg.camera.max_range)
        self.ir = np.zeros(2)
        self.frame = None
        self.t_start = None
        self.steps = 0
        self.loop_hz = 0.0
        self.finish_time = None

    # ------------------------------------------------------------- control
    def tick(self):
        frame, rays, ir = self.hw.read()
        if rays is None:
            if frame is None:
                self.hw.stop()
                self.status = "waiting for camera"
                return
            rays = self.perception.rays(frame)
        obs = self.stack.step(rays, ir, self.last_action)
        a = self.policy(obs)
        a = np.asarray(a, dtype=float).reshape(-1)[:2]
        if self.stack.reached_goal():
            a = np.zeros(2)
            self.hw.stop()
            self.running = False
            self.finish_time = time.time() - (self.t_start or time.time())
            self.status = f"GOAL in {self.finish_time:.2f}s"
            self.hw.led("green")
        else:
            self.hw.set_wheels(a[0], a[1])
        with self.lock:
            self.frame, self.rays, self.ir, self.last_action = frame, rays, ir, a
            self.steps += 1

    def run(self):
        dt = self.cfg.dt
        self.t_start = time.time()
        self.status = "running"
        self.hw.led("blue")
        last = time.time()
        while self.running:
            t0 = time.time()
            try:
                self.tick()
            except Exception as e:  # stop the robot on any error
                self.hw.stop()
                self.running = False
                self.status = f"error: {e}"
                raise
            now = time.time()
            self.loop_hz = 0.9 * self.loop_hz + 0.1 / max(now - last, 1e-6)
            last = now
            if self.realtime:
                time.sleep(max(0.0, dt - (time.time() - t0)))
        self.hw.stop()

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self.hw.stop()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.status.startswith("GOAL"):
            self.status = "stopped"

    # ------------------------------------------------------------ dashboard
    def snapshot(self) -> dict:
        with self.lock:
            pose = self.stack.pose.copy()
            cells = []
            c = cell_of(pose)
            if c in self.stack.planner.dist:
                cells = [c] + self.stack.planner.next_cells(c, 64)
            true_pose = getattr(getattr(self.hw, "robot", None), "pose", None)
            return {
                "status": self.status,
                "running": self.running,
                "pose": pose.tolist(),
                "true_pose": None if true_pose is None else np.asarray(true_pose).tolist(),
                "goal": list(self.goal),
                "goal_xy": cell_center(self.goal).tolist(),
                "bounds": list(self.bounds),
                "walls": [[list(a), list(b)] for a, b in self.stack.belief.walls],
                "true_walls": ([[list(a), list(b)] for a, b in getattr(self.hw, "map").walls]
                               if hasattr(self.hw, "map") else None),
                "path": [list(x) for x in cells],
                "rays": np.asarray(self.rays).tolist(),
                "ray_angles": self.cfg.camera.ray_angles().tolist(),
                "cam_forward": self.cfg.camera.mount_forward,
                "ir": np.asarray(self.ir).tolist(),
                "action": np.asarray(self.last_action).tolist(),
                "steps": self.steps,
                "elapsed": None if self.t_start is None else (self.finish_time or time.time() - self.t_start),
                "loop_hz": round(self.loop_hz, 1),
                "collisions": getattr(self.hw, "collisions", None),
                "localizer_points": self.stack.loc.last_used if self.stack.loc else 0,
            }
