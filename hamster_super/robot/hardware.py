"""Hardware abstraction: the real Hamster (roboid + camera) or a digital twin.

Both expose:
    read() -> (rgb frame or None, rays or None, ir[2])
    set_wheels(left, right)   # [-1, 1]
    stop(), close()
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..config import EnvConfig
from ..maps.grid_map import GridMap, cell_center


class CameraSource:
    """Grabs frames in a background thread so the control loop always gets
    the newest frame (no buffered latency). `src` = device index or URL
    (e.g. an ESP32-CAM MJPEG stream 'http://192.168.0.50:81/stream')."""

    def __init__(self, src, width=None, height=None, flip: bool = False):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {src!r}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.flip = flip
        self.frame = None
        self.stamp = 0.0
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._run:
            ok, bgr = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.flip:
                bgr = self.cv2.flip(bgr, -1)
            self.frame = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            self.stamp = time.time()

    def read(self):
        return self.frame

    def close(self):
        self._run = False
        self.cap.release()


class HamsterHardware:
    """Robomation Hamster over the USB dongle (pip install roboid)."""

    def __init__(self, camera: CameraSource, ir_raw_max: float = 255.0, ir_raw_floor: float = 8.0,
                 port: str | None = None):
        from roboid import Hamster
        self.h = Hamster(port) if port else Hamster()
        self.camera = camera
        self.ir_raw_max, self.ir_raw_floor = ir_raw_max, ir_raw_floor

    def read(self):
        raw = np.array([self.h.left_proximity(), self.h.right_proximity()], dtype=float)
        ir = np.clip((raw - self.ir_raw_floor) / (self.ir_raw_max - self.ir_raw_floor), 0, 1)
        return self.camera.read(), None, ir

    def set_wheels(self, left: float, right: float):
        # Robomation API: hamster.wheels(left, right) takes -100 ... 100, the percentage
        # of the maximum wheel speed; the policy's [-1, 1] maps onto it directly
        # (1.0 = 100 % = top speed, measured 10 cm/s and set by `calibrate speed`)
        self.h.wheels(int(round(float(np.clip(left, -1, 1)) * 100)), int(round(float(np.clip(right, -1, 1)) * 100)))

    def led(self, color: str):
        try:
            self.h.leds(color)
        except Exception:
            pass

    def stop(self):
        self.h.stop()

    def close(self):
        self.stop()
        self.camera.close()
        self.h.dispose()


class SimHardware:
    """Digital twin: simulated Hamster in a known map. With render=True the
    full camera image -> perception pipeline is exercised, like on the real robot."""

    def __init__(self, cfg: EnvConfig, gmap: GridMap, render: bool = True, seed: int = 0):
        from ..sim.camera import Renderer
        from ..sim.robot import SimRobot
        from ..sim.sensors import DepthRays
        self.cfg, self.map = cfg, gmap
        self.rng = np.random.default_rng(seed)
        self.robot = SimRobot(cfg.robot, cfg.dr, self.rng)
        self.boxes = gmap.wall_boxes()
        self.rays = DepthRays(cfg.camera)
        self.renderer = Renderer(cfg.camera) if render else None
        if render:   # a random room around the course (people, posters, lights)
            from ..sim.backgrounds import BackgroundBank
            self.background = BackgroundBank().sample(self.rng)
            b = self.boxes
            self.arena = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()) if len(b) else None
        heading = gmap.start_heading if gmap.start_heading is not None else (gmap.free_headings(gmap.start) or [0.0])[0]
        c = cell_center(gmap.start)
        self.robot.reset((c[0], c[1], heading))
        self.cmd = np.zeros(2)
        self.collisions = 0
        self._last = time.time()

    def read(self):
        ir = self.robot.ir(self.boxes)
        if self.renderer is not None:
            return self.renderer.render(self.robot.pose, self.boxes, self.rng, background=self.background,
                                        arena=self.arena), None, ir
        return None, self.rays.measure(self.robot.pose, self.boxes, self.cfg.dr, self.rng), ir

    def set_wheels(self, left, right):
        self.cmd = np.array([left, right])
        self.collisions += int(self.robot.step(self.cmd, self.cfg.dt, self.cfg.substeps, self.boxes))

    def led(self, color):
        pass

    def stop(self):
        self.cmd = np.zeros(2)

    def close(self):
        pass
