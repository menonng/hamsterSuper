"""Pinhole camera model shared by the simulator renderer and real perception.

Robot frame: x forward, y left, z up (origin = robot center on the floor).
Camera frame: x right, y down, z optical axis. The camera sits at
(mount_forward, 0, mount_height) and is pitched down by `pitch`.

The policy never sees pixels. It sees `n_rays` ground distances to the
nearest wall at fixed bearings, measured from the camera's floor footprint.
`RayLUT` maps every (bearing, distance) to the pixel where that floor point
appears, so a wall mask can be turned into the same rays the simulator emits.
This ray interface is what makes sim-to-real transfer work.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import GRID_SPACING, WALL_HEIGHT, CameraSpec
from .geometry import raycast


class CameraModel:
    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.f = spec.focal
        self.cx = (spec.width - 1) / 2
        self.cy = (spec.height - 1) / 2
        p = spec.pitch
        self.z_c = np.array([math.cos(p), 0.0, -math.sin(p)])   # optical axis
        self.x_c = np.array([0.0, -1.0, 0.0])                    # image right
        self.y_c = np.cross(self.z_c, self.x_c)                  # image down
        self.pos = np.array([spec.mount_forward, 0.0, spec.mount_height])

    def project(self, pts: np.ndarray):
        """Robot-frame 3D points (N, 3) -> pixel (u, v), depth."""
        rel = pts - self.pos
        xc, yc, zc = rel @ self.x_c, rel @ self.y_c, rel @ self.z_c
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f * xc / zc
            v = self.cy + self.f * yc / zc
        return u, v, zc

    def pixel_dirs(self) -> np.ndarray:
        """Unit ray directions in robot frame for every pixel, shape (H, W, 3)."""
        W, H = self.spec.width, self.spec.height
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        d = (self.x_c[None, None] * ((u - self.cx) / self.f)[..., None]
             + self.y_c[None, None] * ((v - self.cy) / self.f)[..., None]
             + self.z_c[None, None])
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


class RayLUT:
    """Pixel locations of floor points along each policy ray bearing."""

    def __init__(self, spec: CameraSpec, step: float = 0.25):
        self.spec = spec
        cam = CameraModel(spec)
        self.angles = spec.ray_angles()
        self.r = np.arange(0.5, spec.max_range + 1e-6, step)
        origin = np.array([spec.mount_forward, 0.0])
        K, N = len(self.angles), len(self.r)
        pts = np.zeros((K, N, 3))
        pts[..., 0] = origin[0] + self.r[None] * np.cos(self.angles)[:, None]
        pts[..., 1] = origin[1] + self.r[None] * np.sin(self.angles)[:, None]
        u, v, z = cam.project(pts.reshape(-1, 3))
        u, v, z = u.reshape(K, N), v.reshape(K, N), z.reshape(K, N)
        inside = (z > 0) & (u >= 0) & (u <= spec.width - 1) & (v >= 0) & (v <= spec.height - 1)
        self.u = np.clip(np.round(u), 0, spec.width - 1).astype(np.int32)
        self.v = np.clip(np.round(v), 0, spec.height - 1).astype(np.int32)
        self.valid = inside
        # nearest visible floor distance per ray (blind zone below the image)
        self.r_min = np.array([self.r[np.argmax(row)] if row.any() else spec.max_range for row in inside])

    def ranges_from_mask(self, wall_mask: np.ndarray, min_run_cm: float = 1.5) -> np.ndarray:
        """Wall mask (H, W) bool -> ray ranges (K,). Small blobs such as the
        floor dots are rejected by requiring a run of `min_run_cm` of
        consecutive wall samples along the ray."""
        step = self.r[1] - self.r[0]
        run = max(1, int(round(min_run_cm / step)))
        samples = wall_mask[self.v, self.u] & self.valid           # (K, N)
        raw = samples
        out = np.full(len(self.angles), self.spec.max_range)
        if run > 1:
            # run-length filter: position i is a wall start if the next `run` samples are wall
            c = np.cumsum(np.pad(samples, ((0, 0), (1, 0))).astype(np.int32), axis=1)
            n = samples.shape[1]
            idx = np.arange(n)
            end = np.minimum(idx + run, n)
            full = (c[:, end] - c[:, idx]) >= (end - idx)
            samples = samples & full
        first = np.argmax(self.valid, axis=1)
        for k in range(len(self.angles)):
            f = first[k]
            if raw[k, f] and raw[k, min(f + 1, raw.shape[1] - 1)]:
                # wall touching the bottom edge of the image: a wall closer than the
                # visible floor shows up as a short strip, keep it despite the run filter
                out[k] = self.r_min[k]
                continue
            hits = np.flatnonzero(samples[k])
            if hits.size:
                out[k] = max(self.r[hits[0]], self.r_min[k])
        return out


# ----------------------------------------------------------------- renderer
# The race course: black floor, white walls. The dot colour is not known yet.
FLOOR = np.array([22, 22, 25], np.uint8)
FLOOR_FAR = np.array([38, 38, 42], np.uint8)
WALL_FACE = np.array([228, 228, 224], np.uint8)
WALL_TOP = np.array([244, 244, 240], np.uint8)
DOT = np.array([150, 150, 150], np.uint8)
SKY = np.array([90, 90, 95], np.uint8)


class Renderer:
    """Approximate first-person camera image of the maze (for tests,
    the digital-twin server and synthetic training data for perception)."""

    def __init__(self, spec: CameraSpec, dot_radius: float = 0.4):
        self.cam = CameraModel(spec)
        self.dirs = self.cam.pixel_dirs().reshape(-1, 3)
        self.dot_radius = dot_radius

    def render(self, pose, boxes: np.ndarray, rng: np.random.Generator | None = None,
               colors: dict | None = None, background: np.ndarray | None = None,
               arena=None) -> np.ndarray:
        """background: panorama (see sim.backgrounds) wrapped around the arena as a dome;
        arena: (x0, y0, x1, y1) cm extent of the black race floor - beyond it the
        line of sight ends in the room (the dome) instead of on black floor."""
        spec = self.cam.spec
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        d = self.dirs
        h = spec.mount_height
        cam_xy = np.array([x, y]) + R @ self.cam.pos[:2]
        img = np.empty((d.shape[0], 3), np.uint8)
        img[:] = SKY
        down = d[:, 2] < -1e-6
        dd = d[down]
        horiz = np.linalg.norm(dd[:, :2], axis=1)
        t_ground = h / -dd[:, 2]
        s_ground = t_ground * horiz                               # horizontal distance to floor hit
        hdir = (dd[:, :2] / horiz[:, None]) @ R.T
        far = 400.0
        slope = h / s_ground                                      # height drop per cm
        is_face = np.zeros(len(dd), bool)
        is_top = np.zeros(len(dd), bool)
        todo = np.arange(len(dd))
        start = np.zeros(len(dd))
        for _ in range(3):   # the line of sight may pass over up to a few walls
            o = cam_xy + hdir[todo] * start[todo, None]
            hit, exit_ = raycast(o, hdir[todo], boxes, far, return_exit=True)
            hit, exit_ = hit + start[todo], exit_ + start[todo]
            sg, sl = s_ground[todo], slope[todo]
            face = (hit < sg) & (h - hit * sl <= WALL_HEIGHT)
            top = (hit < sg) & ~face & (h - exit_ * sl <= WALL_HEIGHT)
            is_face[todo[face]] = True
            is_top[todo[top]] = True
            over = (hit < sg) & ~face & ~top
            start[todo[over]] = exit_[over] + 0.01
            todo = todo[over]
            if todo.size == 0:
                break
        sub = np.empty((dd.shape[0], 3), np.float32)
        gp = cam_xy + hdir * s_ground[:, None]
        g = gp / GRID_SPACING
        dot = np.hypot(g[:, 0] - np.round(g[:, 0]), g[:, 1] - np.round(g[:, 1])) * GRID_SPACING < self.dot_radius
        fade = np.clip(s_ground / 150.0, 0, 1)[:, None]
        floor, face, top, dotc = FLOOR.astype(np.float32), WALL_FACE.astype(np.float32), \
            WALL_TOP.astype(np.float32), DOT.astype(np.float32)
        if rng is not None:   # lighting / material variation: shaded walls, glossy floor, unknown dot colour
            face = face * rng.uniform(0.7, 1.0)
            floor = floor + rng.uniform(-10, 25)
            dotc = np.full(3, rng.uniform(40, 235), np.float32)
        sub[:] = floor * (1 - fade) + FLOOR_FAR * fade
        if rng is not None:   # a soft glare patch on the black floor
            cx_, cy_ = rng.uniform(-20, 20, 2)
            glare = rng.uniform(0, 45) * np.exp(-((gp[:, 0] - cam_xy[0] - cx_) ** 2 + (gp[:, 1] - cam_xy[1] - cy_) ** 2) / 300)
            sub += glare[:, None]
        sub[dot] = dotc
        if background is not None:
            from .backgrounds import lookup
            if arena is not None:
                x0, y0, x1, y1 = arena
                out = (gp[:, 0] < x0) | (gp[:, 0] > x1) | (gp[:, 1] < y0) | (gp[:, 1] > y1)
            else:
                out = s_ground > far
            out &= ~is_top & ~is_face
            if out.any():
                el = np.arctan2(dd[out, 2], horiz[out])
                az = np.arctan2(hdir[out, 1], hdir[out, 0])
                sub[out] = lookup(background, az, el)
        sub[is_top] = top
        sub[is_face] = face
        sub = np.clip(sub, 0, 255).astype(np.uint8)
        img[down] = sub
        if background is not None:
            from .backgrounds import lookup
            up = ~down
            du = d[up]
            hu = du[:, :2] @ R.T
            img[up] = lookup(background, np.arctan2(hu[:, 1], hu[:, 0]),
                             np.arctan2(du[:, 2], np.linalg.norm(du[:, :2], axis=1)))
        img = img.reshape(spec.height, spec.width, 3)
        if rng is not None:  # sensor noise + exposure jitter
            gain = rng.uniform(0.8, 1.2)
            noise = rng.normal(0, 6, img.shape)
            img = np.clip(img * gain + noise, 0, 255).astype(np.uint8)
        return img


def wall_mask_bright(img_rgb: np.ndarray, threshold: int | None = 128) -> np.ndarray:
    """White walls on a black floor. threshold=None picks it per frame (Otsu)."""
    gray = img_rgb.astype(np.float32).mean(axis=-1)
    if threshold is None:   # per-frame threshold, kept in a sane band for wall-free frames
        threshold = float(np.clip(otsu_threshold(gray), 70, 200))
    return gray > threshold


def wall_mask_dark(img_rgb: np.ndarray, threshold: int = 110) -> np.ndarray:
    """For courses with walls darker than the floor."""
    gray = img_rgb.astype(np.float32).mean(axis=-1)
    return gray < threshold


def otsu_threshold(gray: np.ndarray) -> float:
    hist = np.bincount(np.clip(gray, 0, 255).astype(np.int32).ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return float(np.nanargmax(between))
