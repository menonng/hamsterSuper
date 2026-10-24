"""Vectorized 2D geometry against axis-aligned wall boxes."""
from __future__ import annotations

import math

import numpy as np


def raycast(origins: np.ndarray, dirs: np.ndarray, boxes: np.ndarray, max_range: float,
            return_exit: bool = False):
    """Slab-method ray/AABB intersection.

    origins: (R, 2) or (2,), dirs: (R, 2) unit vectors, boxes: (W, 4).
    Returns hit distance per ray (max_range when nothing is hit); with
    return_exit, also the exit distance through the first box hit.
    """
    dirs = np.atleast_2d(dirs)
    origins = np.broadcast_to(np.atleast_2d(origins), dirs.shape)
    R = dirs.shape[0]
    if boxes.shape[0] == 0:
        far = np.full(R, max_range)
        return (far, far) if return_exit else far
    # nudge axis-parallel directions so the slab math stays finite
    dirs = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    inv = 1.0 / dirs                                        # (R, 2)
    ox, oy = origins[:, 0:1], origins[:, 1:2]
    tx1 = (boxes[None, :, 0] - ox) * inv[:, 0:1]
    tx2 = (boxes[None, :, 2] - ox) * inv[:, 0:1]
    ty1 = (boxes[None, :, 1] - oy) * inv[:, 1:2]
    ty2 = (boxes[None, :, 3] - oy) * inv[:, 1:2]
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tmax >= tmin) & (tmax > 0)
    t = np.where(hit, np.maximum(tmin, 0.0), np.inf)
    idx = np.argmin(t, axis=1)
    rows = np.arange(R)
    d = np.minimum(t[rows, idx], max_range)
    if not return_exit:
        return d
    ex = np.where(np.isfinite(t[rows, idx]), tmax[rows, idx], max_range)
    return d, np.minimum(ex, max_range)


def circle_penetration(center: np.ndarray, radius: float, boxes: np.ndarray):
    """Returns (max penetration depth, push-out vector) of a circle vs boxes."""
    if boxes.shape[0] == 0:
        return 0.0, np.zeros(2)
    cx = np.clip(center[0], boxes[:, 0], boxes[:, 2])
    cy = np.clip(center[1], boxes[:, 1], boxes[:, 3])
    dx, dy = center[0] - cx, center[1] - cy
    dist = np.hypot(dx, dy)
    pen = radius - dist
    k = int(np.argmax(pen))
    if pen[k] <= 0:
        return 0.0, np.zeros(2)
    if dist[k] > 1e-9:
        n = np.array([dx[k], dy[k]]) / dist[k]
    else:  # center inside the box: push along the shortest axis
        b = boxes[k]
        opts = np.array([center[0] - b[0], b[2] - center[0], center[1] - b[1], b[3] - center[1]])
        a = int(np.argmin(opts))
        n = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]][a], dtype=float)
        return float(opts[a] + radius), n * (opts[a] + radius)
    return float(pen[k]), n * pen[k]


def resolve_collisions(center: np.ndarray, radius: float, boxes: np.ndarray, iters: int = 4):
    """Push a circle out of all boxes. Returns (new center, collided flag)."""
    c = center.copy()
    collided = False
    for _ in range(iters):
        pen, push = circle_penetration(c, radius, boxes)
        if pen <= 1e-9:
            break
        collided = True
        c = c + push * 1.001
    return c, collided


def rect_penetration(center: np.ndarray, th: float, half_len: float, half_wid: float, boxes: np.ndarray):
    """Oriented rectangle (the robot's footprint) vs axis-aligned boxes, by the
    separating-axis test. Returns (depth, push-out vector) of the deepest overlap."""
    if boxes.shape[0] == 0:
        return 0.0, np.zeros(2)
    c, s = math.cos(th), math.sin(th)
    ac, as_ = abs(c), abs(s)
    bc = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    bx, by = (boxes[:, 2] - boxes[:, 0]) / 2, (boxes[:, 3] - boxes[:, 1]) / 2
    d = bc - center
    axes = np.array([[1.0, 0.0], [0.0, 1.0], [c, s], [-s, c]])
    dist = d @ axes.T                                                  # (B, 4) signed
    reach = np.stack([half_len * ac + half_wid * as_ + bx,             # world x
                      half_len * as_ + half_wid * ac + by,             # world y
                      half_len + bx * ac + by * as_,                   # robot forward
                      half_wid + bx * as_ + by * ac], axis=1)          # robot side
    overlap = reach - np.abs(dist)
    a = np.argmin(overlap, axis=1)
    depth = overlap[np.arange(len(boxes)), a]
    k = int(np.argmax(depth))
    if depth[k] <= 0:
        return 0.0, np.zeros(2)
    ax = axes[a[k]]
    sign = -1.0 if dist[k, a[k]] > 0 else 1.0
    return float(depth[k]), sign * ax * depth[k]


def resolve_rect(center: np.ndarray, th: float, half_len: float, half_wid: float, boxes: np.ndarray,
                 iters: int = 6):
    """Push the robot footprint out of all boxes. Returns (new center, collided flag)."""
    c = center.copy()
    collided = False
    for _ in range(iters):
        pen, push = rect_penetration(c, th, half_len, half_wid, boxes)
        if pen <= 1e-9:
            break
        collided = True
        c = c + push * 1.001
    return c, collided
