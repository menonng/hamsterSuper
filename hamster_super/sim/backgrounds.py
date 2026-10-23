"""Random backgrounds around the course (a dome over the arena, like a sky).

On the real robot the camera looks over the walls into a room: people walk
by, lights change, posters and white furniture appear. During simulation and
perception tests each run draws one panorama and wraps it around the arena,
so nothing learned depends on a particular (e.g. plain white) background.

Panoramas come from PNG/JPG files in `assets/backgrounds/` (or the folder in
$HAMSTER_BG_DIR); procedural ones (gradients, bright blocks, person-like
silhouettes) are mixed in, and used alone when no files exist.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PANO_H, PANO_W = 128, 512
# elevation covered by the panorama rows: +90 deg (top) .. -25 deg (bottom)
ELEV_TOP, ELEV_BOTTOM = np.pi / 2, -np.radians(25)
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "assets" / "backgrounds"


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def load_images(folder=None) -> list:
    folder = Path(folder or os.environ.get("HAMSTER_BG_DIR") or DEFAULT_DIR)
    out = []
    if not folder.is_dir():
        return out
    import cv2
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out.append(_resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), PANO_H, PANO_W))
    return out


def procedural(rng: np.random.Generator) -> np.ndarray:
    """A cluttered room: wall/ceiling gradient, bright and dark blocks, people."""
    h, w = PANO_H, PANO_W
    top, bottom = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    t = np.linspace(0, 1, h)[:, None, None]
    img = np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3)).copy()
    for _ in range(int(rng.integers(3, 12))):            # furniture, posters, windows
        bw, bh = int(rng.integers(10, 120)), int(rng.integers(8, 70))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h - 4))
        col = rng.uniform(0, 255, 3) if rng.random() < 0.6 else np.full(3, rng.uniform(220, 255))
        xs = np.arange(x, x + bw) % w
        img[y:y + bh, xs] = col
    for _ in range(int(rng.integers(0, 6))):             # people: tall blobs reaching the floor
        cx, pw = int(rng.integers(0, w)), int(rng.integers(6, 22))
        ph = int(rng.integers(h // 3, h))
        col = rng.uniform(0, 255, 3)
        yy, xx = np.mgrid[h - ph:h, -pw:pw + 1]
        keep = (xx / pw) ** 2 + ((yy - h) / ph) ** 2 * 0.3 < 1
        img[yy[keep], (cx + xx[keep]) % w] = col
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class BackgroundBank:
    def __init__(self, folder=None, procedural_prob: float = 0.3):
        self.images = load_images(folder)
        self.procedural_prob = procedural_prob if self.images else 1.0

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.procedural_prob:
            return procedural(rng)
        img = self.images[int(rng.integers(len(self.images)))]
        img = np.roll(img, int(rng.integers(PANO_W)), axis=1)          # random orientation
        if rng.random() < 0.5:
            img = img[:, ::-1]
        gain = rng.uniform(0.7, 1.3)
        return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def lookup(pano: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Colour seen in world direction (azimuth, elevation) [rad]."""
    h, w = pano.shape[:2]
    u = ((azimuth / (2 * np.pi)) % 1.0 * w).astype(np.int32) % w
    v = (ELEV_TOP - elevation) / (ELEV_TOP - ELEV_BOTTOM) * (h - 1)
    v = np.clip(v, 0, h - 1).astype(np.int32)
    return pano[v, u]
