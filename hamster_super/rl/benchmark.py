"""Fixed benchmark: the same courses, start conditions and disturbances every time.

Training statistics move with the curriculum level and with every change to
the simulator, so they cannot say whether a change made the robot better.
The benchmark can: a fixed list of race courses (14 x 14 one-way and lap
courses, with hairpin wide sections and with 2-cell-wide straights, plus a
few smaller ones), each driven with the same fixed seeds for the domain
randomization. Results carry the simulator version (config.ENV_VERSION), so
scores are only compared within one version.

    python -m hamster_super.rl.benchmark runs/ppo/last.pt [--out runs/ppo/benchmark.jsonl]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..config import ENV_VERSION, EnvConfig
from ..maps.generator import COURSE_LEVELS, generate_course, generate_wide_course

# (name, level, map seed, style, one-way?) - never change existing entries;
# append new ones and bump BENCH_VERSION instead
BENCH_VERSION = 3          # 2: no widened blocks at the lap line; 3: courses with moving parts
REPEATS = 3          # each course driven with 3 fixed disturbance seeds
SPECS = (
    [(f"race-hairpin-oneway-{i}", COURSE_LEVELS - 1, 1000 + i, "hairpin", True) for i in range(4)]
    + [(f"race-hairpin-lap-{i}", COURSE_LEVELS - 1, 2000 + i, "hairpin", False) for i in range(3)]
    + [(f"race-wide-oneway-{i}", COURSE_LEVELS - 1, 3000 + i, "wide", True) for i in range(4)]
    + [(f"mid-wide-lap-{i}", 4, 4000 + i, "wide", False) for i in range(2)]
    + [(f"small-hairpin-{i}", 2, 5000 + i, "hairpin", bool(i % 2)) for i in range(2)]
    + [(f"race-moving-oneway-{i}", COURSE_LEVELS - 1, 6000 + i, "wide-moving", True) for i in range(3)]
)


def make_map(level: int, seed: int, style: str, one_way: bool):
    for k in range(50):                      # a wide route may not fit a seed: try the next one
        rng = np.random.default_rng(seed + 100000 * k)
        if style.startswith("wide"):
            m = generate_wide_course(rng, level, open_course=one_way)
        else:
            m = generate_course(rng, level, open_course=one_way, wide_straight_prob=0.0)
        if m is not None:
            return m
    raise RuntimeError(f"no {style} course for seed {seed}")


def _run(args):
    ckpt, spec, i, rep = args
    import torch
    from ..env.hamster_env import HamsterMazeEnv
    from .policy import load_policy
    torch.set_num_threads(1)
    name, level, seed, style, one_way = spec
    pol = load_policy(ckpt)
    env = HamsterMazeEnv(EnvConfig(), level=level, seed=10_000 + 10 * i + rep)
    m = make_map(level, seed, style, one_way)
    obs, _ = env.reset(options={"map": m, "dynamic": style.endswith("moving")})
    done = False
    while not done:
        with torch.no_grad():
            a = pol(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]
        obs, _, te, tr, info = env.step(a)
        done = te or tr
    return {"name": name, "success": bool(info["success"]), "time_s": float(info["time_s"]),
            "cells": int(info["opt_cells"]), "collisions": int(info["collisions"])}


def run_benchmark(ckpt, workers: int = 2) -> dict:
    t0 = time.time()
    jobs = [(str(ckpt), s, i, r) for i, s in enumerate(SPECS) for r in range(REPEATS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_run, jobs))
    ok = [r for r in rows if r["success"]]
    groups = {}
    for r in rows:
        g = r["name"].rsplit("-", 1)[0]
        groups.setdefault(g, []).append(r["success"])
    try:
        import torch
        it = torch.load(str(ckpt), map_location="cpu", weights_only=False).get("iter")
    except Exception:
        it = None
    return {
        "env_version": ENV_VERSION, "bench_version": BENCH_VERSION, "iter": it, "checkpoint": str(ckpt),
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "seconds_per_cell": float(np.mean([r["time_s"] / r["cells"] for r in ok])) if ok else None,
        "collisions_per_episode": float(np.mean([r["collisions"] for r in rows])),
        "groups": {g: float(np.mean(v)) for g, v in groups.items()},
        "runs": rows, "wall_seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="fixed benchmark of a checkpoint")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None, help="append the result to this .jsonl file")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    res = run_benchmark(a.checkpoint, a.workers)
    line = json.dumps(res)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")
    summary = {k: v for k, v in res.items() if k != "runs"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
