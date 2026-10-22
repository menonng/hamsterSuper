"""Evaluate a policy on generated or fixed maps.

    python -m hamster_super.rl.evaluate --model runs/x/policy_ts.pt --level 5 --episodes 100
    python -m hamster_super.rl.evaluate --model ... --maps maps/ --full-stack
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from ..config import DomainRandomization, EnvConfig
from ..env.hamster_env import HamsterMazeEnv
from ..maps.parser import load_map
from .policy import load_policy


def load_maps(path) -> list:
    p = Path(path)
    files = sorted(list(p.glob("*.txt")) + list(p.glob("*.json"))) if p.is_dir() else [p]
    return [load_map(f) for f in files]


@torch.no_grad()
def evaluate(policy, cfg: EnvConfig, episodes: int = 50, level: int = 5, maps=None, seed: int = 12345,
             full_stack: bool = False, randomize: bool = True) -> dict:
    if not randomize:
        cfg = dataclasses.replace(cfg, dr=DomainRandomization(enabled=False))
    env = HamsterMazeEnv(cfg, level=level, seed=seed, full_stack=full_stack)
    env.level = level
    results = []
    for ep in range(episodes):
        opts = {"map": maps[ep % len(maps)]} if maps else {"level": level}
        obs, _ = env.reset(options=opts)
        done = False
        while not done:
            a = policy(torch.as_tensor(obs[None])).numpy()[0]
            obs, r, te, tr, info = env.step(a)
            done = te or tr
        results.append(info)
    succ = [i["success"] for i in results]
    ok = [i for i in results if i["success"]]
    return {
        "episodes": episodes,
        "level": level if not maps else "maps",
        "success_rate": float(np.mean(succ)),
        "mean_time_s": float(np.mean([i["time_s"] for i in ok])) if ok else None,
        "mean_speed_cells_per_s": float(np.mean([i["opt_cells"] / i["time_s"] for i in ok])) if ok else None,
        "collisions_per_episode": float(np.mean([i["collisions"] for i in results])),
        "full_stack": full_stack,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--maps", default=None, help="map file or directory (ASCII .txt / .json)")
    ap.add_argument("--full-stack", action="store_true", help="dead-reckoning + grid-snap localization")
    ap.add_argument("--known-map", action="store_true")
    ap.add_argument("--no-dr", action="store_true", help="disable domain randomization")
    args = ap.parse_args()
    cfg = EnvConfig(known_map=args.known_map)
    maps = load_maps(args.maps) if args.maps else None
    pol = load_policy(args.model)
    res = evaluate(pol, cfg, args.episodes, args.level, maps, full_stack=args.full_stack,
                   randomize=not args.no_dr)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
