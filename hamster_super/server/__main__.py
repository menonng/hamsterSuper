"""Run HamsterSuper on the real robot or the digital twin.

Real robot (Hamster USB dongle + camera):
    python -m hamster_super.server --model models/policy.npz --camera 0 \
        --start 0,0,E --goal 5,3 --arena 6,4 [--map maps/race.txt] [--calib calib.json]

Digital twin (no hardware needed):
    python -m hamster_super.server --model models/policy.npz --sim maps/example_02_maze.txt

Then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..config import EnvConfig, from_dict
from ..maps.grid_map import DIRS, cell_center
from ..maps.parser import load_map
from ..robot.perception import Perception, SegmentationConfig
from ..robot.runtime import Runtime
from .app import serve


def numpy_policy(path):
    """Returns obs (np) -> action (np), using the torch-free path when possible."""
    path = str(path)
    if path.endswith(".npz"):
        from ..rl.policy import NumpyPolicy
        pol = NumpyPolicy(path)
        return lambda obs: pol(obs[None])[0]
    import torch
    from ..rl.policy import load_policy
    pol = load_policy(path)

    def f(obs):
        with torch.no_grad():
            return pol(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]
    return f


def parse_pose(s: str):
    parts = s.split(",")
    cx, cy = int(parts[0]), int(parts[1])
    heading = DIRS[parts[2].upper()] if len(parts) > 2 and parts[2].upper() in DIRS else (
        math.radians(float(parts[2])) if len(parts) > 2 else 0.0)
    return (cx, cy), heading


def load_calib(cfg: EnvConfig, path):
    """calib.json: {"robot": {...RobotSpec}, "camera": {...CameraSpec}, "segmentation": {...}, "ir": {...}}"""
    d = json.loads(Path(path).read_text())
    from ..config import CameraSpec, RobotSpec
    if "robot" in d:
        cfg.robot = from_dict(RobotSpec, {**cfg.robot.__dict__, **d["robot"]})
    if "camera" in d:
        cam = dict(d["camera"])
        for k in ("hfov", "pitch", "ray_fov"):       # degrees in the calib file
            if k + "_deg" in cam:
                cam[k] = math.radians(cam.pop(k + "_deg"))
        cfg.camera = from_dict(CameraSpec, {**cfg.camera.__dict__, **cam})
    seg = SegmentationConfig(**d.get("segmentation", {}))
    return seg, d.get("ir", {})


def main():
    ap = argparse.ArgumentParser(description="HamsterSuper server")
    ap.add_argument("--model", required=True, help="policy.npz | policy_ts.pt | checkpoint.pt")
    ap.add_argument("--sim", default=None, help="run the digital twin on this map file")
    ap.add_argument("--sim-fast-rays", action="store_true", help="twin: skip rendering, use ideal rays")
    ap.add_argument("--map", default=None, help="known course map (optional; otherwise mapped online)")
    ap.add_argument("--camera", default="0", help="camera index or stream URL")
    ap.add_argument("--flip", action="store_true", help="rotate camera image 180 deg")
    ap.add_argument("--port-hamster", default=None, help="serial port of the Hamster dongle")
    ap.add_argument("--start", default=None, help="start cell and heading, e.g. 0,0,E")
    ap.add_argument("--goal", default=None, help="goal cell, e.g. 5,3")
    ap.add_argument("--arena", default=None, help="arena size in cells W,H (planning bounds)")
    ap.add_argument("--calib", default=None, help="calibration JSON")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--autostart", action="store_true")
    args = ap.parse_args()

    cfg = EnvConfig()
    seg, ir_cal = (load_calib(cfg, args.calib) if args.calib else (SegmentationConfig(), {}))
    prior = load_map(args.map) if args.map else None

    if args.sim:
        from ..config import DomainRandomization
        from ..robot.hardware import SimHardware
        gmap = load_map(args.sim)
        cfg.dr = DomainRandomization(enabled=True)
        hw = SimHardware(cfg, gmap, render=not args.sim_fast_rays)
        start, heading = gmap.start, hw.robot.th
        goal = gmap.goal
        bounds = gmap.cell_bounds()
        hw.reset_robot = lambda: hw.robot.reset((*cell_center(start), heading))
    else:
        from ..robot.hardware import CameraSource, HamsterHardware
        if prior is None and not (args.start and args.goal):
            ap.error("real robot needs --map or both --start and --goal")
        if prior is not None:
            start, heading, goal = prior.start, prior.start_heading or 0.0, prior.goal
        if args.start:
            start, heading = parse_pose(args.start)
        if args.goal:
            goal = tuple(int(v) for v in args.goal.split(","))
        if args.arena:
            w, h = (int(v) for v in args.arena.split(","))
            bounds = (0, 0, w - 1, h - 1)
        elif prior is not None:
            bounds = prior.cell_bounds()
        else:
            bounds = (min(start[0], goal[0]) - 6, min(start[1], goal[1]) - 6,
                      max(start[0], goal[0]) + 6, max(start[1], goal[1]) + 6)
        cam = CameraSource(args.camera, flip=args.flip)
        hw = HamsterHardware(cam, port=args.port_hamster, **ir_cal)

    start_pose = (*cell_center(start), heading)
    rt = Runtime(cfg, hw, numpy_policy(args.model), start_pose, goal, bounds, prior=prior,
                 perception=Perception(cfg.camera, seg))
    serve(rt, args.host, args.port)
    print(f"HamsterSuper server on http://{args.host}:{args.port}  (start {start} -> goal {goal})", flush=True)
    if args.autostart:
        rt.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        rt.stop()
        hw.close()


if __name__ == "__main__":
    main()
