"""Calibration helpers for the real Hamster + camera. Results go to calib.json.

    python -m hamster_super.robot.calibrate speed            # straight-line speed -> robot.max_speed
    python -m hamster_super.robot.calibrate turn             # spin in place -> robot.track_width
    python -m hamster_super.robot.calibrate ir               # live raw IR values
    python -m hamster_super.robot.calibrate camera --camera 0  # save frame + wall mask + rays
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _save(path, d):
    Path(path).write_text(json.dumps(d, indent=2))
    print(f"saved {path}: {json.dumps(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["speed", "turn", "ir", "camera"])
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--cmd", type=float, default=0.5, help="wheel command used for the test (0..1)")
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()
    calib = _load(args.out)

    if args.what == "camera":
        import cv2
        import numpy as np
        from ..config import EnvConfig
        from .hardware import CameraSource
        from .perception import Perception, SegmentationConfig
        cfg = EnvConfig()
        if "camera" in calib:
            from ..server.__main__ import load_calib
            seg, _ = load_calib(cfg, args.out)
        else:
            seg = SegmentationConfig()
        cam = CameraSource(args.camera)
        time.sleep(1.0)
        frame = cam.read()
        cam.close()
        if frame is None:
            raise SystemExit("no frame")
        per = Perception(cfg.camera, seg)
        rays = per.rays(frame)
        small = cv2.resize(frame, (cfg.camera.width, cfg.camera.height), interpolation=cv2.INTER_AREA)
        over = small.copy()
        over[per.last_mask] = (0.5 * over[per.last_mask] + [127, 0, 35]).astype(np.uint8)
        for k in range(len(rays)):   # mark detected wall points
            idx = np.searchsorted(per.lut.r, rays[k])
            if idx < len(per.lut.r):
                cv2.circle(over, (int(per.lut.u[k, idx]), int(per.lut.v[k, idx])), 2, (57, 197, 187), -1)
        cv2.imwrite("calib_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite("calib_mask.png", cv2.cvtColor(cv2.resize(over, None, fx=4, fy=4,
                    interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
        print("rays (cm, left -> right):", [round(float(r), 1) for r in rays])
        print("wrote calib_frame.png and calib_mask.png; tune 'segmentation' / 'camera' in", args.out)
        return

    from roboid import Hamster
    h = Hamster()
    try:
        if args.what == "ir":
            print("Ctrl+C to quit. Place a wall at 0, 1, 2, ... 6 cm and note the values.")
            while True:
                print(f"\rL {h.left_proximity():3d}  R {h.right_proximity():3d}", end="", flush=True)
                time.sleep(0.1)
        cmd = int(args.cmd * 100)
        if args.what == "speed":
            input(f"Robot will drive straight at {cmd} for {args.seconds}s. Enter to start...")
            h.wheels(cmd, cmd)
            time.sleep(args.seconds)
            h.stop()
            d = float(input("Measured distance travelled (cm): "))
            calib.setdefault("robot", {})["max_speed"] = round(d / args.seconds / args.cmd, 2)
        elif args.what == "turn":
            input(f"Robot will spin in place at {cmd} for {args.seconds}s. Enter to start...")
            h.wheels(-cmd, cmd)
            time.sleep(args.seconds)
            h.stop()
            deg = float(input("Measured rotation (degrees, total incl. full turns): "))
            v = calib.get("robot", {}).get("max_speed", 20.0) * args.cmd
            calib.setdefault("robot", {})["track_width"] = round(2 * v * args.seconds / math.radians(deg), 3)
        _save(args.out, calib)
    except KeyboardInterrupt:
        print()
    finally:
        h.stop()
        h.dispose()


if __name__ == "__main__":
    main()
