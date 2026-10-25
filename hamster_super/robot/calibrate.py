"""Calibration helpers for the real Hamster + camera. Results go to calib.json.

    python -m hamster_super.robot.calibrate speed            # straight-line speed -> robot.max_speed
    python -m hamster_super.robot.calibrate turn             # spin in place -> robot.track_width
    python -m hamster_super.robot.calibrate ir               # live raw IR values
    python -m hamster_super.robot.calibrate pitch --distance 10  # camera tilt from a wall at a known distance
    python -m hamster_super.robot.calibrate camera --camera 0  # save frame + wall mask + rays
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _save(path, d):
    Path(path).write_text(json.dumps(d, indent=2))
    print(f"saved {path}: {json.dumps(d)}")


def wall_base_row(mask, cols, min_rows: int = 8):
    """Lowest image row of the wall in each given column. The wall face is a
    tall run of wall pixels; shorter runs below it are floor dots or glare."""
    rows = []
    for u in cols:
        col = np.r_[False, mask[:, u], False].astype(np.int8)
        d = np.diff(col)
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)   # run = [start, end)
        tall = [(st, en) for st, en in zip(starts, ends) if en - st >= min_rows]
        if tall:
            rows.append(max(en for _, en in tall) - 1)
    return float(np.median(rows)) if rows else None


def pitch_from_base_row(v_base, spec, distance):
    """Tilt that puts a wall base at `distance` cm onto image row v_base."""
    f = spec.focal
    cy = (spec.height - 1) / 2
    below_axis = math.atan((v_base - cy) / f)          # angle of that row below the optical axis
    depression = math.atan(spec.mount_height / distance)
    return depression - below_axis


def calibrate_pitch(args, calib):
    """Put the robot squarely facing a wall, `--distance` cm from the lens.
    The camera tilt is fixed after mounting, so this is done once."""
    from ..config import EnvConfig
    from .hardware import CameraSource
    from .perception import Perception, SegmentationConfig
    cfg = EnvConfig()
    seg = SegmentationConfig()
    if "camera" in calib or "segmentation" in calib:
        from ..server.__main__ import load_calib
        seg, _ = load_calib(cfg, args.out)
    spec = cfg.camera
    per = Perception(spec, seg)
    cam = CameraSource(args.camera)
    time.sleep(1.0)
    vals = []
    import cv2
    for _ in range(15):
        frame = cam.read()
        if frame is not None:
            small = cv2.resize(frame, (spec.width, spec.height), interpolation=cv2.INTER_AREA)
            mask = per.wall_mask(small)
            cx = spec.width // 2
            v = wall_base_row(mask, range(cx - 6, cx + 7))
            if v is not None:
                vals.append(pitch_from_base_row(v, spec, args.distance))
        time.sleep(0.1)
    cam.close()
    if not vals:
        raise SystemExit("no wall found in the image centre; check lighting / segmentation")
    pitch = float(np.median(vals))
    print(f"camera tilt: {math.degrees(pitch):.2f} deg (spread {math.degrees(np.std(vals)):.2f} deg)")
    calib.setdefault("camera", {})["pitch_deg"] = round(math.degrees(pitch), 2)
    _save(args.out, calib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["speed", "turn", "ir", "pitch", "camera"])
    ap.add_argument("--distance", type=float, default=10.0,
                    help="pitch: cm from the camera lens (front of the module) to the wall face")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--cmd", type=float, default=0.5, help="wheel command used for the test (0..1)")
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()
    calib = _load(args.out)

    if args.what == "pitch":
        calibrate_pitch(args, calib)
        return

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
        cmd = int(round(args.cmd * 100))          # hamster.wheels(): % of the maximum speed, -100 ... 100
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
