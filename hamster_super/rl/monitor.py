"""Live training dashboard.

    python -m hamster_super.rl.monitor --run runs/ppo          # http://localhost:8001
    python -m hamster_super.rl.train ... --monitor              # starts it alongside training

Shows the learning curves from <run>/train_log.jsonl as they are written, the
GitHub releases made so far, and a live replay: the newest checkpoint
(<run>/last.pt) drives a random maze in real time, with the true pose, the
robot's own pose estimate, the map it has built and its planned path.
Runs as its own process, so it never slows training down.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PAGE = Path(__file__).with_name("monitor.html")


class LogReader:
    """Incrementally reads train_log.jsonl (the trainer appends one row per iteration)."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list = []
        self._pos = 0
        self._lock = threading.Lock()

    def poll(self):
        with self._lock:
            if not self.path.exists():
                return
            if self.path.stat().st_size < self._pos:   # file restarted
                self.rows, self._pos = [], 0
            with open(self.path) as f:
                f.seek(self._pos)
                while True:
                    line = f.readline()
                    if not line or not line.endswith("\n"):
                        break
                    self._pos = f.tell()
                    try:
                        self.rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def since(self, n: int):
        self.poll()
        return self.rows[n:]


class Replay:
    """Drives random mazes with the newest checkpoint, in real time."""

    def __init__(self, run: Path, level: int | None, full_stack: bool, speed: float, seed: int = 0):
        self.run, self.level_override, self.full_stack, self.speed = run, level, full_stack, speed
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.frame: dict = {"status": "waiting for checkpoint"}
        self.history: deque = deque(maxlen=30)
        self.policy = None
        self.model_iter = None
        self._mtime = None
        self.ck_level = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _maybe_reload(self):
        ck_path = self.run / "last.pt"
        if not ck_path.exists():
            return False
        mtime = ck_path.stat().st_mtime
        if mtime == self._mtime:
            return True
        import torch
        from ..config import EnvConfig, from_dict
        from .policy import ActorCritic, DeployPolicy, RunningNorm
        try:
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        except Exception:
            return self.policy is not None      # being written: keep the old one
        ac = ActorCritic(ck["obs_dim"], ck.get("act_dim", 2), ck.get("hidden", (256, 256)))
        ac.load_state_dict(ck["model"])
        norm = RunningNorm(ck["obs_dim"])
        norm.load_state_dict(ck["obs_norm"])
        pol = DeployPolicy(ac, norm).eval()

        def act(obs):
            with torch.no_grad():
                return pol(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]
        self.policy, self.model_iter, self.ck_level = act, ck.get("iter"), ck.get("level", 0)
        self.cfg = from_dict(EnvConfig, ck["env_config"]) if "env_config" in ck else EnvConfig()
        self._mtime = mtime
        return True

    def _loop(self):
        from ..env.hamster_env import HamsterMazeEnv
        while True:
            try:
                if not self._maybe_reload():
                    time.sleep(2.0)
                    continue
                level = self.ck_level if self.level_override is None else self.level_override
                env = HamsterMazeEnv(self.cfg, level=level, seed=int(self.rng.integers(1 << 30)),
                                     full_stack=self.full_stack)
                self._episode(env, level)
                time.sleep(1.5 / self.speed)
            except Exception as e:   # keep the dashboard alive
                with self.lock:
                    self.frame = {"status": f"replay error: {e}"}
                time.sleep(3.0)

    def _episode(self, env, level):
        from ..maps.grid_map import cell_of
        obs, _ = env.reset(options={"level": level})
        m = env.map
        true_walls = [[list(a), list(b)] for a, b in m.walls]
        trail, est_trail = [], []
        it = self.model_iter
        done, info, t = False, {}, 0
        while not done:
            t0 = time.time()
            obs, r, te, tr, info = env.step(self.policy(obs))
            done = te or tr
            t += 1
            pose, est = env.robot.pose, np.asarray(env.est)
            trail.append([round(float(pose[0]), 2), round(float(pose[1]), 2)])
            est_trail.append([round(float(est[0]), 2), round(float(est[1]), 2)])
            c = cell_of(est)
            path = [list(c)] + [list(x) for x in env.planner.next_cells(c, 64)] if c in env.planner.dist else []
            with self.lock:
                self.frame = {
                    "status": "driving", "model_iter": it, "level": level, "map_name": m.name,
                    "bounds": list(env.bounds), "true_walls": true_walls,
                    "walls": [[list(a), list(b)] for a, b in env.belief.walls],
                    "start": list(m.start), "goal": list(m.goal),
                    "pose": pose.tolist(), "est": est.tolist(),
                    "trail": trail, "est_trail": est_trail, "path": path,
                    **line_info(env.line),
                    "rays": np.asarray(env.last_rays).tolist(),
                    "ray_angles": env.cfg.camera.ray_angles().tolist(),
                    "cam_forward": env.cfg.camera.mount_forward,
                    "cam": camera_info(env.cfg),
                    "action": np.asarray(env.prev_action).tolist(),
                    "step": t, "time_s": round(t * env.cfg.dt, 1), "collisions": env.collisions,
                    "opt_cells": env.opt_cells, "full_stack": bool(env.use_stack),
                }
            time.sleep(max(0.0, env.cfg.dt / self.speed - (time.time() - t0)))
        result = {"model_iter": it, "level": level, "success": bool(info.get("success")),
                  "time_s": info.get("time_s"), "collisions": info.get("collisions"),
                  "opt_cells": info.get("opt_cells")}
        with self.lock:
            self.history.appendleft(result)
            self.frame = {**self.frame, "status": "GOAL" if result["success"] else "failed"}

    def snapshot(self):
        with self.lock:
            return {**self.frame, "history": list(self.history)}


def line_info(line) -> dict:
    """Racing line + per-corner inside/outside call for the dashboard."""
    if line is None or line.line is None:
        return {"line": [], "corners": []}
    return {"line": [[round(float(x), 2), round(float(y), 2)] for x, y in line.line],
            "corners": line.corners, "clearance": round(float(line.clearance), 2)}


def camera_info(cfg) -> dict:
    """Camera / robot geometry for the 3D view."""
    import math
    c, r = cfg.camera, cfg.robot
    return {"h": c.mount_height, "fwd": c.mount_forward, "pitch_deg": round(math.degrees(c.pitch), 2),
            "hfov_deg": round(math.degrees(c.hfov), 2), "attach": c.attach_height, "module_h": c.module_height,
            "max_range": c.max_range,
            "body_h": r.body_height, "body_l": r.body_length, "body_w": r.body_width}


def benchmarks(run: Path) -> list:
    """Fixed-benchmark results (benchmark.jsonl), without the per-run details."""
    f = Path(run) / "benchmark.jsonl"
    out = []
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r.pop("runs", None)
            out.append(r)
    return out


def releases(run: Path):
    out = []
    for f in sorted((run / "releases").glob("*/metrics.json")):
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        er = m.get("eval_random", {})
        out.append({"tag": m.get("tag", f.parent.name), "iter": m.get("iter"),
                    "success_rate": er.get("success_rate"), "collisions": er.get("collisions_per_episode"),
                    "mean_time_s": er.get("mean_time_s"),
                    "maps_success": (m.get("eval_maps") or {}).get("success_rate"),
                    "race_success": (m.get("eval_race") or {}).get("success_rate"),
                    "level": er.get("level")})
    return sorted(out, key=lambda r: r["iter"] or 0)


def make_handler(run: Path, log: LogReader, replay: Replay | None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, _, q = self.path.partition("?")
            params = dict(kv.split("=", 1) for kv in q.split("&") if "=" in kv)
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/log":
                n = int(params.get("since", 0))
                rows = log.since(n)
                return self._send(200, json.dumps({"start": n, "rows": rows}).encode())
            if path == "/api/releases":
                return self._send(200, json.dumps(releases(run)).encode())
            if path == "/api/benchmark":
                return self._send(200, json.dumps(benchmarks(run)).encode())
            if path == "/api/replay":
                snap = replay.snapshot() if replay else {"status": "replay disabled"}
                return self._send(200, json.dumps(snap).encode())
            if path == "/api/info":
                return self._send(200, json.dumps({"run": str(run)}).encode())
            self._send(404, b"not found", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser(description="HamsterSuper live training dashboard")
    ap.add_argument("--run", default="runs/ppo", help="training output directory")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--level", type=int, default=None, help="replay maze level (default: curriculum level)")
    ap.add_argument("--ideal-pose", action="store_true", help="replay with the true pose instead of the full nav stack")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    log = LogReader(run / "train_log.jsonl")
    replay = None if args.no_replay else Replay(run, args.level, not args.ideal_pose, args.speed)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(run, log, replay))
    print(f"training dashboard for {run}: http://localhost:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
