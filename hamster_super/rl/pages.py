"""Publish the training dashboard to GitHub Pages.

GitHub Pages only serves static files, so the trainer periodically exports
a snapshot (learning curves, releases, a few recorded runs of the newest
model) as JSON next to a static copy of the dashboard and force-pushes it as
a single commit to the `gh-pages` branch. The page refreshes the data every
minute and plays the recorded runs back as an animation.

    python -m hamster_super.rl.train ... --pages-every 50      # publish while training
    python -m hamster_super.rl.pages --run runs/ppo            # publish once by hand

Deployment: the pushed commit carries .github/workflows/pages-snapshot.yml;
when it runs, .github/workflows/pages.yml on the default branch deploys the
snapshot with GitHub Actions and switches the repository's Pages source to
"GitHub Actions" by itself, so no settings change is needed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

PAGE = Path(__file__).with_name("monitor.html")
LOG_KEYS = ("iter", "episodes", "level", "fps", "elapsed_min", "success", "ep_return", "collisions",
            "finish_time", "std")
MAX_ROWS = 1500


# ------------------------------------------------------------------ recording
def record_episode(policy, cfg, level: int, seed: int, model_iter=None, full_stack: bool = True,
                   max_steps: int = 1500) -> dict:
    """Run one episode and return a compact, JSON-friendly recording."""
    from ..env.hamster_env import HamsterMazeEnv
    from ..maps.grid_map import cell_of
    from .monitor import camera_info, line_info

    env = HamsterMazeEnv(cfg, level=level, seed=seed, full_stack=full_stack)
    obs, _ = env.reset(options={"level": level})
    m = env.map
    r1 = lambda v: [round(float(x), 1) for x in v]
    frames, events, prev = [], [], set()
    done, info, t = False, {}, 0
    while not done and t < max_steps:
        obs, _, te, tr, info = env.step(policy(obs))
        done = te or tr
        walls = set(env.belief.walls)
        for e in walls - prev:
            events.append([t, 1, list(e[0]), list(e[1])])
        for e in prev - walls:
            events.append([t, -1, list(e[0]), list(e[1])])
        prev = walls
        c = cell_of(env.est)
        path = [list(c)] + [list(x) for x in env.planner.next_cells(c, 64)] if c in env.planner.dist else []
        pose, est = env.robot.pose, np.asarray(env.est)
        frames.append({"p": [round(float(pose[0]), 2), round(float(pose[1]), 2), round(float(pose[2]), 3)],
                       "e": [round(float(est[0]), 2), round(float(est[1]), 2), round(float(est[2]), 3)],
                       "r": r1(env.last_rays), "c": path, "k": env.collisions, **line_info(env.line)})
        t += 1
    return {
        "model_iter": model_iter, "level": level, "map_name": m.name, "bounds": list(env.bounds),
        "true_walls": [[list(a), list(b)] for a, b in sorted(m.walls)],
        "start": list(m.start), "goal": list(m.goal), "dt": env.cfg.dt, "opt_cells": env.opt_cells,
        "ray_angles": [round(float(a), 4) for a in env.cfg.camera.ray_angles()],
        "cam_forward": env.cfg.camera.mount_forward, "cam": camera_info(env.cfg),
        "full_stack": bool(env.use_stack),
        "frames": frames, "wall_events": events,
        "result": {"model_iter": model_iter, "success": bool(info.get("success")),
                   "time_s": info.get("time_s", round(t * env.cfg.dt, 1)), "collisions": env.collisions},
    }


def policy_from_checkpoint(path):
    """(numpy obs -> action, checkpoint dict) without TorchScript."""
    import torch
    from .policy import ActorCritic, DeployPolicy, RunningNorm
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ac = ActorCritic(ck["obs_dim"], ck.get("act_dim", 2), ck.get("hidden", (256, 256)))
    ac.load_state_dict(ck["model"])
    norm = RunningNorm(ck["obs_dim"])
    norm.load_state_dict(ck["obs_norm"])
    pol = DeployPolicy(ac, norm).eval()

    def act(obs):
        with torch.no_grad():
            return pol(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]
    return act, ck


# ------------------------------------------------------------------- export
def read_log(run: Path) -> list:
    path = run / "train_log.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in LOG_KEYS})
    # a resumed run re-logs a few iterations: keep the newest row per iteration
    rows = sorted({r["iter"]: r for r in rows if "iter" in r}.values(), key=lambda r: r["iter"])
    if len(rows) > MAX_ROWS:   # keep the shape, drop points evenly (always keep the newest)
        idx = np.unique(np.r_[np.linspace(0, len(rows) - 1, MAX_ROWS).astype(int), len(rows) - 1])
        rows = [rows[i] for i in idx]
    return rows


def export_site(run: Path, out: Path, episodes: list | None = None, title: str = ""):
    """Write index.html + data/*.json for a static dashboard into `out`."""
    from .monitor import releases
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    data.mkdir(exist_ok=True)
    html = PAGE.read_text(encoding="utf-8").replace(
        '<link rel="icon" href="data:,">', '<link rel="icon" href="data:,">\n<meta name="hs-static" content="1">')
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")
    now = int(time.time())
    (data / "log.json").write_text(json.dumps({"updated": now, "run": title or run.name, "rows": read_log(run)}))
    (data / "releases.json").write_text(json.dumps(releases(run)))
    # room backgrounds for the dome around the 3D course (same images as in training)
    from ..sim.backgrounds import DEFAULT_DIR
    import shutil
    names = []
    if DEFAULT_DIR.is_dir():
        (out / "backgrounds").mkdir(exist_ok=True)
        for f in sorted(DEFAULT_DIR.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                shutil.copy2(f, out / "backgrounds" / f.name)
                names.append("backgrounds/" + f.name)
    (data / "backgrounds.json").write_text(json.dumps(names))
    if episodes is not None:
        (data / "replays.json").write_text(json.dumps({"updated": now, "episodes": episodes},
                                                      separators=(",", ":")))
    elif not (data / "replays.json").exists():
        (data / "replays.json").write_text(json.dumps({"updated": now, "episodes": []}))


# ---------------------------------------------------------------- publishing
def _git(*args, cwd, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


class PagesPublisher:
    """Keeps a git worktree of the `gh-pages` branch under <run>/_pages and
    force-pushes a single-commit snapshot (no history growth)."""

    def __init__(self, run: Path, branch: str = "gh-pages", remote: str = "origin", trailer: str = ""):
        self.run, self.branch, self.remote, self.trailer = Path(run), branch, remote, trailer
        self.repo = Path(_git("rev-parse", "--show-toplevel", cwd=Path.cwd()).stdout.strip())
        self.wt = (self.run / "_pages").resolve()
        self._lock = threading.Lock()
        self._busy = False

    def _ensure_worktree(self):
        if (self.wt / ".git").exists():
            return
        _git("worktree", "prune", cwd=self.repo, check=False)
        self.wt.parent.mkdir(parents=True, exist_ok=True)
        _git("worktree", "add", "--detach", str(self.wt), cwd=self.repo)
        _git("checkout", "--orphan", self.branch + "-tmp", cwd=self.wt)
        _git("rm", "-rf", "--quiet", ".", cwd=self.wt, check=False)

    def publish(self, episodes: list | None, iteration=None) -> str:
        with self._lock:
            self._ensure_worktree()
            for p in self.wt.iterdir():
                if p.name != ".git":
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
            export_site(self.run, self.wt, episodes)
            if episodes is None:      # keep the previous recordings
                old = self.run / "_pages_replays.json"
                if old.exists():
                    shutil.copy(old, self.wt / "data" / "replays.json")
            else:
                shutil.copy(self.wt / "data" / "replays.json", self.run / "_pages_replays.json")
            # the deploy workflow must live in the pushed commit for the push to trigger it
            wf = self.repo / ".github" / "workflows" / "pages-snapshot.yml"
            if wf.exists():
                (self.wt / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
                shutil.copy(wf, self.wt / ".github" / "workflows" / "pages-snapshot.yml")
            _git("add", "-A", cwd=self.wt)
            msg = f"Training dashboard snapshot (iter {iteration})" + (f"\n\n{self.trailer}" if self.trailer else "")
            # a fresh orphan commit every time: the branch never accumulates history
            _git("checkout", "--orphan", self.branch + "-next", cwd=self.wt, check=False)
            _git("commit", "-q", "-m", msg, cwd=self.wt)
            _git("branch", "-D", self.branch, cwd=self.wt, check=False)
            _git("branch", "-m", self.branch, cwd=self.wt)
            res = _git("push", "-f", self.remote, f"{self.branch}:{self.branch}", cwd=self.wt, check=False)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.strip() or res.stdout.strip())
            return pages_url(self.repo)

    def publish_async(self, make_episodes, iteration=None):
        """Record + publish in a background thread; skipped if one is still running."""
        if self._busy:
            return

        def job():
            try:
                url = self.publish(make_episodes(), iteration)
                print(f"[pages] published iter {iteration} -> {url}", flush=True)
            except Exception as e:   # never stop training for the dashboard
                print(f"[pages] publish failed: {e}", flush=True)
            finally:
                self._busy = False
        self._busy = True
        threading.Thread(target=job, daemon=True).start()


def pages_url(repo: Path) -> str:
    try:
        from ..release.github_release import detect_repo
        owner, name = detect_repo(repo).split("/")
        return f"https://{owner.lower()}.github.io/{name}/"
    except Exception:
        return "(GitHub Pages)"


def main():
    ap = argparse.ArgumentParser(description="Publish the training dashboard to GitHub Pages")
    ap.add_argument("--run", default="runs/ppo")
    ap.add_argument("--episodes", type=int, default=3, help="recorded runs of the newest model")
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--out", default=None, help="only export to this folder (no git push)")
    ap.add_argument("--trailer", default="", help="extra text appended to the commit message")
    args = ap.parse_args()
    run = Path(args.run)
    eps = []
    ck_path = run / "last.pt"
    if args.episodes > 0 and ck_path.exists():
        from ..config import EnvConfig, from_dict
        pol, ck = policy_from_checkpoint(ck_path)
        cfg = from_dict(EnvConfig, ck["env_config"]) if "env_config" in ck else EnvConfig()
        level = ck.get("level", 0) if args.level is None else args.level
        eps = [record_episode(pol, cfg, level, seed=int(time.time()) + i, model_iter=ck.get("iter"))
               for i in range(args.episodes)]
    if args.out:
        export_site(run, Path(args.out), eps)
        print(f"exported to {args.out}")
        return
    it = eps[0]["model_iter"] if eps else None
    url = PagesPublisher(run, trailer=args.trailer).publish(eps or None, iteration=it)
    print(f"published -> {url}")


if __name__ == "__main__":
    main()
