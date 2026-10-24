"""PPO training with curriculum and automatic GitHub releases.

    python -m hamster_super.rl.train --iters 5000 --envs 32 --workers 4
    # every 500 iterations: checkpoint + TorchScript export + eval + GitHub Release

One iteration = one PPO update over envs*steps transitions.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import dataclasses
import json
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import EnvConfig, save_json, to_dict
from ..env.vec_env import VecEnv
from ..maps.generator import COURSE_LEVELS, LEVELS, RACE_LEVEL
from ..nav.stack import obs_dim
from .evaluate import evaluate, load_maps
from .policy import ActorCritic, DeployPolicy, RunningNorm, export_numpy


@dataclasses.dataclass
class PPOConfig:
    n_envs: int = 32
    n_steps: int = 128
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden: tuple = (256, 256)


class Curriculum:
    """Promote to the next map level when the recent success rate at the current level is high."""

    def __init__(self, level=0, window=150, promote_at=0.7):
        self.level, self.window, self.promote_at = level, window, promote_at
        self.hist = deque(maxlen=window)

    def record(self, info) -> bool:
        if info.get("level") != self.level:
            return False
        self.hist.append(float(info["success"]))
        if (len(self.hist) == self.window and np.mean(self.hist) >= self.promote_at
                and self.level < COURSE_LEVELS - 1):
            self.level += 1
            self.hist.clear()
            return True
        return False


class Trainer:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = EnvConfig(known_map=args.known_map, full_stack_prob=args.full_stack_prob)
        self.ppo = PPOConfig(n_envs=args.envs, n_steps=args.steps, lr=args.lr)
        torch.manual_seed(args.seed)
        torch.set_num_threads(max(1, args.torch_threads))
        self.n_obs = obs_dim(self.cfg)
        self.n_act = 3            # left wheel, right wheel, racing-line side (left/right = in/out, learned)
        self.ac = ActorCritic(self.n_obs, self.n_act, self.ppo.hidden)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.ppo.lr, eps=1e-5)
        self.norm = RunningNorm(self.n_obs)
        self.cur = Curriculum(level=args.start_level)
        self.iter = 0
        self.episodes = 0
        self.best_eval = -1.0
        if args.resume:
            self.load(args.resume)
        if args.set_level is not None:          # e.g. after changing the task
            self.cur.level = args.set_level
            self.cur.hist.clear()
        self.eval_maps = load_maps(args.eval_maps) if args.eval_maps and Path(args.eval_maps).exists() else None
        self.releaser = None
        if args.release_every > 0 and not args.no_release:
            from ..release.github_release import GitHubReleaser
            try:
                self.releaser = GitHubReleaser(repo=args.repo, dry_run=args.release_dry_run,
                                               method=args.release_method)
            except Exception as e:
                print(f"[release] disabled: {e}")
        save_json(self.cfg, self.out / "env_config.json")
        self.pages = None
        if args.pages_every > 0:
            from .pages import PagesPublisher
            try:
                self.pages = PagesPublisher(self.out)
            except Exception as e:
                print(f"[pages] disabled: {e}")

    # ------------------------------------------------------------ checkpoint
    def state(self):
        return {"model": self.ac.state_dict(), "opt": self.opt.state_dict(), "obs_norm": self.norm.state_dict(),
                "iter": self.iter, "episodes": self.episodes, "level": self.cur.level,
                "obs_dim": self.n_obs, "act_dim": self.n_act, "hidden": self.ppo.hidden,
                "env_config": to_dict(self.cfg)}

    def save(self, path):
        """Atomic write so readers (the live monitor) never see a partial file."""
        tmp = Path(str(path) + ".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("obs_dim", self.n_obs) < self.n_obs:
            # new observation inputs: zero weights, so the old behaviour is kept exactly
            # until training finds a use for them
            extra = self.n_obs - ck["obs_dim"]
            sd = ck["model"]
            for net in ("actor", "critic"):
                k = f"{net}.0.weight"
                sd[k] = torch.cat([sd[k], torch.zeros(sd[k].shape[0], extra)], dim=1)
            ck["opt"] = None
            on = ck["obs_norm"]
            on["mean"] = list(on["mean"]) + [0.0] * extra
            on["var"] = list(on["var"]) + [1.0] * extra
            print(f"expanded {path} from {ck['obs_dim']} to {self.n_obs} observations (optimizer reset)")
            ck["obs_dim"] = self.n_obs
        if ck.get("act_dim", 2) < self.n_act:
            # older 2-action policy: keep everything, add a neutral line-bias output
            sd = ck["model"]
            last = max(k for k in sd if k.startswith("actor.") and k.endswith(".weight"))
            bk = last.replace(".weight", ".bias")
            extra = self.n_act - sd[last].shape[0]
            sd[last] = torch.cat([sd[last], torch.zeros(extra, sd[last].shape[1])])
            sd[bk] = torch.cat([sd[bk], torch.zeros(extra)])
            sd["log_std"] = torch.cat([sd["log_std"], torch.full((extra,), -0.5)])
            self.ac.load_state_dict(sd)
            print(f"expanded {path} from {ck.get('act_dim', 2)} to {self.n_act} actions (optimizer reset)")
        else:
            self.ac.load_state_dict(ck["model"])
            if ck.get("opt") is not None:
                self.opt.load_state_dict(ck["opt"])
        self.norm.load_state_dict(ck["obs_norm"])
        a = self.args
        if getattr(a, "reopen_std", 0) > 0:
            # after a rule change: let the policy try things again (old habits are
            # only unlearned if the alternatives get tried)
            with torch.no_grad():
                self.ac.log_std.clamp_(min=float(np.log(a.reopen_std)))
            print(f"exploration reopened: std >= {a.reopen_std}")
        if getattr(a, "renorm_count", 0) > 0 and self.norm.count > a.renorm_count:
            # keep the statistics but let new observations move them quickly
            print(f"observation normalizer: count {self.norm.count:.3g} -> {a.renorm_count:.3g}")
            self.norm.count = float(a.renorm_count)
        self.iter, self.episodes = ck["iter"], ck.get("episodes", 0)
        self.cur.level = ck.get("level", self.cur.level)
        print(f"resumed {path} at iter {self.iter}, level {self.cur.level}")

    def export(self, path):
        """TorchScript export + a torch-free NumPy copy (policy.npz) next to it."""
        pol = DeployPolicy(self.ac, self.norm).eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.jit.script(pol).save(str(path))
        export_numpy(self.ac, self.norm, Path(path).with_name("policy.npz"))
        return pol

    # --------------------------------------------------------------- rollout
    def train(self):
        a, p = self.args, self.ppo
        t_start = time.time()
        venv = VecEnv(self.cfg, p.n_envs, a.workers, seed=a.seed, level=self.cur.level)
        obs = venv.reset()
        N, T = p.n_envs, p.n_steps
        log_f = open(self.out / "train_log.jsonl", "a")
        ep_ret = np.zeros(N)
        recent = deque(maxlen=200)
        t0 = time.time()
        try:
            while self.iter < a.iters:
                it_t = time.time()
                buf_o = np.zeros((T, N, self.n_obs), np.float32)
                buf_a = np.zeros((T, N, self.n_act), np.float32)
                buf_lp = np.zeros((T, N), np.float32)
                buf_v = np.zeros((T, N), np.float32)
                buf_r = np.zeros((T, N), np.float32)
                buf_d = np.zeros((T, N), np.float32)
                for t in range(T):
                    self.norm.update(obs)
                    on = self.norm(obs)
                    with torch.no_grad():
                        ot = torch.as_tensor(on)
                        dist = self.ac.dist(ot)
                        act = dist.sample()
                        lp = dist.log_prob(act).sum(-1)
                        v = self.ac.value(ot)
                    act_np = act.numpy()
                    nobs, rew, done, infos = venv.step(np.clip(act_np, -1, 1))
                    # bootstrap through time-limit truncation
                    for i, info in enumerate(infos):
                        if done[i] and info.get("truncated"):
                            with torch.no_grad():
                                tv = self.ac.value(torch.as_tensor(self.norm(info["terminal_obs"][None])))[0]
                            rew[i] += p.gamma * float(tv)
                    buf_o[t], buf_a[t], buf_lp[t], buf_v[t] = on, act_np, lp.numpy(), v.numpy()
                    buf_r[t], buf_d[t] = rew, done
                    ep_ret += rew
                    for i, info in enumerate(infos):
                        if done[i]:
                            self.episodes += 1
                            recent.append({**{k: info[k] for k in ("success", "time_s", "collisions", "level")},
                                           "ret": float(ep_ret[i])})
                            ep_ret[i] = 0
                            if self.cur.record(info):
                                print(f"*** curriculum -> level {self.cur.level}", flush=True)
                                venv.set_level(self.cur.level)
                            self._maybe_release_episode()
                    obs = nobs
                with torch.no_grad():
                    last_v = self.ac.value(torch.as_tensor(self.norm(obs))).numpy()
                stats = self._update(buf_o, buf_a, buf_lp, buf_v, buf_r, buf_d, last_v)
                self.iter += 1
                fps = N * T / (time.time() - it_t)
                row = {"iter": self.iter, "episodes": self.episodes, "level": self.cur.level, "fps": round(fps),
                       "elapsed_min": round((time.time() - t0) / 60, 2), **stats}
                if recent:
                    row.update(success=float(np.mean([r["success"] for r in recent])),
                               ep_return=float(np.mean([r["ret"] for r in recent])),
                               collisions=float(np.mean([r["collisions"] for r in recent])))
                    ok = [r["time_s"] for r in recent if r["success"]]
                    if ok:
                        row["finish_time"] = float(np.mean(ok))
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                if self.iter % a.log_every == 0:
                    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                          flush=True)
                if self.iter % a.save_every == 0:
                    self.save(self.out / "last.pt")
                if self.pages and self.iter % a.pages_every == 0:
                    self._publish_pages()
                if a.release_unit == "iter" and a.release_every > 0 and self.iter % a.release_every == 0:
                    self.release()
                if a.max_minutes > 0 and (time.time() - t_start) / 60 > a.max_minutes:
                    print(f"[train] {a.max_minutes} min wall time reached at iter {self.iter}: stopping", flush=True)
                    break
        finally:
            venv.close()
            log_f.close()
            self.save(self.out / "last.pt")
            if self.releaser:
                self.releaser.wait()

    def _publish_pages(self):
        """Snapshot the current policy, then record runs + push to GitHub Pages in the background."""
        import copy
        from .pages import record_episode
        pol = copy.deepcopy(DeployPolicy(self.ac, self.norm)).eval()
        cfg, level, it, n = self.cfg, self.cur.level, self.iter, self.args.pages_episodes

        def act(obs):
            with torch.no_grad():
                return pol(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]

        def make():
            return [record_episode(act, cfg, level, seed=it * 100 + i, model_iter=it) for i in range(n)]
        self.pages.publish_async(make, iteration=it)

    def _benchmark_async(self, ck, rdir):
        """Fixed benchmark of a released checkpoint in the background (2 processes,
        low priority); results go to <run>/benchmark.jsonl and the release folder."""
        if self.args.no_benchmark:
            return
        out = self.out / "benchmark.jsonl"
        cmd = (f"nice -n 10 {sys.executable} -m hamster_super.rl.benchmark {ck} --out {out} --workers 2 "
               f"> {rdir / 'benchmark.txt'} 2>&1")
        subprocess.Popen(cmd, shell=True, cwd=str(Path(__file__).resolve().parents[2]))

    def _maybe_release_episode(self):
        a = self.args
        if a.release_unit == "episode" and a.release_every > 0 and self.episodes % a.release_every == 0:
            self.release()

    def _update(self, o, a, lp, v, r, d, last_v):
        p = self.ppo
        T, N = r.shape
        adv = np.zeros_like(r)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else v[t + 1]
            nonterm = 1.0 - d[t]
            delta = r[t] + p.gamma * nv * nonterm - v[t]
            gae = delta + p.gamma * p.lam * nonterm * gae
            adv[t] = gae
        ret = adv + v
        B = T * N
        O = torch.as_tensor(o.reshape(B, -1))
        A = torch.as_tensor(a.reshape(B, -1))
        LP = torch.as_tensor(lp.reshape(B))
        ADV = torch.as_tensor(adv.reshape(B))
        RET = torch.as_tensor(ret.reshape(B))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        mb = B // p.minibatches
        kls, pls, vls, ents = [], [], [], []
        for _ in range(p.epochs):
            perm = torch.randperm(B)
            for k in range(p.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                dist = self.ac.dist(O[idx])
                nlp = dist.log_prob(A[idx]).sum(-1)
                ratio = (nlp - LP[idx]).exp()
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - p.clip, 1 + p.clip) * ADV[idx]).mean()
                vl = 0.5 * (self.ac.value(O[idx]) - RET[idx]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + p.vf_coef * vl - p.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), p.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kls.append(float((LP[idx] - nlp).mean()))
                pls.append(pg.item())
                vls.append(vl.item())
                ents.append(ent.item())
            if np.mean(kls[-p.minibatches:]) > p.target_kl:
                break
        with torch.no_grad():
            self.ac.log_std.clamp_(-2.5, 0.5)
        return {"pi_loss": float(np.mean(pls)), "v_loss": float(np.mean(vls)), "entropy": float(np.mean(ents)),
                "kl": float(np.mean(kls)), "std": float(self.ac.log_std.exp().mean())}

    # --------------------------------------------------------------- release
    def release(self):
        a = self.args
        n = self.iter if a.release_unit == "iter" else self.episodes
        tag = f"{a.tag_prefix}-{a.release_unit}{n:06d}"
        rdir = self.out / "releases" / tag
        rdir.mkdir(parents=True, exist_ok=True)
        ck = rdir / "checkpoint.pt"
        ts = rdir / "policy_ts.pt"
        self.save(ck)
        pol = self.export(ts)
        t = time.time()
        metrics = {"tag": tag, "iter": self.iter, "episodes": self.episodes, "curriculum_level": self.cur.level}
        # at the size being learned now, and at the real race size (12 x 12 and up)
        metrics["eval_random"] = evaluate(pol, self.cfg, a.eval_episodes, level=self.cur.level)
        metrics["eval_random"]["level"] = self.cur.level
        metrics["eval_race"] = evaluate(pol, self.cfg, max(10, a.eval_episodes // 3), level=RACE_LEVEL)
        if self.eval_maps:
            metrics["eval_maps"] = evaluate(pol, self.cfg, max(a.eval_episodes // 2, len(self.eval_maps)),
                                            maps=self.eval_maps)
        metrics["eval_seconds"] = round(time.time() - t, 1)
        (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        save_json(self.cfg, rdir / "env_config.json")
        sr = metrics["eval_random"]["success_rate"]
        if sr >= self.best_eval:
            self.best_eval = sr
            self.save(self.out / "best.pt")
            self.export(self.out / "best_policy_ts.pt")
        from ..release.github_release import release_notes
        notes = release_notes(metrics)
        print(f"[release] {tag}: eval success {sr:.1%}", flush=True)
        self._benchmark_async(ck, rdir)
        if self.releaser:
            self.releaser.publish(tag, f"HamsterSuper {tag}", notes,
                                  [ts, rdir / "policy.npz", ck, rdir / "metrics.json", rdir / "env_config.json"])


def main():
    ap = argparse.ArgumentParser(description="HamsterSuper PPO training")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="parallel simulation processes (default: one per CPU core)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--start-level", type=int, default=0)
    ap.add_argument("--set-level", type=int, default=None, help="override the curriculum level after --resume")
    ap.add_argument("--known-map", action="store_true", help="the planner knows the full map")
    ap.add_argument("--full-stack-prob", type=float, default=0.0,
                    help="fraction of episodes driven with odometry + particle-filter localization (0: walls judged at the real pose)")
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=60)
    ap.add_argument("--eval-maps", default="maps")
    # Release: every 500 iterations (or episodes) to GitHub Releases
    ap.add_argument("--release-every", type=int, default=500)
    ap.add_argument("--release-unit", choices=["iter", "episode"], default="iter")
    ap.add_argument("--tag-prefix", default="model")
    ap.add_argument("--repo", default=None, help="owner/repo (default: git origin)")
    ap.add_argument("--release-method", choices=["auto", "gh", "api", "branch", "tag"], default="auto",
                    help="auto: gh CLI if logged in, else GITHUB_TOKEN, else push to the model-releases "
                         "branch and let GitHub Actions create the release")
    ap.add_argument("--release-dry-run", action="store_true")
    ap.add_argument("--no-release", action="store_true")
    ap.add_argument("--pages-every", type=int, default=0,
                    help="publish the dashboard to GitHub Pages (gh-pages branch) every N iterations")
    ap.add_argument("--pages-episodes", type=int, default=3, help="recorded runs per Pages update")
    ap.add_argument("--reopen-std", type=float, default=0.0,
                    help="on --resume: raise the policy std to at least this (re-explore after a rule change)")
    ap.add_argument("--renorm-count", type=float, default=0.0,
                    help="on --resume: cap the observation normalizer count so it adapts to new inputs quickly")
    ap.add_argument("--no-benchmark", action="store_true", help="skip the fixed benchmark at each release")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop (and save) after this much wall time, e.g. inside a CI job")
    ap.add_argument("--monitor", action="store_true", help="also start the live training dashboard")
    ap.add_argument("--monitor-port", type=int, default=8001)
    args = ap.parse_args()
    mon = None
    if args.monitor:
        mon = subprocess.Popen([sys.executable, "-m", "hamster_super.rl.monitor", "--run", args.out,
                                "--port", str(args.monitor_port)])
        print(f"live dashboard: http://localhost:{args.monitor_port}", flush=True)
    try:
        Trainer(args).train()
    finally:
        if mon is not None:
            mon.terminate()


if __name__ == "__main__":
    main()
