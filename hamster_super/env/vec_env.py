"""Multiprocess vectorized environment with same-step auto-reset."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from ..config import EnvConfig


def _make_envs(cfg, n, seed, level, maps):
    from .hamster_env import HamsterMazeEnv
    return [HamsterMazeEnv(cfg, maps=maps, level=level, seed=seed + i) for i in range(n)]


def _step_all(envs, actions):
    obs, rew, done, infos = [], [], [], []
    for e, a in zip(envs, actions):
        o, r, te, tr, info = e.step(a)
        if te or tr:
            info["terminal_obs"] = o
            info["truncated"] = bool(tr and not te)
            o, _ = e.reset()
        obs.append(o)
        rew.append(r)
        done.append(te or tr)
        infos.append(info)
    return np.stack(obs), np.asarray(rew, np.float32), np.asarray(done), infos


def _worker(conn, cfg, n, seed, level, maps):
    envs = _make_envs(cfg, n, seed, level, maps)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(np.stack([e.reset()[0] for e in envs]))
            elif cmd == "step":
                conn.send(_step_all(envs, data))
            elif cmd == "set_level":
                for e in envs:
                    e.level = int(data)
                conn.send(None)
            elif cmd == "close":
                break
    finally:
        conn.close()


class VecEnv:
    def __init__(self, cfg: EnvConfig, n_envs: int, n_workers: int = 1, seed: int = 0,
                 level: int = 0, maps=None):
        self.n = n_envs
        self.n_workers = max(1, min(n_workers, n_envs))
        per = np.array_split(np.arange(n_envs), self.n_workers)
        self.slices = [(int(p[0]), int(p[-1]) + 1) for p in per]
        if self.n_workers == 1:
            self.envs = _make_envs(cfg, n_envs, seed, level, maps)
            self.conns = None
        else:
            ctx = mp.get_context("spawn" if mp.get_start_method(allow_none=True) != "fork" else "fork")
            self.conns, self.procs = [], []
            for w, (a, b) in enumerate(self.slices):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, cfg, b - a, seed + a * 1000, level, maps),
                                daemon=True)
                p.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(p)

    def reset(self):
        if self.conns is None:
            return np.stack([e.reset()[0] for e in self.envs])
        for c in self.conns:
            c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        if self.conns is None:
            return _step_all(self.envs, actions)
        for c, (a, b) in zip(self.conns, self.slices):
            c.send(("step", actions[a:b]))
        res = [c.recv() for c in self.conns]
        return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
                np.concatenate([r[2] for r in res]), [i for r in res for i in r[3]])

    def set_level(self, level: int):
        if self.conns is None:
            for e in self.envs:
                e.level = level
            return
        for c in self.conns:
            c.send(("set_level", level))
        for c in self.conns:
            c.recv()

    def close(self):
        if self.conns is None:
            return
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=2)
