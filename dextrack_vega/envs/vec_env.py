"""Vectorized envs: in-process `SyncVectorEnv` and multi-core `ProcessVectorEnv`.

`SyncVectorEnv` runs N VegaTrackingEnv copies in a plain Python loop — simple and
robust, but single-core: the MuJoCo sim of all N envs runs sequentially, so it is
the throughput bottleneck (~1 core). `ProcessVectorEnv` puts each env in its own
process and steps them concurrently, using all CPU cores. Only numpy arrays cross
the pipe (never un-picklable MuJoCo objects): each worker *constructs its own*
env. The parent builds the shared scene XML once (rebuild=True) and workers load
it (rebuild=False) so they don't race writing the same file. mujoco_warp on GPU
is the eventual large-scale path; ProcessVectorEnv is the zero-dependency
multi-core accelerator that works today on Jetson (no JAX/MJX CUDA wheels).

Both auto-reset each sub-env on `done` (cleanrl convention): the returned `obs`
is the *next* episode's first obs, and the terminal obs is stashed in
`infos[i]["terminal_obs"]`.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from .tracking_env import VegaTrackingEnv


class SyncVectorEnv:
    def __init__(self, num_envs: int = 8, seed: int = 0, **env_kwargs):
        self.num_envs = num_envs
        self.envs = [VegaTrackingEnv(seed=seed + i, **env_kwargs)
                     for i in range(num_envs)]
        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs]).astype(np.float32)

    def step(self, actions: np.ndarray):
        n = self.num_envs
        obs = np.zeros((n, self.obs_dim), np.float32)
        rews = np.zeros(n, np.float32)
        dones = np.zeros(n, np.bool_)
        infos: list[dict] = [{} for _ in range(n)]
        for i, env in enumerate(self.envs):
            o, r, d, comp = env.step(actions[i])
            rews[i] = r
            dones[i] = d
            infos[i] = comp
            if d:
                infos[i] = dict(comp)
                infos[i]["terminal_obs"] = o
                infos[i]["episode_return_term"] = comp  # last-step terms
                o = env.reset()
            obs[i] = o
        return obs, rews, dones, infos

    def close(self):
        self.envs.clear()


def _worker(remote, env_kwargs, seed):
    # Each worker owns one env; the scene XML is already on disk (parent built
    # it), so rebuild=False — no race on the shared file.
    env = VegaTrackingEnv(seed=seed, rebuild=False, **env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                o, r, d, info = env.step(data)
                if d:
                    info = dict(info)
                    info["terminal_obs"] = o
                    o = env.reset()
                remote.send((o, r, d, info))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        remote.close()


class ProcessVectorEnv:
    """Multi-core vectorized env. Steps every sub-env concurrently across
    processes (one core each). Drop-in for SyncVectorEnv."""

    def __init__(self, num_envs: int = 8, seed: int = 0, **env_kwargs):
        self.num_envs = num_envs
        # Build the (identical) scene once in the parent and read back the
        # action/obs sizes; workers then load the file without rebuilding.
        probe = VegaTrackingEnv(seed=seed, rebuild=True, **env_kwargs)
        self.obs_dim, self.action_dim = probe.obs_dim, probe.action_dim
        del probe

        ctx = mp.get_context("fork")  # fork: workers inherit the built XML path
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.procs = []
        for i, wr in enumerate(work_remotes):
            p = ctx.Process(target=_worker, args=(wr, env_kwargs, seed + i),
                            daemon=True)
            p.start()
            wr.close()  # parent keeps only its end
            self.procs.append(p)
        self.closed = False

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes]).astype(np.float32)

    def step(self, actions: np.ndarray):
        # send all actions first, then collect — this is what overlaps the N
        # sims across cores (vs SyncVectorEnv's sequential loop).
        for remote, a in zip(self.remotes, actions):
            remote.send(("step", a))
        results = [r.recv() for r in self.remotes]
        n = self.num_envs
        obs = np.zeros((n, self.obs_dim), np.float32)
        rews = np.zeros(n, np.float32)
        dones = np.zeros(n, np.bool_)
        infos: list[dict] = [{} for _ in range(n)]
        for i, (o, r, d, info) in enumerate(results):
            obs[i] = o
            rews[i] = r
            dones[i] = d
            infos[i] = info
        return obs, rews, dones, infos

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        self.closed = True


def make_vec_env(backend: str, num_envs: int, seed: int = 0, **env_kwargs):
    """Factory: backend in {'sync','process'}."""
    if backend == "process":
        return ProcessVectorEnv(num_envs=num_envs, seed=seed, **env_kwargs)
    return SyncVectorEnv(num_envs=num_envs, seed=seed, **env_kwargs)
