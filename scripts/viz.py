"""Interactive MuJoCo viewer for the Vega + f5d6 tracking env.

Plays the reference trajectory in real time. By default the policy is the
zero-residual "pure kinematic bias" controller, so you watch the arm/hand
follow the reference joints while physics integrates the object.

Usage:
    python scripts/viz.py                 # play synthetic reference, looping
    python scripts/viz.py --ref path.npz  # play a saved ReferenceTrajectory
    python scripts/viz.py --random        # random residual actions (stress test)
"""
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from dextrack_vega.envs import VegaTrackingEnv
from dextrack_vega.utils import trajectory as tj
from dextrack_vega import config as C


def _load_policy(ckpt_path, obs_dim, act_dim):
    """Load a trained ActorCritic; returns a deterministic action fn."""
    import torch
    from dextrack_vega.learning import ActorCritic
    ck = torch.load(ckpt_path, map_location="cpu")
    agent = ActorCritic(ck["obs_dim"], ck["act_dim"])
    agent.load_state_dict(ck["model"])
    agent.eval()
    assert ck["obs_dim"] == obs_dim and ck["act_dim"] == act_dim, "ckpt/env mismatch"

    def policy(obs):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return agent.act_deterministic(x).squeeze(0).numpy()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="R")
    ap.add_argument("--ref", default=None, help="path to a ReferenceTrajectory .npz")
    ap.add_argument("--random", action="store_true", help="random residual actions")
    ap.add_argument("--ckpt", default=None, help="run a trained policy checkpoint")
    ap.add_argument("--playback", action="store_true",
                    help="kinematic playback of --ref (drive recorded qpos directly)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = ap.parse_args()

    ref = None
    if args.ref:
        ref = tj.load_npz(args.ref, C.controlled_joints(args.side))
    env = VegaTrackingEnv(side=args.side, ref=ref)
    rng = np.random.default_rng(0)

    policy = None
    if args.ckpt:
        policy = _load_policy(args.ckpt, env.obs_dim, env.action_dim)
    mode = ("playback" if args.playback else
            "policy" if policy else
            "random" if args.random else "zero-residual")

    print(f"[viz] obs={env.obs_dim} act={env.action_dim} ref_dur={env.ref.duration:.2f}s "
          f"ctrl_dt={env.dt:.3f}s mode={mode}  (Ctrl-C or close window to quit)")

    obs = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # frame the right arm workspace
        viewer.cam.lookat[:] = [0.55, -0.15, 0.85]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        if args.playback:
            _run_playback(env, viewer, args.speed)
            return

        while viewer.is_running():
            t0 = time.time()
            if policy is not None:
                action = policy(obs)
            elif args.random:
                action = rng.uniform(-1, 1, env.action_dim)
            else:
                action = np.zeros(env.action_dim)
            obs, reward, done, comp = env.step(action)
            viewer.sync()
            if done:
                obs = env.reset()
            # real-time pacing
            dt = env.dt / max(args.speed, 1e-3)
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


def _run_playback(env, viewer, speed):
    """Drive the recorded reference qpos (hand + object) directly — shows the
    exact demo that was generated, with no dynamics/contact drift."""
    ref = env.ref
    m, d = env.model, env.data
    while viewer.is_running():
        for f in range(ref.n_frames):
            t0 = time.time()
            d.qpos[env._jnt_qposadr] = ref.hand_qpos[f]
            d.qpos[env._obj_qadr:env._obj_qadr + 3] = ref.obj_pos[f]
            d.qpos[env._obj_qadr + 3:env._obj_qadr + 7] = ref.obj_quat[f]
            for j, v in C.HOME_POSTURE.items():
                d.qpos[m.jnt_qposadr[env._jid(j)]] = v
            mujoco.mj_forward(m, d)
            viewer.sync()
            if not viewer.is_running():
                break
            sleep = ref.dt / max(speed, 1e-3) - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
