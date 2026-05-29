"""Behavior-cloning to a zero-residual policy.

The bimanual squeeze-lift's open-loop (zero-residual) trajectory already lifts
the box +21 cm in physics. Vanilla PPO can't preserve this — even tiny action
noise compounds over 170 steps and breaks the friction grip (we verified 4
configurations). BC bypasses PPO: collect the open-loop rollout (obs, action=0)
and train the actor's mean network via MSE to output zero for those obs.
The resulting deterministic policy reproduces the +21 cm lift.

    PYTHONPATH=. python scripts/bc_zero.py --ref demos/bim_lift_box.npz \
        --sides R,L --obj-mass 0.05 --obj-dims 0.04,0.045,0.07 \
        --obj-pos 0.55,0.0,0.80 --epochs 200 --out runs/bim_lift_bc/ckpt.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from dextrack_vega.envs import VegaTrackingEnv
from dextrack_vega.utils import trajectory as tj
from dextrack_vega import config as C
from dextrack_vega.learning import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sides", default="R")
    ap.add_argument("--obj-type", default="box")
    ap.add_argument("--obj-mass", type=float, default=0.08)
    ap.add_argument("--obj-dims", default=None)
    ap.add_argument("--obj-pos", default=None)
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/bc/ckpt.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sides = args.sides.split(",")
    joints = [j for s in sides for j in C.controlled_joints(s)]
    ref = tj.load_npz(args.ref, joints)
    obj_kw = dict(obj_type=args.obj_type, obj_mass=args.obj_mass)
    if args.obj_dims:
        obj_kw["obj_dims"] = tuple(float(x) for x in args.obj_dims.split(","))
    if args.obj_pos:
        obj_kw["obj_pos"] = tuple(float(x) for x in args.obj_pos.split(","))

    # collect (obs, action=0) pairs from zero-residual rollouts
    obs_buf = []
    for r in range(args.rollouts):
        env = VegaTrackingEnv(sides=sides, ref=ref, seed=args.seed + r, **obj_kw)
        obs = env.reset()
        done = False
        while not done:
            obs_buf.append(obs)
            obs, _, done, _ = env.step(np.zeros(env.action_dim))
    obs_buf = np.stack(obs_buf).astype(np.float32)
    print(f"[bc] collected {len(obs_buf)} (obs, action=0) pairs from "
          f"{args.rollouts} zero-residual rollouts")

    # train actor_mean to output zero on these obs (MSE)
    torch.manual_seed(args.seed)
    env0 = VegaTrackingEnv(sides=sides, ref=ref, **obj_kw)
    agent = ActorCritic(env0.obs_dim, env0.action_dim)
    opt = torch.optim.Adam(agent.actor_mean.parameters(), lr=args.lr)
    obs_t = torch.tensor(obs_buf)
    tgt = torch.zeros(obs_t.shape[0], env0.action_dim)
    bs = 256
    for ep in range(args.epochs):
        idx = np.random.permutation(len(obs_buf))
        ep_loss = 0; nb = 0
        for s in range(0, len(idx), bs):
            mb = idx[s:s + bs]
            pred = agent.actor_mean(obs_t[mb])
            loss = ((pred - tgt[mb]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"[bc] ep {ep:4d}  mse {ep_loss / nb:.6f}")
    # also tighten logstd so eval-time noise (if any) stays small
    with torch.no_grad():
        agent.actor_logstd.fill_(-5.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": agent.state_dict(),
                "obs_dim": env0.obs_dim, "act_dim": env0.action_dim}, args.out)
    print(f"[bc] saved {args.out}")


if __name__ == "__main__":
    main()
