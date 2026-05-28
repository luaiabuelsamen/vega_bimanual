"""cleanrl-style PPO for Vega + f5d6 single-trajectory tracking.

Trains the residual policy against the env's reference (synthetic by default).
This validates the env + reward + action space before the retargeting workstream.

Run:
    PYTHONPATH=. python scripts/train.py --total-steps 500000 --num-envs 8
Checkpoints + a metrics CSV land in runs/<run-name>/.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dextrack_vega.envs.vec_env import make_vec_env
from dextrack_vega.learning import ActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None)
    p.add_argument("--ref", default=None, help="ReferenceTrajectory .npz to track")
    p.add_argument("--sides", default="R", help="arms to control, e.g. 'R' or 'R,L'")
    p.add_argument("--vec", default="sync", choices=["sync", "process"],
                   help="'process' runs each env on its own core (multi-core)")
    # The training env's object MUST match the one the reference was generated
    # with (geometry, mass, spawn) — otherwise the policy squeezes a different
    # object than the reference poses were designed for.
    p.add_argument("--obj-type", default="box")
    p.add_argument("--obj-mass", type=float, default=0.08)
    p.add_argument("--obj-dims", default=None, help="comma list, e.g. 0.04,0.045,0.07")
    p.add_argument("--obj-pos", default=None, help="comma list x,y,z")
    p.add_argument("--obj-friction", default=None,
                   help="MuJoCo friction tuple 'slide spin roll', e.g. '4.0 0.1 0.002'")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--update-epochs", type=int, default=5)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p.parse_args()


def main():
    args = parse_args()
    args.run_name = args.run_name or f"track_{int(time.time())}"
    out = Path("runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[train] run={args.run_name} device={device}")

    sides = args.sides.split(",")
    ref = None
    if args.ref:
        from dextrack_vega.utils import trajectory as tj
        from dextrack_vega import config as C
        joints = [j for s in sides for j in C.controlled_joints(s)]
        ref = tj.load_npz(args.ref, joints)
        print(f"[train] tracking reference {args.ref} ({ref.n_frames} frames, "
              f"sides={sides}, {len(joints)}-DoF)")
    obj_kwargs = dict(obj_type=args.obj_type, obj_mass=args.obj_mass)
    if args.obj_dims:
        obj_kwargs["obj_dims"] = tuple(float(x) for x in args.obj_dims.split(","))
    if args.obj_pos:
        obj_kwargs["obj_pos"] = tuple(float(x) for x in args.obj_pos.split(","))
    if args.obj_friction:
        obj_kwargs["obj_friction"] = args.obj_friction
    envs = make_vec_env(args.vec, num_envs=args.num_envs, seed=args.seed,
                        sides=sides, ref=ref, **obj_kwargs)
    print(f"[train] vec={args.vec} num_envs={args.num_envs} obj={obj_kwargs}")
    obs_dim, act_dim = envs.obs_dim, envs.action_dim
    agent = ActorCritic(obs_dim, act_dim).to(device)
    opt = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    N, T = args.num_envs, args.rollout_steps
    batch_size = N * T
    mb_size = batch_size // args.num_minibatches

    # rollout storage
    obs_buf = torch.zeros((T, N, obs_dim), device=device)
    act_buf = torch.zeros((T, N, act_dim), device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    csv_path = out / "metrics.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["global_step", "ep_return", "obj_pos_err_cm", "rot_err_rad",
                     "hand_qpos_err", "value_loss", "policy_loss", "sps"])

    next_obs = torch.tensor(envs.reset(), device=device)
    next_done = torch.zeros(N, device=device)
    global_step = 0
    start = time.time()
    num_updates = args.total_steps // batch_size

    # running episode-return tracker (per env)
    ep_ret = np.zeros(N)
    recent_returns, recent_poserr, recent_roterr, recent_qerr = [], [], [], []

    for update in range(1, num_updates + 1):
        for step in range(T):
            global_step += N
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            val_buf[step] = value.flatten()
            act_buf[step] = action
            logp_buf[step] = logp

            a_np = torch.clamp(action, -1, 1).cpu().numpy()
            obs_np, rew_np, done_np, infos = envs.step(a_np)
            rew_buf[step] = torch.tensor(rew_np, device=device)
            next_obs = torch.tensor(obs_np, device=device)
            next_done = torch.tensor(done_np.astype(np.float32), device=device)

            ep_ret += rew_np
            for i in range(N):
                if done_np[i]:
                    recent_returns.append(ep_ret[i]); ep_ret[i] = 0.0
                    recent_poserr.append(infos[i].get("_obj_pos_err", np.nan) * 100)
                    recent_roterr.append(infos[i].get("_obj_rot_err", np.nan))
                    recent_qerr.append(infos[i].get("_hand_qpos_err", np.nan))

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            adv = torch.zeros_like(rew_buf, device=device)
            lastgae = 0
            for t in reversed(range(T)):
                nextnonterm = 1.0 - (next_done if t == T - 1 else done_buf[t + 1])
                nextval = next_value if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nextval * nextnonterm - val_buf[t]
                lastgae = delta + args.gamma * args.gae_lambda * nextnonterm * lastgae
                adv[t] = lastgae
            returns = adv + val_buf

        # flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)

        idx = np.arange(batch_size)
        v_loss = p_loss = 0.0
        for _ in range(args.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, batch_size, mb_size):
                mb = idx[s:s + mb_size]
                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb], b_act[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss_mb = 0.5 * ((newval.flatten() - b_ret[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss_mb
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
                v_loss, p_loss = float(v_loss_mb), float(pg_loss)

        sps = int(global_step / (time.time() - start))
        mret = np.mean(recent_returns[-50:]) if recent_returns else np.nan
        mpos = np.nanmean(recent_poserr[-50:]) if recent_poserr else np.nan
        mrot = np.nanmean(recent_roterr[-50:]) if recent_roterr else np.nan
        mqer = np.nanmean(recent_qerr[-50:]) if recent_qerr else np.nan
        writer.writerow([global_step, mret, mpos, mrot, mqer, v_loss, p_loss, sps])
        csv_f.flush()
        if update % 5 == 0 or update == 1:
            print(f"upd {update:4d}  step {global_step:>8d}  ret {mret:7.1f}  "
                  f"objErr {mpos:5.1f}cm  rotErr {mrot:4.2f}  qErr {mqer:.3f}  "
                  f"vloss {v_loss:6.2f}  {sps} sps", flush=True)
        # periodic checkpoint so a crash (e.g. an OOM-killed worker) doesn't
        # discard the whole run — always leaves a recent, loadable ckpt.pt.
        if update % 100 == 0:
            torch.save({"model": agent.state_dict(), "obs_dim": obs_dim,
                        "act_dim": act_dim}, out / "ckpt.pt")

    torch.save({"model": agent.state_dict(), "obs_dim": obs_dim,
                "act_dim": act_dim}, out / "ckpt.pt")
    csv_f.close()
    print(f"[train] saved {out/'ckpt.pt'}", flush=True)


if __name__ == "__main__":
    main()
