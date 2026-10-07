"""Roll out the BC policies in physics and record what the object actually does.

Writes results/verified_rollouts.json: for each bimanual BC policy, the object's
net height change and yaw change in a deterministic physics rollout, next to
the kinematic reference's targets. This is the source of the README's numbers.

    PYTHONPATH=. python scripts/verify_policies.py
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from dextrack_vega import config as C
from dextrack_vega.envs import VegaTrackingEnv
from dextrack_vega.learning import ActorCritic
from dextrack_vega.utils import trajectory as tj

POLICIES = {
    "bim_lift_bc": dict(ref="demos/bim_lift_box.npz", ckpt="runs/bim_lift_bc/ckpt.pt",
                        obj_mass=0.05, obj_dims=(0.04, 0.045, 0.07), obj_pos=(0.55, 0.0, 0.80)),
    "bim_reorient_bc": dict(ref="demos/bim_reorient_k_box.npz", ckpt="runs/bim_reorient_bc/ckpt.pt",
                            obj_mass=0.1, obj_dims=(0.045, 0.045, 0.06), obj_pos=(0.55, 0.0, 0.80)),
}


def yaw_deg(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def rollout(name, ref, ckpt, **obj):
    sides = ["R", "L"]
    joints = [j for s in sides for j in C.controlled_joints(s)]
    traj = tj.load_npz(ref, joints)
    env = VegaTrackingEnv(sides=sides, ref=traj, obj_type="box", **obj)
    d, adr = env.data, env._obj_qadr
    ck = torch.load(ckpt, map_location="cpu")
    agent = ActorCritic(ck["obs_dim"], ck["act_dim"])
    agent.load_state_dict(ck["model"])
    agent.eval()

    obs = env.reset()
    z0, yaw0 = float(d.qpos[adr + 2]), yaw_deg(d.qpos[adr + 3:adr + 7])
    zs, yaws, done = [], [], False
    while not done:
        with torch.no_grad():
            a = agent.act_deterministic(torch.tensor(obs).float()[None]).squeeze(0).numpy()
        obs, _, done, _ = env.step(a)
        zs.append(float(d.qpos[adr + 2]))
        yaws.append(yaw_deg(d.qpos[adr + 3:adr + 7]))

    ref_dz = (traj.obj_pos[:, 2] - traj.obj_pos[0, 2]) * 100
    ref_dyaw = [wrap(yaw_deg(q) - yaw_deg(traj.obj_quat[0])) for q in traj.obj_quat]
    dyaw = [wrap(y - yaw0) for y in yaws]
    return {
        "ref": ref, "ckpt": ckpt, "object": obj, "steps": len(zs), "rollouts": 1,
        "physics_max_lift_cm": round((max(zs) - z0) * 100, 2),
        "physics_final_lift_cm": round((zs[-1] - z0) * 100, 2),
        "physics_final_yaw_deg": round(dyaw[-1], 1),
        "physics_max_abs_yaw_deg": round(max(dyaw, key=abs), 1),
        "reference_max_lift_cm": round(float(np.max(ref_dz)), 2),
        "reference_final_yaw_deg": round(ref_dyaw[-1], 1),
    }


def main():
    out = {name: rollout(name, **cfg) for name, cfg in POLICIES.items()}
    Path("results").mkdir(exist_ok=True)
    Path("results/verified_rollouts.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
