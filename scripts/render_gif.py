"""Render a ReferenceTrajectory playback to an animated GIF (headless).

Plays the recorded qpos through the env model with mj_forward (no dynamics
drift) and renders each frame offscreen with mujoco.Renderer, then writes a GIF.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_gif.py \
        --ref demos/reorient_box.npz --out media/reorient.gif
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

from dextrack_vega.envs import VegaTrackingEnv
from dextrack_vega.utils import trajectory as tj
from dextrack_vega import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="media/demo.gif")
    ap.add_argument("--side", default="R")
    ap.add_argument("--sides", default=None,
                    help="comma list e.g. R,L for a bimanual reference")
    ap.add_argument("--ckpt", default=None,
                    help="render a trained policy ROLLOUT (real physics) instead "
                         "of the kinematic reference playback")
    ap.add_argument("--openloop", action="store_true",
                    help="render the reference executed OPEN-LOOP in physics "
                         "(zero residual) — real dynamics, no policy, no locking")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    # object must match the one the ref was made with (see train.py)
    ap.add_argument("--obj-type", default="box")
    ap.add_argument("--obj-mass", type=float, default=0.08)
    ap.add_argument("--obj-dims", default=None, help="comma list e.g. 0.04,0.045,0.07")
    ap.add_argument("--obj-pos", default=None, help="comma list x,y,z")
    args = ap.parse_args()

    sides = args.sides.split(",") if args.sides else [args.side]
    joints = [j for s in sides for j in C.controlled_joints(s)]
    ref = tj.load_npz(args.ref, joints)
    obj_kwargs = dict(obj_type=args.obj_type, obj_mass=args.obj_mass)
    if args.obj_dims:
        obj_kwargs["obj_dims"] = tuple(float(x) for x in args.obj_dims.split(","))
    if args.obj_pos:
        obj_kwargs["obj_pos"] = tuple(float(x) for x in args.obj_pos.split(","))
    env = VegaTrackingEnv(sides=sides, ref=ref, **obj_kwargs)
    m, d = env.model, env.data

    cam = mujoco.MjvCamera()
    if len(sides) > 1:  # bimanual: object on midline; oblique 3/4 so both arms
        cam.lookat[:] = [0.5, 0.0, 0.9]   # show (azimuth 90 collapses them)
        cam.distance = 1.5
        cam.azimuth = 35
        cam.elevation = -20
    else:
        cam.lookat[:] = [0.58, -0.15, 0.82]
        cam.distance = 1.3
        cam.azimuth = 140
        cam.elevation = -22

    frames = []
    with mujoco.Renderer(m, height=args.h, width=args.w) as r:
        if args.openloop:
            # reference executed open-loop in physics: zero residual -> the
            # actuator target IS the reference qpos; the object moves only as a
            # genuine dynamical consequence of the hands (no locking).
            import numpy as np
            obs = env.reset(); z0 = d.qpos[env._obj_qadr + 2]; done = False; i = 0
            while not done:
                obs, _, done, _ = env.step(np.zeros(env.action_dim))
                if i % args.stride == 0:
                    r.update_scene(d, camera=cam); frames.append(r.render())
                i += 1
            print(f"[render_gif] open-loop physics: object net lift "
                  f"{(d.qpos[env._obj_qadr + 2] - z0) * 100:+.1f}cm")
        elif args.ckpt:
            # POLICY ROLLOUT: step the env with the trained deterministic policy
            # and render the real physics — this is what the tracker actually
            # achieves, not the kinematic target.
            import torch
            from dextrack_vega.learning import ActorCritic
            ck = torch.load(args.ckpt, map_location="cpu")
            assert ck["obs_dim"] == env.obs_dim and ck["act_dim"] == env.action_dim, \
                "ckpt/env mismatch (sides?)"
            agent = ActorCritic(ck["obs_dim"], ck["act_dim"])
            agent.load_state_dict(ck["model"]); agent.eval()
            obs = env.reset(); done = False; i = 0
            while not done:
                with torch.no_grad():
                    a = agent.act_deterministic(torch.tensor(obs).float()[None]).squeeze(0).numpy()
                obs, _, done, _ = env.step(a)
                if i % args.stride == 0:
                    r.update_scene(d, camera=cam); frames.append(r.render())
                i += 1
            lifted = (d.qpos[env._obj_qadr + 2] - ref.obj_pos[0][2]) * 100
            print(f"[render_gif] policy rollout: object net height "
                  f"{lifted:+.1f}cm at episode end")
        else:
            for f in range(0, ref.n_frames, args.stride):
                d.qpos[env._jnt_qposadr] = ref.hand_qpos[f]
                d.qpos[env._obj_qadr:env._obj_qadr + 3] = ref.obj_pos[f]
                d.qpos[env._obj_qadr + 3:env._obj_qadr + 7] = ref.obj_quat[f]
                for j, v in C.HOME_POSTURE.items():
                    d.qpos[m.jnt_qposadr[env._jid(j)]] = v
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera=cam)
                frames.append(r.render())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"[render_gif] wrote {args.out}  ({len(frames)} frames, {args.w}x{args.h})")


if __name__ == "__main__":
    main()
