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
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    ref = tj.load_npz(args.ref, C.controlled_joints(args.side))
    env = VegaTrackingEnv(side=args.side, ref=ref)
    m, d = env.model, env.data

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.58, -0.15, 0.82]
    cam.distance = 1.3
    cam.azimuth = 140
    cam.elevation = -22

    frames = []
    with mujoco.Renderer(m, height=args.h, width=args.w) as r:
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
