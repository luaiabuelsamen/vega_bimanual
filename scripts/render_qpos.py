"""Render a saved MJX-policy rollout (qpos[T, nq] from the runs volume) to a GIF.

Replays the recorded qpos through the same primitive scene the policy trained on
(kinematic playback: set qpos, mj_forward, render). The hand shows as fingertip
spheres — that's the training collision model.

    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_qpos.py \
        --rollout runs_dl/push_mjx_v5/rollout.npz --out media/push_mjx.gif
"""
import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

from dextrack_vega import assets, config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out", default="media/mjx_policy.gif")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--collision", default="mesh",
                    help="scene to render on: 'mesh' shows the full hand/arm "
                         "(nicer); 'primitive' shows the bare capsules the "
                         "policy actually trained on. qpos replays identically.")
    args = ap.parse_args()

    r = np.load(args.rollout, allow_pickle=True)
    qpos = r["qpos"]                         # [T, nq]
    sides = list(str(r["sides"]).split(","))
    obj_dims = tuple(float(x) for x in r["obj_dims"])
    obj_pos0 = tuple(float(x) for x in r["obj_pos0"])
    obj_mass = float(r["obj_mass"])

    assets.build_scene(sides=sides, obj_type="box", obj_dims=obj_dims,
                       obj_pos=obj_pos0, obj_mass=obj_mass, collision=args.collision)
    m = mujoco.MjModel.from_xml_path(str(C.GENERATED_SCENE))
    d = mujoco.MjData(m)
    assert qpos.shape[1] == m.nq, f"qpos nq {qpos.shape[1]} != model nq {m.nq}"

    cam = mujoco.MjvCamera()
    if len(sides) > 1:                       # bimanual: side-on near the box
        cam.lookat[:] = [0.55, 0.0, 0.93]; cam.distance = 0.8
        cam.azimuth = 10; cam.elevation = -8
    else:                                    # single arm: angled over the table
        cam.lookat[:] = [0.58, -0.15, 0.85]; cam.distance = 0.8
        cam.azimuth = 135; cam.elevation = -18

    frames = []
    with mujoco.Renderer(m, height=args.h, width=args.w) as ren:
        for f in range(0, qpos.shape[0], args.stride):
            d.qpos[:] = qpos[f]
            mujoco.mj_forward(m, d)
            ren.update_scene(d, camera=cam)
            frames.append(ren.render())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"[render_qpos] wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
