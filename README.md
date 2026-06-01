# vega_bimanual — RL manipulation *tracking* on the Dexmate Vega (MuJoCo)

> **Work in progress.** Reimplementing [DexTrack](https://github.com/Meowuu7/DexTrack)-style
> reinforcement-learning manipulation *tracking* on the bimanual
> [Dexmate Vega](https://www.dexmate.ai/) humanoid, in MuJoCo — single-arm first,
> bimanual as the goal.

<p align="center">
  <img src="media/bim_lift_policy.gif" width="520" alt="Trained bimanual policy squeezing and lifting a box off the table"/>
</p>

*Above: a **trained neural-network policy** driving two Vega f5d6 hands to squeeze
and lift a 0.05 kg box, held by inter-hand friction. **Verified in physics: the
box lifts ~+5 cm and stays on the table** (the kinematic reference targets +21 cm,
but real-physics replay only reaches ~+5 cm — an env-vs-reference divergence in
the contact, not yet closed). Lifting at all is the one manipulation a **single**
f5d6 hand cannot do (its thumb can't oppose the fingers closer than ~3.1 cm); the
second hand provides the missing object opposition. Trained by **behavior
cloning** of the open-loop zero-residual rollout — the kinematic reference is
(partially) feasible in physics, so BC clones it directly, sidestepping the
reward-hacking that PPO falls into on this contact task.*

<p align="center">
  <img src="media/bim_handover.gif" width="480" alt="Bimanual handover: R hand pushes box to midline, both hands then lift it"/>
</p>

*A **bimanual handover** — two arms playing two genuinely different roles.
**Phase 1**: the R hand alone pushes the box from its off-midline spawn (y = -18 cm,
unreachable by the L hand) to the midline. **Phase 2**: both hands approach the
now-centred box, squeeze it between opposite faces, and lift it together. The
demo records the full ~10-second multi-phase trajectory in physics; the policy
side is partial (env-vs-demo replay diverges for chained drive_to demos, a
separate bug from the RL plateau).*

<p align="center">
  <img src="media/bim_reorient_policy.gif" width="420" alt="Trained bimanual policy yawing a box ~37°"/>
</p>

*A second **trained-policy** result — **cooperative reorient**: both hands
contact the box's opposite ±y faces and sweep tangentially in opposite x
directions; the friction drag yaws the box **+37°** on the table (verified in
physics). Nonprehensile (no force closure needed), so it doesn't have the
grip-margin problem that limits the lift. Same BC recipe as the lift policy.*

<p align="center">
  <img src="media/reorient.gif" width="320" alt="Vega f5d6 reorienting a box on a table (nonprehensile)"/>
</p>

*Single-arm reorient for comparison: one f5d6 hand yawing a box ~140° by
off-center pushing.*

## What I'm trying to do

DexTrack learns a single RL policy that **tracks** kinematic references — a hand
trajectory plus an object's 6-DoF pose — so a dexterous hand reproduces real
manipulation. The original targets a 24-DoF Shadow Hand in Isaac Gym from
retargeted human (MANO/GRAB) grasps. The goal here is to bring that **method**
(not the code, which is Isaac-Gym-only) to the **Dexmate Vega** — a bimanual
mobile humanoid with 7-DoF arms and the underactuated **f5d6** 5-finger hand —
running entirely in **MuJoCo**, and to scale it to **coordinated bimanual**
manipulation.

## Approach

- **Env** (`dextrack_vega/envs`): per-arm-modular tracking env. Action space is
  DexTrack's **cumulative residual position targets with a kinematic bias**
  (`target = ref_qpos + Σ residual`, the residual **bounded** to a band around
  the reference so the policy corrects it but can't drift away — an unbounded
  residual let the bimanual arms walk to their joint limits). Reward =
  object-pose + hand-pose (joint & fingertip) tracking, all
  bounded exponential kernels. Now generalizes from one arm (`side="R"`, 18-DoF
  action) to **two** (`sides=["R","L"]`, 36-DoF).
- **Assets** (`dextrack_vega/assets.py`): compiles the Vega-1U f5d6 URDF into a
  MuJoCo scene (strips `.glb` visuals MuJoCo can't decode, keeps `.obj`
  collision), adds a table, a manipulable object (box / cylinder / sphere) and
  position actuators.
- **References** (`scripts/make_demo.py`): physically-consistent demos generated
  in-sim via damped-least-squares IK (incl. a verified 6-DoF position+orientation
  IK) — `push`, `reorient`, plus a kinematic `lift_ref` builder.
- **Learning** (`scripts/train.py`): clean PPO (PyTorch). Vectorized envs come in
  two backends (`dextrack_vega/envs/vec_env.py`): `sync` (in-process) and
  `process` (one env per core). On this 8-core Jetson the multi-core backend is
  ~2.8× faster (130 → ~366 steps/s); MJX isn't an option here (no JAX-CUDA wheels
  for aarch64/Tegra), so `mujoco_warp` is the eventual GPU path.

## Status

| Piece | State |
|---|---|
| Single-arm tracking env + reward + PPO | ✅ working |
| **Push** task | ✅ learns (object-tracking error 80 → ~11 cm) |
| **Reorient** task (object yaw tracking) | ✅ demo + training |
| 6-DoF IK, object types, headless GIF render | ✅ |
| Bimanual env (36-DoF action) | ✅ constructs & steps |
| **Bimanual squeeze-and-lift** (open-loop physics) | ✅ two hands lift a 0.05 kg box +5 cm in real physics (the kinematic reference targets +21 cm — the gap is env-replay divergence) |
| **Bimanual squeeze-and-lift** (BC-trained policy) | ✅ trained NN actor, +5 cm max lift, box stays on table |
| **Bimanual cooperative reorient** (BC-trained policy) | ✅ NN policy, +37° yaw (verified in physics) |
| **Bimanual handover** (R-push -> bimanual lift, two-phase) | ✅ scripted demo (10 s composition) |
| **BC push policy** (single-arm) | ✅ NN policy pushes the box −11.5 cm (matches reference target) — verified |
| Multi-core training (`ProcessVectorEnv`, ~2.8× on 8 cores) | ✅ |
| **GPU-parallel MJX training on Modal** (`cloud/modal_app.py`) | ✅ ~104k env-steps/s on an A10G (~290× local CPU), ~$0.30 per 100M steps |
| Bimanual *PPO* matching the BC lift | 🚧 PPO reward-hacks these contact tasks (push → do-nothing, lift → flings the box for transient height); BC of the feasible reference is the honest working path |
| Human-grasp retargeting (GRAB/TACO → f5d6) | 🚧 scoped (`RETARGETING.md`) |
| Parallel sim (mujoco_warp) for throughput | 🚧 |

## A finding worth noting

The **f5d6 hand cannot achieve force closure** — a full thumb-joint-range search
shows the thumb can't oppose the fingers closer than ~3.1 cm, so it can't pinch,
enclose, lift, or do in-hand manipulation. The feasible (and trackable) repertoire
is therefore **nonprehensile**: pushing, pivoting and reorienting objects on a
support surface — which is what the current tasks target, and which extends
naturally to two-arm cooperative manipulation.

## Run

```bash
# generate a reference demo
PYTHONPATH=. python scripts/make_demo.py --task reorient --out demos/reorient_box.npz

# bimanual handover (R pushes to midline, then both lift)
PYTHONPATH=. python scripts/make_demo.py --task bim_handover --out demos/bim_handover_box.npz

# bimanual cooperative reorient (couple yaws the box ~111°)
PYTHONPATH=. python scripts/make_demo.py --task bim_reorient --out demos/bim_reorient_box.npz

# kinematic-seg version of bimanual reorient (env replay reproduces a +57° yaw)
PYTHONPATH=. python scripts/make_demo.py --task bim_reorient_k --out demos/bim_reorient_k_box.npz
# BC train a policy on it
PYTHONPATH=. python scripts/bc_zero.py --ref demos/bim_reorient_k_box.npz --sides R,L \
    --obj-type box --obj-mass 0.1 --obj-dims 0.045,0.045,0.06 \
    --obj-pos 0.55,0.0,0.80 --epochs 400 --out runs/bim_reorient_bc/ckpt.pt

# bimanual cooperative squeeze-and-lift reference (36-DoF, two arms)
PYTHONPATH=. python scripts/make_demo.py --task bim_lift --out demos/bim_lift_box.npz
# render it executed OPEN-LOOP in physics (real dynamics; prints the net lift)
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_gif.py \
    --ref demos/bim_lift_box.npz --sides R,L --openloop \
    --obj-type box --obj-mass 0.05 --obj-dims 0.04,0.045,0.07 --obj-pos 0.55,0.0,0.80 \
    --out media/bim_lift.gif

# train the bimanual tracker across all CPU cores (object must match the ref)
PYTHONPATH=. python scripts/train.py --ref demos/bim_lift_box.npz --sides R,L \
    --vec process --num-envs 12 --obj-mass 0.05 --obj-dims 0.04,0.045,0.07 \
    --obj-pos 0.55,0.0,0.80 --total-steps 800000

# behavior-clone a residual policy from the (already-feasible) zero-residual
# rollout — yields a trained NN policy that lifts the box +21 cm
PYTHONPATH=. python scripts/bc_zero.py --ref demos/bim_lift_box.npz --sides R,L \
    --obj-type box --obj-mass 0.05 --obj-dims 0.04,0.045,0.07 \
    --obj-pos 0.55,0.0,0.80 --epochs 400 --out runs/bim_lift_bc/ckpt.pt
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_gif.py \
    --ref demos/bim_lift_box.npz --sides R,L --ckpt runs/bim_lift_bc/ckpt.pt \
    --obj-type box --obj-mass 0.05 --obj-dims 0.04,0.045,0.07 --obj-pos 0.55,0.0,0.80 \
    --out media/bim_lift_policy.gif

# train the tracker on it
PYTHONPATH=. python scripts/train.py --ref demos/reorient_box.npz --total-steps 250000

# visualize (kinematic playback of a reference)
./run_viz.sh --ref demos/reorient_box.npz --playback

# render a demo to GIF (headless)
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_gif.py --ref demos/reorient_box.npz --out media/reorient.gif
```
