# Retargeting workstream — design scope

Goal: produce **real dexterous** `ReferenceTrajectory` objects (the env's tracking
command) from human manipulation data, so the RL tracker learns actual grasps —
not scripted pushes. This is the step that turns "push a box" into *dexterous
manipulation*, and the proper route to the pick-and-lift that scripted top-down
grasping can't achieve (no force closure on a table-flush box).

## What we're producing
A `ReferenceTrajectory` (see `utils/trajectory.py`):
- `obj_pos[T,3]`, `obj_quat[T,4]` — object 6-DoF pose over time
- `hand_qpos[T, n_ctrl]` — controlled-joint targets = **[arm 7 DoF ; hand 11 DoF]**
- `dt`, `joint_names`

The env already FK's reference fingertips for the reward and tracks the object
pose, so anything in this format plugs straight into training/viz.

## How DexTrack does it (reference)
- Target hand: **Shadow Hand, 24 DoF** (`DexTrack/assets/hand/mjcf/shadow_hand_24_retarget*.xml`).
- Source: **MANO** hand params from **GRAB** (human grasps of objects, with object
  meshes + 6-DoF pose).
- Retarget MANO → Shadow via keypoint matching, then RL-track the result.

Our difference: **f5d6 has ~6 actuated DoF (11 joints, underactuated)** vs Shadow's
24 — so the human→robot map is a lossy approximation. We accept a fidelity ceiling
(already a known project constraint) and let RL close the dynamics gap.

## The core problem: MANO (~20+ DoF) → f5d6 (~6 actuated DoF)
NOT a joint copy. Standard solution = **keypoint / fingertip vector retargeting**
(DexPilot / haosulab `dex-retargeting`): per frame, solve a small nonlinear
optimization over f5d6's controlled joints to minimize the error between
human keypoint *vectors* (e.g. wrist→fingertip, thumb→fingertip) and the
corresponding f5d6 link vectors, plus smoothness + joint-limit terms.

We already have the building blocks in-repo:
- f5d6 URDF/MJCF + joint/limit bookkeeping (`config.py`, env indices).
- A working DLS IK on fingertip targets (`make_demo.py: ik_arm_to`) — extend to
  multi-target (per-finger) and add wrist orientation for the arm.

## Proposed pipeline
1. **Load human sequence** → per-frame: fingertip + key knuckle 3D positions
   (wrist-relative) and object 6-DoF pose. (From GRAB/MANO, or a proxy — see below.)
2. **Arm IK**: map the human **wrist 6-DoF** trajectory to the Vega arm with our
   DLS IK (position **+** orientation on `R_arm_l7`) → arm 7 DoF. Must clamp to
   joint limits and to the reachable workspace (x∈[0.5,0.8], y∈[-0.45,0.1]).
3. **Finger retarget**: per frame, optimize f5d6's 11 hand joints to match the
   human fingertip-vector targets (dex-retargeting or hand-rolled MuJoCo IK).
4. **Object pose**: transform the human-data object 6-DoF into the Vega table frame.
5. **Assemble** `ReferenceTrajectory` and feed the env.

## Key decisions / risks
- **Underactuation** → some grasps won't reproduce; pick objects/grasps f5d6 can
  actually achieve force closure on (sphere / cylinder / handle, not a flat box).
- **Frame alignment** human→Vega: calibration transform + workspace scaling;
  trajectories that leave the arm workspace must be retargeted/clipped.
- **Kinematic ref need not be dynamically consistent** (penetration/float is OK) —
  RL closes that gap. That's the whole point of tracking.
- **Data access**: GRAB + MANO need registration/license + model files; that's a
  setup gate (not code).

## Bootstrap (decouple from the data gate) — get a LIFT now
Before wiring the full MANO/GRAB pipeline, validate **lift-tracking** with a
"retargeted-quality" reference authored directly in sim:
- Use a **graspable object** (sphere / cylinder / vertical handle) where f5d6 *can*
  get force closure — the box failed only because top-down has none.
- Find a real force-closure grasp pose for f5d6 on it (search/IK in sim), author
  approach→grasp→lift keyframes, and record the physically-consistent result the
  way `make_demo.py` already does.
- This produces a genuine pick-and-lift `ReferenceTrajectory` to train against —
  answering the earlier "actually pick it up" via the proper (grasp-based) route —
  while the real human-data pipeline is set up in parallel.

## Milestones
1. **Bootstrap lift**: graspable object + force-closure grasp → lift reference →
   confirm the tracker learns a lift (proves grasp-tracking, not just push).
2. **dex-retargeting integration**: add f5d6 as a target hand (we have the URDF),
   retarget one GRAB grasp → `ReferenceTrajectory` → train.
3. **Dataset loop**: batch-retarget a set of grasps; train a multi-trajectory /
   generalist tracker (DexTrack's setting).
4. **Bimanual**: two coordinated arms + dual references (env already keys off `side`).

## Throughput note (separate axis)
torch = the RL/NN layer (we use it like DexTrack uses PyTorch). The sim backend
is the orthogonal axis. Current bottleneck is CPU `mj_step` (~87 sps), not the
MLP, so GPU-torch alone ≈ +10%. The real lever is parallel sim: **mujoco_warp**
(GPU sim on Jetson; JAX/MJX has no reliable aarch64 wheels) paired with the
CUDA-working torch in `../clean_env/venv` (torch 2.7.1, CUDA 12.6, Orin).
