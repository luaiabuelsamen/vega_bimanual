"""Modal app for GPU-parallel MJX training of the Vega f5d6 tracker.

The Jetson has no JAX-CUDA wheels (aarch64), so MJX runs here on a Modal GPU.
Provides: the GPU image, repo/mesh mounts mirroring the local layout (so
assets.build_scene regenerates path-correct), staged smoke tests, a throughput
benchmark, and brax PPO training.

    modal run cloud/modal_app.py                 # 3 staged smoke probes
    modal run cloud/modal_app.py::bench          # scan-rollout throughput
    modal run cloud/modal_app.py::train --task push
"""
from pathlib import Path

import modal

# --- paths (local) -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent                       # ~/projects/dextrack_vega
PROJECTS = REPO.parent                   # ~/projects
DEXMATE = PROJECTS / "dexmate" / "dexmate-urdf"

# --- paths (container) -------------------------------------------------------
# Reproduce the local ~/projects layout so config.py path math resolves:
#   PKG_ROOT = .../dextrack_vega ; PROJECTS = its parent ; URDF under dexmate/.
C_PROJECTS = "/root/projects"
C_REPO = f"{C_PROJECTS}/dextrack_vega"
C_DEXMATE = f"{C_PROJECTS}/dexmate/dexmate-urdf"

# --- images ------------------------------------------------------------------
# Smoke image is the minimal JAX-CUDA + MJX stack (no brax). jax 0.4.38: 0.4.35
# had an nvidia-cuda-nvcc namespace-package import bug.
_MOUNTS = lambda img: (
    img
    .add_local_dir(str(REPO), C_REPO, ignore=["~*", ".venv", ".git", "runs", "media", "*.log"])
    .add_local_dir(str(DEXMATE), C_DEXMATE)
)

smoke_image = _MOUNTS(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]==0.4.38",
        "mujoco==3.8.0",
        "mujoco-mjx==3.8.0",
        "numpy<2",
    )
)

# Training image adds brax (JAX PPO) + wandb.
train_image = _MOUNTS(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]==0.4.38",
        "mujoco==3.8.0",
        "mujoco-mjx==3.8.0",
        "brax==0.12.1",
        "wandb",
        "numpy<2",
    )
)

app = modal.App("dextrack-vega-mjx")

# Persisted volume for checkpoints / metrics so runs survive container teardown.
vol = modal.Volume.from_name("dextrack-vega-runs", create_if_missing=True)
VOL_MNT = "/runs"


def _add_repo_to_path():
    import sys
    if C_REPO not in sys.path:
        sys.path.insert(0, C_REPO)


# One combined smoke in a single GPU container: GPU check -> trivial MJX ->
# Vega scene. Later stages only run if earlier pass; failures are captured.
@app.function(image=smoke_image, gpu="A10G", timeout=300)
def smoke(n_envs_builtin: int = 4096, n_envs_vega: int = 2048, n_steps: int = 50):
    import time, traceback
    import jax
    import jax.numpy as jnp
    out = {}

    # --- Stage 1: JAX sees the GPU ---
    print("[stage1] jax", jax.__version__, "devices:", jax.devices(), flush=True)
    gpu_ok = any(d.platform == "gpu" for d in jax.devices())
    out["stage1_gpu"] = gpu_ok
    if not gpu_ok:
        print("[stage1] FAIL: no GPU device", flush=True)
        return out
    x = jnp.ones((2048, 2048)); print("[stage1] matmul ->", float((x @ x).sum()), flush=True)

    # --- Stage 2: MJX steps a trivial vmapped model ---
    import mujoco
    from mujoco import mjx
    try:
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep="0.005"/><worldbody>'
            '<body><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>'
            '<geom type="plane" size="5 5 0.1"/></worldbody></mujoco>')
        mx = mjx.put_model(m)
        step = jax.jit(lambda dx: jax.vmap(lambda d: mjx.step(mx, d))(dx))
        dx = jax.vmap(lambda _: mjx.make_data(mx))(jnp.arange(n_envs_builtin))
        dx = step(dx); jax.block_until_ready(dx)
        t0 = time.time()
        for _ in range(n_steps):
            dx = step(dx)
        jax.block_until_ready(dx)
        sps = n_envs_builtin * n_steps / (time.time() - t0)
        out["stage2_sps"] = sps
        print(f"[stage2] builtin MJX {n_envs_builtin} envs -> {sps:,.0f} env-steps/s", flush=True)
    except Exception:
        out["stage2_err"] = traceback.format_exc()
        print("[stage2] FAIL:\n", out["stage2_err"], flush=True)
        return out

    # --- Stage 3: MJX loads + steps the real Vega f5d6 scene ---
    _add_repo_to_path()
    from dextrack_vega import assets, config as C
    assets.build_scene(sides=["R", "L"], obj_type="box",
                       obj_dims=(0.04, 0.045, 0.07), obj_pos=(0.55, 0.0, 0.80),
                       obj_mass=0.05, collision="primitive")

    def _run_variant(label, mutate=None):
        """put_model + jit-vmap-step the Vega model; return env-steps/s or raise.
        `mutate(m)` optionally tweaks MjModel options (cone/integrator) for an
        MJX-friendlier variant."""
        m = mujoco.MjModel.from_xml_path(str(C.GENERATED_SCENE))
        if mutate:
            mutate(m)
        print(f"[stage3:{label}] nq={m.nq} nu={m.nu} ngeom={m.ngeom} "
              f"cone={m.opt.cone} integrator={m.opt.integrator}", flush=True)
        mx = mjx.put_model(m)
        print(f"[stage3:{label}] put_model OK -> on GPU", flush=True)
        step = jax.jit(lambda dx: jax.vmap(lambda d: mjx.step(mx, d))(dx))
        dx = jax.vmap(lambda _: mjx.make_data(mx))(jnp.arange(n_envs_vega))
        t_c = time.time()
        dx = step(dx); jax.block_until_ready(dx)
        print(f"[stage3:{label}] first compile+step {time.time()-t_c:.1f}s", flush=True)
        t0 = time.time()
        for _ in range(n_steps):
            dx = step(dx)
        jax.block_until_ready(dx)
        sps = n_envs_vega * n_steps / (time.time() - t0)
        print(f"[stage3:{label}] {n_envs_vega} envs -> {sps:,.0f} env-steps/s "
              f"(~{sps/366:,.0f}x local CPU)", flush=True)
        return sps

    # Probe the scene as-is (elliptic cone + implicitfast) first.
    try:
        out["stage3_orig_sps"] = _run_variant("orig")
    except Exception:
        out["stage3_orig_err"] = traceback.format_exc()
        print("[stage3:orig] FAIL:\n", out["stage3_orig_err"], flush=True)
        # same container: retry with the MJX-friendliest options
        def _friendly(m):
            m.opt.cone = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
            m.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
        try:
            out["stage3_pyramidal_sps"] = _run_variant("pyramidal+euler", _friendly)
        except Exception:
            out["stage3_pyramidal_err"] = traceback.format_exc()
            print("[stage3:pyramidal+euler] FAIL:\n", out["stage3_pyramidal_err"], flush=True)
    return out


# Throughput benchmark: jax.lax.scan rollout (training pattern) at a few env counts.
@app.function(image=smoke_image, gpu="A10G", timeout=300)
def bench(env_counts: str = "2048,8192", rollout: int = 100):
    import time
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx
    _add_repo_to_path()
    from dextrack_vega import assets, config as C
    assets.build_scene(sides=["R", "L"], obj_type="box",
                       obj_dims=(0.04, 0.045, 0.07), obj_pos=(0.55, 0.0, 0.80),
                       obj_mass=0.05, collision="primitive")
    m = mujoco.MjModel.from_xml_path(str(C.GENERATED_SCENE))
    mx = mjx.put_model(m)

    def rollout_scan(dx0):
        def body(dx, _):
            return jax.vmap(lambda d: mjx.step(mx, d))(dx), None
        dx, _ = jax.lax.scan(body, dx0, None, length=rollout)
        return dx
    rollout_jit = jax.jit(rollout_scan)

    results = {}
    for n in [int(x) for x in env_counts.split(",")]:
        dx0 = jax.vmap(lambda _: mjx.make_data(mx))(jnp.arange(n))
        t_c = time.time()
        dx = rollout_jit(dx0); jax.block_until_ready(dx)        # compile + 1 rollout
        compile_s = time.time() - t_c
        t0 = time.time()
        for _ in range(3):
            dx = rollout_jit(dx0); jax.block_until_ready(dx)
        dt = (time.time() - t0) / 3
        sps = n * rollout / dt
        results[n] = sps
        print(f"[bench] {n:5d} envs x {rollout} (scan) compile {compile_s:.1f}s "
              f"-> {sps:,.0f} env-steps/s ({sps*3600/1e6:.1f}M/hr, "
              f"~{sps/366:,.0f}x CPU)", flush=True)
    return results


def _load_ref(rel):
    """Load a demo .npz (mounted with the repo) into the array dict the MJX env
    expects."""
    import os
    import numpy as np
    d = np.load(os.path.join(C_REPO, rel))
    return {"obj_pos": d["obj_pos"], "obj_quat": d["obj_quat"],
            "hand_qpos": d["hand_qpos"],
            "hand_ctrl": d["hand_ctrl"] if "hand_ctrl" in d.files else None,
            "dt": float(d["dt"])}


# Default object params per task (must match how each demo was generated).
TASK_OBJ = {
    "push":     dict(sides="R",   obj_dims=(0.03, 0.03, 0.03),  obj_pos=(0.6, -0.15, 0.80), obj_mass=0.08),
    "reorient": dict(sides="R",   obj_dims=(0.03, 0.03, 0.03),  obj_pos=(0.6, -0.15, 0.80), obj_mass=0.08),
    "bim_lift": dict(sides="R,L", obj_dims=(0.04, 0.045, 0.07), obj_pos=(0.55, 0.0, 0.80),  obj_mass=0.05),
}


# Env smoke: vmap reset/step the MJX env, check shapes + finite rewards.
@app.function(image=train_image, gpu="A10G", timeout=300)
def env_smoke(task: str = "push", n: int = 64, steps: int = 5):
    import time
    import jax
    import jax.numpy as jnp
    _add_repo_to_path()
    from dextrack_vega.envs.mjx_tracking_env import MjxTrackingEnv
    cfg = TASK_OBJ[task]
    ref = _load_ref(f"demos/{task}_box.npz")
    env = MjxTrackingEnv(sides=cfg["sides"].split(","), ref=ref, obj_type="box",
                         obj_dims=cfg["obj_dims"], obj_pos=cfg["obj_pos"],
                         obj_mass=cfg["obj_mass"])
    print(f"[env_smoke] obs={env.observation_size} act={env.action_size} "
          f"ep_len={env._episode_length}", flush=True)
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    t0 = time.time(); st = reset(keys); jax.block_until_ready(st)
    print(f"[env_smoke] reset compile {time.time()-t0:.1f}s obs.shape={st.obs.shape}", flush=True)
    assert st.obs.shape == (n, env.observation_size), "obs size mismatch!"
    act = jnp.zeros((n, env.action_size))
    t0 = time.time()
    for _ in range(steps):
        st = step(st, act)
    jax.block_until_ready(st)
    rew_ok = bool(jnp.isfinite(st.reward).all())
    obs_ok = bool(jnp.isfinite(st.obs).all())
    print(f"[env_smoke] {steps} steps compile+run {time.time()-t0:.1f}s "
          f"reward(mean)={float(st.reward.mean()):.3f} finite={rew_ok} obs_finite={obs_ok}", flush=True)
    print(f"[env_smoke] obj_pos_err(mean)={float(st.metrics['obj_pos_err'].mean()):.3f} "
          f"hand_qpos_err={float(st.metrics['hand_qpos_err'].mean()):.4f}", flush=True)
    return {"obs": env.observation_size, "act": env.action_size,
            "reward_finite": rew_ok, "obs_finite": obs_ok}


# brax PPO training on the MJX env.
def _make_network_factory():
    """Shared PPO network shape so training and rollout/render reconstruct the
    same architecture (params alone don't carry the layer sizes)."""
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256))


def _rollout_qpos(env, make_inference_fn, params, n_steps=None):
    """Deterministic policy rollout; returns full qpos[T, nq] + obj_pos[T,3] for
    rendering, plus the net object displacement/lift."""
    import jax
    import numpy as np
    inference = make_inference_fn(params, deterministic=True)
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    jstep = jax.jit(env.step)
    jact = jax.jit(inference)
    T = n_steps or env._episode_length
    qs, ops = [], []
    oq = env._obj_qadr
    for _ in range(T):
        act, _ = jact(state.obs, rng)
        state = jstep(state, act)
        qs.append(np.array(state.pipeline_state.qpos))
        ops.append(np.array(state.pipeline_state.qpos[oq:oq + 3]))
    qs = np.stack(qs); ops = np.stack(ops)
    disp = ops[-1] - ops[0]
    return qs, ops, disp


@app.function(image=train_image, gpu="A10G", timeout=3600, volumes={VOL_MNT: vol},
              secrets=[modal.Secret.from_name("wandb-secret")])
def train(task: str = "push", num_timesteps: int = 20_000_000, num_envs: int = 2048,
          seed: int = 0, run_name: str | None = None, use_wandb: bool = True,
          entropy_cost: float = 1e-3, learning_rate: float = 3e-4,
          w_lift_bonus: float = 0.0):
    import os
    import time
    import numpy as np
    import jax
    _add_repo_to_path()
    from dextrack_vega.envs.mjx_tracking_env import MjxTrackingEnv
    from brax.training.agents.ppo import train as ppo
    from brax.io import model

    cfg = TASK_OBJ[task]
    ref = _load_ref(f"demos/{task}_box.npz")
    env = MjxTrackingEnv(sides=cfg["sides"].split(","), ref=ref, obj_type="box",
                         obj_dims=cfg["obj_dims"], obj_pos=cfg["obj_pos"],
                         obj_mass=cfg["obj_mass"], w_lift_bonus=w_lift_bonus)
    run_name = run_name or f"{task}_mjx_{int(time.time())}"
    print(f"[train] task={task} run={run_name} obs={env.observation_size} "
          f"act={env.action_size} ep_len={env._episode_length} envs={num_envs}", flush=True)

    if use_wandb:
        try:
            import wandb
            wandb.login()  # reads WANDB_API_KEY from the wandb-secret env
            wandb.init(project="dextrack-vega-mjx", name=run_name,
                       config=dict(task=task, num_timesteps=num_timesteps,
                                   num_envs=num_envs, seed=seed,
                                   entropy_cost=entropy_cost, lr=learning_rate,
                                   w_lift_bonus=w_lift_bonus))
        except Exception as e:
            print(f"[train] wandb disabled ({e})", flush=True)
            use_wandb = False

    t_start = time.time()

    L = env._episode_length  # eval metrics are episode SUMS; /L -> per-step

    def progress(step, metrics):
        er = metrics.get("eval/episode_reward", float("nan"))
        pe = metrics.get("eval/episode_obj_pos_err", float("nan"))
        re = metrics.get("eval/episode_obj_rot_err", float("nan"))
        print(f"[train] step {step:>9d}  ep_reward {er:8.2f}  "
              f"objErr {pe/L*100 if pe==pe else pe:6.2f}cm/step  "
              f"rotErr {re/L if re==re else re:5.3f}/step  "
              f"{int(step/max(time.time()-t_start,1e-9))} sps", flush=True)
        if use_wandb:
            import wandb
            wandb.log({k: float(v) for k, v in metrics.items()}, step=int(step))

    # brax PPO: collected env-steps per iter = num_envs*unroll_length, reshaped
    # into num_minibatches of batch_size (must multiply to the same count).
    unroll_length = 16
    num_minibatches = 32
    batch_size = num_envs * unroll_length // num_minibatches
    make_inference, params, _ = ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=env._episode_length,
        num_envs=num_envs,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        unroll_length=unroll_length,
        num_updates_per_batch=4,
        discounting=0.98,
        gae_lambda=0.95,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        normalize_observations=True,
        reward_scaling=1.0,
        num_evals=20,
        seed=seed,
        network_factory=_make_network_factory(),
        progress_fn=progress,
    )

    out_dir = os.path.join(VOL_MNT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model.save_params(os.path.join(out_dir, "params"), params)
    # save a deterministic policy rollout (full qpos) for offline rendering
    qs, ops, disp = _rollout_qpos(env, make_inference, params)
    np.savez(os.path.join(out_dir, "rollout.npz"), qpos=qs, obj_pos=ops,
             sides=cfg["sides"], obj_dims=np.array(cfg["obj_dims"]),
             obj_pos0=np.array(cfg["obj_pos"]), obj_mass=cfg["obj_mass"])
    vol.commit()
    print(f"[train] done in {time.time()-t_start:.0f}s -> {out_dir}/params", flush=True)
    print(f"[train] rollout net obj displacement: dx={disp[0]*100:+.1f} "
          f"dy={disp[1]*100:+.1f} dz(lift)={disp[2]*100:+.1f} cm", flush=True)
    if use_wandb:
        import wandb
        wandb.finish()
    return run_name


@app.local_entrypoint()
def main():
    r = smoke.remote()
    print("\n===== SMOKE RESULT =====")
    for k, v in r.items():
        print(f"  {k}: {v if 'err' not in k else '(see log above)'}")
