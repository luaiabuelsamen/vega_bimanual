"""Step the tracking env with zero and random actions; print shapes + reward.

Run:  python -m scripts.smoke_test
"""
import numpy as np

from dextrack_vega.envs import VegaTrackingEnv


def rollout(env, policy, label):
    obs = env.reset()
    total, terms, n = 0.0, {}, 0
    done = False
    while not done:
        a = policy(obs)
        obs, r, done, comp = env.step(a)
        total += r
        n += 1
        for k, v in comp.items():
            if not k.startswith("_"):
                terms[k] = terms.get(k, 0.0) + v
    print(f"\n[{label}] steps={n}  return={total:.2f}  "
          f"final obj_pos_err={comp['_obj_pos_err']*100:.1f}cm  "
          f"rot_err={comp['_obj_rot_err']:.2f}rad")
    print("   mean term/step:", {k: round(v / n, 3) for k, v in terms.items()})


def main():
    env = VegaTrackingEnv(side="R")
    print(f"obs_dim={env.obs_dim}  action_dim={env.action_dim}  "
          f"ctrl_dt={env.dt:.3f}s  ref_dur={env.ref.duration:.2f}s")

    rollout(env, lambda o: np.zeros(env.action_dim), "zero residual (pure kinematic bias)")
    rng = np.random.default_rng(0)
    rollout(env, lambda o: rng.uniform(-1, 1, env.action_dim), "random residual")


if __name__ == "__main__":
    main()
