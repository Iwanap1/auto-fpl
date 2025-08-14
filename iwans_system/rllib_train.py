import os
import time
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from xp import load_models
import fpl_env
from ray import tune

tune.register_env("FPLEnv-v0", fpl_env.fpl_env_creator)
# ---- Edit these to match your setup ----
BASE_DIR  = "../data/Fantasy-Premier-League"
SEASONS   = ["2020-21", "2021-22", "2023-24"]
models = load_models("../models/rand_forest/classifiers")

ENV_CONFIG = dict(
    base_dir=BASE_DIR,
    seasons=SEASONS,
    start_gw=2,
    budget=100.0,
    temperature=0.7,
    transfer_hit=-4.0,
    max_free_transfers=5,
    models=models,
)

# Optional quick smoke test (creates locally, then closes)
def _smoke_test():
    # Test with the creator function directly
    env = fpl_env.fpl_env_creator(ENV_CONFIG)  # Pass as positional arg
    obs, info = env.reset()
    print(f"✅ Smoke test passed. Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    env.close()
    
if __name__ == "__main__":
    # Start Ray locally (change to address="auto" only if you have a Ray cluster to join)
    ray.init(ignore_reinit_error=True)

    # (Optional) verify env is discoverable and constructible
    _smoke_test()

    # Build PPO
    algo = (
        PPOConfig()
        .environment(env="FPLEnv-v0", env_config=ENV_CONFIG)
        .framework("torch")
        .env_runners(                  # replaces deprecated `.rollouts(...)`
            num_env_runners=8,         # was: num_rollout_workers
            num_envs_per_env_runner=1, # vector envs per worker
            rollout_fragment_length=512,
        )
        .training(
            lr=3e-4,
            gamma=0.995,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            train_batch_size=256_000,     # aggregated across runners
            sgd_minibatch_size=32_000,
            num_sgd_iter=10,
        )
        .resources(num_gpus=0)  # set >0 if you actually want GPUs
        .build()
    )

    # Simple training loop with periodic checkpoints
    os.makedirs("./checkpoints_rllib", exist_ok=True)
    target_timesteps = 10_000_000
    last_save = time.time()
    save_every_sec = 300  # save every 5 min

    while True:
        result = algo.train()
        ts = result.get("timesteps_total", 0)
        iter_ = result.get("training_iteration", 0)
        ep_rew_mean = result.get("episode_reward_mean", float("nan"))
        print(f"[itr {iter_:4d}] ts={ts:,}  meanR={ep_rew_mean:.3f}")

        now = time.time()
        if now - last_save > save_every_sec:
            cp = algo.save(checkpoint_dir="./checkpoints_rllib")
            print(f"Saved checkpoint: {cp}")
            last_save = now

        if ts >= target_timesteps:
            break

    # Final save
    cp = algo.save(checkpoint_dir="./checkpoints_rllib")
    print(f"Final checkpoint: {cp}")

