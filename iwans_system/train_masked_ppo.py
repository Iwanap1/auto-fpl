# train_sb3_masked_parallel.py
import os
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from xp import load_models
from data_utils import load_season_fn, load_gw_fn
from env import FPLEnv
from masked_ppo_wrapper import DiscreteActionSetWrapper, mask_fn
import gym

class RandomizeBudgetOnReset(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._vals = np.round(np.arange(99.5, 102.5 + 1e-9, 0.1), 1)

    def reset(self, **kwargs):
        self.env.unwrapped.budget = float(np.random.choice(self._vals))
        return self.env.reset(**kwargs)

def make_env_fn(BASE_DIR, SEASONS, MODELS, seed=None):
    def _thunk():
        env = FPLEnv(
            load_season_fn=load_season_fn,
            load_gw_fn=load_gw_fn,
            seasons=SEASONS,
            base_dir=BASE_DIR,
            start_gw=2,
            budget=100.0,           # overwritten on every reset by wrapper
            temperature=0.8,
            transfer_hit=-4.0,
            max_free_transfers=5,
            models=MODELS,
        )
        env = RandomizeBudgetOnReset(env)   # randomize budget each episode
        env = DiscreteActionSetWrapper(env) # MultiDiscrete -> Discrete choices
        env = ActionMasker(env, mask_fn)    # action masking
        return env
    return _thunk

def main():
    BASE_DIR = "../data/Fantasy-Premier-League"
    SEASONS  = ["2020-21", "2021-22", "2022-23", "2023-24"]
    MODELS   = load_models("../models/rand_forest/classifiers")

    # parallel envs
    ncpu = int(os.environ.get("NCPUS", os.cpu_count() or 16))
    NUM_ENVS = max(8, min(ncpu - 2, 14))   # e.g., 12 on a 16-core node

    env_fns = [make_env_fn(BASE_DIR, SEASONS, MODELS) for _ in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns, start_method="forkserver")
    vec_env = VecMonitor(vec_env)
    vec_env.seed(42)                        # ← reproducible (optional)
    set_random_seed(42)

    os.makedirs("./checkpoints", exist_ok=True)  # ← ensure exists
    rollout_size = NUM_ENVS * 2048
    ckpt = CheckpointCallback(
        save_freq=rollout_size * 5,
        save_path="./checkpoints",
        name_prefix="ppo_fpl_masked"
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        tensorboard_log="./tb_masked",
        # Optional: slightly bigger net
        # policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(total_timesteps=2_000_000, callback=ckpt)
    model.save("ppo_fpl_masked_discrete_parallel")

    # quick sanity rollout
    test_env = make_env_fn(BASE_DIR, SEASONS, MODELS)()
    obs, info = test_env.reset()
    for _ in range(10):
        mask = mask_fn(test_env)
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)  # ← pass mask here
        assert mask[action] == 1, f"Picked illegal action index {action}"
        obs, reward, terminated, truncated, info = test_env.step(action)
        if terminated or truncated:
            obs, info = test_env.reset()
    test_env.close()

    vec_env.close()

if __name__ == "__main__":
    main()
