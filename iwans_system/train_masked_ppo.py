# train_sb3_masked_parallel.py
import os
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker

from xp import load_models
from data_utils import load_season_fn, load_gw_fn
from env import FPLEnv

from masked_ppo_wrapper import DiscreteActionSetWrapper, mask_fn


# --------- single-env factory (wrapped) ----------
def make_env_fn(BASE_DIR, SEASONS, MODELS, seed=None):
    def _thunk():
        env = FPLEnv(
            load_season_fn=load_season_fn,
            load_gw_fn=load_gw_fn,
            seasons=SEASONS,
            base_dir=BASE_DIR,
            start_gw=2,
            budget=100.0,
            temperature=0.8,
            transfer_hit=-4.0,
            max_free_transfers=5,
            models=MODELS,
        )
        # 1) turn MultiDiscrete into Discrete with dynamic legal set
        env = DiscreteActionSetWrapper(env)
        # 2) plug in action masking for the Discrete space
        env = ActionMasker(env, mask_fn)
        return env
    return _thunk


def main():
    BASE_DIR = "../data/Fantasy-Premier-League"
    SEASONS  = ["2019-20", "2020-21", "2021-22", "2023-24"]
    MODELS   = load_models("../models/rand_forest/classifiers")

    # ---- parallel envs
    NUM_ENVS = os.cpu_count() // 2 or 4  # choose what you like
    env_fns = [make_env_fn(BASE_DIR, SEASONS, MODELS) for _ in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns, start_method="forkserver")  # or "spawn" on Windows
    vec_env = VecMonitor(vec_env)

    # ---- build MaskablePPO (no PyTorch hacking needed)
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
    )

    model.learn(total_timesteps=2_000_000)
    model.save("ppo_fpl_masked_discrete_parallel")
    test_env = make_env_fn(BASE_DIR, SEASONS, MODELS)()
    obs, info = test_env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=False)
        # hard assert the chosen action is valid
        mask = mask_fn(test_env)
        assert mask[action] == 1, f"Picked illegal action index {action}"
        obs, reward, terminated, truncated, info = test_env.step(action)
        if terminated or truncated:
            obs, info = test_env.reset()
    test_env.close()

    vec_env.close()


if __name__ == "__main__":
    main()
