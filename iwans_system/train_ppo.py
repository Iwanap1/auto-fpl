# train_ppo.py
import os
from functools import partial
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from xp import load_models
from env import FPLEnv
from data_utils import load_season_fn, load_gw_fn

BASE_DIR = "../data/Fantasy-Premier-League"
SEASONS  = ["2020-21", "2021-22", "2023-24"]   # whatever seasons you have
START_GW = 2                                   # start at GW2 so features aren’t trivially zero
models = load_models("../models/rand_forest/classifiers")

def make_env(seed: int = 0):
    def _thunk():
        env = FPLEnv(
            load_season_fn=load_season_fn,
            load_gw_fn=load_gw_fn,
            seasons=SEASONS,
            base_dir=BASE_DIR,
            start_gw=START_GW,
            budget=100.0,
            temperature=0.7,         # controls semi-random initial 15
            transfer_hit=-4.0,
            max_free_transfers=5,
            models=models
        )
        env = Monitor(env)  # tracks episode rewards/lengths
        env.reset(seed=seed)
        return env
    return _thunk

N_ENVS = 8
SEED   = 42

vec_env = SubprocVecEnv([make_env(SEED + i) for i in range(N_ENVS)])

log_dir = "./logs/ppo_fpl"
os.makedirs(log_dir, exist_ok=True)
new_logger = configure(log_dir, ["stdout", "tensorboard"])  # optional

policy_kwargs = dict(
    net_arch=[256, 256],  # two-layer MLP; tweak as needed
    # You can also add ortho_init=False to speed-up convergence a bit
)

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,              # per env; total batch = n_steps * N_ENVS
    batch_size=4096,
    n_epochs=10,
    gamma=0.995,               # long-ish horizon
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,             # tiny entropy to encourage exploration
    vf_coef=0.5,
    target_kl=0.03,
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
)

model.set_logger(new_logger)

# Callbacks: evaluate & checkpoint while training
eval_env = DummyVecEnv([make_env(SEED + 1000)])  # single env for eval
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./checkpoints/best",
    log_path="./eval_logs",
    eval_freq=50_000 // N_ENVS,   # every ~50k steps
    n_eval_episodes=5,
    deterministic=True
)
ckpt_callback = CheckpointCallback(
    save_freq=100_000 // N_ENVS,
    save_path="./checkpoints/periodic",
    name_prefix="ppo_fpl"
)

TOTAL_STEPS = 2_000_000
model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_callback, ckpt_callback])
model.save("./ppo_fpl_final")

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print("Eval reward:", mean_reward, "±", std_reward)

# Or run one full episode to inspect behavior:
env = make_env(SEED+999)()
obs, info = env.reset()
done = False
episode_reward = 0.0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
print("Episode reward:", episode_reward)
