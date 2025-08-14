from env import FPLEnv
from data_utils import load_season_fn, load_gw_fn
from xp import load_models
import gymnasium as gym

def fpl_env_creator(env_config):  # Changed: removed ** to accept positional arg
    """
    Environment creator function for RLlib.
    RLlib passes env_config as a positional argument.
    """
    # Handle both pre-loaded models and model paths
    if "models" in env_config:
        models = env_config["models"]  # Use pre-loaded models
    elif "models_path" in env_config:
        models = load_models(env_config["models_path"])  # Load from path
    else:
        raise ValueError("Must provide either 'models' or 'models_path' in env_config")
    
    return FPLEnv(
        load_season_fn=load_season_fn,
        load_gw_fn=load_gw_fn,
        seasons=env_config["seasons"],
        base_dir=env_config["base_dir"],
        start_gw=int(env_config.get("start_gw", 2)),
        budget=float(env_config.get("budget", 100.0)),
        temperature=float(env_config.get("temperature", 0.7)),
        transfer_hit=float(env_config.get("transfer_hit", -4.0)),
        max_free_transfers=int(env_config.get("max_free_transfers", 5)),
        models=models,
    )

# # Optional: Keep gym registration if you want to use gym.make() for testing
# gym.register(id="FPLEnv-v0", entry_point="fpl_env:fpl_env_creator")