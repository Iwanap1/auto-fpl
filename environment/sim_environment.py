import gymnasium as gym
import numpy as np
from helpers import *
import pickle
from gymnasium import spaces
import pandas as pd
from typing import List, Tuple, Dict

# --- you already have these from earlier turn ---
# build_candidate_pool_from_gw(...), random_initial_squad_from_pool(...),
# row_to_player(...), Squad, Player, etc.

POS2ID = {"GK":0, "DEF":1, "MID":2, "FWD":3}
FORMATIONS = [(3,4,3), (3,5,2), (4,3,3), (4,4,2), (5,3,2), (5,4,1)]

class FPLEnv(gym.Env):
    metadata = {"render_modes":[]}

    def __init__(self, season_root: str, gw_loader, per_pos: int = 20, seed: int = 42):
        """
        season_root: path to '.../<season>/' containing teams.csv and fixtures.csv
        gw_loader: callable(gw) -> (gw_df, fixtures_df, teams_df)
        """
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.season_root = season_root
        self.gw_loader = gw_loader
        self.per_pos = per_pos
        self.current_gw = 0
        self.free_transfers = 0
        self.running = True
        
    def step(self):
        if self.current_gw >= 38:
            self.running = False
        self.current_gw += 1
        self.free_transfers += 1
