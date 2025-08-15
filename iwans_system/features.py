"""
You fill these two.
- Decide & compute your 5 features per player (return a 5-tuple of floats).
- Compute expected points xP for your horizon (e.g., next GW).
"""

from typing import Tuple
from player import Player

# form: 
# ict_index: ict_index from FPL
# transfer_ratio: transfers_in / (transfers_in + transfers_out) for upcoming GW



def get_features(feature_df, pid, gw) -> Tuple:
    """
    raw_row: a pandas Series or dict with whatever columns you need.
    Return: [form, ict_index, transfer_in_out_ratio, fixtures, prob_playing]
    """
    player_features = feature_df.loc[feature_df["id"] == pid]
    keys = [f"form_{gw}", 
            f"ict_{gw}",
            f"upcoming_fixture_{gw}",
            f"later_fixtures_{gw}",
            f"gws_until_return_{gw}",
            f"transfer_ratio_{gw}",
            f"availability_{gw}"]
    if player_features.empty:
        raise ValueError(f"Player {pid} not found in feature_df for GW {gw}.")
    if not all(key in player_features.columns for key in keys):
        raise ValueError(f"Feature DataFrame missing required columns for player {pid} and GW {gw}.")
    values = player_features[keys].values.flatten().tolist()
    return tuple(float(v) for v in values)



