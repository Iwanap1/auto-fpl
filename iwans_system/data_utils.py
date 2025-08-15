from typing import Tuple, Dict, List, Any
import os, numpy as np, pandas as pd
from candidate_pool import build_candidate_pool_from_gw
from hyperparams import PER_POS, TEMP
from xp import ScorePredictor

def load_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")


def load_season_fn(base_dir: str, season: str, models: List = None) -> Tuple[Dict[str, Any], int]:
    season_dir = f"{base_dir}/data/{season}"

    fixtures = load_csv(f"{season_dir}/fixtures.csv")
    teams    = load_csv(f"{season_dir}/teams.csv")
    players_raw = load_csv(f"{season_dir}/players_raw.csv")  # exists in vaastav repo

    # total GWs
    total_gws = int(fixtures["event"].max()) if "event" in fixtures.columns else 38

    season_ctx: Dict[str, Any] = {
        "season": season,
        "season_dir": season_dir,
        "fixtures": fixtures,
        "teams": teams,
        "players_raw": players_raw,
    }

    # If models are provided, construct the predictor here
    if models is not None:
        season_ctx["predictor"] = ScorePredictor(
            season=season,
            players_raw_df=players_raw,
            teams_raw_df=teams,
            fixtures_raw_df=fixtures,
            models=models
        )

    return season_ctx, total_gws

def load_gw_fn(season_ctx, gw):
    season_dir = season_ctx["season_dir"]
    season     = season_ctx["season"]
    fixtures   = season_ctx["fixtures"]
    teams      = season_ctx["teams"]
    features_df = load_csv(f"feature_data/{season}_features.csv")  # feature file
    gw_df = load_csv(f"{season_dir}/gws/gw{gw}.csv")

    predictor = season_ctx.get("predictor", None)

    players, index_map, pool_df = build_candidate_pool_from_gw(
        gw_df=gw_df,
        fixtures_df=fixtures,
        features_df=features_df,
        teams_df=teams,
        season_path=season_dir,
        gw=gw,
        per_pos=PER_POS,
        temperature=TEMP,
        predictor=predictor,  # <-- pass it in
    )
    return players, index_map, gw_df
