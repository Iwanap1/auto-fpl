from typing import Tuple, Dict, List, Any
import os, numpy as np, pandas as pd
from candidate_pool import build_candidate_pool_from_gw
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

def load_gw_fn(season_ctx, gw, per_pos=20, temperature=1.0):
    """
    Returns: players (List[Player]), index_map (Dict[str, List[int]]), gw_df (pd.DataFrame)
    Ensures gw_df has 'element_type' (and 'position') even for older seasons that lack them.
    """
    season_dir = season_ctx["season_dir"]
    season     = season_ctx["season"]
    fixtures   = season_ctx["fixtures"]
    teams      = season_ctx["teams"]
    players_raw = season_ctx["players_raw"]  # must be in season_ctx (loaded in load_season_fn)

    # Load GW csv first
    gw_df = load_csv(f"{season_dir}/gws/gw{gw}.csv")

    # If older data is missing element_type/position, enrich from players_raw
    if ("position" not in gw_df.columns) and ("element_type" not in gw_df.columns):
        # merge element_type onto gw_df by player id
        add = players_raw.loc[:, ["id", "element_type"]]
        gw_df = gw_df.merge(add, left_on="element", right_on="id", how="left")
        gw_df.drop(columns=["id"], inplace=True)
        if "element_type" not in gw_df.columns:
            raise ValueError(f"Failed to enrich gw_df for {season} GW{gw}: no element_type after merge")

    # If position still missing, derive it from element_type
    if "position" not in gw_df.columns and "element_type" in gw_df.columns:
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        gw_df["position"] = gw_df["element_type"].map(pos_map).astype(str)

    # Load per-season feature file
    features_df = load_csv(f"feature_data/{season}_features.csv")

    predictor = season_ctx.get("predictor", None)
    if predictor is None:
        # you said you don't want fallbacks; fail fast if models weren't passed to FPLEnv
        raise RuntimeError("Predictor is missing in season_ctx. Make sure FPLEnv received 'models' so reset() builds it.")

    players, index_map, pool_df = build_candidate_pool_from_gw(
        gw_df=gw_df,
        fixtures_df=fixtures,
        features_df=features_df,
        teams_df=teams,
        season_path=season_dir,
        gw=gw,
        per_pos=per_pos,
        temperature=temperature,
        predictor=predictor,
        players_raw_df=players_raw,
    )
    return players, index_map, gw_df

