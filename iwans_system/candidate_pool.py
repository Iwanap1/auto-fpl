from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from player import Player
from squad import POS_ORDER
from features import get_features
from xp import ScorePredictor
from typing import Optional

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

def get_team_name(fid: int, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, was_home: bool) -> Tuple[str, int]:
    fixture = fixtures_df.loc[fixtures_df["id"].eq(fid)]
    if fixture.empty:
        raise KeyError(f"Fixture id {fid} not found in fixtures_df")
    row = fixture.iloc[0]
    team_id = int(row["team_h"] if was_home else row["team_a"])
    trow = teams_df.loc[teams_df["id"] == team_id]
    team_name = str(trow["name"].iloc[0]) if not trow.empty else ""
    return team_name, team_id

def _choose_metric_cols(gw: int, cols: List[str]) -> Tuple[str, List[str]]:
    if gw == 1:
        primary = "selected" if "selected" in cols else ("selected_by_percent" if "selected_by_percent" in cols else None)
        fallbacks = ["selected_by_percent", "transfers_in_event", "transfers_in"]
    else:
        primary = "transfers_in_event" if "transfers_in_event" in cols else ("transfers_in" if "transfers_in" in cols else None)
        fallbacks = ["transfers_in", "selected", "selected_by_percent"]
    return primary, fallbacks


def build_candidate_pool_from_gw(
    gw_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    features_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    season_path: str,
    gw: int,
    per_pos: int,
    temperature: float,
    predictor,                    # MUST be not None
    players_raw_df: pd.DataFrame, # MUST be provided
) -> Tuple[List[Player], Dict[str, List[int]], pd.DataFrame]:

    assert "element" in gw_df.columns, "gw_df must have 'element' (player id)"
    assert ("position" in gw_df.columns) or ("element_type" in gw_df.columns), \
        "gw_df missing 'position' or 'element_type'"

    # Ensure a 'position' column
    if "position" not in gw_df.columns:
        map_pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        id_to_pos = dict(zip(
            players_raw_df["id"].astype(int),
            players_raw_df["element_type"].map(map_pos)
        ))
        gw_df = gw_df.copy()
        gw_df["position"] = gw_df["element"].astype(int).map(id_to_pos)

    if predictor is None:
        raise RuntimeError("Predictor is required but is None.")

    df = gw_df.copy()

    # Price
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["price"] = df["value"] / 10.0
    elif "now_cost" in df.columns:
        df["price"] = pd.to_numeric(df["now_cost"], errors="coerce") / 10.0
    else:
        raise ValueError("GW df missing 'value'/'now_cost' for price.")

    # Ranking metric
    primary, fallbacks = _choose_metric_cols(gw, df.columns.tolist())
    if primary is None:
        df["metric"] = 0.0
    else:
        df["metric"] = pd.to_numeric(df[primary], errors="coerce")
        for fb in fallbacks:
            if fb in df.columns:
                df["metric"] = df["metric"].fillna(pd.to_numeric(df[fb], errors="coerce"))
        df["metric"] = df["metric"].fillna(0.0)

    # Top-N per position
    parts = []
    for pos in POS_ORDER:
        sub = df.loc[df["position"] == pos].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["metric", "price"], ascending=[False, True]).head(per_pos)
        parts.append(sub)
    if not parts:
        raise ValueError("No candidates found for any position.")

    pool_df = pd.concat(parts, axis=0).reset_index(drop=True)
    pool_df = (
        pool_df.sort_values(["position", "metric", "price"], ascending=[True, False, True])
               .drop_duplicates(subset=["element"], keep="first")
               .reset_index(drop=True)
    )

    # Team info (requires 'fixture' and 'was_home')
    if not {"fixture", "was_home"}.issubset(pool_df.columns):
        raise ValueError("GW df must include 'fixture' and 'was_home' to resolve team names/ids.")
    team_names, team_ids = [], []
    for _, r in pool_df.iterrows():
        tname, tid = get_team_name(int(r["fixture"]), fixtures_df, teams_df, bool(r["was_home"]))
        team_names.append(tname); team_ids.append(tid)
    pool_df["team"] = team_names
    pool_df["team_id"] = team_ids

    # Features (7)
    feats_list = []
    for _, r in pool_df.iterrows():
        pid = int(r["element"])
        # If you want *strict* behavior for features too, replace try/except with a hard raise.
        try:
            feats = get_features(features_df, pid, gw)
        except Exception:
            feats = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        feats_list.append(feats)
    pool_df["features"] = feats_list

    # Strict xP for candidates only
    cand_pids = pool_df["element"].astype(int).tolist()
    xp_map = predictor.xp_map_strict_for_candidates(
        gw=gw,
        candidate_pids=cand_pids,
        players_raw_df=players_raw_df,
        fixtures_df=fixtures_df,
    )

    # Build Player objects
    players: List[Player] = []
    for _, r in pool_df.iterrows():
        pid = int(r["element"])
        feats = tuple(r["features"])
        xp = float(xp_map[pid])  # guaranteed (0.0 if blank)
        players.append(Player(
            pid=pid,
            name=str(r["name"]),
            pos=str(r["position"]),
            team=str(r["team"]),
            team_id=int(r["team_id"]),
            price=float(r["price"]),
            features=feats,
            pooling_metric=float(r["metric"]),
            xP=xp,
        ))

    index_map = {pos: [i for i, p in enumerate(players) if p.pos == pos] for pos in POS_ORDER}
    return players, index_map, pool_df
