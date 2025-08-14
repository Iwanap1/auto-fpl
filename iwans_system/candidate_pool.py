from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from player import Player
from squad import POS_ORDER
from features import get_features, get_xp
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
    per_pos: int = 20,
    temperature: float = 1.0,
    predictor: Optional[ScorePredictor] = None,   # <-- NEW
) -> Tuple[List[Player], Dict[str, List[int]], pd.DataFrame]:

    df = gw_df.copy()

    # Normalize/ensure position column is string POS
    if "position" in df.columns:
        if pd.api.types.is_numeric_dtype(df["position"]):
            df["position"] = pd.to_numeric(df["position"], errors="coerce").map(POS_MAP).astype(str)
        else:
            df["position"] = df["position"].astype(str)
    elif "element_type" in df.columns:
        df["position"] = pd.to_numeric(df["element_type"], errors="coerce").map(POS_MAP).astype(str)
    else:
        raise ValueError("GW df missing 'position' or 'element_type'.")

    # Required basics
    required = ["element", "name", "position", "fixture", "was_home"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GW df missing required columns: {missing}")

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
        if sub.empty:  # no players of that pos this GW (rare)
            continue
        # You can also inject temperature here if you want the pool itself to be semi-random
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

    # Resolve teams from fixtures
    team_names, team_ids = [], []
    for _, r in pool_df.iterrows():
        tname, tid = get_team_name(int(r["fixture"]), fixtures_df, teams_df, bool(r["was_home"]))
        team_names.append(tname); team_ids.append(tid)
    pool_df["team"] = team_names
    pool_df["team_id"] = team_ids

    # ------- Compute 5 features per player (your function) -------
    feats_list = []
    pids, poss = [], []
    skipped = []
    for _, r in pool_df.iterrows():
        pid = int(r["element"])
        try:
            feats = get_features(features_df, pid, gw)
            feats_list.append(feats)
            pids.append(pid); poss.append(str(r["position"]))
        except Exception as e:
            print(e)
            feats_list.append((0.0,0.0,0.0,0.0,1.0))  # safe default
            pids.append(pid); poss.append(str(r["position"]))
            skipped.append(pid)
    pool_df["features"] = feats_list  # for inspection/debug

    # ------- Use predictor (if provided) to get model xP in batch -------
    xp_map: Dict[int, float] = {}
    if predictor is not None:
        pid_pos_tuples = list(zip(pids, poss))
        try:
            xp_map = predictor.predict_batch(pid_pos_tuples, gw)
        except Exception:
            xp_map = {}

    # Build Player objects
    players: List[Player] = []
    for _, r in pool_df.iterrows():
        pid = int(r["element"])
        feats = tuple(r["features"])
        # xP priority: predictor -> gw_df column 'xP' -> feature proxy
        if pid in xp_map:
            xp = float(xp_map[pid])
        elif "xP" in r and pd.notna(r["xP"]):
            xp = float(r["xP"])
        else:
            xp = float(feats[0])  # fallback to 'form' as a weak proxy

        p = Player(
            pid=pid,
            name=str(r["name"]),
            pos=str(r["position"]),
            team=str(r["team"]),
            team_id=int(r["team_id"]),
            price=float(r["price"]),
            features=feats,
            pooling_metric=float(r["metric"]),
            xP=xp,
        )
        players.append(p)

    index_map = {pos: [i for i, p in enumerate(players) if p.pos == pos] for pos in POS_ORDER}
    return players, index_map, pool_df