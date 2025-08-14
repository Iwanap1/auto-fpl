import pandas as pd
import numpy as np
from typing import Sequence

def load_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                path,
                engine="python",       # more tolerant
                on_bad_lines="skip",   # or "skip"
                encoding=enc
            )
        except UnicodeDecodeError:
            continue

    return pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )

def convert_season_to_int(season: str):
    """uses last 2 digits"""
    year = int(season[-2:])
    return int(year)

def calculate_transfer_ratio(gw_row):
    """Calculate the transfer ratio based on the number of transfers."""
    t_in = float(gw_row['transfers_in'].values[0])
    t_out = float(gw_row['transfers_out'].values[0])
    if t_in + t_out == 0:
        return 0.0
    return t_in / (t_in + t_out)

def estimate_availability_bucket(minutes_series: pd.Series, gw: int) -> int:
    """
    Deterministic FPL-style availability bucket for a given GW using the actual minutes
    of that GW plus recent/season context. Returns one of {0, 25, 50, 75, 100}.
    """
    LABELS = np.array([0, 25, 50, 75, 100])

    # Base distributions (kept for fallback decisions)
    DIST_FULL = np.array([0.00, 0.02, 0.03, 0.10, 0.85])   # fully fit baseline
    DIST_60_84 = np.array([0.01, 0.04, 0.08, 0.37, 0.50])
    DIST_20_59_regular = np.array([0.02, 0.08, 0.20, 0.40, 0.30])
    DIST_20_59_regular_sub = np.array([0.00, 0.05, 0.10, 0.25, 0.60]) # regular bench but fit
    DIST_1_19 = np.array([0.03, 0.12, 0.30, 0.35, 0.20])
    DIST_DNP_ROTATION = np.array([0.00, 0.05, 0.10, 0.25, 0.60])      # fit but not picked
    DIST_DNP_UNCERTAIN = np.array([0.05, 0.10, 0.20, 0.35, 0.30])
    DIST_DNP_INJURY = np.array([0.80, 0.15, 0.05, 0.00, 0.00])

    # Extra distributions to *force* some 50/25 modes when appropriate
    DIST_RETURNING_20_35 = np.array([0.02, 0.08, 0.34, 0.33, 0.23])   # mode=50
    DIST_CAMEO_KNOCK = np.array([0.05, 0.20, 0.40, 0.25, 0.10])       # mode=50
    DIST_DNP_ONE_OFF_STARTER = np.array([0.10, 0.25, 0.40, 0.20, 0.05]) # mode=50
    DIST_DNP_MINOR_UNCERTAIN = np.array([0.25, 0.35, 0.25, 0.10, 0.05]) # mode=25

    def softclip_probs(p: np.ndarray) -> np.ndarray:
        p = np.clip(p.astype(float), 0, None)
        s = p.sum()
        return p / s if s > 0 else np.ones_like(p) / len(p)

    def zero_run_ahead(series: pd.Series, idx: int) -> int:
        n = 0
        for j in range(idx, len(series)):
            if float(series.iloc[j]) == 0.0:
                n += 1
            else:
                break
        return n

    def zero_run_behind(series: pd.Series, idx: int, limit: int = 3) -> int:
        n = 0
        j = idx - 1
        while j >= 0 and n < limit and float(series.iloc[j]) == 0.0:
            n += 1
            j -= 1
        return n

    idx = gw - 1
    if idx < 0 or idx >= len(minutes_series):
        return 0

    minutes_now = float(minutes_series.iloc[idx]) if pd.notna(minutes_series.iloc[idx]) else 0.0

    # Prior window (exclude current GW)
    prior = minutes_series.iloc[max(0, idx-5):idx].astype(float)
    prior_app_rate = (prior > 0).mean() if len(prior) else 0.0
    prior_over60_rate = (prior >= 60).mean() if len(prior) else 0.0
    prior_avg_min = prior.mean() if len(prior) else 0.0
    last_min = float(prior.iloc[-1]) if len(prior) else np.nan

    # Season context
    ms = minutes_series.fillna(0).astype(float)
    season_app_rate = (ms > 0).mean()
    season_over60_rate = (ms >= 60).mean()
    season_sub_lt30_rate = ((ms > 0) & (ms < 30)).mean()

    zr_ahead = zero_run_ahead(minutes_series, idx) if minutes_now == 0 else 0
    zr_behind = zero_run_behind(minutes_series, idx, limit=3)

    # Risk score (for full-game but 75 cases)
    risk_score = 0
    if prior_over60_rate >= 0.6 and zr_behind >= 1:
        risk_score += 2
    if prior_over60_rate >= 0.6 and zr_behind >= 2:
        risk_score += 2
    if not np.isnan(last_min) and last_min < 12 and prior_over60_rate >= 0.6:
        risk_score += 2
    if prior_avg_min < 40 and season_over60_rate >= 0.6:
        risk_score += 1
    if gw <= 3:
        risk_score = max(0, risk_score - 1)

    # --- Map actual minutes to a bucket (deterministically) ---
    # FULL GAME
    if minutes_now >= 85:
        if risk_score >= 3:
            return 75                          # classic: yellow-flag but starts/plays 90
        elif risk_score >= 5:
            return 50                          # very rare: heavy pre-match doubt, still played 90
        dist = DIST_FULL

    # 60–84 minutes
    elif 60 <= minutes_now < 85:
        # slight tilt toward 75 if there was recent doubt
        if risk_score >= 3:
            return 75
        dist = DIST_60_84

    # 20–59 minutes
    elif 20 <= minutes_now < 60:
        # Regular bench profile → treated as fit (100-ish)
        if season_sub_lt30_rate >= 0.6 and season_app_rate >= 0.3:
            dist = DIST_20_59_regular_sub
        # Starter returning with limited minutes (esp. 20–35) → force some 50s
        elif prior_over60_rate >= 0.6 and (minutes_now < 35 or zr_behind >= 1 or (not np.isnan(last_min) and last_min < 12)):
            dist = DIST_RETURNING_20_35
        else:
            dist = DIST_20_59_regular

    # 1–19 minutes
    elif 1 <= minutes_now < 20:
        # If usually a starter and just a cameo → "doubt but might play" (50)
        if prior_over60_rate >= 0.6:
            dist = DIST_CAMEO_KNOCK
        else:
            dist = DIST_1_19

    # DNP (0 minutes)
    else:
        # Clear injury pattern: regular starter -> now multiple-GW DNP
        if prior_app_rate >= 0.7 and prior_avg_min >= 60 and zr_ahead >= 2:
            dist = DIST_DNP_INJURY
        # One-off DNP for a starter → often flagged around 50
        elif prior_over60_rate >= 0.6 and zr_behind == 0:
            dist = DIST_DNP_ONE_OFF_STARTER
        # Fringe/uncommon pick → "fit but not picked"
        elif season_app_rate < 0.4 and season_over60_rate < 0.3:
            dist = DIST_DNP_ROTATION
        # Middling usage with recent noise → occasionally 25 as the mode
        elif 0.4 <= season_app_rate <= 0.6 and prior_app_rate <= 0.5:
            dist = DIST_DNP_MINOR_UNCERTAIN
        else:
            dist = DIST_DNP_UNCERTAIN

    # Deterministic: choose the highest-probability bucket
    dist = softclip_probs(dist)
    return int(LABELS[np.argmax(dist)])


BUCKETS = [0, 25, 50, 75, 100]
_RANKS  = np.array([0, 1, 2, 3, 4], dtype=float)

def compute_minutes_series(gw_df: pd.DataFrame) -> pd.Series:
    """
    Collapse per-fixture rows -> summed minutes per GW 1..38.
    Expects columns: 'round', 'minutes'.
    """
    per_gw = gw_df.groupby("round")["minutes"].sum()
    return per_gw.reindex(range(1, 39), fill_value=0).astype(int)

def compute_dimishing_hist(
    player_gw_df: pd.DataFrame,
    gw: int,
    key: str,
    prev_season_value: float,
    half_life_fixtures: float = 3.0,     # decay measured in *fixtures* (smaller = more reactive)
    max_lookback_fixtures: int = 20      # 0 for unlimited; else last N fixtures only
) -> float:
    """
    Diminishing-weight average of `key` using *past fixtures* only (no GW aggregation).
    Includes prior-season average as a GW0 sample older than all in-season fixtures.

    Parameters
    ----------
    player_gw_df : DataFrame
        Must include columns: 'round' (int), `key`, and ideally 'kickoff_time' for ordering.
        Each row is a fixture (double GWs appear as 2+ rows).
    gw : int
        Target gameweek; use fixtures with round < gw.
    key : str
        Column name to average (e.g. 'total_points' for form, 'ict_index' for ICT).
    prev_season_value : float
        Prior-season average to use as GW0.
    half_life_fixtures : float
        Exponential half-life measured in fixtures (weight = 0.5 ** (age/half_life_fixtures)).
    max_lookback_fixtures : int
        Only consider the last N fixtures before `gw` (0 = unlimited).

    Returns
    -------
    float
        Exponentially weighted average by fixture.
    """
    if "round" not in player_gw_df.columns:
        raise ValueError("player_gw_df must contain a 'round' column")

    df = player_gw_df.copy()
    df[key] = pd.to_numeric(df[key], errors="coerce").fillna(0.0)
    df["round"] = pd.to_numeric(df["round"], errors="coerce")

    # prior fixtures only (strictly before target GW)
    prev_fx = df.loc[df["round"] < gw].copy()

    # order by actual time if available; else by (round, index) as a stable fallback
    if "kickoff_time" in prev_fx.columns:
        with pd.option_context("mode.chained_assignment", None):
            prev_fx["_kt"] = pd.to_datetime(prev_fx["kickoff_time"], errors="coerce")
        prev_fx = prev_fx.sort_values(["_kt", "round"]).reset_index(drop=True)
    else:
        prev_fx = prev_fx.sort_values(["round"]).reset_index(drop=True)

    # keep only last N fixtures if requested
    if max_lookback_fixtures and max_lookback_fixtures > 0:
        prev_fx = prev_fx.tail(max_lookback_fixtures)

    k = len(prev_fx)
    vals = prev_fx[key].to_numpy(dtype=float)

    # age in fixtures: most recent previous fixture has age=1, older ones increase
    # (so blanks/doubles naturally don't matter; each fixture is one step back)
    if k > 0:
        ages = np.arange(k, 0, -1, dtype=float)  # oldest=k ... newest=1
        weights = np.power(0.5, ages / float(half_life_fixtures))
    else:
        ages = np.array([], dtype=float)
        weights = np.array([], dtype=float)

    # add GW0 (prior-season average) as one more (oldest) sample with age=k+1
    v0 = float(0 if pd.isna(prev_season_value) else prev_season_value)
    vals = np.append(vals, v0)
    age0 = float(k + 1)
    w0 = 0.5 ** (age0 / float(half_life_fixtures))
    weights = np.append(weights, w0)

    # safe weighted average
    wsum = float(weights.sum())
    if wsum <= 0:
        return v0
    return float(np.dot(vals, weights) / wsum)

def compute_upcoming_gws(upcoming_3_gws_df: pd.DataFrame, start_gw: int) -> float:
    """
    Computes a diminishing-weight difficulty score over the next 3 gameweeks
    starting at `start_gw` (inclusive).

    Rules per GW:
      - Single GW: value = game_difficulty
      - Double GW: value = diff1 + diff2 - 5
      - Blank GW:  value = 7
      - (Generalised) k fixtures: value = sum(diffs) - 5*(k-1)

    Weighting:
      - Exponential decay by GW distance (nearest GW weighted highest).

    Args
    ----
    upcoming_3_gws_df : DataFrame
        Must include columns: 'round' (int), 'difficulty' (int),
        where each row is a fixture.
    start_gw : int
        The starting gameweek (inclusive).

    Returns
    -------
    float
        Diminishing-weight aggregate difficulty across the next 3 GWs.
    """
    # Tunables
    HORIZON = 3          # number of upcoming GWs to include
    HALF_LIFE_GW = 1.5   # smaller -> more weight on the immediate next GW
    BLANK_VALUE = 7
    DOUBLE_PENALTY = 5

    if not {"round", "difficulty"}.issubset(upcoming_3_gws_df.columns):
        raise ValueError("upcoming_3_gws_df must have 'round' and 'difficulty' columns")

    df = upcoming_3_gws_df.copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")
    df = df.dropna(subset=["round", "difficulty"])

    rounds = [start_gw + i for i in range(HORIZON)]
    per_gw_scores = []

    for r in rounds:
        gw_rows = df.loc[df["round"] == r, "difficulty"]
        if gw_rows.empty:
            score = float(BLANK_VALUE)
        else:
            diffs = gw_rows.to_numpy(dtype=float)
            k = len(diffs)
            if k == 1:
                score = float(diffs[0])
            elif k == 2:
                score = float(diffs.sum() - DOUBLE_PENALTY)
            else:
                # generalised multi-fixture rule
                score = float(diffs.sum() - DOUBLE_PENALTY * (k - 1))
        per_gw_scores.append(score)

    ages = np.arange(0, HORIZON, dtype=float)  # 0 for start_gw, 1 for start_gw+1, ...
    weights = np.power(0.5, ages / float(HALF_LIFE_GW))
    return float(np.dot(per_gw_scores, weights) / weights.sum())

def get_upcoming_gw_df(player_gw_df, fixture_df, gw):
    next_3 = player_gw_df.loc[(player_gw_df["round"] >= gw) & (player_gw_df["round"] < (gw + 3))].copy()
    next_3_gws = next_3["round"].values
    fixture_ids = next_3["fixture"].values
    was_homes = next_3["was_home"].values
    diffs = []
    for i, fix in enumerate(fixture_ids):
        fixture_row = fixture_df.loc[fixture_df["id"] == fix]
        if was_homes[i]:
            diffs.append(fixture_row["team_a_difficulty"].values[0])
        else:
            diffs.append(fixture_row["team_h_difficulty"].values[0])
    return pd.DataFrame({
        "round": next_3_gws,
        "difficulty": diffs
    })

if __name__ == "__main__":
    seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

    for season in seasons:
        season_path = f"../data/Fantasy-Premier-League/data/{season}"
        fixtures_df = load_csv(f"{season_path}/fixtures.csv")
        season_int = convert_season_to_int(season)
        players_raw_df = load_csv(f"{season_path}/players_raw.csv")
        features = []
        for _, player_row in players_raw_df.iterrows():
            player = {}
            first_name = player_row["first_name"]
            last_name = player_row["second_name"]
            pid = player_row["id"]
            player["id"] = pid
            player["name"] = first_name + " " + last_name
            player_path = f"{season_path}/players/{first_name}_{last_name}_{pid}"
            player["path"] = player_path
            gw_df = load_csv(player_path + "/gw.csv")
            #--- GW 0 stuff (prev season) ---
            try:
                history_df = load_csv(player_path + "/history.csv")
                history_df["season_norm"] = history_df["season"].apply(convert_season_to_int)
                prev_season_row = history_df[history_df["season_norm"] == season_int - 1].iloc[-1]
                tp = pd.to_numeric(prev_season_row.get("total_points", 0), errors="coerce")
                ict = pd.to_numeric(prev_season_row.get("ict_index", 0), errors="coerce")
                form_gw0 = float(tp) / 38.0 if pd.notna(tp) else 0.0
                ict_gw0  = float(ict) / 38.0 if pd.notna(ict) else 0.0
            except:
                form_gw0 = 0
                ict_gw0 = 0
            minutes_series = compute_minutes_series(gw_df)
            for gw in range(1, 39):
                upcoming_fixtures = get_upcoming_gw_df(gw_df, fixtures_df, gw)
                player[f"upcoming_difficulty_gw{gw}"] = compute_upcoming_gws(upcoming_fixtures, gw)

                # 2) ALWAYS compute historical form/ict based on prior fixtures (GW0 feeds GW1)
                player[f"form_gw{gw}"] = compute_dimishing_hist(
                    gw_df, gw, "total_points", form_gw0, half_life_fixtures=3.0, max_lookback_fixtures=20
                )
                player[f"ict_gw{gw}"] = compute_dimishing_hist(
                    gw_df, gw, "ict_index",  ict_gw0,  half_life_fixtures=3.0, max_lookback_fixtures=20
                )

                # 3) Availability + transfer ratio depend on whether there is a row for this GW
                if gw not in gw_df["round"].values:  # blank GW or not at club
                    player[f"availability_gw{gw}"] = 0.0
                    player[f"transfer_ratio_gw{gw}"] = 0.0
                else:
                    gw_row = gw_df[gw_df["round"] == gw]
                    player[f"availability_gw{gw}"] = estimate_availability_bucket(minutes_series, gw) / 100.0
                    player[f"transfer_ratio_gw{gw}"] = calculate_transfer_ratio(gw_row)
            features.append(player)
        features_df = pd.DataFrame(features)
        features_df.to_csv(f"../iwans_system/{season}.csv", index=False)