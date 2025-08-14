import numpy as np
import pandas as pd


# Use to calculate form and ICT index for a player based on their past fixtures.
# player_gw_df: DataFrame with columns 'round', 'kickoff_time', and the key to calculate
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
