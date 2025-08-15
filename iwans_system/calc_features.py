import numpy as np
import pandas as pd

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

def get_path_from_row(root:str, row: pd.Series) -> str:
    first_name = row.get("first_name", "")
    last_name = row.get("second_name", "")
    id = row.get("id", "")
    return f"{root}/players/{first_name}_{last_name}_{id}"

def calc_transfer_ratio(gw_df: pd.DataFrame, pid:int) -> float:
    player_row = gw_df[gw_df["element"] == pid]
    t_in = player_row["transfers_in"].sum()
    t_out = player_row["transfers_out"].sum()
    if t_in + t_out == 0:
        return 0.0
    return t_in / (t_in + t_out)


def determine_avail(
    player_df: pd.DataFrame,
    gw: int,
    *,
    # starter/sub & rotation profiling
    min_start_minutes: int = 60,
    window: int = 5,
    sub_min: int = 10, sub_max: int = 45,
    starter_rate_cut: float = 0.4,
    doubt_prob: float = 0.05,
    rot_start_rate_lo: float = 0.2, rot_start_rate_hi: float = 0.6,
    rot_zero_streak_max: int = 2,
    rot_flip_rate_min: float = 0.4,
    rot_min_var_minutes: float = 400,
    # NEW: injury onset peek (handles GW16 case + blanks)
    onset_k_existing_zeros: int = 2,  # require this many consecutive zero-minute *entries* from GW
    onset_search_span: int = 5,       # check first N existing rows from GW onward
    seed: int | None = 42,
) -> tuple[int, int]:
    df = player_df.copy().sort_values("round").reset_index(drop=True)
    if "round" not in df or "minutes" not in df:
        raise ValueError("player_df must contain columns: 'round', 'minutes'")

    rounds = df["round"].astype(int).to_numpy()
    minutes = df["minutes"].fillna(0).astype(float).to_numpy()

    hist_mask = rounds < gw
    fut_mask = rounds >= gw
    if not hist_mask.any():
        return 100, 0

    hist_idx = np.where(hist_mask)[0]
    last_i = hist_idx[-1]
    m_last = minutes[last_i]

    ser_min = pd.Series(minutes)
    started = (ser_min >= min_start_minutes).astype(int)

    # prior-only rolling stats (shifted)
    roll_mean = ser_min.rolling(window, min_periods=1).mean().shift(1).fillna(0).to_numpy()
    roll_var  = ser_min.rolling(window, min_periods=1).var().shift(1).fillna(0).to_numpy()
    roll_start_rate = started.rolling(window, min_periods=1).mean().shift(1).fillna(0).to_numpy()

    def prior_flip_rate(i: int) -> float:
        lo = max(0, i - window); hi = i
        s = started.iloc[lo:hi].to_numpy()
        if len(s) <= 1: return 0.0
        return np.sum(np.abs(np.diff(s))) / (len(s) - 1)

    def prior_max_zero_streak(i: int) -> int:
        lo = max(0, i - window); hi = i
        z = (ser_min.iloc[lo:hi].to_numpy() == 0).astype(int)
        max_run = cur = 0
        for v in z:
            if v: cur += 1; max_run = max(max_run, cur)
            else: cur = 0
        return int(max_run)

    flip_rate = prior_flip_rate(last_i)
    max_zero_streak = prior_max_zero_streak(last_i)

    prof_sub = (sub_min <= roll_mean[last_i] <= sub_max) and (roll_start_rate[last_i] < starter_rate_cut)
    prof_starter = roll_start_rate[last_i] >= starter_rate_cut
    prof_rotation = (
        (rot_start_rate_lo <= roll_start_rate[last_i] <= rot_start_rate_hi) and
        (roll_var[last_i] >= rot_min_var_minutes) and
        (flip_rate >= rot_flip_rate_min) and
        (max_zero_streak <= rot_zero_streak_max)
    )
    prof_regular = prof_starter or prof_sub or prof_rotation

    # trailing zero streak up to gw-1 (injury candidate)
    injury_streak = 0
    j = last_i
    while j >= 0 and minutes[j] == 0:
        lo = max(0, j - window)
        rsr_j = started.iloc[lo:j].mean() if j > lo else 0.0
        rmean_j = ser_min.iloc[lo:j].mean() if j > lo else 0.0
        reg_sub_j = (sub_min <= rmean_j <= sub_max) and (rsr_j < starter_rate_cut)
        was_recent_regular_not_rotation = (rsr_j >= starter_rate_cut) or reg_sub_j
        if was_recent_regular_not_rotation:
            injury_streak += 1; j -= 1
        else:
            break

    # first future return GW (>= gw) — used when availability == 0
    return_gw = 0
    if fut_mask.any():
        for k in np.where(fut_mask)[0]:
            if minutes[k] > 0:
                return_gw = int(rounds[k]); break

    rng = np.random.default_rng(seed)

    # BASE decision using only history up to gw-1
    if m_last >= min_start_minutes:
        availability = 75 if rng.random() < doubt_prob else 100
    elif 0 < m_last < min_start_minutes:
        if prof_sub and (sub_min <= m_last <= sub_max):
            availability = 75 if rng.random() < doubt_prob else 100
        elif injury_streak > 0:
            availability = 75  # easing back
        else:
            availability = (75 if rng.random() < doubt_prob else 100) if prof_rotation else 50
    else:  # m_last == 0
        if prof_rotation:
            availability = 75 if rng.random() < doubt_prob else 100
        elif prof_regular and injury_streak > 0:
            if return_gw == gw:       availability = 75
            elif return_gw == gw + 1: availability = 25
            elif return_gw > gw + 1:  availability = 0
            else:                     availability = 0
        else:
            availability = 0

    # INJURY ONSET PEEK ...
    if onset_k_existing_zeros > 0 and availability == 100 and m_last >= min_start_minutes:
        fut_indices = np.where(rounds >= gw)[0][:onset_search_span]
        if len(fut_indices) > 0 and rounds[fut_indices[0]] == gw:
            fut_minutes = [minutes[i] for i in fut_indices]
            if len(fut_minutes) >= onset_k_existing_zeros and all(m == 0 for m in fut_minutes[:onset_k_existing_zeros]):
                availability = 0
                # BEFORE (bug): gws_until_return = return_gw if return_gw >= gw else 0
                gws_until_return = (return_gw - gw) if return_gw >= gw else 0   # <-- FIX
                return int(availability), int(gws_until_return)


    # final gws_until_return rule
    if availability > 0:
        gws_until_return = 0
    else:
        # BEFORE (other variant you may have tried): return_gw - gw + 1
        gws_until_return = (return_gw - gw) if return_gw >= gw else 0          # <-- FIX


    return int(availability), int(gws_until_return)

def calc_form(
    player_df: pd.DataFrame,
    gw: int,
    *,
    points_col: str = "total_points",
    decay: str = "exp",          # 'exp' or 'linear'
    half_life: float = 6.0,      # for exp decay: weight halves every `half_life` GWs
    linear_window: int | None = None,  # for linear decay: include only last N GWs
    return_nan_if_empty: bool = True
) -> float:
    prior_fixtures = player_df[(player_df["round"] < gw) & (player_df["round"] >= gw-5)].copy()
    prior_fixtures["n_gws_before"] = prior_fixtures["round"].max() - prior_fixtures["round"]
    if prior_fixtures.empty:
        return 0.0
    
    by_round = (
        prior_fixtures
        .groupby("round", as_index=True)
        .agg(
            round_points=(points_col, "sum"),
            n_gws_before=("n_gws_before", "first"),
        )
        .sort_index()
    )

    if by_round.empty:
        return np.nan if return_nan_if_empty else 0.0

    # Build weights
    n = by_round["n_gws_before"].astype(float)

    if decay == "exp":
        # Most recent prior round has n=0 -> weight=1.0
        weights = 0.5 ** (n / float(half_life))
    elif decay == "linear":
        if not linear_window or linear_window <= 0:
            raise ValueError("For linear decay, provide a positive linear_window.")
        # Keep only the last `linear_window` rounds (n = 0..linear_window-1)
        by_round = by_round.loc[n <= (linear_window - 1)]
        if by_round.empty:
            return np.nan if return_nan_if_empty else 0.0
        n = by_round["n_gws_before"].astype(float)
        # Most recent prior (n=0) gets weight=linear_window; farthest in window (n=linear_window-1) gets weight=1
        weights = (linear_window - n)
    else:
        raise ValueError("decay must be 'exp' or 'linear'.")

    # Weighted mean of per-ROUND points
    num = float(np.dot(by_round["round_points"].astype(float), weights))
    den = float(weights.sum())


    return num / den

def get_fixtures_info(fixtures_df, player_df, gw):
    diff_scores = []
    for g in range(gw, gw+5):
        fixtures_from_player_df = player_df[player_df["round"] == g]

        # arbitrarily high difficulty if a blank gw
        if fixtures_from_player_df.empty:
            diff_scores.append(8) 
            continue
        current_score = 0
        
        # double gameweek starts at -5 difficulty
        if len(fixtures_from_player_df) > 1:
            current_score -= 5
        
        for _, row in fixtures_from_player_df.iterrows():
            was_home = row["was_home"]
            fix_id = row["fixture"]
            fixture_row = fixtures_df[fixtures_df["id"] == fix_id]
            if was_home:
                current_score += fixture_row["team_h_difficulty"].values[0]
            else:
                current_score += fixture_row["team_a_difficulty"].values[0]
        diff_scores.append(current_score)
    
    return float(diff_scores[0]), np.array(diff_scores[1:]).mean()

    
def get_player_features(fixtures_df: pd.DataFrame, player_df: pd.DataFrame, gw_df: pd.DataFrame, upcoming_gw: int, pid: int, avail=None) -> List:
    """
    Returns a list of all features of a player at a specific gw

    ARGS:
    fixtures_df: DataFrame containing 
    player_df: DataFrame containing fixture data for a specific player. Must have columns 'round', 'total_points', 'ict_index', 'fixture', 'was_home'
    gw_df: DataFrame containing GW data for all players. Has cols transfers_in, transfers_out
    avail: Optional; provide if availability is known, otherwise will be estimated by looking at past, present and future mins
    upcoming_gw: The upcoming gameweek
    """
    if avail is None:
        avail = determine_avail(player_df, upcoming_gw)

    fixtures = get_fixtures_info(fixtures_df, player_df, upcoming_gw)
    return {
        "transfer_ratio": calc_transfer_ratio(gw_df, pid), # t_in / (t_in + t_out) or 0
        "form": calc_form(player_df, upcoming_gw), # weighted last 5 gameweeks according to how many GWs apart
        "ict": calc_form(player_df, upcoming_gw, points_col="ict_index"),
        "gws_until_return": avail[1],
        "availability": avail[0],
        "upcoming_fixture": fixtures[0], # difficulty of next fixture
        "later_fixtures": fixtures[1], # average difficulty of next 4 fixtures
    }

if __name__ == "__main__":
    seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
    for season in seasons:
        root = f"../data/Fantasy-Premier-League/data/{season}"
        fixtures_df = load_csv(f"{root}/fixtures.csv")
        players_df = load_csv(f"{root}/players_raw.csv")
        pids = list(players_df["id"].unique())
        players = []
        for pid in pids:
            player_row = players_df[players_df["id"] == pid].iloc[0]
            player = {}
            first_name = player_row["first_name"]
            last_name = player_row["second_name"]
            pid = player_row["id"]
            player["id"] = pid
            player["name"] = first_name + " " + last_name
            player_path = f"{root}/players/{first_name}_{last_name}_{pid}"
            player["path"] = player_path
            player_df = load_csv(player_path + "/gw.csv")
            for gw in range(1, 39):
                gw_df = load_csv(f"{root}/gws/gw{gw}.csv")
                feats = get_player_features(
                    fixtures_df,
                    player_df,
                    gw_df,
                    gw,
                    pid
                )
                for key, value in feats.items():
                    player[f"{key}_{gw}"] = value
            players.append(player)
        df = pd.DataFrame(players)
        df.to_csv(f"../iwans_system/feature_data/{season}_features.csv")