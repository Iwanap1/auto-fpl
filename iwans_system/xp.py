# classifiers.py
from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
import pandas as pd
import numpy as np
from joblib import load as joblib_load

def load_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")

def load_models(path):
    """
    Load the models for each position from the specified directory.
    """
    models = []
    for pos in ["goalkeepers", "defenders", "midfielders", "forwards"]:
        model_path = f"{path}/{pos}.pkl"
        models.append(joblib_load(model_path))
    return models


POS_TO_IDX = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


class ScorePredictor:
    """
    Builds per-player feature rows compatible with your saved models and runs
    per-position inference in batch. Returns probability (binary clf) or
    regressed xP (regressor). You can convert probability->xP if you wish.
    """

    def __init__(self,
                 season: str,
                 players_raw_df: pd.DataFrame,
                 teams_raw_df: pd.DataFrame,
                 fixtures_raw_df: pd.DataFrame,
                 models: List):
        self.season = season
        self.players_raw_df = players_raw_df
        self.teams_raw_df = teams_raw_df
        self.fixtures_raw_df = fixtures_raw_df
        self.models = models  # [GK, DEF, MID, FWD]

    # --------- PUBLIC: batch predict for a list of (pid, pos) ----------
    def predict_batch(self,
                      pid_pos: Iterable[Tuple[int, str]],
                      gw: int) -> Dict[int, float]:
        """
        pid_pos: iterable of (player_id, position_str) with position in {"GK","DEF","MID","FWD"}
        Returns: {pid: xp_value}
        """
        by_pos: Dict[str, List[int]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for pid, pos in pid_pos:
            if pos in by_pos:
                by_pos[pos].append(pid)

        out: Dict[int, float] = {}
        for pos, pids in by_pos.items():
            if not pids:
                continue
            model = self.models[POS_TO_IDX[pos]]
            # Build features for this position in batch
            rows = []
            pids_ok = []
            for pid in pids:
                try:
                    rows.append(self.prepare_data(pid, gw))
                    pids_ok.append(pid)
                except Exception:
                    # silently skip player if feature build fails
                    continue
            if not rows:
                continue
            X = pd.DataFrame(rows)

            # Align to training feature set
            feat_cols = getattr(model, "feature_names_in_", None)
            if feat_cols is None:
                # If your saved model doesn't have feature_names_in_, you must
                # hardcode/ship the feature list used at training.
                raise ValueError("Model missing feature_names_in_. Save models with it, or provide a fixed feature list.")
            # reindex and fill
            X = X.reindex(columns=feat_cols, fill_value=0.0)
            # Predict
            if hasattr(model, "predict_proba"):
                # Binary classifier: use class-1 probability as xP-like score
                proba = model.predict_proba(X)[:, 1]
                xp = proba.astype(float)
            else:
                preds = model.predict(X)
                xp = np.asarray(preds, dtype=float)

            out.update({pid: float(v) for pid, v in zip(pids_ok, xp)})

        return out

    _POS_INV = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    
    def xp_map_strict_for_candidates(
        self,
        gw: int,
        candidate_pids: list[int],
        players_raw_df,
        fixtures_df,
    ) -> dict[int, float]:
        """
        Compute xP only for candidate_pids.
        - If a candidate's team has a fixture in GW, they MUST have xP (raise if missing).
        - If no fixture, xP = 0.0 is allowed.
        """
        pr = players_raw_df.set_index("id")
        # teams that play in this GW
        f = fixtures_df.loc[fixtures_df["event"] == gw, ["team_h", "team_a"]]
        teams_with_fixture = set(pd.concat([f["team_h"], f["team_a"]]).astype(int).tolist())

        # split candidates by whether their team plays
        with_fx, no_fx = [], []
        for pid in candidate_pids:
            team_id = int(pr.at[pid, "team"])
            (with_fx if team_id in teams_with_fixture else no_fx).append(pid)

        # predict only for players whose teams play
        pid_pos = [(pid, self._POS_INV[int(pr.at[pid, "element_type"])]) for pid in with_fx]
        xp_pred = self.predict_batch(pid_pos, gw)  # {pid: xp}

        # strict check: all with fixtures must be present
        missing = set(with_fx) - set(xp_pred.keys())
        if missing:
            raise ValueError(f"Missing xP for candidates with fixtures: {sorted(list(missing))[:8]} ...")

        # assemble final map (0.0 for blanks)
        xp_map = {pid: 0.0 for pid in candidate_pids}
        xp_map.update(xp_pred)
        return xp_map


    # --------- Your feature builders (kept as you wrote them) ----------
    def prepare_data(self, pid, gw):
        data = self.data_dict()
        data["gw"] = gw
        data["year"] = self.season
        data["player"] = pid

        # players_raw_df row
        player_row = self.players_raw_df[self.players_raw_df["id"] == pid]
        if player_row.empty:
            raise ValueError(f"Player {pid} not in players_raw_df")
        first_name = player_row["first_name"].values[0]
        last_name  = player_row["second_name"].values[0]

        # per-player gw file
        player_df = load_csv(f"../data/Fantasy-Premier-League/data/{self.season}/players/{first_name}_{last_name}_{pid}/gw.csv")

        # Cast kickoff_time if present
        if "kickoff_time" in player_df.columns:
            player_df["kickoff_time"] = pd.to_datetime(player_df["kickoff_time"], errors="coerce")

        # CURRENT GW rows (handles doubles)
        gw_rows = player_df[player_df["round"] == gw]
        if gw_rows.empty:
            # Let caller handle skip; we raise to be caught by predict_batch
            raise ValueError(f"No data for GW {gw} for player {pid}")

        data["n_games_in_gw"] = len(gw_rows)
        data["points_scored"] = gw_rows["total_points"].sum()
        data["avg_home"] = float(gw_rows["was_home"].mean()) if "was_home" in gw_rows else 0.0

        # Fixture difficulty & strengths (averaged if double)
        playing_against_difficulty = 0.0
        playing_against_defence = 0.0
        playing_against_attack = 0.0
        playing_for_difficulty = None
        playing_for_defence = None
        playing_for_attack = None

        for _, row in gw_rows.iterrows():
            fix = self.fixtures_raw_df[self.fixtures_raw_df["id"] == row["fixture"]]
            if fix.empty:
                continue
            fixture = fix.iloc[0]
            home = bool(row.get("was_home", False))
            team_id = fixture["team_h"] if home else fixture["team_a"]
            opp_id  = fixture["team_a"] if home else fixture["team_h"]

            if home:
                playing_against_difficulty += float(fixture.get("team_a_difficulty", 0.0))
                pf_diff = float(fixture.get("team_h_difficulty", 0.0))
                pf_def  = float(self.teams_raw_df.loc[self.teams_raw_df["id"] == team_id, "strength_defence_home"].values[0])
                pf_att  = float(self.teams_raw_df.loc[self.teams_raw_df["id"] == team_id, "strength_attack_home"].values[0])
                playing_against_defence += float(self.teams_raw_df.loc[self.teams_raw_df["id"] == opp_id, "strength_defence_away"].values[0])
                playing_against_attack  += float(self.teams_raw_df.loc[self.teams_raw_df["id"] == opp_id, "strength_attack_away"].values[0])
            else:
                playing_against_difficulty += float(fixture.get("team_h_difficulty", 0.0))
                pf_diff = float(fixture.get("team_a_difficulty", 0.0))
                pf_def  = float(self.teams_raw_df.loc[self.teams_raw_df["id"] == team_id, "strength_defence_away"].values[0])
                pf_att  = float(self.teams_raw_df.loc[self.teams_raw_df["id"] == team_id, "strength_attack_away"].values[0])
                playing_against_defence += float(self.teams_raw_df.loc[self.teams_raw_df["id"] == opp_id, "strength_defence_home"].values[0])
                playing_against_attack  += float(self.teams_raw_df.loc[self.teams_raw_df["id"] == opp_id, "strength_attack_home"].values[0])

            if playing_for_difficulty is None:
                playing_for_difficulty = pf_diff
                playing_for_defence = pf_def
                playing_for_attack = pf_att

        n_games = data["n_games_in_gw"]
        if n_games:
            data["playing_against_mean_difficulty"] = playing_against_difficulty / n_games
            data["playing_against_mean_defence"] = playing_against_defence / n_games
            data["playing_against_mean_attack"] = playing_against_attack / n_games
        else:
            data["playing_against_mean_difficulty"] = 0.0
            data["playing_against_mean_defence"] = 0.0
            data["playing_against_mean_attack"] = 0.0

        data["playing_for_difficulty"] = playing_for_difficulty or 0.0
        data["playing_for_defence"] = playing_for_defence or 0.0
        data["playing_for_attack"] = playing_for_attack or 0.0

        # Prior GW stats
        prior_gw_rows = player_df[player_df["round"] < gw]
        data.update(self.get_prev_gw_data(prior_gw_rows))

        # Player static data
        data.update(self.get_player_data(self.players_raw_df, pid))

        # Form (moving 30-day avg prior to the GW)
        data["form"] = self.calc_form(gw, player_df)

        return data

    def calc_form(self, gw: int, player_gw_df: pd.DataFrame) -> float:
        ref_time = player_gw_df.loc[player_gw_df["round"] == gw, "kickoff_time"].min()
        if pd.isna(ref_time):
            return 0.0
        window_start = ref_time - pd.Timedelta(days=30)
        prior = player_gw_df.loc[
            (player_gw_df["kickoff_time"] >= window_start) &
            (player_gw_df["kickoff_time"] < ref_time),
            "total_points"
        ]
        return float(prior.mean()) if len(prior) else 0.0

    def get_prev_gw_data(self, prior_gw_rows: pd.DataFrame):
        data = {
            "avg_points_when_playing": 0.0,
            "avg_minutes_when_playing": 0.0,
            "avg_assists_p_game_when_playing": 0.0,
            "avg_goals_p_game_when_playing": 0.0,
            "avg_bps_p_game_when_playing": 0.0,
            "avg_yellows_p_game_when_playing": 0.0,
            "avg_reds_p_game_when_playing": 0.0,
            "avg_clean_sheets_when_playing": 0.0,
            "avg_goals_conceded_when_playing": 0.0,
            "avg_starts_when_playing": 0.0,
            "avg_saves_when_playing": 0.0
        }
        p = prior_gw_rows[prior_gw_rows.get("minutes", 0) > 0]
        if p.empty:
            return data
        def m(col): return float(p[col].mean()) if col in p.columns else 0.0
        data["avg_points_when_playing"] = m("total_points")
        data["avg_minutes_when_playing"] = m("minutes")
        data["avg_assists_p_game_when_playing"] = m("assists")
        data["avg_goals_p_game_when_playing"] = m("goals_scored")
        data["avg_bps_p_game_when_playing"] = m("bps")
        data["avg_yellows_p_game_when_playing"] = m("yellow_cards")
        data["avg_reds_p_game_when_playing"] = m("red_cards")
        data["avg_clean_sheets_when_playing"] = m("clean_sheets")
        data["avg_goals_conceded_when_playing"] = m("goals_conceded")
        data["avg_starts_when_playing"] = m("starts")
        data["avg_saves_when_playing"] = m("saves")
        return data

    def get_player_data(self, players_raw_df, id):
        player = players_raw_df[players_raw_df["id"] == id]
        if player.empty:
            raise ValueError(f"Player with ID {id} not found in players_raw_df.")
        def g(col, default=0.0):
            return float(player[col].fillna(default).values[0]) if col in player.columns else default
        result = {
            "position": int(g("element_type", 0.0)),
            "creativity": g("creativity"),
            "influence": g("influence"),
            "threat": g("threat"),
            "selected": g("selected_by_percent"),
            "playing_chance": g("chance_of_playing_this_round"),
            "birth_date": player["birth_date"].values[0] if "birth_date" in player.columns else None,
            "corners_and_free_kicks_order": g("corners_and_indirect_freekicks_order"),
            "penalties_order": g("penalties_order"),
        }
        return result

    def data_dict(self):
        cols = [
            'gw','year','player',
            'n_games_in_gw','points_scored','avg_home',
            'playing_against_mean_difficulty','playing_against_mean_defence','playing_against_mean_attack',
            'playing_for_difficulty','playing_for_defence','playing_for_attack',
            'form','position','creativity','influence','threat','selected','playing_chance','birth_date',
            'corners_and_free_kicks_order','penalties_order',
            'avg_minutes_when_playing','avg_points_when_playing','avg_yellows_p_game_when_playing',
            'avg_reds_p_game_when_playing','avg_bps_p_game_when_playing','avg_goals_p_game_when_playing',
            'avg_assists_p_game_when_playing','avg_clean_sheets_when_playing','avg_goals_conceded_when_playing',
            'avg_starts_when_playing','avg_saves_when_playing'
        ]
        return {c: 0 for c in cols}
