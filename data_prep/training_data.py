import os
import re
import pandas as pd
from typing import Dict, Any



def norm_name(name_path):
    """
    Normalize player names by removing special characters and converting to lowercase.
    """
    name = os.path.basename(name_path).replace(".csv", "")
    name = name.replace("_", " ")
    name = re.sub(r"\d*", "", name)
    return name.strip()

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
    # last-resort: decode with utf-8 replacing bad bytes (keeps data length)
    return pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )


def calc_form(gw: int, df: pd.DataFrame) -> float:
    """
    Return the average total_points in the 30 days strictly before GW `gw`.
    Assumes `df` is a single player's gw.csv with columns:
      - 'round' (GW number)
      - 'kickoff_time' (ISO timestamp)
      - 'total_points'
    """

    # Reference time = earliest kickoff of the target GW (handles double GWs)
    ref_time = df.loc[df["round"] == gw, "kickoff_time"].min()
    if pd.isna(ref_time):
        raise ValueError(f"GW {gw} not found in dataframe")

    window_start = ref_time - pd.Timedelta(days=30)
    prior = df.loc[(df["kickoff_time"] >= window_start) & (df["kickoff_time"] < ref_time), "total_points"]

    return float(prior.mean()) if len(prior) else 0.0


def get_team_name(year, gw, season_data, fixture_df, opp):
    teams = load_csv(f"../data/Fantasy-Premier-League/data/{year}/teams.csv")
    was_home = season_data[season_data["round"] == gw]["was_home"].values[0]
    fixture = fixture_df[fixture_df["id"] == season_data[season_data["round"] == gw]["fixture"].values[0]]
    if opp:
        team_id = fixture["team_h"].values[0] if not was_home else fixture["team_a"].values[0]
    else:
        team_id = fixture["team_h"].values[0] if was_home else fixture["team_a"].values[0]
    date = pd.to_datetime(fixture["kickoff_time"]).values[0]
    team_name = teams[teams["id"] == team_id]["name"].values[0]
    return team_name
    


def get_player_data(year, id):
    data = load_csv(f"../data/Fantasy-Premier-League/data/{year}/players_raw.csv")
    player = data[data["id"] == id]
    if player.empty:
        raise ValueError(f"Player with ID {id} not found in {year} data.")
    result = {
        "position": player["element_type"].values[0],
        "creativity": player["creativity"].values[0],
        "influence": player["influence"].values[0],
        "threat": player["threat"].values[0],
        "selected": player["selected_by_percent"].values[0],
        "playing_chance": player["chance_of_playing_this_round"].values[0],
        "birth_date": player["birth_date"].values[0] if "birth_date" in player.columns else None,
        "corners_and_free_kicks_order": player["corners_and_indirect_freekicks_order"].fillna(0).values[0],
        "penalties_order": player["penalties_order"].fillna(0).values[0],
    }
    need_to_norm = {
        "team_join_date": player["team_join_date"].values[0] if "team_join_date" in player.columns else None,
        "bps": player["bps"],
        "minutes_total": player["minutes"].values[0],
        "yellow_cards": player["yellow_cards"].values[0],
        "red_cards": player["red_cards"].values[0],
    }
    return result, need_to_norm

def get_team_data(team_name, teams_df, home):
    team_r = teams_df[teams_df["name"] == team_name]
    if home:
        team_def = team_r["strength_defence_away"].values[0]
        team_att = team_r["strength_attack_away"].values[0]
    else:
        team_def = team_r["strength_defence_away"].values[0]
        team_att = team_r["strength_attack_away"].values[0]
    strength = team_r["strength"].values[0]
    return strength, team_att, team_def

def generate_data2():
    results = []
    years = ["2021-22", "2022-23", "2023-24", "2024-25"]
    for year in years:
        root = f"../data/Fantasy-Premier-League/data/{year}/players"
        dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        fixture_df = load_csv(f"../data/Fantasy-Premier-League/data/{year}/fixtures.csv").copy()
        teams = load_csv(f"../data/Fantasy-Premier-League/data/{year}/teams.csv").copy()

        # Index for fast, reliable lookups
        fixtures_idx = fixture_df.set_index("id", drop=True)
        teams_idx = teams.set_index("id", drop=True)

        for player in dirs:
            player_path = os.path.join(root, player)
            season_data = load_csv(os.path.join(player_path, "gw.csv")).copy()

            # Robust types
            if "kickoff_time" in season_data.columns:
                season_data["kickoff_time"] = pd.to_datetime(season_data["kickoff_time"], errors="coerce")
            for col in ["minutes", "round", "fixture", "team", "opponent_team", "bps",
                        "yellow_cards", "red_cards", "goals_scored", "assists",
                        "clean_sheets", "goals_conceded", "saves", "total_points", "starts"]:
                if col in season_data.columns:
                    season_data[col] = pd.to_numeric(season_data[col], errors="coerce").fillna(0)

            played_games = season_data[season_data["minutes"] > 0]
            player_name = norm_name(player)

            for gw in range(1, 39):
                played_games_prior = played_games[played_games["round"] < gw]

                gw_player_data = season_data[season_data["round"] == gw]
                if gw_player_data.empty:
                    continue
                if gw_player_data["minutes"].sum() == 0:
                    continue

                # Basic row
                result = {"gw": gw, "year": year, "player": player_name}

                # Player ID from gw.csv (FPL element id)
                if "element" in gw_player_data.columns:
                    id_ = int(gw_player_data["element"].iloc[0])
                else:
                    id_ = None

                n_games = len(gw_player_data)
                result["n_games_in_gw"] = n_games
                result["points_scored"] = float(gw_player_data["total_points"].sum())

                fixtures = gw_player_data["fixture"].astype(int).tolist()
                was_home_flags = gw_player_data["was_home"].astype(bool).tolist() if "was_home" in gw_player_data.columns else [True]*n_games
                avg_home = sum(was_home_flags) / n_games if n_games > 0 else 0.0
                result["avg_home"] = avg_home
                # Accumulators across (possible) multiple fixtures
                playing_against_difficulty = 0.0
                playing_against_defence = 0.0
                playing_against_attack = 0.0

                # We'll also record the *player's* team context for the first fixture of the GW
                playing_for_difficulty = None
                playing_for_defence = None
                playing_for_attack = None

                for i, fixture_id in enumerate(fixtures):
                    if fixture_id not in fixtures_idx.index:
                        # Skip if fixture not found
                        continue
                    frow = fixtures_idx.loc[fixture_id]
                    home = bool(was_home_flags[i])

                    # Determine our team_id and opponent team_id from the fixture + home flag
                    team_id = int(frow["team_h"]) if home else int(frow["team_a"])
                    opp_id  = int(frow["team_a"]) if home else int(frow["team_h"])

                    # Difficulties from fixtures.csv
                    if home:
                        playing_against_difficulty += float(frow["team_a_difficulty"])
                        pf_diff = float(frow["team_h_difficulty"])
                    else:
                        playing_against_difficulty += float(frow["team_h_difficulty"])
                        pf_diff = float(frow["team_a_difficulty"])

                    # Strengths from teams.csv (be careful with home/away columns)
                    if team_id in teams_idx.index:
                        if home:
                            pf_def = float(teams_idx.at[team_id, "strength_defence_home"])
                            pf_att = float(teams_idx.at[team_id, "strength_attack_home"])
                        else:
                            pf_def = float(teams_idx.at[team_id, "strength_defence_away"])
                            pf_att = float(teams_idx.at[team_id, "strength_attack_away"])
                    else:
                        pf_def = pf_att = float("nan")

                    if opp_id in teams_idx.index:
                        if home:
                            # opponent is away
                            playing_against_defence += float(teams_idx.at[opp_id, "strength_defence_away"])
                            playing_against_attack  += float(teams_idx.at[opp_id, "strength_attack_away"])
                        else:
                            # opponent is home
                            playing_against_defence += float(teams_idx.at[opp_id, "strength_defence_home"])
                            playing_against_attack  += float(teams_idx.at[opp_id, "strength_attack_home"])
                    else:
                        # If unknown, skip contributions
                        pass

                    # Save "playing_for_*" once (first fixture context)
                    if playing_for_difficulty is None:
                        playing_for_difficulty = pf_diff
                        playing_for_defence = pf_def
                        playing_for_attack = pf_att

                # Averages over the number of fixtures in the GW
                if n_games > 0:
                    result["playing_against_mean_difficulty"] = playing_against_difficulty / n_games
                    result["playing_against_mean_defence"] = playing_against_defence / n_games
                    result["playing_against_mean_attack"] = playing_against_attack / n_games
                else:
                    result["playing_against_mean_difficulty"] = 0.0
                    result["playing_against_mean_defence"] = 0.0
                    result["playing_against_mean_attack"] = 0.0

                result["playing_for_difficulty"] = playing_for_difficulty
                result["playing_for_defence"] = playing_for_defence
                result["playing_for_attack"] = playing_for_attack

                # Optional: enrich with your player metadata
                if id_ is not None:
                    player_data, _ = get_player_data(year, id_)
                    for key, value in player_data.items():
                        result[key] = value

                # Prior-form features (when played)
                denom = max(len(played_games_prior), 1)
                total_minutes = float(played_games_prior["minutes"].sum())
                result["avg_minutes_when_playing"] = total_minutes / denom if len(played_games_prior) > 0 else 0.0
                result["avg_points_when_playing"] = float(played_games_prior["total_points"].sum()) / denom if len(played_games_prior) > 0 else 0.0
                result["form"] = calc_form(gw, season_data)

                # Rates from prior games
                def avg(col):
                    return float(played_games_prior[col].sum()) / denom if len(played_games_prior) > 0 else 0.0

                for col, out in [
                    ("yellow_cards", "avg_yellows_p_game_when_playing"),
                    ("red_cards", "avg_reds_p_game_when_playing"),
                    ("bps", "avg_bps_p_game_when_playing"),
                    ("goals_scored", "avg_goals_p_game_when_playing"),
                    ("assists", "avg_assists_p_game_when_playing"),
                    ("clean_sheets", "avg_clean_sheets_when_playing"),
                    ("goals_conceded", "avg_goals_conceded_when_playing"),
                    ("starts", "avg_starts_when_playing"),
                    ("saves", "avg_saves_when_playing"),
                ]:
                    if col in played_games_prior.columns:
                        result[out] = avg(col)
                    else:
                        result[out] = 0.0

                results.append(result)

    return pd.DataFrame(results)
