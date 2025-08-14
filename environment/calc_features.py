def determine_opposition_ids_in_next_3(pid):
    return [1, 2, 3]  # Example opposition IDs for the next 3 gameweeks


def calc_form(pid):
    ## GET LAST UPDATED DATA OF API AND SUBTRACT 30 DAYS
    return 5.0

def calc_next_gws_data(pid):
    return {
        "next3_sum__n_games_in_gw": 0,
        "next3_mean__avg_home": 0,
        "next3_mean__playing_against_mean_difficulty": 0,
        "next3_mean__playing_against_mean_defence": 0,
        "next3_mean__playing_against_mean_attack": 0,
        "next3_mean__playing_for_mean_difficulty": 0,
        "next3_mean__playing_for_mean_defence": 0,
        "next3_mean__playing_for_mean_attack": 0,
    }

def player_data(pid):
    player = {}
    return {
        "position": 0,
        "creativity": 0,
        "influence": 0,
        "threat": 0,
        "selected": 0,
        "playing_chance": 0,
        "birth_date": 0,
        "corners_and_free_kicks_order": 0,
        "penalties_order": 0
    }

def previous_gw_data(pid):
    # each of these are averages of all the previous gameweeks.
    return {
        "avg_points_when_playing": 50,
        "avg_minutes_when_playing": 1.5,
        "avg_assists_p_game_when_playing": 0.5,
        "avg_goals_p_game_when_playing": 0.3,
        "avg_bps_p_game_when_playing": 2.0,
        "avg_yellows_p_game_when_playing": 0.1,
        "avg_reds_p_game_when_playing": 0.05,
        "avg_clean_sheets_when_playing": 0.2,
        "avg_goals_conceded_when_playing": 1.0,
        "avg_starts_when_playing": 0.8,
        "avg_saves_when_playing": 3.0,
    }