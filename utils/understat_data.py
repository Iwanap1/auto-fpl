"""
Build a random (fixture, player) dataset from FPL fixtures + Understat match data.

Usage:
    pip install understat aiohttp pandas
    python this_script.py

Notes:
- Internet access required (Understat endpoints).
- If name mismatches occur, extend TEAM_NAME_MAP below.
"""

import asyncio
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import aiohttp
import pandas as pd
from understat import Understat


# ---------- Config ----------
FPL_PLAYERS_PATH = "../data/Fantasy-Premier-League/data/2024-25/cleaned_players.csv"
FPL_FIXTURES_PATH = "../data/Fantasy-Premier-League/data/2024-25/fixtures.csv"
N_SAMPLES = 10
RANDOM_SEED = 42
SAVE_CSV = "random_understat_player_fixture_dataset.csv"  # set to None to skip saving

# Map FPL team names to Understat canonical names (extend as needed)
TEAM_NAME_MAP = {
    "Brighton and Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle United",
    "Nottingham Forest": "Nottingham Forest",
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Sheffield United": "Sheffield United",
    "AFC Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Burnley": "Burnley",
    "Aston Villa": "Aston Villa",
    "Leicester City": "Leicester",
    "Southampton": "Southampton",
    "Leeds United": "Leeds",
    "Norwich City": "Norwich",
    "Watford": "Watford",
    "West Bromwich Albion": "West Bromwich Albion",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Luton Town": "Luton",
    "Huddersfield Town": "Huddersfield",
    "Swansea City": "Swansea",
    "Middlesbrough": "Middlesbrough",
    "Hull City": "Hull",
    "Cardiff City": "Cardiff",
    # add any older PL teams you need
}


# ---------- Helpers ----------

def season_from_kickoff(kickoff_iso: str) -> int:
    """
    Understat uses the starting year of the season (e.g., 2023 for 2023/24).
    If month >= July -> season = year; else season = year - 1.
    """
    if kickoff_iso.endswith("Z"):
        kickoff_iso = kickoff_iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(kickoff_iso)
    return dt.year if dt.month >= 7 else dt.year - 1


def to_understat_team_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


async def get_fpl_team_map() -> Dict[int, str]:
    """Fetch FPL team_id -> team_name map from bootstrap (read-only, public)."""
    import aiohttp
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    async with aiohttp.ClientSession() as s:
        async with s.get(url, timeout=30) as r:
            r.raise_for_status()
            data = await r.json()
    return {t["id"]: t["name"] for t in data["teams"]}


async def find_understat_match_id(u: Understat, home_us: str, away_us: str, season: int, kickoff_iso: str) -> Optional[int]:
    """
    Find the Understat match id by scanning the home team's results for the season
    and matching (home, away, date). Falls back to nearest by time if exact date fails.

    Fix: make both FPL (kickoff_iso) and Understat ('datetime') naive for comparison.
    """
    try:
        results = await u.get_team_results(home_us, season)
    except Exception as e:
        print(f"[WARN] Could not fetch results for {home_us} {season}: {e}")
        return None

    # FPL gives ISO with Z; make it naive
    if kickoff_iso.endswith("Z"):
        kickoff_iso = kickoff_iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(kickoff_iso).replace(tzinfo=None)  # strip tz
    target_date = dt.date()

    candidates: List[Dict[str, Any]] = []
    for m in results:
        h = m.get("h", {}).get("title")
        a = m.get("a", {}).get("title")
        if not h or not a:
            continue
        if not ((h == home_us and a == away_us) or (h == away_us and a == home_us)):
            continue

        # Understat 'datetime' is naive string like "2023-05-07 15:30:00"
        md = datetime.strptime(m["datetime"], "%Y-%m-%d %H:%M:%S")
        if md.date() == target_date:
            candidates.append(m)

    if not candidates:
        # pick closest datetime match among the same teams
        closest = None
        min_abs = None
        for m in results:
            h = m.get("h", {}).get("title")
            a = m.get("a", {}).get("title")
            if not h or not a:
                continue
            if not ((h == home_us and a == away_us) or (h == away_us and a == home_us)):
                continue
            md = datetime.strptime(m["datetime"], "%Y-%m-%d %H:%M:%S")
            diff = abs((md - dt).total_seconds())  # both naive now
            if min_abs is None or diff < min_abs:
                min_abs = diff
                closest = m
        if closest:
            candidates.append(closest)

    if not candidates:
        return None

    # prefer exact home/away orientation match
    for m in candidates:
        if m.get("h", {}).get("title") == home_us and m.get("a", {}).get("title") == away_us:
            return int(m["id"])
    return int(candidates[0]["id"])




@dataclass
class PlayerFixtureStats:
    match_id: int
    season: int
    gw: Optional[int]
    kickoff_time: str

    fpl_home_team_id: int
    fpl_away_team_id: int
    fpl_home_team: str
    fpl_away_team: str

    understat_home_team: str
    understat_away_team: str

    player_id: Optional[int]
    player_name: str
    player_team: str
    player_position: Optional[str]
    player_minutes: Optional[int]
    starter: Optional[bool]

    opponent_team: str

    shots: int
    shots_on_target: int
    goals: int
    xg_sum: float
    avg_xg_per_shot: float


def aggregate_player_shots(shots: List[Dict[str, Any]], player_name: str) -> Dict[str, Any]:
    """Aggregate shot events for the chosen player."""
    ps = [s for s in shots if s.get("player") == player_name]
    if not ps:
        return dict(shots=0, shots_on_target=0, goals=0, xg_sum=0.0, avg_xg_per_shot=0.0)
    shots_n = len(ps)
    goals = sum(1 for s in ps if s.get("result") == "Goal")
    shots_ot = sum(1 for s in ps if s.get("result") in ("SavedShot", "Goal", "ShotOnPost"))  # conservative
    xg_sum = float(sum(float(s.get("xG", 0.0)) for s in ps))
    return dict(
        shots=shots_n,
        shots_on_target=shots_ot,
        goals=goals,
        xg_sum=round(xg_sum, 4),
        avg_xg_per_shot=round(xg_sum / shots_n, 4),
    )


async def sample_random_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)

    # Load CSVs
    fixtures = pd.read_csv(FPL_FIXTURES_PATH)
    # Filter to fixtures with a kickoff time (played/scheduled) and finished or started
    mask = fixtures["kickoff_time"].notna() & (fixtures["finished"] | fixtures["started"])
    fixtures = fixtures.loc[mask].reset_index(drop=True)

    # Map team IDs -> names from FPL, then to Understat names
    fpl_id_to_name = await get_fpl_team_map()

    rows: List[PlayerFixtureStats] = []

    async with aiohttp.ClientSession() as session:
        u = Understat(session)

        # Pre-fetch a small cache for team name canonicalization (optional)

        for _ in range(n):
            # 1) randomly pick a fixture row
            fx = fixtures.sample(1).iloc[0]

            home_name_fpl = fpl_id_to_name[int(fx["team_h"])]
            away_name_fpl = fpl_id_to_name[int(fx["team_a"])]

            home_us = to_understat_team_name(home_name_fpl)
            away_us = to_understat_team_name(away_name_fpl)

            kickoff = str(fx["kickoff_time"])
            season = season_from_kickoff(kickoff)

            # 2) find Understat match id
            match_id = await find_understat_match_id(u, home_us, away_us, season, kickoff)
            if match_id is None:
                print(f"[SKIP] No match found: {home_us} vs {away_us} ({season}) {kickoff}")
                continue

            # 3) get match players & shots
            players_data = await u.get_match_players(match_id)   # dict with 'h' and 'a' lists
            shots_data = await u.get_match_shots(match_id)       # list of shot events

            # 4) randomly pick a player FROM THIS MATCH (prefer players who actually appeared)
            all_players = []
            for side_key, side_players in players_data.items():  # 'h' and 'a'
                if isinstance(side_players, list):
                    for p in side_players:
                        if isinstance(p, dict):
                            all_players.append(p)

            # only those with minutes > 0 (ensure 'time' is numeric)
            eligible = []
            for p in all_players:
                try:
                    if int(p.get("time") or 0) > 0:
                        eligible.append(p)
                except (ValueError, AttributeError):
                    continue

            if not eligible:
                continue
            chosen = random.choice(eligible)

            player_name = chosen.get("player")
            player_team = chosen.get("team")
            opponent_team = away_us if player_team == home_us else home_us
            player_minutes = int(chosen.get("time", 0)) if chosen.get("time") is not None else None
            player_position = chosen.get("position")
            starter = bool(chosen.get("is_substitute") == "False" or chosen.get("is_substitute") is False)

            shot_aggs = aggregate_player_shots(shots_data, player_name)

            row = PlayerFixtureStats(
                match_id=match_id,
                season=season,
                gw=int(fx["event"]) if pd.notna(fx["event"]) else None,
                kickoff_time=kickoff,

                fpl_home_team_id=int(fx["team_h"]),
                fpl_away_team_id=int(fx["team_a"]),
                fpl_home_team=home_name_fpl,
                fpl_away_team=away_name_fpl,

                understat_home_team=home_us,
                understat_away_team=away_us,

                player_id=int(chosen.get("player_id")) if chosen.get("player_id") else None,
                player_name=player_name,
                player_team=player_team,
                player_position=player_position,
                player_minutes=player_minutes,
                starter=starter,

                opponent_team=opponent_team,

                **shot_aggs
            )
            rows.append(row)

    df = pd.DataFrame([asdict(r) for r in rows])
    if SAVE_CSV:
        df.to_csv(SAVE_CSV, index=False)
    return df


# ---------- Run directly ----------
if __name__ == "__main__":
    df_out = asyncio.run(sample_random_dataset(N_SAMPLES, RANDOM_SEED))
    print(df_out.head())
    if SAVE_CSV:
        print(f"\nSaved {len(df_out)} rows to {SAVE_CSV}")
