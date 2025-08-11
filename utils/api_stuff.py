import requests
from datetime import datetime, timezone

ENDPOINTS = {
    'fixtures': "https://fantasy.premierleague.com/api/fixtures/",
    'static': "https://fantasy.premierleague.com/api/bootstrap-static/",
    'login': 'https://users.premierleague.com/accounts/login/'
}

POSITIONS = {
        1: "goalkeeper",
        2: "defender",
        3: "midfielder",
        4: "forward"
    }


def login_to_fpl_session(email, password):
    """
    Log in to the Fantasy Premier League session. Needed to submit a transfer or check current team.
    
    Args:
        email (str): User's email address
        password (str): User's password
    
    Returns:
        requests.Session: A session object with the logged-in state
    """
    session = requests.Session()
    payload = {
        'password': password,
        'login': email,
        'redirect_uri': 'https://fantasy.premierleague.com/a/login',
        'app': 'plfpl-web'
    }
    
    response = session.post(ENDPOINTS['login'], data=payload)
    if response.status_code != 200:
        raise Exception("Login failed. Please check your credentials.")
    
    return session


def next_gw():
    """
    Get the next game week number.
    
    Returns:
        int: The next game week number
    """
    response = requests.get(ENDPOINTS['bootstrap_static'])
    data = response.json()
    for gw in data['events']:
        if gw['is_next']:
            return gw['id']


def get_current_team(email, password, manager_id):

    """
    Get the current team sorted by position
    
    Returns:
        dict: A dictionary of the IDs of each player in each position as of the most recent GW: {
            "goalkeepers": [int, int],
            "defenders": [int, int, int, int, int],
            "midfielders": [int, int, int, int, int],
            "forwards": [int, int, int]
            }
    """
    session = login_to_fpl_session(email, password)
    url = f"https://fantasy.premierleague.com/api/my-team/{manager_id}/"
    response = session.get(url)
    if response.status_code != 200:
        raise Exception("Failed to retrieve current team data.")
    data = response.json()
    team = {
        "goalkeepers": [],
        "defenders": [],
        "midfielders": [],
        "forwards": []
    }
    for player in data['picks']:
        position = POSITIONS[player['element_type']]
        team[position + 's'].append(player['element'])
    return team


def check_injury(player_id):
    """
    Given a player's element ID, returns injury/availability info.
    Output: dict with keys 'status', 'news', 'news_added'.
    Example: {
    'status_code': 'i',
    'status': 'Injured',
    'news': 'Hamstring injury - Expected back 24 Aug',
    'news_added': '2025-08-08T10:32:00Z'
}
    """
    r = requests.get(ENDPOINTS["bootstrap_static"], timeout=15)
    r.raise_for_status()
    data = r.json()
    elements = data.get("elements", [])
    
    player = next((p for p in elements if p["id"] == player_id), None)
    if not player:
        raise ValueError(f"No player found with id={player_id}")
    
    status_map = {
        "a": "Available",
        "i": "Injured",
        "d": "Doubtful",
        "s": "Suspended",
        "n": "Not in squad",
        "u": "Unavailable",
    }
    
    return {
        "status_code": player["status"],
        "status": status_map.get(player["status"], "Unknown"),
        "news": player.get("news", ""),
        "news_added": player.get("news_added", None)
    }


def fixture_list_by_gw(team, n_gameweeks):
    """
    Get fixtures for the next N gameweeks for a given team.
    Excludes postponed and TBD fixtures entirely.
    Includes all matches in a DGW, blanks will have 0 fixtures.

    Args:
        team (str|int): Team name ('Arsenal') or ID (1–20)
        n_gameweeks (int): Number of upcoming GWs

    Returns:
        list[dict]: [
          {'event': 3, 'deadline_utc': '...', 'fixtures': [...], 'is_double': False},
          ...
        ]
    """
    bt = requests.get(ENDPOINTS['static'], timeout=15).json()
    teams = bt["teams"]
    events = bt["events"]

    # Resolve team_id
    if isinstance(team, str):
        t = next((x for x in teams if x["name"].lower() == team.lower()), None)
        if not t:
            raise ValueError(f"Team '{team}' not found.")
        team_id = t["id"]
    else:
        team_id = int(team)

    # Target events
    start_event = next_gw()
    start_id = start_event["id"]
    target_events = [e for e in events if e["id"] >= start_id][:n_gameweeks]
    target_ids = {e["id"] for e in target_events}

    # Fixtures
    fx = requests.get(ENDPOINTS["fixtures"], timeout=15).json()
    id_to_name = {t["id"]: t["name"] for t in teams}

    # Prepare output buckets
    out_by_event = {
        e["id"]: {
            "event": e["id"],
            "deadline_utc": e.get("deadline_time"),
            "fixtures": []
        }
        for e in target_events
    }

    for f in fx:
        if f.get("postponed", False):
            continue  # skip postponed
        if f.get("kickoff_time") is None:
            continue  # skip TBD
        if f.get("event") not in target_ids:
            continue
        if not (f["team_h"] == team_id or f["team_a"] == team_id):
            continue

        home = (f["team_h"] == team_id)
        opp_id = f["team_a"] if home else f["team_h"]

        out_by_event[f["event"]]["fixtures"].append({
            "opponent": id_to_name[opp_id],
            "home": home,
            "kickoff_time": f["kickoff_time"],
            "status": "Scheduled"
        })

    # Final ordering & double flags
    ordered = []
    for e in target_events:
        rec = out_by_event[e["id"]]
        rec["fixtures"].sort(key=lambda x: x["kickoff_time"])
        rec["is_double"] = len(rec["fixtures"]) > 1
        ordered.append(rec)

    return ordered
