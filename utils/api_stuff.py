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


def is_injured():
    """
    Check if the player is injured.
    
    Returns:
        bool: True if the player is injured, False otherwise.
    """
    # Placeholder for actual injury check logic
    return False


def fixture_list(team, n_fixtures):
    """
    Get a list of upcoming fixtures for a given team.
    
    Args:
        n_fixtures (int): Number of upcoming fixtures to retrieve
    
    Returns:
        list: A list of n upcoming fixtures
    """

    return [f"Fixture {i + 1}" for i in range(n_fixtures)]