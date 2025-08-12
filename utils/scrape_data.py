import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin

BASE = "https://fbref.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; fbref-scraper/1.0; +https://example.com/bot)"
}

def get(url, retries=3, backoff=1.5):
    for i in range(retries):
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            # polite pause to avoid hammering their servers
            time.sleep(0.6 + random.random() * 0.4)
            return r.text
        time.sleep(backoff ** i)
    r.raise_for_status()

def team_page(team_slug="18bb7c10/Arsenal-Stats"):
    # You can store these slugs for the 20 PL teams in a dict once.
    return urljoin(BASE, f"/en/squads/{team_slug}")

def find_match_links(team_slug, season_label="2024-2025"):
    """
    Return list of {date, opponent, comp, venue, match_url} for a team & season.
    Works from the team 'Scores & Fixtures' table on the club page.
    """
    html = get(team_page(team_slug))
    soup = BeautifulSoup(html, "lxml")

    # Find the Scores & Fixtures table for this season on the team page
    # FBref renders a single table listing upcoming/full season rows with 'Match Report' links
    table = soup.find("table")  # first large fixtures table on the club page is ok for most teams
    rows = table.tbody.find_all("tr")

    out = []
    for tr in rows:
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        # Defensive: some rows are separators
        date = tr.find("td", {"data-stat": "date"})
        opponent = tr.find("td", {"data-stat": "opponent"})
        comp = tr.find("td", {"data-stat": "comp"}) or tr.find("td", {"data-stat": "competition"})
        venue = tr.find("td", {"data-stat": "venue"})
        report = tr.find("a", string=lambda s: s and "Match Report" in s)
        if not (date and opponent and report):
            continue
        out.append({
            "date": date.get_text(strip=True),
            "opponent": opponent.get_text(strip=True),
            "comp": comp.get_text(strip=True) if comp else "",
            "venue": venue.get_text(strip=True) if venue else "",
            "match_url": urljoin(BASE, report.get("href"))
        })
    return out

def read_tables_including_comments(match_html):
    """
    Return list of DataFrames from all tables on a match page,
    successfully reading tables embedded inside HTML comments.
    """
    dfs = []
    # First pass: normal tables
    dfs.extend(pd.read_html(match_html))
    # Second pass: commented tables
    soup = BeautifulSoup(match_html, "lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        txt = c.strip()
        if "<table" in txt:
            try:
                dfs.extend(pd.read_html(txt))
            except ValueError:
                pass
    return dfs

def get_match_shots_table(match_url):
    """
    Return a tidy 'shots by player' dataframe for both teams from an FBref match page.
    """
    html = get(match_url)
    soup = BeautifulSoup(html, "lxml")

    # Try to directly find a shots table by id or caption text
    # Fallback: read all tables and filter by expected columns
    candidates = []
    for df in read_tables_including_comments(html):
        cols = [c.lower() for c in df.columns.map(str)]
        if {"sh", "player"} <= set(col[:2] for col in cols) or "shots" in " ".join(cols):
            candidates.append(df)

    if not candidates:
        # heuristic: look for tables labelled "Shooting"
        for df in read_tables_including_comments(html):
            if any("shoot" in str(c).lower() for c in df.columns):
                candidates.append(df)

    if not candidates:
        raise RuntimeError("Could not find a shots table on the page.")

    # Clean the first plausible table
    shots = candidates[0].copy()
    # Common FBref tidy-ups
    shots = shots.rename(columns=lambda x: str(x).strip())
    # Remove header subrows that get parsed as data
    shots = shots[~shots['Player'].astype(str).str.contains("Player", na=False)]
    # Keep core columns if present
    keep = [c for c in ["Player", "Sh", "SoT", "Dist", "xG", "xG Assisted", "Notes", "Team"] if c in shots.columns]
    if keep:
        shots = shots[keep]
    return shots

def avg_player_shots_vs_opponent(team_slug, season_label, player_name, opponent_name):
    """
    Walk season fixtures, open each match report, extract shots per player,
    and compute the player's average shots vs an opponent.
    """
    matches = find_match_links(team_slug, season_label=season_label)
    # filter this season + opponent
    target = [m for m in matches if opponent_name.lower() in m["opponent"].lower()]
    if not target:
        return 0.0, []

    per_match = []
    for m in target:
        shots = get_match_shots_table(m["match_url"])
        # FBref often duplicates player names across sides; filter by exact player
        sub = shots[shots["Player"].str.lower() == player_name.lower()]
        if not sub.empty and "Sh" in sub.columns:
            # sum shots across entries (rare but safe)
            per_match.append(int(pd.to_numeric(sub["Sh"], errors="coerce").fillna(0).sum()))

    avg = float(sum(per_match)) / len(per_match) if per_match else 0.0
    return avg, per_match
