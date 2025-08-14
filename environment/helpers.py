from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from load_csv import load_csv
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from player import Player
from squad import *

# --- helpers you suggested + robust wrappers ---

def get_team_name(fid: int, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, was_home: bool) -> Tuple[str, int]:
    fixture = fixtures_df.loc[fixtures_df["id"].eq(fid)]
    if fixture.empty:
        # Some seasons use "id" for fixture id; others "code" or "event"—adjust here if needed
        raise KeyError(f"Fixture id {fid} not found in fixtures_df")
    row = fixture.iloc[0]
    # Most vaastav seasons: columns team_h/team_a (ids), plus optionally team_h_name/team_a_name
    team_id = int(row["team_h"] if was_home else row["team_a"])
    # Get name from teams_df
    trow = teams_df.loc[teams_df["id"] == team_id]
    team_name = str(trow["name"].iloc[0]) if not trow.empty else ""
    return team_name, team_id


def row_to_player(row: pd.Series, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, gw: int) -> Player:
    # price: vaastav 'value' is tenths of a million
    price = float(row["value"]) / 10.0 if "value" in row and pd.notna(row["value"]) else float(row.get("price", 0.0))
    # team resolution via fixture+home/away
    fid = int(row["fixture"])
    was_home = bool(row["was_home"])
    team_name, team_id = get_team_name(fid, fixtures_df, teams_df, was_home)
    metric = float(row.get("metric", 0.0))
    return Player(
        pid=int(row["element"]),
        name=str(row["name"]),
        pos=str(row["position"]),
        team=team_name,
        team_id=team_id,
        price=price,
        metric=metric,
    )

# --- pool builder (GW file only) ---
def _choose_metric_cols(gw: int, df_cols: List[str]) -> Tuple[str, List[str]]:
    if gw == 1:
        primary = "selected" if "selected" in df_cols else ("selected_by_percent" if "selected_by_percent" in df_cols else None)
        fallbacks = ["selected_by_percent", "transfers_in_event", "transfers_in"]
    else:
        primary = "transfers_in_event" if "transfers_in_event" in df_cols else ("transfers_in" if "transfers_in" in df_cols else None)
        fallbacks = ["transfers_in", "selected", "selected_by_percent"]
    return primary, fallbacks

def build_candidate_pool_from_gw(
    gw_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    gw: int,
    per_pos: int = 20
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """
    Build top-N per position from a single GW df.
    Resolves team via fixtures_df + teams_df (no season master).
    """
    df = gw_df.copy()

    # Ensure required basics exist
    required = ["element", "name", "position", "fixture", "was_home", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GW df missing required columns: {missing}")

    # numeric coercions
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    metric_primary, metric_fallbacks = _choose_metric_cols(gw, df.columns.tolist())
    # Create 'metric' column
    if metric_primary is None:
        df["metric"] = 0.0
    else:
        df["metric"] = pd.to_numeric(df[metric_primary], errors="coerce").fillna(0.0)
        for fb in metric_fallbacks:
            if df["metric"].notna().sum() >= per_pos and fb in df.columns:
                # optional: boost missing with fallback
                mask = df["metric"].isna() | (df["metric"] == 0)
                df.loc[mask, "metric"] = pd.to_numeric(df.loc[mask, fb], errors="coerce").fillna(df.loc[mask, "metric"])

    # Top-N per position
    parts = []
    for pos in POS_ORDER:
        sub = df.loc[df["position"] == pos].copy()
        if sub.empty:
            continue
        # Sort by metric (desc), then value ascending to slightly bias toward affordability
        sub = sub.sort_values(["metric", "value"], ascending=[False, True]).head(per_pos)
        parts.append(sub)
    if not parts:
        raise ValueError("No candidates found for any position in this GW file.")
    pool_df = pd.concat(parts, axis=0).reset_index(drop=True)

    # Resolve team fields for each row (vectorize by applying once; this is fine for ≤80 rows)
    teams = []
    team_ids = []
    for _, r in pool_df.iterrows():
        tname, tid = get_team_name(int(r["fixture"]), fixtures_df, teams_df, bool(r["was_home"]))
        teams.append(tname); team_ids.append(tid)
    pool_df["team"] = teams
    pool_df["team_id"] = team_ids
    pool_df["price"] = pool_df["value"] / 10.0

    # Index map by position (into pool_df)
    index_map = {pos: pool_df.index[pool_df["position"] == pos].tolist()
                 for pos in POS_ORDER if (pool_df["position"] == pos).any()}
    return pool_df, index_map

# --- random squad from pool ---

def random_initial_squad_from_pool(
    pool_df: pd.DataFrame, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, rng: np.random.Generator, tries: int = 2000
) -> Squad:
    by_pos = {pos: pool_df.loc[pool_df["position"] == pos].reset_index(drop=True) for pos in POS_ORDER}
    for _ in range(tries):
        s = Squad(bank=START_BANK)
        ok = True
        for pos, need in SQUAD_REQUIREMENTS.items():
            if by_pos[pos].empty:
                ok = False; break
            for idx in rng.permutation(len(by_pos[pos])):
                if s.pos_counts[pos] == need:
                    break
                p = row_to_player(by_pos[pos].iloc[idx], fixtures_df, teams_df)
                s.add(p)
            if s.pos_counts[pos] < need:
                ok = False; break
        if ok and s.is_complete():
            return s
    raise RuntimeError("Could not form a legal squad from this pool (check budget or per_pos).")


def can_apply_transfer(
    squad: Squad,
    out_pid: int,
    incoming: Player,
    *,
    max_per_club: int = MAX_PER_CLUB
) -> tuple[bool, str]:
    # locate outgoing
    out_idx = next((i for i, p in enumerate(squad.players) if p.pid == out_pid), None)
    if out_idx is None:
        return False, f"out_pid {out_pid} not in squad"

    outgoing = squad.players[out_idx]

    # same-position constraint keeps 2/5/5/3 intact
    if incoming.pos != outgoing.pos:
        return False, f"pos mismatch: {incoming.pos} != {outgoing.pos}"

    # no duplicates
    if any(p.pid == incoming.pid for p in squad.players):
        return False, f"incoming {incoming.pid} already in squad"

    # compute post-swap bank
    new_bank = squad.bank + outgoing.price - incoming.price
    if new_bank < 0:
        return False, "insufficient bank for swap"

    # club limit after swap
    # simulate club counts
    out_team = outgoing.team_id
    in_team = incoming.team_id
    club_counts = dict(squad.club_counts)

    # decrement outgoing team
    club_counts[out_team] = club_counts.get(out_team, 0) - 1
    if club_counts[out_team] <= 0:
        club_counts.pop(out_team, None)

    # increment incoming team
    club_counts[in_team] = club_counts.get(in_team, 0) + 1
    if club_counts[in_team] > max_per_club:
        return False, f"club limit exceeded for team_id {in_team}"

    # position counts unchanged (same-pos swap), no need to check SQUAD_REQUIREMENTS
    return True, "ok"



def apply_transfer_same_pos(squad: Squad, out_pid: int, incoming: Player) -> tuple[bool, str]:
    ok, reason = can_apply_transfer(squad, out_pid, incoming)
    if not ok:
        return False, reason

    # remove outgoing
    out_idx = next(i for i, p in enumerate(squad.players) if p.pid == out_pid)
    outgoing = squad.players[out_idx]

    # mutate squad
    # update bank
    squad.bank += outgoing.price - incoming.price

    # update club counts
    out_team = outgoing.team_id
    in_team = incoming.team_id

    squad.club_counts[out_team] = squad.club_counts.get(out_team, 0) - 1
    if squad.club_counts[out_team] <= 0:
        squad.club_counts.pop(out_team, None)

    squad.club_counts[in_team] = squad.club_counts.get(in_team, 0) + 1

    # position counts unchanged (same-pos swap)

    # replace player in list to keep order stable (helps with debugging)
    squad.players[out_idx] = incoming
    return True, "ok"


def apply_transfers(
    squad: Squad,
    transfers: List[Tuple[int, Player]],   # [(out_pid, incoming), ...]
    free_transfers: int
) -> tuple[bool, str]:
    """
    Apply up to 3 same-position swaps (pairwise), bounded by free_transfers.
    All validation is done on a temporary copy; the real squad is mutated only on success.
    """
    # caps
    if len(transfers) > 3:
        return False, "cannot make more than 3 transfers"
    if len(transfers) > free_transfers:
        return False, f"only {free_transfers} free transfers available"

    # quick shape checks
    if len(transfers) == 0:
        return True, "no transfers requested"

    # simulate on a copy
    tmp = deepcopy(squad)

    # Optional pre-check: total affordability with bank
    freed_cash = 0.0
    need_cash = 0.0
    out_players = []
    for out_pid, inc in transfers:
        op = next((p for p in tmp.players if p.pid == out_pid), None)
        if op is None:
            return False, f"out_pid {out_pid} not found in squad"
        out_players.append(op)
        freed_cash += op.price
        need_cash  += inc.price
    if tmp.bank + freed_cash - need_cash < 0:
        return False, "insufficient bank for batch (after swaps)"

    # apply pairwise same-position swaps in order
    for out_pid, incoming in transfers:
        ok, reason = apply_transfer_same_pos(tmp, out_pid, incoming)
        if not ok:
            return False, f"failed on swap ({out_pid} -> {incoming.pid}): {reason}"

    # success -> commit temp state to real squad
    squad.players = tmp.players
    squad.bank = tmp.bank
    squad.pos_counts = dict(tmp.pos_counts)
    squad.club_counts = dict(tmp.club_counts)
    return True, "ok"