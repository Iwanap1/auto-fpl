from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from player import Player
from squad import Squad, SQUAD_REQUIREMENTS, POS_ORDER, MAX_PER_CLUB


def _score(p: Player) -> float:
    """Primary score to pick players by. Default: expected points xP."""
    return float(p.xP) if np.isfinite(p.xP) else float(p.pooling_metric)


def build_initial_squad(pool_df: pd.DataFrame, budget: float = 100.0) -> Squad:
    """
    Greedy, constraint-aware squad builder.
    Assumes pool_df already has Player-compatible columns & xP populated.
    """
    # Convert rows to Player objects
    players: List[Player] = []
    for _, r in pool_df.iterrows():
        players.append(
            Player(
                pid=int(r["element"]),
                name=str(r["name"]),
                pos=str(r["position"]),
                team=str(r["team"]),
                team_id=int(r["team_id"]),
                price=float(r["price"]),
                features=tuple(r["features"]),
                pooling_metric=float(r["metric"]),
                xP=float(r["xP"]),
            )
        )

    # Group by position, sort by score desc then price asc
    by_pos: Dict[str, List[Player]] = {pos: [] for pos in POS_ORDER}
    for p in players:
        by_pos[p.pos].append(p)
    for pos in POS_ORDER:
        by_pos[pos].sort(key=lambda p: (_score(p), -p.price), reverse=True)

    # Greedy with budget & club caps
    chosen: List[Player] = []
    bank = budget
    club_counts: Dict[int, int] = {}

    # Helper: minimal remaining cost lower bound (cheapest available)
    cheapest_by_pos = {
        pos: sorted([p.price for p in by_pos[pos]])[:SQUAD_REQUIREMENTS[pos]]
        for pos in POS_ORDER
    }
    def feasible_with(p: Player, picked: List[Player], bank_left: float, need_map: Dict[str, int]) -> bool:
        # club cap check
        new_cc = club_counts.copy()
        new_cc[p.team_id] = new_cc.get(p.team_id, 0) + 1
        if new_cc[p.team_id] > MAX_PER_CLUB:
            return False
        # budget bound
        # optimistic lower bound: sum of cheapest prices for all remaining slots after taking p
        next_need = need_map.copy()
        next_need[p.pos] -= 1
        lb = 0.0
        for pos in POS_ORDER:
            need = max(next_need[pos], 0)
            cheap_list = cheapest_by_pos[pos]
            if need > len(cheap_list):
                return False  # not enough players for that position
            lb += sum(cheap_list[:need])
        return (bank_left - p.price - lb) >= -1e-9

    need_map = SQUAD_REQUIREMENTS.copy()
    for pos in POS_ORDER:
        for cand in by_pos[pos]:
            if need_map[pos] <= 0:
                break
            if feasible_with(cand, chosen, bank, need_map):
                chosen.append(cand)
                bank -= cand.price
                club_counts[cand.team_id] = club_counts.get(cand.team_id, 0) + 1
                need_map[pos] -= 1
            # keep scanning; if we can’t fit, try next candidate

    if len(chosen) != sum(SQUAD_REQUIREMENTS.values()):
        raise RuntimeError("Failed to build a legal 15-man squad with given pool/budget.")

    return Squad(players=chosen, bank=bank, free_transfers=1)


def suggest_best_single_transfer(squad: Squad, pool_df: pd.DataFrame) -> Tuple[int, Player, float]:
    """
    Naive 1-transfer improvement: try all legal in-position swaps and return the best delta xP.
    Returns: (out_idx, incoming_player, delta_xP). If no improvement, delta_xP <= 0.
    """
    # Build Player objects from pool_df
    pool_players: List[Player] = []
    for _, r in pool_df.iterrows():
        pool_players.append(
            Player(
                pid=int(r["element"]),
                name=str(r["name"]),
                pos=str(r["position"]),
                team=str(r["team"]),
                team_id=int(r["team_id"]),
                price=float(r["price"]),
                features=tuple(r["features"]),
                pooling_metric=float(r["metric"]),
                xP=float(r["xP"]),
            )
        )

    current_xp = sum(p.xP for p in squad.players)
    best: Tuple[int, Player, float] = (-1, None, 0.0)  # type: ignore

    for i, out_p in enumerate(squad.players):
        for cand in pool_players:
            if cand.pos != out_p.pos:
                continue
            if not squad.can_swap(i, cand):
                continue
            delta = cand.xP - out_p.xP
            if delta > best[2]:
                best = (i, cand, delta)

    return best
