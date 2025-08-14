from typing import Dict, List
from player import Player
import numpy as np

POS_ORDER = ["GK", "DEF", "MID", "FWD"]
SQUAD_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3

class Squad:
    """Simple 15-man squad with budget, FT tracking, and constraints.
    Assumes invariant counts per position equal to SQUAD_REQUIREMENTS.
    """
    def __init__(self, players: List[Player], bank: float, free_transfers: int = 1):
        assert len(players) == 15, "Squad must have 15 players"
        self.players: List[Player] = list(players)
        self.bank = float(bank)
        self.free_transfers = int(np.clip(free_transfers, 1, 2))

    # --- helpers ---
    def club_counts(self) -> Dict[int, int]:
        cc: Dict[int, int] = {}
        for p in self.players:
            cc[p.team_id] = cc.get(p.team_id, 0) + 1
        return cc

    def pos_counts(self) -> Dict[str, int]:
        pc = {k: 0 for k in SQUAD_REQUIREMENTS}
        for p in self.players:
            pc[p.pos] += 1
        return pc

    def can_swap(self, out_idx: int, incoming: Player) -> bool:
        if out_idx < 0 or out_idx >= 15:
            return False
        out_p = self.players[out_idx]
        if incoming.pos != out_p.pos:
            return False
        # club limit
        cc = self.club_counts()
        # removing out_p frees one club slot
        cc[out_p.team_id] -= 1
        if cc.get(incoming.team_id, 0) + 1 > MAX_PER_CLUB:
            return False
        # budget check (simplified: selling price = current price)
        cost_delta = incoming.price - out_p.price
        return self.bank - cost_delta >= -1e-9

    def apply_swap(self, out_idx: int, incoming: Player) -> bool:
        if not self.can_swap(out_idx, incoming):
            return False
        out_p = self.players[out_idx]
        self.players[out_idx] = incoming
        self.bank -= (incoming.price - out_p.price)
        return True