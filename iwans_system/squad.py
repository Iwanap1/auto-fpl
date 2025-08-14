from typing import Dict, List
from player import Player
import numpy as np

POS_ORDER = ["GK", "DEF", "MID", "FWD"]
SQUAD_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3

class Squad:
    def __init__(self, players: List[Player], bank: float, free_transfers: int = 1):
        assert len(players) == 15, "Squad must have 15 players"
        self.players: List[Player] = list(players)
        self.bank = float(bank)
        self.free_transfers = int(np.clip(free_transfers, 1, 2))

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
        # must match position
        if incoming.pos != out_p.pos:
            return False
        # NEW: forbid duplicates (except replacing same pid)
        if incoming.pid != out_p.pid and any(
            p.pid == incoming.pid for i, p in enumerate(self.players) if i != out_idx
        ):
            return False

        # club limit
        cc = self.club_counts()
        cc[out_p.team_id] -= 1  # freeing one slot from the outgoing player's club
        if cc.get(incoming.team_id, 0) + 1 > MAX_PER_CLUB:
            return False

        # budget (sell price == current price for simplicity)
        cost_delta = incoming.price - out_p.price
        return self.bank - cost_delta >= -1e-9

    def apply_swap(self, out_idx: int, incoming: Player) -> bool:
        if not self.can_swap(out_idx, incoming):
            return False
        out_p = self.players[out_idx]
        self.players[out_idx] = incoming
        self.bank -= (incoming.price - out_p.price)
        return True
