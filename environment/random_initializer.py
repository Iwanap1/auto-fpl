import numpy as np
import pandas as pd
from player import Player
from squad import *
from typing import List, Tuple, Dict, Optional

import pandas as pd
from typing import Tuple

class TeamInitializer:
    """Utility to construct a candidate pool from a single GW file and sample a legal XI.
    Designed to play nicely with this env's `Player` and `SquadXI` types.
    """
    def __init__(self, per_pos: int = 20, seed: int = 42):
        self.per_pos = int(per_pos)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _choose_metric_cols(gw: int, df_cols: List[str]) -> Tuple[Optional[str], List[str]]:
        if gw == 1:
            primary = "selected" if "selected" in df_cols else ("selected_by_percent" if "selected_by_percent" in df_cols else None)
            fallbacks = ["selected_by_percent", "transfers_in_event", "transfers_in"]
        else:
            primary = "transfers_in_event" if "transfers_in_event" in df_cols else ("transfers_in" if "transfers_in" in df_cols else None)
            fallbacks = ["transfers_in", "selected", "selected_by_percent"]
        return primary, fallbacks

    @staticmethod
    def get_team_name(fid: int, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, was_home: bool) -> Tuple[str, int]:
        fixture = fixtures_df.loc[fixtures_df["id"].eq(fid)]
        if fixture.empty:
            raise KeyError(f"Fixture id {fid} not found in fixtures_df")
        row = fixture.iloc[0]
        team_id = int(row["team_h"] if was_home else row["team_a"]) if "team_h" in row and "team_a" in row else int(row["team_h"] if was_home else row["team_a"])
        trow = teams_df.loc[teams_df["id"].eq(team_id)]
        team_name = str(trow["name"].iloc[0]) if not trow.empty else ""
        return team_name, team_id

    @classmethod
    def row_to_player(cls, row: pd.Series, fixtures_df: pd.DataFrame, teams_df: pd.DataFrame) -> Player:
        price = float(row["value"]) / 10.0 if "value" in row and pd.notna(row["value"]) else float(row.get("price", 0.0))
        fid = int(row["fixture"]) if "fixture" in row else int(row.get("fixture_id", 0))
        was_home = bool(row.get("was_home", True))
        team, team_id = cls.get_team_name(fid, fixtures_df, teams_df, was_home)
        return Player(
            pid=int(row["element"] if "element" in row else row.get("pid", 0)),
            name=str(row.get("name", "")),
            pos=str(row["position"] if "position" in row else row.get("pos", "")),
            team=team,
            team_id=team_id,
            price=price,
            xP=0,
            metric=float(row["metric"]) if "metric" in row else float(row.get("metric_value", 0.0))
        )

    def build_candidate_pool_from_gw(
        self,
        gw_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        gw: int,
        per_pos: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
        """Build a top-N-per-position pool from a single GW dataframe.
        Returns pool_df and an index_map {pos: [row_indices]} into pool_df.
        Required columns (at minimum): element, name, position, fixture, was_home, value
        """
        df = gw_df.copy()
        required = ["element", "name", "position", "fixture", "was_home", "value"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"GW df missing required columns: {missing}")

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        primary, fallbacks = self._choose_metric_cols(gw, df.columns.tolist())
        if primary is None:
            df["metric"] = 0.0
        else:
            df["metric"] = pd.to_numeric(df[primary], errors="coerce")
            for fb in fallbacks:
                if fb in df.columns:
                    mask = df["metric"].isna() | (df["metric"] == 0)
                    df.loc[mask, "metric"] = pd.to_numeric(df.loc[mask, fb], errors="coerce").fillna(df.loc[mask, "metric"]) 
        df["metric"] = df["metric"].fillna(0.0)

        N = int(per_pos or self.per_pos)
        parts = []
        for pos in ["GK", "DEF", "MID", "FWD"]:
            sub = df.loc[df["position"] == pos].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["metric", "value"], ascending=[False, True]).head(N)
            parts.append(sub)
        if not parts:
            raise ValueError("No candidates found for any position in this GW file.")
        pool_df = pd.concat(parts, axis=0).reset_index(drop=True)

        # Resolve team ids/names for pool rows
        teams, team_ids = [], []
        for _, r in pool_df.iterrows():
            tname, tid = self.get_team_name(int(r["fixture"]), fixtures_df, teams_df, bool(r["was_home"]))
            teams.append(tname); team_ids.append(tid)
        pool_df["team_id"] = team_ids
        pool_df["price"] = pool_df["value"].astype(float) / 10.0

        index_map = {pos: pool_df.index[pool_df["position"] == pos].tolist() for pos in ["GK", "DEF", "MID", "FWD"] if (pool_df["position"] == pos).any()}
        return pool_df, index_map

    def random_initial_xi_from_pool(
        self,
        pool_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        tries: int = 2000,
        bank: float = 100.0,
    ) -> Squad:
        """Sample a legal XI from a candidate pool under budget and 3-per-club.
        Uses a random formation from allowed set and same-position picks.
        """
        # Allowed formations (DEF, MID, FWD)
        formations = [(3,4,3), (3,5,2), (4,4,2), (4,3,3), (5,4,1), (5,3,2), (4,5,1)]
        by_pos = {pos: pool_df.loc[pool_df["position"] == pos].reset_index(drop=True) for pos in ["GK", "DEF", "MID", "FWD"]}
        if any(df.empty for df in by_pos.values()):
            raise RuntimeError("Pool must contain candidates for all positions")

        for _ in range(tries):
            DEFc, MIDc, FWDc = formations[self.rng.integers(len(formations))]
            xi_rows: List[pd.Series] = []
            cc: Dict[int, int] = {}
            spent = 0.0

            # 1 GK
            gk_rows = by_pos["GK"].sample(n=1, random_state=None)
            for _, r in gk_rows.iterrows():
                p = self.row_to_player(r, fixtures_df, teams_df)
                cc[p.team_id] = cc.get(p.team_id, 0) + 1
                xi_rows.append(r); spent += p.price

            def fill(pos: str, n: int) -> bool:
                nonlocal spent
                # random order each try
                shuf = by_pos[pos].sample(frac=1.0, random_state=None)
                cnt = 0
                for _, r in shuf.iterrows():
                    p = self.row_to_player(r, fixtures_df, teams_df)
                    if cc.get(p.team_id, 0) >= 3:  # club limit
                        continue
                    if spent + p.price > bank:     # budget
                        continue
                    cc[p.team_id] = cc.get(p.team_id, 0) + 1
                    xi_rows.append(r); spent += p.price; cnt += 1
                    if cnt == n:
                        return True
                return False

            if not fill("DEF", DEFc):
                continue
            if not fill("MID", MIDc):
                continue
            if not fill("FWD", FWDc):
                continue

            if len(xi_rows) != 11:
                continue

            players: List[Player] = [self.row_to_player(r, fixtures_df, teams_df) for r in xi_rows]
            remaining = bank - spent
            return Squad(players, bank=remaining, free_transfers=2)

        raise RuntimeError("Could not form a legal XI from this pool (check budget or per_pos)")

def random_initial_squad_from_pool(
    self,
    pool_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    tries: int = 2000,
    bank: float = 100.0,
) -> Squad:
    formations = [(3,4,3), (3,5,2), (4,4,2), (4,3,3), (5,4,1), (5,3,2), (4,5,1)]
    by_pos = {pos: pool_df.loc[pool_df["position"] == pos].reset_index(drop=True) for pos in ["GK", "DEF", "MID", "FWD"]}
    if any(df.empty for df in by_pos.values()):
        raise RuntimeError("Pool must contain candidates for all positions")

    for _ in range(tries):
        DEFc, MIDc, FWDc = formations[self.rng.integers(len(formations))]
        chosen_rows: List[pd.Series] = []
        club_counts: Dict[int, int] = {}
        spent = 0.0

        def can_take(p: Player, budget_left: float) -> bool:
            if club_counts.get(p.team_id, 0) >= 3: return False
            if spent + p.price > bank: return False
            return True

        def push(p: Player, r: pd.Series):
            nonlocal spent
            club_counts[p.team_id] = club_counts.get(p.team_id, 0) + 1
            chosen_rows.append(r)
            spent += p.price

        # GK starter
        gk_shuf = by_pos["GK"].sample(frac=1.0)
        took = False
        for _, r in gk_shuf.iterrows():
            p = self.row_to_player(r, fixtures_df, teams_df)
            if can_take(p, bank - spent):
                push(p, r); took = True; break
        if not took: continue

        # DEF/MID/FWD starters
        def fill_starters(pos: str, n: int) -> bool:
            shuf = by_pos[pos].sample(frac=1.0)
            cnt = 0
            for _, r in shuf.iterrows():
                p = self.row_to_player(r, fixtures_df, teams_df)
                if can_take(p, bank - spent):
                    push(p, r); cnt += 1
                    if cnt == n: return True
            return False

        if not fill_starters("DEF", DEFc): continue
        if not fill_starters("MID", MIDc): continue
        if not fill_starters("FWD", FWDc): continue
        if len(chosen_rows) != 11: continue  # safety

        # Bench to reach squad requirements (2/5/5/3)
        need = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        have = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        # Count current starters by pos
        for r in chosen_rows:
            have[str(r["position"])] += 1

        def fill_bench(pos: str, extra: int) -> bool:
            if extra <= 0: return True
            # Prefer cheaper first to help budget
            shuf = by_pos[pos].copy()
            shuf = shuf.sort_values("value")  # cheapest first
            cnt = 0
            # avoid duplicates: skip already chosen pid
            chosen_pids = {int(rr["element"]) for rr in chosen_rows}
            for _, r in shuf.iterrows():
                if int(r["element"]) in chosen_pids: continue
                p = self.row_to_player(r, fixtures_df, teams_df)
                if can_take(p, bank - spent):
                    push(p, r); cnt += 1
                    chosen_pids.add(p.pid)
                    if cnt == extra: return True
            return False

        if not fill_bench("GK",  need["GK"]  - have["GK"]):  continue
        if not fill_bench("DEF", need["DEF"] - have["DEF"]): continue
        if not fill_bench("MID", need["MID"] - have["MID"]): continue
        if not fill_bench("FWD", need["FWD"] - have["FWD"]): continue

        if len(chosen_rows) != 15: continue  # must reach full squad

        players: List[Player] = [self.row_to_player(r, fixtures_df, teams_df) for r in chosen_rows]
        remaining = bank - spent
        # construct your Squad (adjust signature to your actual Squad class)
        s = Squad()
        for p in players: s.add(p)
        s.bank = remaining
        return s

    raise RuntimeError("Could not form a legal squad from this pool (check budget or per_pos)")
