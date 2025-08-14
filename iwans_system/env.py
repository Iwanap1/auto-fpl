import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Tuple, Callable, Any
import random
import copy
from player import Player
from squad import Squad, POS_ORDER, SQUAD_REQUIREMENTS, MAX_PER_CLUB

def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mu, sd = float(np.nanmean(x)), float(np.nanstd(x) + 1e-8)
    z = (x - mu) / sd
    z = z / max(temp, 1e-6)
    z -= np.nanmax(z)
    e = np.exp(z)
    return e / (np.nansum(e) + 1e-12)

def _p_play(p: Player) -> float:
    """
    features[4] is a discrete probability in {0, 0.25, 0.5, 0.75, 1}.
    We snap to the nearest allowed value to be robust to any float noise.
    """
    try:
        v = float(p.features[4])
    except Exception:
        return 1.0  # safe default: assume they play

    if not np.isfinite(v):
        return 1.0

    v = float(np.clip(v, 0.0, 1.0))
    allowed = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    return float(allowed[np.argmin(np.abs(allowed - v))])



class FPLEnv(gym.Env):
    """
    FPL multi-GW RL environment.

    Episode flow:
      - reset(): pick a RANDOM season from `seasons`; load season context ONCE via `load_season_fn`.
                 detect total GWs and set episode length to full season (e.g., 38).
                 build semi-random initial 15-man squad from GW `start_gw`.
      - step():  per GW, agent chooses transfer or skip; apply free transfer banking / hits;
                 auto-pick best XI + captain + vice; expected autosubs; reward is Δ expected GW points (+ hit).
                 advance to next GW; refresh squad players' xP/features from the preloaded season context.

    Args:
      load_season_fn: (base_dir, season) -> season_ctx, total_gws
      load_gw_fn:     (season_ctx, gw)   -> (pool_players, index_map)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        load_season_fn: Callable[[str, str], Tuple[Any, int]],
        load_gw_fn: Callable[[Any, int], Tuple[List[Player], Dict[str, List[int]]]],
        seasons: List[str],
        base_dir: str,
        models: List,
        start_gw: int = 1,
        budget: float = 100.0,
        illegal_penalty: float = -0.2,
        temperature: float = 1.0,
        transfer_hit: float = -4.0,
        max_free_transfers: int = 5
    ):
        super().__init__()
        self.seasons = list(seasons)
        self.base_dir = base_dir
        self.load_season_fn = load_season_fn
        self.load_gw_fn = load_gw_fn
        self.models = models
        self.start_gw = int(start_gw)
        self.budget = float(budget)
        self.illegal_penalty = float(illegal_penalty)
        self.temperature = float(temperature)
        self.transfer_hit = float(transfer_hit)
        self.max_free_transfers = int(max_free_transfers)

        # Will be set at reset()
        self.season: str = ""
        self.season_ctx: Any = None
        self.total_gws: int = 38
        self.current_gw: int = self.start_gw
        self.step_in_episode: int = 0

        # Pool for current GW
        self.pool: List[Player] = []
        self.index_map: Dict[str, List[int]] = {}
        self.pool_size: int = 0

        # Squad & transfers
        self.squad: Squad = None  # type: ignore
        self.free_transfers: int = 1

        self.action_space = spaces.MultiDiscrete([
            16,                    # out1: 0..14 are squad slots, 15 = "no transfer"
            self.max_pool_size+1,  # in1:  0..max_pool_size-1, max_pool_size = "no transfer"
            16,                    # out2: 0..14, 15 = "no second transfer"
            self.max_pool_size+1,  # in2:  sentinel as above
            15,                    # cap:  0..14
            15,                    # vc:   0..14
        ])

        # Observation: 15 × (price, xP, 5 features) + bank + free transfers
        self.obs_dim = 15 * 9 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_dim,), dtype=np.float32)

        # Cache for XI/captain/VC/score
        self._xi_idx: List[int] = []
        self._captain_idx: int = -1
        self._vice_idx: int = -1
        self._team_score_exp: float = 0.0

    # ===== Helpers =====
    def _pvec(self, p: Player, is_c=False, is_v=False) -> np.ndarray:
        return np.array([p.price, p.xP, *p.features, float(is_c), float(is_v)], dtype=np.float32)

    def _gw_points_maps_from_df(self, gw_df):
        """Build maps for realized scoring from a GW dataframe."""
        import numpy as np
        pid = gw_df["element"].astype(int)
        pts = gw_df.get("total_points", 0.0)
        mins = gw_df.get("minutes", 0.0)
        pts = np.asarray(pts, dtype=float)
        mins = np.asarray(mins, dtype=float)

        points_map = {int(e): float(p) for e, p in zip(pid, pts)}
        played_map = {int(e): bool(m > 0.0) for e, m in zip(pid, mins)}
        return points_map, played_map

    def _realized_team_score(self, players, xi_idx, cap_idx, vc_idx, points_map, played_map) -> float:
        """Compute actual points with captain/VC + autosubs using real results."""
        # Base points from XI (whoever actually played)
        base = sum(points_map.get(players[i].pid, 0.0) for i in xi_idx)

        # Captain/VC: if C didn't play, VC gets the armband (double)
        cap_played = played_map.get(players[cap_idx].pid, False)
        vc_played  = played_map.get(players[vc_idx].pid, False)
        if cap_played:
            armband = points_map.get(players[cap_idx].pid, 0.0)  # extra points to double cap
        else:
            # C blanked: give VC the double if he played (extra = pts(vc))
            armband = points_map.get(players[vc_idx].pid, 0.0) if vc_played else 0.0

        # Autosubs: replace non-playing starters with best bench of SAME POSITION that played
        bench_idx = [i for i in range(len(players)) if i not in xi_idx]
        # rank bench same-position by their actual points (desc)
        bench_by_pos = {pos: [] for pos in POS_ORDER}
        for i in bench_idx:
            bench_by_pos[players[i].pos].append(i)
        for pos in POS_ORDER:
            bench_by_pos[pos].sort(key=lambda i: points_map.get(players[i].pid, 0.0), reverse=True)

        used = set()
        autosub_gain = 0.0
        needs = []
        for i in xi_idx:
            if played_map.get(players[i].pid, False):
                continue  # starter played → no sub needed
            pos = players[i].pos
            # best bench of same pos who actually played and not yet used
            candidates = [j for j in bench_by_pos[pos]
                        if j not in used and played_map.get(players[j].pid, False)]
            if candidates:
                best_bench = candidates[0]
                gain = points_map.get(players[best_bench].pid, 0.0)
                needs.append((gain, best_bench))
        # If multiple starters missed, greedily assign the largest bench gains first
        needs.sort(reverse=True, key=lambda t: t[0])
        for gain, b in needs:
            if b in used:
                continue
            autosub_gain += gain
            used.add(b)

        return base + armband + autosub_gain

    def _best_xi(self) -> List[int]:
        players = self.squad.players
        pos_lists = {pos: sorted([(i, p) for i, p in enumerate(players) if p.pos == pos],
                                key=lambda t: t[1].xP, reverse=True)
                    for pos in POS_ORDER}
        best_xi, best_score = [], -1e9
        for d_cnt, m_cnt, f_cnt in self._valid_formations():
            if not pos_lists["GK"] or len(pos_lists["DEF"]) < d_cnt or len(pos_lists["MID"]) < m_cnt or len(pos_lists["FWD"]) < f_cnt:
                continue
            xi_idx = [pos_lists["GK"][0][0]] + \
                    [i for i, _ in pos_lists["DEF"][:d_cnt]] + \
                    [i for i, _ in pos_lists["MID"][:m_cnt]] + \
                    [i for i, _ in pos_lists["FWD"][:f_cnt]]
            score = sum(players[i].xP for i in xi_idx)
            if score > best_score:
                best_xi, best_score = xi_idx, score
        if not best_xi:
            best_xi = sorted(range(len(players)), key=lambda i: players[i].xP, reverse=True)[:11]
        return best_xi


    def _obs(self) -> np.ndarray:
        vecs = [self._pvec(p, is_c=(i == self._captain_idx), is_v=(i == self._vice_idx))
                for i, p in enumerate(self.squad.players)]
        flat = np.concatenate(vecs, axis=0)
        extras = np.array([self.squad.bank, float(self.free_transfers)], dtype=np.float32)
        return np.concatenate([flat, extras], axis=0)


    def _semi_random_initial_squad(self) -> Squad:
        rng = np.random.default_rng()

        cheapest_by_pos = {
            pos: sorted([self.pool[i].price for i in self.index_map[pos]]) for pos in POS_ORDER
        }
        if any(len(v) < SQUAD_REQUIREMENTS[pos] for pos, v in cheapest_by_pos.items()):
            raise RuntimeError("Pool too small for at least one position")

        chosen: List[Player]
        chosen_pids: set  # <-- NEW
        for attempt in range(50):
            chosen = []
            chosen_pids = set()  # <-- NEW
            bank = self.budget
            club_counts: Dict[int, int] = {}

            try:
                for pos in POS_ORDER:
                    need = SQUAD_REQUIREMENTS[pos]
                    # filter out already chosen PIDs for this position sampling  <-- NEW
                    cand = [self.pool[i] for i in self.index_map[pos] if self.pool[i].pid not in chosen_pids]
                    if len(cand) < need:
                        raise RuntimeError("retry")

                    metrics = np.array([c.pooling_metric for c in cand], dtype=np.float32)
                    probs = _softmax(metrics, self.temperature)

                    picks: List[Player] = []
                    tries = 0
                    while len(picks) < need and tries < 5000:
                        tries += 1
                        j = int(rng.choice(len(cand), p=probs))
                        p = cand[j]
                        if p.pid in chosen_pids:
                            continue  # <-- NEW safety
                        if club_counts.get(p.team_id, 0) + 1 > MAX_PER_CLUB:
                            continue

                        # budget lower-bound guard (same as before)
                        req_remaining = {}
                        for pos2 in POS_ORDER:
                            have = sum(1 for q in (chosen + picks) if q.pos == pos2)
                            req_remaining[pos2] = SQUAD_REQUIREMENTS[pos2] - have - (1 if pos2 == pos else 0)

                        min_future = 0.0
                        for pos2 in POS_ORDER:
                            r = req_remaining[pos2]
                            if r > 0:
                                min_future += float(np.sum(cheapest_by_pos[pos2][:r]))

                        if bank - p.price < -1e-9:
                            continue
                        if (bank - p.price) < (min_future - 1e-9):
                            continue

                        # accept
                        picks.append(p)
                        chosen.append(p)
                        chosen_pids.add(p.pid)  # <-- NEW
                        bank -= p.price
                        club_counts[p.team_id] = club_counts.get(p.team_id, 0) + 1

                    if len(picks) < need:
                        # fallback fill with cheapest legal not-yet-chosen  <-- NEW
                        remaining = [c for c in cand if c.pid not in chosen_pids and club_counts.get(c.team_id,0) < MAX_PER_CLUB]
                        remaining.sort(key=lambda x: (x.price, -x.xP))
                        for p in remaining:
                            if len(picks) >= need:
                                break
                            # repeat budget guard
                            req_remaining = {}
                            for pos2 in POS_ORDER:
                                have = sum(1 for q in (chosen + picks) if q.pos == pos2)
                                req_remaining[pos2] = SQUAD_REQUIREMENTS[pos2] - have - (1 if pos2 == pos else 0)
                            min_future = 0.0
                            for pos2 in POS_ORDER:
                                r = req_remaining[pos2]
                                if r > 0:
                                    min_future += float(np.sum(cheapest_by_pos[pos2][:r]))
                            if bank - p.price >= min_future - 1e-9:
                                picks.append(p)
                                chosen.append(p)
                                chosen_pids.add(p.pid)  # <-- NEW
                                bank -= p.price
                                club_counts[p.team_id] = club_counts.get(p.team_id, 0) + 1

                    if len(picks) < need:
                        raise RuntimeError("retry")

                # success
                # sanity: ensure no duplicate pids
                assert len({p.pid for p in chosen}) == len(chosen), "duplicate PIDs in chosen"
                return Squad(players=chosen, bank=bank, free_transfers=1)

            except RuntimeError:
                continue  # try again

        raise RuntimeError("Initial sampling failed (after multiple attempts). Consider adding cheaper candidates or more budget.")


    @staticmethod
    def _valid_formations() -> List[Tuple[int, int, int]]:
        return [(d, m, f) for d in range(3, 6) for m in range(2, 6) for f in range(1, 4) if d + m + f == 10]

    def _best_xi_and_band(self) -> Tuple[List[int], int, int]:
        players = self.squad.players
        pos_lists = {pos: sorted([(i, p) for i, p in enumerate(players) if p.pos == pos],
                                 key=lambda t: t[1].xP, reverse=True)
                     for pos in POS_ORDER}

        best_xi, best_score = [], -1e9
        for d_cnt, m_cnt, f_cnt in self._valid_formations():
            if not pos_lists["GK"] or len(pos_lists["DEF"]) < d_cnt or len(pos_lists["MID"]) < m_cnt or len(pos_lists["FWD"]) < f_cnt:
                continue
            xi_idx = [pos_lists["GK"][0][0]] + \
                     [i for i, _ in pos_lists["DEF"][:d_cnt]] + \
                     [i for i, _ in pos_lists["MID"][:m_cnt]] + \
                     [i for i, _ in pos_lists["FWD"][:f_cnt]]
            score = sum(players[i].xP for i in xi_idx)
            if score > best_score:
                best_xi, best_score = xi_idx, score

        if not best_xi:
            best_xi = sorted(range(len(players)), key=lambda i: players[i].xP, reverse=True)[:11]

        xi_sorted = sorted(best_xi, key=lambda i: players[i].xP, reverse=True)
        cap_idx = xi_sorted[0]
        vc_idx = xi_sorted[1] if len(xi_sorted) > 1 else xi_sorted[0]
        return best_xi, cap_idx, vc_idx

    def _expected_team_score(self, xi_idx: List[int], cap_idx: int, vc_idx: int) -> float:
        players = self.squad.players
        base = sum(players[i].xP for i in xi_idx)
        pC = _p_play(players[cap_idx])
        extra_armband = players[cap_idx].xP + (1.0 - pC) * players[vc_idx].xP

        bench_idx = [i for i in range(len(players)) if i not in xi_idx]
        bench_by_pos = {pos: [] for pos in POS_ORDER}
        for i in bench_idx:
            bench_by_pos[players[i].pos].append(i)
        for pos in POS_ORDER:
            bench_by_pos[pos].sort(key=lambda i: players[i].xP, reverse=True)

        autosub_gain, used_bench = 0.0, set()
        starter_slots = []
        for i in xi_idx:
            pos = players[i].pos
            bench_list = [j for j in bench_by_pos[pos] if j not in used_bench]
            if not bench_list:
                continue
            best_bench = bench_list[0]
            need = (1.0 - _p_play(players[i])) * players[best_bench].xP
            starter_slots.append((need, best_bench))
        starter_slots.sort(reverse=True, key=lambda t: t[0])

        for need, b in starter_slots:
            if b in used_bench:
                continue
            autosub_gain += need
            used_bench.add(b)

        return base + extra_armband + autosub_gain

    # ===== Gym API =====

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.season = random.choice(self.seasons)
        self.season_ctx, self.total_gws = self.load_season_fn(self.base_dir, self.season)
        if ("predictor" not in self.season_ctx) and (self.models is not None):
            from xp import ScorePredictor
            players_raw = self.season_ctx.get("players_raw")
            teams      = self.season_ctx.get("teams")
            fixtures   = self.season_ctx.get("fixtures")
            if players_raw is not None and teams is not None and fixtures is not None:
                self.season_ctx["predictor"] = ScorePredictor(
                    season=self.season,
                    players_raw_df=players_raw,
                    teams_raw_df=teams,
                    fixtures_raw_df=fixtures,
                    models=self.models
                )
        self.current_gw = 1
        self.episode_gws = int(self.total_gws - 1)
        self.step_in_episode = 0
        self.free_transfers = 1
        result_gw1 = self.load_gw_fn(self.season_ctx, 1)
        if isinstance(result_gw1, tuple) and len(result_gw1) == 3:
            pool_gw1, index_map_gw1, gw1_df = result_gw1
            self._points_map, self._played_map = self._gw_points_maps_from_df(gw1_df)
        else:
            pool_gw1, index_map_gw1 = result_gw1
            self._points_map, self._played_map = {}, {}

        self.pool = pool_gw1
        self.index_map = index_map_gw1
        self.pool_size = len(self.pool)
        # self.action_space = spaces.MultiDiscrete([15 + 1, max(1, self.pool_size) + 1, 15, 15])

        self.squad = self._semi_random_initial_squad()
        self._xi_idx = self._best_xi()
        xi_sorted = sorted(self._xi_idx, key=lambda i: self.squad.players[i].xP, reverse=True)
        self._captain_idx = xi_sorted[0]
        self._vice_idx    = xi_sorted[1] if len(xi_sorted) > 1 else xi_sorted[0]
        self._team_score_exp = self._expected_team_score(self._xi_idx, self._captain_idx, self._vice_idx)
        self.current_gw = 2
        result_gw2 = self.load_gw_fn(self.season_ctx, 2)
        if isinstance(result_gw2, tuple) and len(result_gw2) == 3:
            self.pool, self.index_map, gw2_df = result_gw2
            self._points_map, self._played_map = self._gw_points_maps_from_df(gw2_df)
        else:
            self.pool, self.index_map = result_gw2
            self._points_map, self._played_map = {}, {}

        self.pool_size = len(self.pool)
        # self.action_space = spaces.MultiDiscrete([15 + 1, self.pool_size + 1, 15, 15])

        pid_to_pool = {pl.pid: pl for pl in self.pool}
        for i, p in enumerate(self.squad.players):
            if p.pid in pid_to_pool:
                self.squad.players[i] = pid_to_pool[p.pid]

        return self._obs(), self._info_dict(action="init_gw2")



    def _info_dict(self, action: str, points_hit: float = 0.0):
        return {
            "season": self.season,
            "current_gw": self.current_gw,
            "team_xP_expected": self._team_score_exp,
            "captain": self.squad.players[self._captain_idx].name if self._captain_idx >= 0 else None,
            "vice_captain": self.squad.players[self._vice_idx].name if self._vice_idx >= 0 else None,
            "xi_indices": self._xi_idx,
            "bank": self.squad.bank,
            "free_transfers": self.free_transfers,
            "action": action,
            "points_hit": points_hit
        }


    def step(self, action: np.ndarray):
        """
        Action = (out1, in1, out2, in2, cap_idx, vc_idx)
        - out* in {0..14}, 15 = 'no transfer'
        - in*  in {0..pool_size-1}, pool_size = 'no transfer'
        - cap_idx, vc_idx in {0..14}

        Both transfers are decided relative to the ORIGINAL squad of this GW.
        We validate/apply them on a temporary copy, then commit once both pass.
        """
        # --- episode end guard ---
        if self.step_in_episode >= self.episode_gws:
            return self._obs(), 0.0, True, False, self._info_dict(action="already_done")

        self.step_in_episode += 1
        done = False
        points_hit = 0.0

        # --- parse & normalize ---
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.size != 6:
            raise ValueError(f"Expected 6 ints (out1,in1,out2,in2,cap,vc); got {a} with shape {a.shape}")
        out1, in1, out2, in2, cap_idx, vc_idx = map(int, a)

        # Bring incoming sentinels in line with current pool size
        if in1 >= self.pool_size: in1 = self.pool_size
        if in2 >= self.pool_size: in2 = self.pool_size

        # Snapshot ORIGINAL state for planning + baseline comparison
        baseline_players = list(self.squad.players)
        baseline_squad   = copy.deepcopy(self.squad)  # deep copy to trial swaps

        # --- helpers ---
        def is_skip(o, i) -> bool:
            return (o >= 15) or (i >= self.pool_size)

        def validate_bounds(o_idx: int, i_idx: int) -> Tuple[bool, Dict[str, str]]:
            if o_idx < 0 or o_idx > 15:
                return False, {"illegal": "out_of_range"}
            if i_idx < 0 or i_idx > self.pool_size:
                return False, {"illegal": "in_of_range"}
            return True, {}

        # --- guards to forbid weird combos (based on ORIGINAL squad) ---
        # If both transfers requested, disallow using the same out-slot twice
        if not is_skip(out1, in1) and not is_skip(out2, in2):
            if out1 == out2:
                return self._obs(), self.illegal_penalty, False, False, {"illegal": "duplicate_out_slot"}
            # Forbid buy-back of the other swap’s sold player (based on original PID)
            sold1_pid = baseline_players[out1].pid if out1 < 15 else None
            sold2_pid = baseline_players[out2].pid if out2 < 15 else None
            if in2 < self.pool_size and sold1_pid is not None and self.pool[in2].pid == sold1_pid:
                return self._obs(), self.illegal_penalty, False, False, {"illegal": "buy_back_same_player"}
            if in1 < self.pool_size and sold2_pid is not None and self.pool[in1].pid == sold2_pid:
                return self._obs(), self.illegal_penalty, False, False, {"illegal": "buy_back_same_player"}

        # --- validate bounds early ---
        for (o_idx, i_idx) in [(out1, in1), (out2, in2)]:
            ok, info = validate_bounds(o_idx, i_idx)
            if not ok:
                return self._obs(), self.illegal_penalty, False, False, info

        # --- attempt swaps on the TEMP copy of the baseline squad ---
        def try_apply_on_temp(temp_squad: "Squad", o_idx: int, i_idx: int) -> Tuple[bool, Dict[str, str]]:
            if is_skip(o_idx, i_idx):
                return False, {}
            if o_idx >= 15 or i_idx >= self.pool_size:
                return False, {"illegal": "index_out_of_range"}
            incoming = self.pool[i_idx]
            if not temp_squad.can_swap(o_idx, incoming):
                return False, {"illegal": "cannot_swap"}
            temp_squad.apply_swap(o_idx, incoming)  # in-place replace at slot o_idx
            return True, {}

        # Work entirely on temp copy; this ensures both swaps are planned off ORIGINAL indices
        temp_squad = copy.deepcopy(baseline_squad)
        applied1, info1 = try_apply_on_temp(temp_squad, out1, in1)
        if info1.get("illegal"):
            return self._obs(), self.illegal_penalty, False, False, info1

        applied2, info2 = try_apply_on_temp(temp_squad, out2, in2)
        if info2.get("illegal"):
            return self._obs(), self.illegal_penalty, False, False, info2

        # --- compute hits (relative to current free_transfers) ---
        transfers_made = int(applied1) + int(applied2)
        ft_consumed = min(self.free_transfers, transfers_made)
        extra_transfers = transfers_made - ft_consumed
        if extra_transfers > 0:
            points_hit += self.transfer_hit * extra_transfers

        # --- commit temp result to real squad ---
        if transfers_made > 0:
            self.squad = temp_squad
            self.free_transfers -= ft_consumed
            if self.free_transfers < 0:
                self.free_transfers = 0

        # --- (Re)pick XI & apply C/VC (must be in XI and distinct) ---
        self._xi_idx = self._best_xi()
        if not (0 <= cap_idx < 15 and 0 <= vc_idx < 15):
            return self._obs(), self.illegal_penalty, False, False, {"illegal": "cap_vc_out_of_range"}
        if cap_idx == vc_idx:
            return self._obs(), self.illegal_penalty, False, False, {"illegal": "cap_eq_vc"}
        if cap_idx not in self._xi_idx or vc_idx not in self._xi_idx:
            return self._obs(), self.illegal_penalty, False, False, {"illegal": "band_not_in_XI"}
        self._captain_idx = cap_idx
        self._vice_idx    = vc_idx

        # --- realized vs skip (counterfactual) ---
        points_map = getattr(self, "_points_map", {})
        played_map = getattr(self, "_played_map", {})

        actual_realized = self._realized_team_score(
            self.squad.players, self._xi_idx, self._captain_idx, self._vice_idx, points_map, played_map
        )

        # Baseline (no transfers) greedy XI & band on original
        def _best_xi_and_band_tmp(players):
            pos_lists = {pos: sorted([(i, q) for i, q in enumerate(players) if q.pos == pos],
                                    key=lambda t: t[1].xP, reverse=True) for pos in POS_ORDER}
            best_xi, best_score = [], -1e9
            for d_cnt, m_cnt, f_cnt in self._valid_formations():
                if not pos_lists["GK"] or len(pos_lists["DEF"]) < d_cnt or len(pos_lists["MID"]) < m_cnt or len(pos_lists["FWD"]) < f_cnt:
                    continue
                xi_idx = [pos_lists["GK"][0][0]] + \
                        [i for i, _ in pos_lists["DEF"][:d_cnt]] + \
                        [i for i, _ in pos_lists["MID"][:m_cnt]] + \
                        [i for i, _ in pos_lists["FWD"][:f_cnt]]
                score = sum(players[i].xP for i in xi_idx)
                if score > best_score:
                    best_xi, best_score = xi_idx, score
            if not best_xi:
                best_xi = sorted(range(len(players)), key=lambda i: players[i].xP, reverse=True)[:11]
            xi_sorted = sorted(best_xi, key=lambda i: players[i].xP, reverse=True)
            cap = xi_sorted[0]
            vc  = xi_sorted[1] if len(xi_sorted) > 1 else xi_sorted[0]
            return best_xi, cap, vc

        xi_b, cap_b, vc_b = _best_xi_and_band_tmp(baseline_players)
        skip_realized = self._realized_team_score(
            baseline_players, xi_b, cap_b, vc_b, points_map, played_map
        )

        reward = (actual_realized - skip_realized) + points_hit
        self._team_score_exp = actual_realized  # for info panel

        # --- advance GW ---
        self.current_gw += 1
        if self.step_in_episode < self.episode_gws:
            result = self.load_gw_fn(self.season_ctx, self.current_gw)
            if isinstance(result, tuple) and len(result) == 3:
                self.pool, self.index_map, next_gw_df = result
                self._points_map, self._played_map = self._gw_points_maps_from_df(next_gw_df)
            else:
                self.pool, self.index_map = result
                self._points_map, self._played_map = {}, {}

            self.pool_size = len(self.pool)

            # Refresh squad players with next-GW stats (by PID)
            pid_to_pool = {pl.pid: pl for pl in self.pool}
            for i, p in enumerate(self.squad.players):
                if p.pid in pid_to_pool:
                    self.squad.players[i] = pid_to_pool[p.pid]

            # Bank a FT only if no transfers were made this GW
            if transfers_made == 0:
                self.free_transfers = min(self.free_transfers + 1, self.max_free_transfers)
        else:
            done = True

        return self._obs(), reward, done, False, self._info_dict(action="step", points_hit=points_hit)

