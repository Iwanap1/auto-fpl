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
    try:
        v = float(p.features[-1])  # last element in features tuple
    except Exception:
        return 1.0
    return max(0.0, min(1.0, v))



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
        illegal_penalty: float = -8,
        temperature: float = 1.0,
        pool_per_pos: int = 20, # number of players to choose from in each position
        transfer_hit: float = -4.0,
        max_free_transfers: int = 5,
        n_features: int = 7,
        randomize_start: bool = True,
        max_episode_gws: int = 12
    ):
        
        super().__init__()
        self.seasons = list(seasons)
        self.base_dir = base_dir
        self.load_season_fn = load_season_fn
        self.load_gw_fn = load_gw_fn
        self.models = models
        self.randomize_start = randomize_start
        self.max_episode_gws = max_episode_gws
        self.start_gw = int(start_gw)
        self.start_gw_actual: int = self.start_gw
        self.budget = float(budget)
        self.illegal_penalty = float(illegal_penalty)
        self.temperature = float(temperature)
        self.transfer_hit = float(transfer_hit)
        self.max_free_transfers = int(max_free_transfers)
        self.max_pool_size = 4 * pool_per_pos
        self.SKIP_IN = self.max_pool_size
        self.n_features = int(n_features)
        self.per_pos = int(pool_per_pos)
        # Will be set at reset()
        self.season: str = ""
        self.season_ctx: Any = None
        self.total_gws: int = 38
        self.end_gw: int = self.total_gws
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
            16,                    # out: 0..14 , 15 = skip
            self.max_pool_size+1,  # in:  0..max_pool_size , +1 = skip
        ])
        self.phase = 1 # 1 = transfer 1, 2 = transfer 2
        self._pending_out_in = None  

        # Observation: 15 × (price, xP, 5 features) + bank + free transfers
        self.obs_dim = 15 * (2 + self.n_features) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_dim,), dtype=np.float32)

        # Cache for XI/captain/VC/score
        self._xi_idx: List[int] = []
        self._captain_idx: int = -1
        self._vice_idx: int = -1
        self._team_score_exp: float = 0.0

    # ===== Helpers =====
    # Masks for legal out/in choices for second transfer phase
    def _skip_sentinel(self) -> Tuple[int, int]:
        # out=15, in=self.SKIP_IN means "skip"
        return 15, self.SKIP_IN

    def _is_skip_pair(self, out_idx: int, in_idx: int) -> bool:
        # only the fixed SKIP_IN is skip; values >= pool_size but < SKIP_IN are invalid, not skip
        return (out_idx >= 15) or (in_idx == self.SKIP_IN)


    def _legal_incoming_indices(self, temp_squad: "Squad", out_idx: int) -> np.ndarray:
        mask_in = np.zeros(self.max_pool_size + 1, dtype=np.int8)
        mask_in[self.SKIP_IN] = 1  # always allow skip sentinel

        if out_idx >= 15:
            return mask_in  # no real out -> only skip is allowed

        # Try only the actually-present pool candidates: 0..pool_size-1
        for j in range(self.pool_size):
            incoming = self.pool[j]
            trial = copy.deepcopy(temp_squad)
            if not trial.can_swap(out_idx, incoming):
                continue
            trial.apply_swap(out_idx, incoming)
            if trial.bank >= -1e-9:
                mask_in[j] = 1

        # Note: indices in [pool_size, max_pool_size) remain 0 (invalid), SKIP_IN is 1
        return mask_in



    def _build_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build masks for the current phase.
        Returns (mask_out[16], mask_in[self.pool_size+1]).
        """
        mask_out = np.zeros(16, dtype=np.int8)
        mask_in  = np.zeros(self.max_pool_size + 1, dtype=np.int8)

        mask_out[15] = 1
        mask_in[self.SKIP_IN] = 1

        incoming1_pid = None
        if self.phase == 2 and (self._pending_out_in is None or self._is_skip_pair(*self._pending_out_in)):
            return mask_out, mask_in
        
        # Which squad context do we validate against?
        # Phase 1: current squad
        # Phase 2: hypothetical squad after applying pending out/in (if any)
        if self.phase == 1 or self._pending_out_in is None or self._is_skip_pair(*self._pending_out_in):
            temp_squad = self.squad
            sold1_pid = None
            used_out_slot = None
        else:
            # simulate first swap on a copy
            out1, in1 = self._pending_out_in
            temp_squad = copy.deepcopy(self.squad)
            if not self._is_skip_pair(out1, in1):
                if in1 < self.pool_size and out1 < 15 and temp_squad.can_swap(out1, self.pool[in1]):
                    sold1_pid = temp_squad.players[out1].pid
                    incoming1_pid = self.pool[in1].pid
                    used_out_slot = out1
                    temp_squad.apply_swap(out1, self.pool[in1])
                else:
                    sold1_pid = None
                    incoming1_pid = None
                    used_out_slot = None
            else:
                sold1_pid = None
                incoming1_pid = None
                used_out_slot = None

        for out_idx in range(15):
            if self.phase == 2 and used_out_slot is not None and out_idx == used_out_slot:
                continue
            in_mask = self._legal_incoming_indices(temp_squad, out_idx)
            if self.phase == 2 and sold1_pid is not None:
                for j in range(self.pool_size):           # only iterate real pool indices
                    if in_mask[j] and self.pool[j].pid == sold1_pid:
                        in_mask[j] = 0
            # enable out only if some non-skip incoming remains
            if np.any(in_mask[:self.pool_size]):          # non-skip, real candidates
                mask_out[out_idx] = 1

        if self.phase == 2 and self._pending_out_in is not None:
            out1, in1 = self._pending_out_in
            if not self._is_skip_pair(out1, in1):
                mask_in = self._legal_incoming_indices(temp_squad, out1)
                if sold1_pid is not None:
                    for j in range(self.pool_size):
                        if mask_in[j] and self.pool[j].pid == sold1_pid:
                            mask_in[j] = 0
                if incoming1_pid is not None:
                    for j in range(self.pool_size):
                        if mask_in[j] and self.pool[j].pid == incoming1_pid:
                            mask_in[j] = 0
                mask_in[self.SKIP_IN] = 1

        return mask_out, mask_in



    # ===== Squad & Pool Helpers =====
    def _pvec(self, p: Player) -> np.ndarray:
        return np.array([p.price, p.xP, *p.features], dtype=np.float32)

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
        vecs = [self._pvec(p) for p in self.squad.players]
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

        # Load season
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

        # Pick a start GW
        if self.randomize_start:
            # cap at 33 so you can fit up to 12 GWs without passing 38; also guard by season length
            hi = min(33, max(1, self.total_gws)) 
            self.start_gw_actual = random.randint(1, hi)
        else:
            self.start_gw_actual = max(1, min(self.start_gw, self.total_gws))

        # Define end_gw and length
        self.end_gw = min(self.total_gws, self.start_gw_actual + self.max_episode_gws - 1)
        self.current_gw = self.start_gw_actual
        self.step_in_episode = 0
        self.free_transfers = 1

        # Load pool + realized maps for the CURRENT GW
        result = self.load_gw_fn(self.season_ctx, self.current_gw, per_pos=self.per_pos, temperature=self.temperature)
        if isinstance(result, tuple) and len(result) == 3:
            self.pool, self.index_map, gw_df = result
            self._points_map, self._played_map = self._gw_points_maps_from_df(gw_df)
        else:
            self.pool, self.index_map = result
            self._points_map, self._played_map = {}, {}

        self.pool_size = len(self.pool)

        # Build initial squad from *this* GW’s pool
        self.squad = self._semi_random_initial_squad()

        # Best XI & band for info (expected score not strictly required but handy)
        self._xi_idx = self._best_xi()
        self._captain_idx, self._vice_idx = self._best_band_for_xi(self._xi_idx)
        self._team_score_exp = self._expected_team_score(self._xi_idx, self._captain_idx, self._vice_idx)

        # Set phase/masks for sequential action picking
        self.phase = 1
        self._pending_out_in = None

        obs = self._obs()
        info = self._info_dict(action="init")
        info.update({
            "phase": self.phase,
            "start_gw": self.start_gw_actual,
            "end_gw": self.end_gw,
        })
        mask_out, mask_in = self._build_masks()
        info["action_mask_out"] = mask_out
        info["action_mask_in"] = mask_in
        return obs, info


    def _best_band_for_xi(self, xi_idx):
        # Sort XI by xP, highest first
        sorted_xi = sorted(xi_idx, key=lambda i: self.squad.players[i].xP, reverse=True)
        cap_idx = sorted_xi[0]
        vc_idx  = sorted_xi[1] if len(sorted_xi) > 1 else cap_idx
        return cap_idx, vc_idx

    def _info_dict(self, action: str, points_hit: float = 0.0):
        return {
            "season": self.season,
            "current_gw": self.current_gw,
            "team_xP_expected": self._team_score_exp,
            "captain": self.squad.players[self._captain_idx].name if self._captain_idx >= 0 else None,
            "vice_captain": self.squad.players[self._vice_idx].name if self._vice_idx >= 0 else None,
            "xi_indices": self._xi_idx,
            "bank": self.squad.bank,
            "phase": self.phase,
            "action_mask_out": '',
            "action_mask_in": '',
            "free_transfers": self.free_transfers,
            "action": action,
            "points_hit": points_hit
        }


    def step(self, action: np.ndarray):
        # Early termination guard
        if self.current_gw > self.end_gw or self.step_in_episode >= (self.end_gw - self.start_gw_actual + 1):
            info = self._info_dict(action="already_done")
            mask_out, mask_in = self._build_masks()
            info.update({"phase": self.phase, "action_mask_out": mask_out, "action_mask_in": mask_in,
                        "start_gw": self.start_gw_actual, "end_gw": self.end_gw})
            return self._obs(), 0.0, True, False, info

        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.size != 2:
            raise ValueError(...)
        out, inn = map(int, a)

        # clamp to action space range
        if inn < 0: inn = 0
        if inn > self.max_pool_size: inn = self.SKIP_IN

        # OPTIONAL: normalize values ≥ current pool_size to skip to avoid “phantom” indices
        if inn >= self.pool_size and inn != self.SKIP_IN:
            inn = self.SKIP_IN

        # Build masks and sanitize
        # Build masks and sanitize 'out' first
        mask_out, _ = self._build_masks()
        if out < 0 or out > 15 or mask_out[out] == 0:
            out = 15

        # If we're in phase 1, we don't sanitize 'inn' further (skip-only mask is fine)
        # If we're in phase 2, recompute an 'in' mask for the actually chosen out
        if self.phase == 1:
            # ✅ Phase-1: validate IN against per-OUT legality, not the coarse skip-only mask
            mask_in1 = self._legal_incoming_indices(self.squad, out)
            if inn < 0 or inn > self.max_pool_size or mask_in1[inn] == 0:
                inn = self.SKIP_IN
        else:
            # Build phase-2 context (same logic as in _build_masks)
            sold1_pid = None
            incoming1_pid = None
            used_out_slot = None
            temp_squad2 = copy.deepcopy(self.squad)
            if self._pending_out_in is not None:
                out1, in1 = self._pending_out_in
                if not self._is_skip_pair(out1, in1):
                    if in1 < self.pool_size and out1 < 15 and temp_squad2.can_swap(out1, self.pool[in1]):
                        sold1_pid = temp_squad2.players[out1].pid
                        incoming1_pid = self.pool[in1].pid
                        used_out_slot = out1
                        temp_squad2.apply_swap(out1, self.pool[in1])

            # Now compute the legal incoming mask for the ACTUAL phase-2 'out'
            mask_in2 = self._legal_incoming_indices(temp_squad2, out)

            # Enforce no buy-back of sold1 and no duplicate incoming
            if sold1_pid is not None:
                for j in range(self.pool_size):
                    if mask_in2[j] and self.pool[j].pid == sold1_pid:
                        mask_in2[j] = 0
            if incoming1_pid is not None:
                for j in range(self.pool_size):
                    if mask_in2[j] and self.pool[j].pid == incoming1_pid:
                        mask_in2[j] = 0

            mask_in2[self.SKIP_IN] = 1
            if inn < 0 or inn > self.max_pool_size or mask_in2[inn] == 0:
                inn = self.SKIP_IN


        # ===== Phase 1: store choice, no reward yet =====
        if self.phase == 1:
            self._pending_out_in = (out, inn)
            self.phase = 2

            obs = self._obs()
            info = self._info_dict(action="phase1_choice")
            # next masks (for phase 2)
            mask_out2, mask_in2 = self._build_masks()
            info.update({"phase": self.phase, "action_mask_out": mask_out2, "action_mask_in": mask_in2})
            return obs, 0.0, False, False, info

        # ===== Phase 2: apply both, compute reward, advance GW =====
        self.step_in_episode += 1
        done = False
        points_hit = 0.0

        # Snapshot ORIGINAL state for baseline comparison
        baseline_players = list(self.squad.players)
        baseline_squad   = copy.deepcopy(self.squad)

        # Resolve first and second choices
        out1, in1 = self._pending_out_in if self._pending_out_in is not None else self._skip_sentinel()
        out2, in2 = out, inn

        # Helper to try a swap on temp
        def try_apply_on_temp(temp_squad: "Squad", o_idx: int, i_idx: int) -> bool:
            if self._is_skip_pair(o_idx, i_idx):
                return False
            if o_idx >= 15 or i_idx >= self.pool_size:
                return False
            incoming = self.pool[i_idx]
            if not temp_squad.can_swap(o_idx, incoming):
                return False
            temp_squad.apply_swap(o_idx, incoming)
            return True

        # Work on a temp copy
        temp_squad = copy.deepcopy(baseline_squad)
        applied1 = try_apply_on_temp(temp_squad, out1, in1)
        applied2 = try_apply_on_temp(temp_squad, out2, in2)

        # Hits relative to free transfers
        transfers_made = int(applied1) + int(applied2)
        ft_consumed = min(self.free_transfers, transfers_made)
        extra_transfers = transfers_made - ft_consumed
        if extra_transfers > 0:
            points_hit += self.transfer_hit * extra_transfers

        # Commit
        if transfers_made > 0:
            self.squad = temp_squad
            self.free_transfers -= ft_consumed
            if self.free_transfers < 0:
                self.free_transfers = 0

        # (Re)pick XI & band
        self._xi_idx = self._best_xi()
        self._captain_idx, self._vice_idx = self._best_band_for_xi(self._xi_idx)

        # Realized vs skip (counterfactual)
        points_map = getattr(self, "_points_map", {})
        played_map = getattr(self, "_played_map", {})

        actual_realized = self._realized_team_score(
            self.squad.players, self._xi_idx, self._captain_idx, self._vice_idx, points_map, played_map
        )

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
        self._team_score_exp = actual_realized

        # Advance GW
        self.current_gw += 1  # advance to next GW
        # done if we’ve stepped past end_gw
        done = self.current_gw > self.end_gw

        if not done:
            # Load next GW pool/maps
            result = self.load_gw_fn(self.season_ctx, self.current_gw, per_pos=self.per_pos, temperature=self.temperature)
            if isinstance(result, tuple) and len(result) == 3:
                self.pool, self.index_map, next_gw_df = result
                self._points_map, self._played_map = self._gw_points_maps_from_df(next_gw_df)
            else:
                self.pool, self.index_map = result
                self._points_map, self._played_map = {}, {}
            self.pool_size = len(self.pool)
            # Refresh by PID
            pid_to_pool = {pl.pid: pl for pl in self.pool}
            for i, p in enumerate(self.squad.players):
                if p.pid in pid_to_pool:
                    self.squad.players[i] = pid_to_pool[p.pid]
            # Bank FT if no transfers this GW
            if transfers_made == 0:
                self.free_transfers = min(self.free_transfers + 1, self.max_free_transfers)

        # Reset the two-phase state for the next GW (or for the terminal info)
        self.phase = 1
        self._pending_out_in = None

        obs = self._obs()
        info = self._info_dict(action="step", points_hit=points_hit)
        mask_out_next, mask_in_next = self._build_masks()
        info.update({
            "phase": self.phase,
            "action_mask_out": mask_out_next,
            "action_mask_in": mask_in_next,
            "start_gw": self.start_gw_actual,
            "end_gw": self.end_gw
        })
        return obs, reward, done, False, info

