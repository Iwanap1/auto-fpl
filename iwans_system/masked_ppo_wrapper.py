# masked_ppo_wrapper.py
import copy
import numpy as np
import gymnasium as gym
from typing import List, Tuple

class DiscreteActionSetWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # generous upper bound for flat action set
        self.max_pool_size = env.max_pool_size
        self.MAX_CHOICES = 15 * max(1, self.max_pool_size) + 1
        self.action_space = gym.spaces.Discrete(self.MAX_CHOICES)
        self._choices: List[Tuple[int, int]] = []
        self._last_mask = np.zeros(self.MAX_CHOICES, dtype=np.int8)

    def _rebuild_choices_from_masks(self):
        SKIP_IN = self.env.SKIP_IN
        mask_out, mask_in_global = self.env._build_masks()
        choices: List[Tuple[int, int]] = []

        if self.env.phase == 1:
            # Phase-1: enumerate ALL valid (out, in) pairs for transfer #1.
            temp_squad = self.env.squad  # current squad
            for out_idx in range(15):
                if mask_out[out_idx] != 1:
                    continue
                # per-OUT incoming mask
                in_mask = self.env._legal_incoming_indices(temp_squad, out_idx)
                for j in range(min(self.env.pool_size, self.max_pool_size)):
                    if in_mask[j] == 1:
                        choices.append((out_idx, j))
            # also allow full skip for transfer #1
            choices.append((15, SKIP_IN))
        else:
            # Phase-2: enumerate legal (out, in) for transfer #2,
            # simulating transfer #1 on a copy of the squad.
            temp_squad = copy.deepcopy(self.env.squad)
            sold1_pid = None
            incoming1_pid = None
            used_out_slot = None

            if self.env._pending_out_in is not None:
                out1, in1 = self.env._pending_out_in
                # if transfer #1 was not skipped, apply it to the temp squad
                if not self.env._is_skip_pair(out1, in1):
                    if in1 < self.env.pool_size and out1 < 15 and temp_squad.can_swap(out1, self.env.pool[in1]):
                        sold1_pid = temp_squad.players[out1].pid
                        incoming1_pid = self.env.pool[in1].pid
                        used_out_slot = out1
                        temp_squad.apply_swap(out1, self.env.pool[in1])

            for out_idx in range(15):
                if mask_out[out_idx] != 1:
                    continue  # env mask already forbids reusing used_out_slot, etc.
                in_mask = self.env._legal_incoming_indices(temp_squad, out_idx)

                # forbid buy-back of sold1 and duplicate incoming of incoming1
                if sold1_pid is not None:
                    for j in range(self.env.pool_size):
                        if in_mask[j] and self.env.pool[j].pid == sold1_pid:
                            in_mask[j] = 0
                if incoming1_pid is not None:
                    for j in range(self.env.pool_size):
                        if in_mask[j] and self.env.pool[j].pid == incoming1_pid:
                            in_mask[j] = 0

                for j in range(min(self.env.pool_size, self.max_pool_size)):
                    if in_mask[j] == 1:
                        choices.append((out_idx, j))

            # always allow full skip for transfer #2
            choices.append((15, SKIP_IN))

        # clip and build flat mask
        if len(choices) > self.MAX_CHOICES:
            choices = choices[:self.MAX_CHOICES]
        mask = np.zeros(self.MAX_CHOICES, dtype=np.int8)
        mask[:len(choices)] = 1
        self._choices = choices
        self._last_mask = mask

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._rebuild_choices_from_masks()
        return obs, info

    def step(self, action: int):
        if not (0 <= action < len(self._choices)):
            action = len(self._choices) - 1  # map OOB → (skip, skip)
        out_idx, in_idx = self._choices[action]
        obs, rew, term, trunc, info = self.env.step(np.array([out_idx, in_idx], dtype=np.int64))
        self._rebuild_choices_from_masks()
        info["action_mask"] = self._last_mask
        info["choices"] = self._choices
        return obs, rew, term, trunc, info

def mask_fn(env: DiscreteActionSetWrapper) -> np.ndarray:
    return env._last_mask
