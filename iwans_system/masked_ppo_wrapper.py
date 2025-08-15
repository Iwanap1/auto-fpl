# wrappers_discrete_mask.py
import numpy as np
import gymnasium as gym
from typing import List, Tuple

class DiscreteActionSetWrapper(gym.ActionWrapper):
    """
    Maps a dynamic set of legal (out, in) pairs to a fixed Discrete action space.
    - action_space: Discrete(MAX_CHOICES), where MAX_CHOICES is a safe upper bound.
    - On reset/step, we rebuild the legal list from env._build_masks() and expose a binary mask.
    - Use with sb3-contrib's ActionMasker (mask_fn reads self._last_mask).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.max_pool_size = env.max_pool_size
        self.MAX_CHOICES = 15 * self.max_pool_size + 1
        self.action_space = gym.spaces.Discrete(self.MAX_CHOICES)
        self._choices: List[Tuple[int, int]] = []
        self._last_mask = np.zeros(self.MAX_CHOICES, dtype=np.int8)

    # ---- helpers ----
    def _rebuild_choices_from_masks(self):
        """Build the flat list of legal (out, in) indices and a mask vector."""
        mask_out, mask_in = self.env._build_masks()

        choices: List[Tuple[int, int]] = []
        SKIP_IN = self.env.SKIP_IN
        for out_idx in range(15):
            if mask_out[out_idx] == 1:
                for j in range(min(self.env.pool_size, self.max_pool_size)):
                    if mask_in[j] == 1:
                        choices.append((out_idx, j))
        choices.append((15, SKIP_IN))
        if len(choices) > self.MAX_CHOICES:
            choices = choices[:self.MAX_CHOICES]
        mask = np.zeros(self.MAX_CHOICES, dtype=np.int8)
        mask[:len(choices)] = 1

        self._choices = choices
        self._last_mask = mask

    # ---- wrappers ----
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._rebuild_choices_from_masks()
        return obs, info

    def step(self, action: int):
        # Clamp into current range; if invalid, map to skip.
        if not (0 <= action < len(self._choices)):
            # last element is (skip, skip)
            action = len(self._choices) - 1

        out_idx, in_idx = self._choices[action]
        obs, rew, terminated, truncated, info = self.env.step(np.array([out_idx, in_idx], dtype=np.int64))

        # Rebuild choices for the next decision *from the new env state*
        self._rebuild_choices_from_masks()
        # Expose mask via info if you like (optional)
        info["action_mask"] = self._last_mask
        info["choices"] = self._choices  # handy for debugging

        return obs, rew, terminated, truncated, info


# ---- sb3-contrib ActionMasker expects a callable(env) -> mask ndarray ----
def mask_fn(env: DiscreteActionSetWrapper) -> np.ndarray:
    # simply return the last computed mask
    return env._last_mask
