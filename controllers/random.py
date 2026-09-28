"""Random baseline: uniformly random action, held for a few decisions.

Uses its own RNG seeded independently of the environment seed.
"""
from __future__ import annotations

import random
from typing import Optional

from arena.action import ALL_ACTIONS, Action
from arena.observation import Observation
from controllers.base import SyncController


class RandomController(SyncController):
    name = "random"

    def __init__(self, seed: Optional[int] = 12345, persistence: int = 3):
        super().__init__()
        self._seed = seed
        self._rng = random.Random(seed)
        self.persistence = max(1, persistence)
        self._current = Action.STAY
        self._left = 0

    def decide(self, observation: Observation) -> Action:
        if self._left <= 0:
            self._current = self._rng.choice(ALL_ACTIONS)
            self._left = self.persistence
        self._left -= 1
        return self._current
