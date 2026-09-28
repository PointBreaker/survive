"""Diagnostic controller that takes ``delay_ms`` of real wall time per decision.

Runs on a worker thread (like a network API client would). Used to verify
that a slow controller never pauses the world.
"""
from __future__ import annotations

import time

from arena.action import Action
from arena.observation import Observation
from controllers.base import ThreadedController


class SleepController(ThreadedController):
    name = "sleep"

    def __init__(self, delay_ms: float = 500.0, action: Action = Action.E):
        super().__init__()
        self.delay_s = delay_ms / 1000.0
        self.action = Action(action)

    def decide(self, observation: Observation) -> Action:
        time.sleep(self.delay_s)
        return self.action
