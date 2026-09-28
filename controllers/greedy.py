"""Greedy baseline: head for the target, ignore obstacles entirely.

Steers the velocity toward the target with standard "arrive" behaviour
(slows down near the target) so that inertia plus decision delay does not
trap it in an orbit around the target. Completely obstacle-blind.
"""
from __future__ import annotations

import math

from arena.action import Action
from arena.observation import Observation
from controllers.base import SyncController


class GreedyController(SyncController):
    name = "greedy"

    def __init__(self, slow_radius: float = 120.0):
        super().__init__()
        self.slow_radius = slow_radius

    def decide(self, observation: Observation) -> Action:
        p, t = observation.player, observation.target
        dx, dy = t["x"] - p["x"], t["y"] - p["y"]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return Action.STAY
        vmax = self.info.player_max_speed
        speed = vmax * min(1.0, dist / self.slow_radius)
        desired_vx, desired_vy = dx / dist * speed, dy / dist * speed
        return Action.from_vector(desired_vx - p["vx"], desired_vy - p["vy"], dead_zone=0.1 * vmax)
