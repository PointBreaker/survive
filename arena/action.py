"""Discrete action space.

An action only expresses the controller's *desired thrust direction*.
The environment turns it into acceleration; velocity and position follow
from the physics (see ``arena/physics.py``).
"""
from __future__ import annotations

import math
from enum import Enum


class Action(str, Enum):
    STAY = "STAY"
    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"

    @property
    def direction(self) -> tuple[float, float]:
        """Unit thrust vector in screen coordinates (y grows downward)."""
        return _DIRECTIONS[self]

    @classmethod
    def from_vector(cls, dx: float, dy: float, dead_zone: float = 1e-9) -> "Action":
        """Quantize an arbitrary vector to the nearest of the 8 compass actions.

        Pure geometry helper so controllers share one quantization rule.
        It knows nothing about the world.
        """
        if math.hypot(dx, dy) <= dead_zone:
            return cls.STAY
        # Screen y grows downward, so "north" is -y.
        angle = math.atan2(-dy, dx)  # 0 = east, counter-clockwise positive
        sector = int(round(angle / (math.pi / 4))) % 8
        return _SECTORS[sector]


_D = math.sqrt(0.5)
_DIRECTIONS: dict[Action, tuple[float, float]] = {
    Action.STAY: (0.0, 0.0),
    Action.N: (0.0, -1.0),
    Action.NE: (_D, -_D),
    Action.E: (1.0, 0.0),
    Action.SE: (_D, _D),
    Action.S: (0.0, 1.0),
    Action.SW: (-_D, _D),
    Action.W: (-1.0, 0.0),
    Action.NW: (-_D, -_D),
}
_SECTORS = [Action.E, Action.NE, Action.N, Action.NW, Action.W, Action.SW, Action.S, Action.SE]

ALL_ACTIONS: tuple[Action, ...] = tuple(Action)
