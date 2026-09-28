"""Internal mutable world entities.

These objects belong to the environment. Controllers never receive them;
they only receive the copied, primitive-valued ``Observation``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Player:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    max_speed: float


@dataclass
class Obstacle:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float


@dataclass
class Target:
    x: float
    y: float
    radius: float
