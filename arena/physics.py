"""Pure physics functions. No decisions, no randomness.

The action pipeline is: action -> acceleration -> velocity -> position.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from arena.entities import Obstacle, Player

PHYSICS_HZ = 60
TICK_SECONDS = 1.0 / PHYSICS_HZ  # controller-clock seconds per physics tick


def integrate_player(
    player: Player,
    thrust: tuple[float, float],
    acceleration: float,
    drag: float,
    dt: float,
    width: float,
    height: float,
) -> float:
    """Advance the player by ``dt`` world seconds. Returns distance moved."""
    tx, ty = thrust
    player.vx += tx * acceleration * dt
    player.vy += ty * acceleration * dt

    damp = max(0.0, 1.0 - drag * dt)
    player.vx *= damp
    player.vy *= damp

    speed = math.hypot(player.vx, player.vy)
    if speed > player.max_speed:
        k = player.max_speed / speed
        player.vx *= k
        player.vy *= k

    old_x, old_y = player.x, player.y
    player.x += player.vx * dt
    player.y += player.vy * dt

    # Walls are solid but harmless: clamp and kill the normal velocity.
    r = player.radius
    if player.x < r:
        player.x, player.vx = r, 0.0
    elif player.x > width - r:
        player.x, player.vx = width - r, 0.0
    if player.y < r:
        player.y, player.vy = r, 0.0
    elif player.y > height - r:
        player.y, player.vy = height - r, 0.0

    return math.hypot(player.x - old_x, player.y - old_y)


def move_obstacle(ob: Obstacle, dt: float, width: float, height: float) -> None:
    """Straight-line motion with elastic bounce off arena edges."""
    ob.x += ob.vx * dt
    ob.y += ob.vy * dt
    r = ob.radius
    if ob.x < r:
        ob.x = 2 * r - ob.x
        ob.vx = abs(ob.vx)
    elif ob.x > width - r:
        ob.x = 2 * (width - r) - ob.x
        ob.vx = -abs(ob.vx)
    if ob.y < r:
        ob.y = 2 * r - ob.y
        ob.vy = abs(ob.vy)
    elif ob.y > height - r:
        ob.y = 2 * (height - r) - ob.y
        ob.vy = -abs(ob.vy)


def circles_overlap(ax: float, ay: float, ar: float, bx: float, by: float, br: float) -> bool:
    dx = ax - bx
    dy = ay - by
    rr = ar + br
    return dx * dx + dy * dy < rr * rr


def first_collision(player: Player, obstacles: Iterable[Obstacle]) -> Optional[Obstacle]:
    """Return the lowest-id obstacle currently overlapping the player, if any."""
    px, py, pr = player.x, player.y, player.radius
    for ob in obstacles:
        if circles_overlap(px, py, pr, ob.x, ob.y, ob.radius):
            return ob
    return None
