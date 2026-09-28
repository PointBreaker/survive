"""SimpleAvoid baseline: target attraction + obstacle repulsion.

All risk reasoning lives here, computed from raw positions/velocities in
the Observation. The environment provides none of it.

For each obstacle, the controller extrapolates straight-line relative
motion to the time of closest approach (within a short horizon). Obstacles
whose predicted miss distance is small push the player away from that
predicted point, more strongly the sooner and closer the approach. A
static term also pushes away from anything already very close, and walls
repel mildly so the player is not cornered. The summed vector is a desired
velocity; the action is the compass direction that best steers the current
velocity toward it.

Latency compensation: the controller cannot see its own latency, but it
can infer its effective decision interval from the tick spacing of the
observations it receives. It extrapolates the whole scene forward by that
interval (straight lines, no bounces) before reasoning, since that is
roughly how stale the snapshot will be by the time its action takes effect.
"""
from __future__ import annotations

import math

from arena.action import Action
from arena.observation import ArenaInfo, Observation
from controllers.base import SyncController


class SimpleAvoidController(SyncController):
    name = "simple_avoid"

    def __init__(
        self,
        margin: float = 18.0,
        buffer: float = 45.0,
        base_horizon: float = 1.0,
        repulsion_gain: float = 3.0,
        proximity_range: float = 55.0,
        wall_range: float = 60.0,
        slow_radius: float = 120.0,
        compensate_latency: bool = True,
        max_lead: float = 0.4,
        lead_rule: str = "latency_hold",
    ):
        super().__init__()
        self.margin = margin
        self.buffer = buffer
        self.base_horizon = base_horizon
        self.repulsion_gain = repulsion_gain
        self.proximity_range = proximity_range
        self.wall_range = wall_range
        self.slow_radius = slow_radius
        self.compensate_latency = compensate_latency
        self.max_lead = max_lead
        self.lead_rule = lead_rule

    def reset(self, info: ArenaInfo) -> None:
        super().reset(info)
        # Look further ahead when decisions are sparse in world time.
        decision_period_world = info.world_speed_scale / info.decision_hz
        self.horizon = self.base_horizon + 2.0 * decision_period_world
        self._prev_tick = None

    def decide(self, observation: Observation) -> Action:
        p = observation.player
        px, py, pvx, pvy, pr = p["x"], p["y"], p["vx"], p["vy"], p["radius"]
        vmax = self.info.player_max_speed

        lead = 0.0
        if self.compensate_latency and self._prev_tick is not None:
            tick_s = self.info.world_speed_scale / self.info.physics_hz
            gap_world = (observation.tick - self._prev_tick) * tick_s
            lat_world = observation.control.get("applied_latency_world_s")
            if self.lead_rule == "gap" or lat_world is None:
                lead = gap_world
            elif self.lead_rule == "latency":
                lead = lat_world
            else:  # latency + half the interval this action will be held
                lead = lat_world + 0.5 * gap_world
            lead = min(self.max_lead, lead)
        self._prev_tick = observation.tick
        px, py = px + pvx * lead, py + pvy * lead

        # Attraction toward target ("arrive": slow down when close so inertia
        # and decision delay do not make the player orbit the target).
        tx, ty = observation.target["x"] - px, observation.target["y"] - py
        td = math.hypot(tx, ty) or 1.0
        arrive = min(1.0, td / self.slow_radius)
        ax, ay = tx / td * arrive, ty / td * arrive

        # Obstacle repulsion.
        rx = ry = 0.0
        for o in observation.obstacles:
            dx, dy = o["x"] + o["vx"] * lead - px, o["y"] + o["vy"] * lead - py
            vx, vy = o["vx"] - pvx, o["vy"] - pvy
            R = o["radius"] + pr + self.margin
            reach = R + self.buffer

            vv = vx * vx + vy * vy
            tca = 0.0 if vv < 1e-9 else max(0.0, min(self.horizon, -(dx * vx + dy * vy) / vv))
            cx, cy = dx + vx * tca, dy + vy * tca
            d = math.hypot(cx, cy)
            if d < reach:
                if d > 1e-6:
                    ux, uy = -cx / d, -cy / d
                else:  # dead-on: sidestep perpendicular to relative motion
                    s = math.sqrt(vv) or 1.0
                    ux, uy = -vy / s, vx / s
                w = (reach - d) / reach / (1.0 + 2.0 * tca)
                rx += ux * w
                ry += uy * w

            gap = math.hypot(dx, dy) - (o["radius"] + pr)
            if gap < self.proximity_range:
                dist = math.hypot(dx, dy) or 1.0
                w = (self.proximity_range - max(gap, 0.0)) / self.proximity_range
                rx -= dx / dist * w
                ry -= dy / dist * w

        # Mild wall repulsion.
        W, H = self.info.arena_width, self.info.arena_height
        for dist, sx, sy in ((px - pr, 1, 0), (W - px - pr, -1, 0), (py - pr, 0, 1), (H - py - pr, 0, -1)):
            if dist < self.wall_range:
                w = 0.5 * (self.wall_range - dist) / self.wall_range
                rx += sx * w
                ry += sy * w

        rep = math.hypot(rx, ry)
        # Back off the attraction when threatened.
        attract = 1.0 / (1.0 + self.repulsion_gain * rep)
        dvx = (ax * attract + rx * self.repulsion_gain) * vmax
        dvy = (ay * attract + ry * self.repulsion_gain) * vmax
        n = math.hypot(dvx, dvy)
        if n > vmax:
            dvx, dvy = dvx / n * vmax, dvy / n * vmax
        return Action.from_vector(dvx - pvx, dvy - pvy, dead_zone=0.1 * vmax)
