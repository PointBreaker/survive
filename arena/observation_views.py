"""Observation modes for ablation experiments: RAW, RELATIVE, PHYSICS.

The environment produces one canonical observation (``Environment.observe``).
A view transforms it into the controller-facing observation:

    canonical Observation --> view(mode) --> controller

Environment, physics, collision and the action space are identical in every
mode. A view is a pure function of the one snapshot it is given: it never sees
the environment, its RNG, the spawn schedule or any future state.

RAW       the canonical observation, returned unchanged (the default, and the
          strictest setting: the controller builds all geometry itself).

RELATIVE  RAW plus exact coordinate transforms of the snapshot (ego-centric
          geometry). Added fields:
            target:    dx, dy, distance
            obstacle:  dx, dy, dvx, dvy, distance, gap, relative_speed
            player:    wall_distance {left, right, top, bottom}
          where dx = other.x - player.x, dy = other.y - player.y,
          dvx = other.vx - player.vx, dvy = other.vy - player.vy,
          distance = sqrt(dx^2 + dy^2) (centre to centre),
          gap = distance - other.radius - player.radius,
          relative_speed = sqrt(dvx^2 + dvy^2),
          wall_distance = centre to each arena edge (x, W - x, y, H - y).

PHYSICS   RELATIVE plus a straight-line constant-velocity projection for the
          target and each obstacle (no wall bounces, no acceleration, no
          simulator):
            r = (dx, dy), v = (dvx, dvy)
            time_to_closest_approach  t* = clamp(-(r.v) / (v.v), 0, H)   (0 if v.v = 0)
            closest_approach_distance     = |r + v t*|
            closest_approach_gap          = closest_approach_distance - radii
          with H = ``prediction_horizon_s`` (world seconds, published in the
          observation).

Neither derived mode contains any judgement: no risk labels, no ordering by
danger (obstacles stay in id order), no recommended or safe direction, no
action. Those words are checked by tests (``POLICY_HINT_WORDS``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from arena.observation import ArenaInfo, Observation
from controllers.base import Controller

MODES = ("raw", "relative", "physics")
DEFAULT_MODE = "raw"
PREDICTION_HORIZON_S = 3.0  # world seconds; fixed and published with every PHYSICS observation

# Substrings that must never appear in any key of a derived observation.
POLICY_HINT_WORDS = ("safe", "danger", "threat", "risk", "recommend", "avoid", "best", "should", "dodge",
                     "escape", "likely", "collision", "advice", "action", "priority", "nearest", "rank")

RELATIVE_TARGET_FIELDS = ("dx", "dy", "distance")
RELATIVE_OBSTACLE_FIELDS = ("dx", "dy", "dvx", "dvy", "distance", "gap", "relative_speed")
PHYSICS_FIELDS = ("time_to_closest_approach", "closest_approach_distance", "closest_approach_gap")


@dataclass(frozen=True)
class ViewObservation(Observation):
    """An Observation with derived fields. Same interface; ``to_dict`` also names the mode."""

    observation_mode: str = DEFAULT_MODE
    prediction_horizon_s: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["observation_mode"] = self.observation_mode
        if self.prediction_horizon_s is not None:
            d["prediction_horizon_s"] = self.prediction_horizon_s
        return d


def closest_approach(dx: float, dy: float, dvx: float, dvy: float, horizon: float) -> tuple[float, float]:
    """(t*, |r + v t*|) for relative position r and velocity v, t* clamped to [0, horizon]."""
    vv = dvx * dvx + dvy * dvy
    t = 0.0 if vv <= 1e-12 else max(0.0, min(horizon, -(dx * dvx + dy * dvy) / vv))
    return t, math.hypot(dx + dvx * t, dy + dvy * t)


def _relative(p: dict[str, float], o: dict[str, float], moving: bool) -> dict[str, float]:
    dx, dy = o["x"] - p["x"], o["y"] - p["y"]
    dist = math.hypot(dx, dy)
    if not moving:  # the target is static
        return {"dx": dx, "dy": dy, "distance": dist}
    dvx, dvy = o["vx"] - p["vx"], o["vy"] - p["vy"]
    return {"dx": dx, "dy": dy, "dvx": dvx, "dvy": dvy, "distance": dist,
            "gap": dist - o["radius"] - p["radius"], "relative_speed": math.hypot(dvx, dvy)}


def _physics(p: dict[str, float], o: dict[str, float], rel: dict[str, float], horizon: float) -> dict[str, float]:
    dvx = rel.get("dvx", -p["vx"])  # static target: relative velocity is minus the player's
    dvy = rel.get("dvy", -p["vy"])
    t, d = closest_approach(rel["dx"], rel["dy"], dvx, dvy, horizon)
    return {"time_to_closest_approach": t, "closest_approach_distance": d,
            "closest_approach_gap": d - o["radius"] - p["radius"]}


def view(obs: Observation, mode: str, horizon_s: float = PREDICTION_HORIZON_S) -> Observation:
    """The controller-facing observation for ``mode``. Pure function of ``obs``."""
    if mode == "raw":
        return obs
    if mode not in MODES:
        raise ValueError(f"unknown observation mode {mode!r} (choose from {', '.join(MODES)})")
    physics = mode == "physics"
    p = dict(obs.player)
    W, H = obs.arena["width"], obs.arena["height"]
    player = {**p, "wall_distance": {"left": p["x"], "right": W - p["x"], "top": p["y"], "bottom": H - p["y"]}}
    t_rel = _relative(p, obs.target, moving=False)
    target = {**obs.target, **t_rel}
    if physics:
        target.update(_physics(p, obs.target, t_rel, horizon_s))
    obstacles = []
    for o in obs.obstacles:
        rel = _relative(p, o, moving=True)
        row = {**o, **rel}
        if physics:
            row.update(_physics(p, o, rel, horizon_s))
        obstacles.append(row)
    return ViewObservation(
        timestamp=obs.timestamp, tick=obs.tick, player=player, target=target, obstacles=tuple(obstacles),
        arena=dict(obs.arena), score=obs.score, control=dict(obs.control),
        observation_mode=mode, prediction_horizon_s=horizon_s if physics else None,
    )


# Neutral field definitions, appended to a language-model controller's
# instructions so the added fields are not unexplained numbers. Formulas only.
FIELD_DEFINITIONS = {
    "raw": "",
    "relative": (
        " Extra fields in this observation are exact functions of the same snapshot: "
        "target.dx = target.x - player.x, target.dy = target.y - player.y, target.distance = "
        "sqrt(dx^2 + dy^2). For each obstacle: dx, dy (obstacle minus player position), dvx, dvy "
        "(obstacle minus player velocity), distance (centre to centre), gap = distance - "
        "obstacle.radius - player.radius (0 or less means touching), relative_speed = "
        "sqrt(dvx^2 + dvy^2). player.wall_distance: distance from the player's centre to each arena edge."
    ),
}
FIELD_DEFINITIONS["physics"] = FIELD_DEFINITIONS["relative"] + (
    " For the target and each obstacle, assuming both keep their current velocity in a straight "
    "line (no bounces, no acceleration): time_to_closest_approach = clamp(-(dx*dvx + dy*dvy) / "
    "(dvx^2 + dvy^2), 0, prediction_horizon_s) in world seconds (0 when the relative velocity is 0; "
    "for the static target dvx, dvy = -player.vx, -player.vy), closest_approach_distance = centre "
    "distance at that time, closest_approach_gap = closest_approach_distance minus both radii. "
    "These are linear extrapolations of the snapshot, not forecasts of the simulator."
)


class ObservationView(Controller):
    """Controller wrapper: hands ``inner`` the ``mode`` view of every observation.

    Transport-level only: it forwards reset/poll/close unchanged and never
    sees or passes on anything but the observation it was handed.
    """

    def __init__(self, inner, mode: str, horizon_s: float = PREDICTION_HORIZON_S):
        if mode not in MODES:
            raise ValueError(f"unknown observation mode {mode!r} (choose from {', '.join(MODES)})")
        self.inner = inner
        self.mode = mode
        self.horizon_s = horizon_s
        self.name = inner.name
        setter = getattr(inner, "set_observation_mode", None)
        if setter:
            setter(mode)

    def reset(self, info: ArenaInfo) -> None:
        super().reset(info)
        self.inner.reset(info)

    def request(self, observation: Observation, request_id: Optional[int] = None) -> None:
        self.inner.request(view(observation, self.mode, self.horizon_s), request_id)

    def poll(self):
        return self.inner.poll()

    def close(self) -> None:
        self.inner.close()
