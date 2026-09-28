"""Shared command-line options for main.py and arena.benchmark."""
from __future__ import annotations

import argparse
from typing import Optional

from arena.difficulty import PRESETS, DifficultyConfig


def add_difficulty_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("difficulty")
    g.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    g.add_argument("--obstacles", type=int, dest="obstacle_count")
    g.add_argument("--speed-min", type=float, dest="obstacle_speed_min")
    g.add_argument("--speed-max", type=float, dest="obstacle_speed_max")
    g.add_argument("--radius-min", type=float, dest="obstacle_radius_min")
    g.add_argument("--radius-max", type=float, dest="obstacle_radius_max")
    g.add_argument("--player-speed", type=float, dest="player_max_speed")
    g.add_argument("--spawn-rate", type=float, dest="spawn_rate", help="extra obstacles per world second")
    g.add_argument("--world-speed", type=float, dest="world_speed_scale")
    g.add_argument("--decision-hz", type=float, dest="decision_hz")
    g.add_argument("--deadline-ms", type=float, dest="decision_deadline_ms")
    g.add_argument("--target-timeout", type=float, dest="target_timeout")
    g.add_argument("--target-goal", type=int, dest="target_goal")
    g.add_argument("--max-duration", type=float, dest="max_duration")

    c = ap.add_argument_group("controller")
    c.add_argument("--latency-ms", type=float, default=0.0, help="extra latency added to every decision")
    c.add_argument(
        "--real-latency",
        action="store_true",
        help="withhold decisions for --latency-ms of real wall time instead of charging it in sim ticks",
    )
    c.add_argument("--controller-seed", type=int, default=12345, help="RNG seed of the random controller")
    c.add_argument("--sleep-ms", type=float, default=500.0, help="delay of the diagnostic sleep controller")
    c.add_argument("--endpoint", default=None,
                   help="decision service URL (jev default: $JEV_ENDPOINT or OpenRouter; remote default: http://127.0.0.1:8765)")


DIFFICULTY_FIELDS = (
    "obstacle_count",
    "obstacle_speed_min",
    "obstacle_speed_max",
    "obstacle_radius_min",
    "obstacle_radius_max",
    "player_max_speed",
    "spawn_rate",
    "world_speed_scale",
    "decision_hz",
    "decision_deadline_ms",
    "target_timeout",
    "target_goal",
    "max_duration",
)


def config_from_args(args: argparse.Namespace, **defaults: Optional[float]) -> DifficultyConfig:
    base = PRESETS[args.preset]
    overrides = {k: v for k, v in defaults.items() if v is not None}
    overrides.update({f: getattr(args, f) for f in DIFFICULTY_FIELDS if getattr(args, f) is not None})
    cfg = base.with_overrides(**overrides)
    cfg.validate()
    return cfg


def controller_factory(name: str, args: argparse.Namespace, latency_ms: Optional[float] = None):
    from controllers import make_controller
    from controllers.base import LatencyWrapper

    kwargs = {}
    if name == "random":
        kwargs["seed"] = args.controller_seed
    elif name == "sleep":
        kwargs["delay_ms"] = args.sleep_ms
    elif name in ("jev", "remote") and args.endpoint:
        kwargs["endpoint"] = args.endpoint
    delay = args.latency_ms if latency_ms is None else latency_ms

    try:  # fail fast with a readable message (e.g. missing API token)
        make_controller(name, **kwargs).close()
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"error: {e}")

    def factory():
        ctrl = make_controller(name, **kwargs)
        if delay and delay > 0:
            ctrl = LatencyWrapper(ctrl, delay, simulated=not args.real_latency)
        return ctrl

    return factory
