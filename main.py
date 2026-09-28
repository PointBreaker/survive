"""Decision Arena desktop console (one page: parameters · arena · inspector).

    python main.py                                   # open the console, pick and press Start
    python main.py --controller jev --max-inflight 3 # preselect and start immediately
    python main.py --controller human

In a game: the right-hand inspector panel shows every request/response live.
The faded "ghost" in the arena is the snapshot the controller is currently
deciding on; the gap between ghost and reality is the latency being paid.

Keys: Space start/stop, WASD / arrows move (human), R restart, N next seed,
- / = world speed (restarts on the same seed), J cycle panel, G ghost,
P probability compass, F1 debug overlay, V replay the finished episode.
"""
from __future__ import annotations

import argparse
import sys

from arena.cli import add_difficulty_args, config_from_args
from arena.dotenv import load_dotenv

INTERACTIVE_CONTROLLERS = ("human", "random", "greedy", "simple_avoid", "sleep", "jev", "remote")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=INTERACTIVE_CONTROLLERS, default=None,
                    help="preselect this controller and start immediately")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-log", action="store_true", help="do not write run logs")
    ap.add_argument("--panel", choices=("decisions", "request", "response"), default="decisions",
                    help="initial inspector panel (J cycles)")
    ap.add_argument("--window", default=None, help="window size in points, e.g. 1600x900")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)  # smoke tests: quit after N frames
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    add_difficulty_args(ap)
    args = ap.parse_args(argv)
    load_dotenv()  # e.g. OPENROUTER_API_KEY for jev

    from arena.app import Settings, run_app

    window = tuple(int(v) for v in args.window.lower().split("x")) if args.window else None
    human_hz = 60.0 if args.controller == "human" and args.decision_hz is None else None
    cfg = config_from_args(args, decision_hz=human_hz)
    # Only pin an exact base config if flags go beyond what the launcher exposes.
    extra = ("obstacle_speed_min", "obstacle_speed_max", "obstacle_radius_min", "obstacle_radius_max",
             "player_max_speed", "spawn_rate", "decision_deadline_ms", "target_timeout", "target_goal")
    pinned = any(getattr(args, f) is not None for f in extra)
    settings = Settings.from_config(
        cfg, controller=args.controller or "simple_avoid", preset=args.preset, latency_ms=args.latency_ms,
        real_latency=args.real_latency, seed=args.seed, save_logs=not args.no_log, out=args.out,
        endpoint=args.endpoint,
    )
    if not pinned:
        settings.base = None
    return run_app(settings, autostart=args.controller is not None, panel=args.panel,
                   max_frames=args.frames, screenshot=args.screenshot, window=window)


if __name__ == "__main__":
    sys.exit(main())
