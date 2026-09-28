"""Headless benchmark CLI (never opens a window, never imports pygame).

Examples::

    python -m arena.benchmark --controller simple_avoid --episodes 100 \\
        --obstacles 20 --world-speed 4

    python -m arena.benchmark --controller simple_avoid --episodes 20 \\
        --sweep world_speed_scale=1,2,4,8

    python -m arena.benchmark --controller simple_avoid --adaptive world_speed_scale
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from arena.cli import add_difficulty_args, config_from_args, controller_factory
from arena.recorder import new_run_dir
from benchmark.adaptive import find_frontier
from benchmark.metrics import aggregate, threshold_level
from benchmark.runner import run_episodes

HEADLESS_CONTROLLERS = ("random", "greedy", "simple_avoid", "sleep", "jev", "remote")
SWEEPABLE = (
    "world_speed_scale",
    "obstacle_count",
    "obstacle_speed_max",
    "spawn_rate",
    "latency_ms",
)
INTEGER_PARAMS = {"obstacle_count"}


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def print_summary(title: str, agg: dict[str, Any]) -> None:
    lo, hi = agg["success_rate_ci95"]
    print(f"\n== {title} ==")
    print(f"episodes          {agg['episodes']}")
    print(f"success rate      {agg['success_rate']:.3f}  (95% CI {lo:.2f}-{hi:.2f})")
    print(f"average targets   {_fmt(agg['mean_targets'])}")
    print(f"average survival  {_fmt(agg['mean_survival_time'])} s (world)")
    print(f"mean latency      {_fmt(agg['mean_latency_ms'], 3)} ms")
    print(f"p95 latency       {_fmt(agg['p95_latency_ms'], 3)} ms")
    print(f"missed slots/ep   {_fmt(agg['mean_missed_slots'])}")
    print(f"failure reasons   {agg['failure_reasons'] or '-'}")


def _level_setup(args, base_cfg, param: str):
    def make_config(level: float):
        if param == "latency_ms":
            return base_cfg
        v = int(round(level)) if param in INTEGER_PARAMS else level
        extra = {}
        if param == "obstacle_speed_max" and v < base_cfg.obstacle_speed_min:
            extra["obstacle_speed_min"] = v
        return replace(base_cfg, **{param: v}, **extra)

    def make_ctrl(level: float):
        lat = level if param == "latency_ms" else None
        return controller_factory(args.controller, args, latency_ms=lat)

    return make_config, make_ctrl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arena.benchmark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=HEADLESS_CONTROLLERS, default="simple_avoid")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="base seed; episode i uses seed+i")
    ap.add_argument("--out", default="runs", help="output root directory")
    ap.add_argument("--no-save", action="store_true", help="do not write anything to disk")
    ap.add_argument("--save-events", action="store_true", help="write full per-episode event logs")
    ap.add_argument("--no-observations", action="store_true", help="omit observations from event logs")
    ap.add_argument("--realtime", action="store_true", help="pace the simulation to the wall clock")
    ap.add_argument("--sweep", metavar="PARAM=V1,V2,...", help=f"sweep one of {SWEEPABLE}")
    ap.add_argument("--adaptive", metavar="PARAM", choices=SWEEPABLE, help="staircase search for failure frontier")
    ap.add_argument("--pass-rate", type=float, default=0.5, help="adaptive: success rate that counts as pass")
    ap.add_argument("--start", type=float, default=1.0, help="adaptive: start level")
    ap.add_argument("--max-level", type=float, default=64.0)
    ap.add_argument("--refine-steps", type=int, default=4)
    ap.add_argument("--quiet", action="store_true")
    add_difficulty_args(ap)
    args = ap.parse_args(argv)

    base_cfg = config_from_args(args)
    out_dir = None if args.no_save else new_run_dir(args.out, args.controller, "bench")
    summary: dict[str, Any] = {
        "controller": args.controller,
        "base_seed": args.seed,
        "episodes": args.episodes,
        "latency_ms": args.latency_ms,
        "real_latency": args.real_latency,
        "config": base_cfg.to_dict(),
        "argv": sys.argv[1:] if argv is None else argv,
    }
    t0 = time.perf_counter()

    def progress(i, r):
        if not args.quiet:
            mark = "ok " if r["success"] else "FAIL"
            print(f"  ep {i:4d} seed {r['seed']:6d} {mark} {r['reason']:<14} "
                  f"t={r['survival_time']:6.2f}s targets={r['targets_collected']:3d}", flush=True)

    if args.adaptive:
        param = args.adaptive
        make_config, make_ctrl = _level_setup(args, base_cfg, param)

        def lvl_progress(rec):
            print(f"  {param}={rec['level']:g}: success {rec['success_rate']:.2f} "
                  f"targets {rec['mean_targets']:.1f} survival {rec['mean_survival_time']:.1f}s", flush=True)

        fr = find_frontier(param, make_config, make_ctrl, start=args.start, episodes_per_level=args.episodes,
                           pass_rate=args.pass_rate, max_level=args.max_level, refine_steps=args.refine_steps,
                           integer=param in INTEGER_PARAMS, base_seed=args.seed, progress=lvl_progress)
        summary["adaptive"] = fr.to_dict()
        print(f"\nfailure frontier for {param} (pass = success >= {args.pass_rate}):")
        print(f"  highest passing level: {_fmt(fr.highest_pass)}")
        print(f"  lowest failing level:  {_fmt(fr.lowest_fail)}")
    elif args.sweep:
        param, _, vals = args.sweep.partition("=")
        if param not in SWEEPABLE:
            ap.error(f"--sweep param must be one of {SWEEPABLE}")
        levels = [float(v) for v in vals.split(",") if v]
        make_config, make_ctrl = _level_setup(args, base_cfg, param)
        rows = []
        for lv in levels:
            res = run_episodes(make_ctrl(lv), make_config(lv), args.episodes, args.seed)
            agg = aggregate(res)
            rows.append({"level": lv, **agg})
            print(f"  {param}={lv:g}: success {agg['success_rate']:.2f} targets {agg['mean_targets']:.1f} "
                  f"survival {agg['mean_survival_time']:.1f}s latency {_fmt(agg['mean_latency_ms'], 3)}ms",
                  flush=True)
        curve = [(r["level"], r["success_rate"]) for r in rows]
        summary["sweep"] = {"param": param, "levels": rows,
                            "D90": threshold_level(curve, 0.9), "D50": threshold_level(curve, 0.5),
                            "D10": threshold_level(curve, 0.1)}
        print(f"\n{param}: D90={_fmt(summary['sweep']['D90'])} D50={_fmt(summary['sweep']['D50'])} "
              f"D10={_fmt(summary['sweep']['D10'])}  (interpolated; lower bound if never crossed)")
    else:
        events_dir = out_dir / "episodes" if (out_dir and args.save_events) else None
        results = run_episodes(controller_factory(args.controller, args), base_cfg, args.episodes, args.seed,
                               events_dir=events_dir, record_observations=not args.no_observations,
                               realtime=args.realtime, progress=progress)
        agg = aggregate(results)
        summary["aggregate"] = agg
        print_summary(f"{args.controller}  obstacles={base_cfg.obstacle_count} "
                      f"world_speed={base_cfg.world_speed_scale:g}x decision_hz={base_cfg.decision_hz:g}", agg)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "episodes.jsonl", "w") as f:
                for r in results:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")

    summary["wall_time_s"] = time.perf_counter() - t0
    print(f"wall time         {summary['wall_time_s']:.1f} s")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"saved to          {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
