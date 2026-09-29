"""Benchmark suite: several controllers x one swept parameter x paired seeds.

This is orchestration only. Every episode runs through
``benchmark.runner.run_episodes`` (the same EpisodeRunner, rules and latency
accounting as ``python -m arena.benchmark``); levels are applied exactly as
``arena.benchmark --sweep`` applies them. What it adds is a complete,
self-describing artifact for analysis tools (e.g. the web dashboard):

    runs/<stamp>_suite_<param>/
        suite.json      manifest: controllers, param, levels, seeds, base config, status
        results.jsonl   one line per episode (raw result fields + controller/level/seed)
        summary.json    per controller: per-level aggregate (benchmark.metrics.aggregate)
                        and empirical thresholds D90/D50/D10 (benchmark.metrics.threshold_summary)
        episodes/<controller>/<level>/seed<k>/{config.json, events.jsonl, result.json}

Episode events are written without observations by default (decisions,
latencies, targets, collisions); ``--observations`` keeps them. Replays do
not need observations: an episode is re-simulated exactly from config + seed
+ the logged per-tick actions.

    python -m benchmark.suite --controllers jev,simple_avoid,greedy,random \\
        --param world_speed_scale --levels 0.25,0.5,1,2,4,8 --episodes 20

Controller specs: a registered name, optionally with added simulated latency,
e.g. ``simple_avoid+310ms`` (a latency-matched baseline).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from arena.benchmark import INTEGER_PARAMS, SWEEPABLE, _level_setup
from arena.cli import add_difficulty_args, config_from_args, controller_factory
from arena.dotenv import load_dotenv
from arena.recorder import new_run_dir
from benchmark.metrics import aggregate, threshold_summary
from benchmark.runner import run_episodes

SUITE_VERSION = 1
THRESHOLDS = {"D90": 0.9, "D50": 0.5, "D10": 0.1}
_SPEC = re.compile(r"^([a-z_]+)(?:\+(\d+(?:\.\d+)?)ms)?$")


def parse_spec(spec: str) -> tuple[str, float]:
    m = _SPEC.match(spec.strip())
    if not m:
        raise ValueError(f"bad controller spec {spec!r} (expected name or name+NNNms)")
    return m.group(1), float(m.group(2) or 0.0)


def level_key(level: float) -> str:
    return f"{level:g}"


def build_summary(manifest: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"param": manifest["param"], "levels": manifest["levels"], "controllers": {}}
    for spec in manifest["controllers"]:
        points = []
        for lv in manifest["levels"]:
            rs = [r for r in results if r["controller_spec"] == spec and r["level"] == lv]
            if rs:
                points.append({"level": lv, **aggregate(rs)})
        curve = [(p["level"], p["success_rate"]) for p in points]
        out["controllers"][spec] = {
            "points": points,
            "thresholds": {k: threshold_summary(curve, p) for k, p in THRESHOLDS.items()},
        }
    return out


class Cancelled(Exception):
    pass


def _atomic_write(path: Path, data: Any) -> None:
    """Readers (the dashboard) never see a half-written JSON file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmark.suite", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controllers", default="simple_avoid,greedy,random",
                    help="comma-separated controller specs, e.g. jev,simple_avoid,simple_avoid+300ms")
    ap.add_argument("--param", choices=SWEEPABLE, default="world_speed_scale")
    ap.add_argument("--levels", default="0.25,0.5,1,2,4,8")
    ap.add_argument("--episodes", type=int, default=20, help="episodes per (controller, level) point")
    ap.add_argument("--seed", type=int, default=0, help="base seed; episode i uses seed+i at every point")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--run-dir", default=None, help="exact output directory (must not exist or be empty)")
    ap.add_argument("--observations", action="store_true", help="also log full observations per request")
    ap.add_argument("--quiet", action="store_true")
    add_difficulty_args(ap)
    args = ap.parse_args(argv)
    load_dotenv()

    specs = [s.strip() for s in args.controllers.split(",") if s.strip()]
    levels = [float(v) for v in args.levels.split(",") if v.strip()]
    if args.param in INTEGER_PARAMS:
        levels = [float(int(round(v))) for v in levels]
    base_cfg = config_from_args(args)

    factories = {}
    for spec in specs:
        name, extra = parse_spec(spec)
        ns = SimpleNamespace(**{**vars(args), "controller": name, "latency_ms": args.latency_ms + extra})
        factories[spec] = (name, ns, extra)
        controller_factory(name, ns)  # fail fast (e.g. missing Jev token) before any episode runs

    if args.run_dir:
        out_dir = Path(args.run_dir)
        if out_dir.exists() and any(out_dir.iterdir()):
            ap.error(f"--run-dir {out_dir} is not empty")
    else:
        out_dir = new_run_dir(args.out, "suite", args.param)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_episodes = len(levels) * len(specs) * args.episodes
    manifest: dict[str, Any] = {
        "kind": "suite",
        "version": SUITE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "running",
        "pid": os.getpid(),
        "param": args.param,
        "levels": levels,
        "controllers": specs,
        "episodes_per_point": args.episodes,
        "base_seed": args.seed,
        "seeds": [args.seed + i for i in range(args.episodes)],
        "config": base_cfg.to_dict(),
        "added_latency_ms": args.latency_ms,
        "observations_logged": bool(args.observations),
        "argv": sys.argv[1:] if argv is None else argv,
        "progress": {"episodes_done": 0, "episodes_total": total_episodes, "points_done": 0,
                     "points_total": len(levels) * len(specs), "current": None},
        "updated": time.time(),
    }
    last_write = [0.0]

    def write_manifest(force: bool = True):
        now = time.time()
        if force or now - last_write[0] > 0.5:
            manifest["updated"] = now
            _atomic_write(out_dir / "suite.json", manifest)
            last_write[0] = now

    def on_sigterm(signum, frame):
        raise Cancelled()

    prev_handler = signal.signal(signal.SIGTERM, on_sigterm) if hasattr(signal, "SIGTERM") else None
    write_manifest()
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        with open(out_dir / "results.jsonl", "w") as rf:
            for lv in levels:
                for spec in specs:
                    name, ns, extra = factories[spec]
                    make_config, make_ctrl = _level_setup(ns, base_cfg, args.param)
                    # latency_ms as the swept param adds to the spec's own latency
                    factory = (controller_factory(name, ns, latency_ms=lv + extra) if args.param == "latency_ms"
                               else make_ctrl(lv))
                    ep_dir = out_dir / "episodes" / spec / level_key(lv)
                    manifest["progress"]["current"] = {"controller": spec, "level": lv}
                    base_done = manifest["progress"]["episodes_done"]

                    def progress(i, r, base_done=base_done):
                        manifest["progress"]["episodes_done"] = base_done + i + 1
                        write_manifest(force=False)

                    rs = run_episodes(factory, make_config(lv), args.episodes, args.seed,
                                      events_dir=ep_dir, record_observations=args.observations, progress=progress)
                    for i, r in enumerate(rs):
                        rec = {k: v for k, v in r.items() if k != "action_changes"}
                        rec.update({"controller_spec": spec, "level": lv,
                                    "episode_dir": str((ep_dir / f"episode_{i:04d}_seed{r['seed']}").relative_to(out_dir))})
                        rf.write(json.dumps(rec, separators=(",", ":")) + "\n")
                        results.append(rec)
                    rf.flush()
                    agg = aggregate(rs)
                    if not args.quiet:
                        print(f"  {args.param}={lv:g}  {spec:<22} success {agg['success_rate']:.2f}  "
                              f"targets {agg['mean_targets']:.1f}  survival {agg['mean_survival_time']:.1f}s", flush=True)
                    _atomic_write(out_dir / "summary.json", build_summary(manifest, results))
                    manifest["progress"]["points_done"] += 1
                    write_manifest()
    except (Cancelled, KeyboardInterrupt):
        # Keep every completed point; the partial episode in flight is discarded.
        manifest["status"] = "cancelled"
        manifest["progress"]["current"] = None
        manifest["wall_time_s"] = time.perf_counter() - t0
        _atomic_write(out_dir / "summary.json", build_summary(manifest, results))
        write_manifest()
        print(f"\ncancelled; {manifest['progress']['points_done']} complete points kept in {out_dir}", flush=True)
        return 130
    finally:
        if prev_handler is not None:
            signal.signal(signal.SIGTERM, prev_handler)

    manifest["status"] = "complete"
    manifest["progress"]["current"] = None
    manifest["wall_time_s"] = time.perf_counter() - t0
    write_manifest()
    summary = build_summary(manifest, results)
    _atomic_write(out_dir / "summary.json", summary)
    print(f"\nsaved to {out_dir}")
    for spec, c in summary["controllers"].items():
        d50 = c["thresholds"]["D50"]
        print(f"  {spec:<22} D50 {args.param} = {d50['value']}  ({d50['kind']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
