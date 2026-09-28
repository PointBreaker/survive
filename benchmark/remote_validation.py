"""Validate the out-of-process path: does real remote latency behave exactly
like simulated latency?

For each latency level L:
  * remote:    start ``remote.fake_server`` as a separate OS process whose
               policy is SimpleAvoid and which holds each reply for L ms of
               real wall time; run JevController against it.
  * simulated: run in-process SimpleAvoid behind LatencyWrapper(L), which
               charges L in physics ticks without sleeping.
Both use the same seeds. Where the effective delay in ticks
(``request_tick + ceil(latency / tick)``) is the same, the two runs must be
bit-identical, since same observations -> same policy -> same actions. So the
report includes per-seed agreement, not just matching averages.

    python -m benchmark.remote_validation --latencies 50,100,150,200 --episodes 10

Remote episodes run in real time while a request is in flight, so this
takes roughly (latency share x world duration) of wall time per episode.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from arena.difficulty import PRESETS
from arena.recorder import new_run_dir
from benchmark.metrics import aggregate
from benchmark.runner import run_episodes
from controllers.base import LatencyWrapper
from controllers.jev import JevController
from controllers.simple_avoid import SimpleAvoidController

ROOT = Path(__file__).resolve().parents[1]
COMPARE_KEYS = ("success", "reason", "targets_collected", "ticks")


def start_server(latency_ms: float, jitter_ms: float) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "remote.fake_server", "--port", "0", "--policy", "simple_avoid",
         "--latency-ms", str(latency_ms), "--jitter-ms", str(jitter_ms)],
        cwd=ROOT, stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if not line.startswith("LISTENING "):
        proc.kill()
        raise RuntimeError(f"fake server failed to start: {line!r}")
    return proc, line.split(" ", 1)[1]


def compare_level(latency_ms: float, cfg, episodes: int, seed: int, jitter_ms: float) -> dict[str, Any]:
    proc, url = start_server(latency_ms, jitter_ms)
    try:
        t0 = time.perf_counter()
        remote = run_episodes(lambda: JevController(endpoint=url), cfg, episodes, seed)
        remote_wall = time.perf_counter() - t0
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    t0 = time.perf_counter()
    sim = run_episodes(lambda: LatencyWrapper(SimpleAvoidController(), latency_ms), cfg, episodes, seed)
    sim_wall = time.perf_counter() - t0

    identical = sum(all(r[k] == s[k] for k in COMPARE_KEYS) for r, s in zip(remote, sim))
    same_success = sum(r["success"] == s["success"] for r, s in zip(remote, sim))
    overhead = [r["mean_decision_latency_ms"] - latency_ms for r in remote if r["mean_decision_latency_ms"]]

    def delay(rs):
        xs = [r["mean_delay_ticks"] for r in rs if r["mean_delay_ticks"] is not None]
        return sum(xs) / len(xs) if xs else None

    return {
        "latency_ms": latency_ms,
        "remote": aggregate(remote),
        "simulated": aggregate(sim),
        "remote_mean_delay_ticks": delay(remote),
        "simulated_mean_delay_ticks": delay(sim),
        "remote_overhead_ms": sum(overhead) / len(overhead) if overhead else None,
        "remote_failed_decisions": sum(r["failed_decisions"] for r in remote),
        "identical_episodes": identical,
        "same_success": same_success,
        "episodes": episodes,
        "remote_wall_s": remote_wall,
        "simulated_wall_s": sim_wall,
        "per_seed": [
            {"seed": r["seed"], **{f"remote_{k}": r[k] for k in COMPARE_KEYS},
             **{f"sim_{k}": s[k] for k in COMPARE_KEYS}}
            for r, s in zip(remote, sim)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latencies", default="50,100,150,200")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-duration", type=float, default=20.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    cfg = replace(PRESETS[args.preset], max_duration=args.max_duration)
    levels = [float(x) for x in args.latencies.split(",") if x]
    rows = []
    hdr = (f"{'lat ms':>6} | {'remote succ':>11} {'targets':>7} {'surv':>6} {'lat':>7} {'dly':>5} | "
           f"{'sim succ':>8} {'targets':>7} {'surv':>6} {'lat':>7} {'dly':>5} | {'identical':>9} {'fail':>4}")
    print(hdr)
    print("-" * len(hdr))
    for lv in levels:
        row = compare_level(lv, cfg, args.episodes, args.seed, args.jitter_ms)
        rows.append(row)
        r, s = row["remote"], row["simulated"]
        print(f"{lv:6g} | {r['success_rate']:11.2f} {r['mean_targets']:7.1f} {r['mean_survival_time']:6.1f} "
              f"{r['mean_latency_ms']:7.1f} {row['remote_mean_delay_ticks'] or 0:5.2f} | "
              f"{s['success_rate']:8.2f} {s['mean_targets']:7.1f} {s['mean_survival_time']:6.1f} "
              f"{s['mean_latency_ms']:7.1f} {row['simulated_mean_delay_ticks'] or 0:5.2f} | "
              f"{row['identical_episodes']:>4}/{row['episodes']:<4} {row['remote_failed_decisions']:4d}", flush=True)

    total = sum(r["identical_episodes"] for r in rows)
    n = sum(r["episodes"] for r in rows)
    ov = [r["remote_overhead_ms"] for r in rows if r["remote_overhead_ms"] is not None]
    print(f"\nidentical episodes: {total}/{n}   mean transport overhead: {sum(ov) / len(ov):.2f} ms")
    print("(lat = measured mean latency; dly = mean effective delay in physics ticks)")
    if not args.no_save:
        out = new_run_dir(args.out, "jev", "remote_validation")
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(
            {"config": cfg.to_dict(), "jitter_ms": args.jitter_ms, "levels": rows}, indent=2))
        print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
