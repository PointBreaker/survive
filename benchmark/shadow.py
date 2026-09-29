"""Shadow-mode decision benchmark (lockstep, latency-free by default).

One controller drives; the shadows answer the identical observation at every
decision point but never act. Decision rate is pinned (one decision per
``--interval`` world seconds for everyone) and the world waits for answers,
so latency does not affect what is measured. ``--delay`` optionally adds a
fixed world-time delay.

Decision quality is graded by counterfactual takeover branches (see
``benchmark.takeover``): every ``--branch-every`` world seconds along the
driver's trajectory, each controller takes over from an identical copy of the
state for ``--takeover`` world seconds and is judged on its own consequences.

The default driver ``explorer`` is a neutral, unscored state generator (see
``benchmark.takeover`` on driver bias).

    python -m benchmark.shadow --shadows jev,simple_avoid,greedy,random \\
        --episodes 10 --interval 0.1

Artifact: runs/<stamp>_shadow_<driver>/
    shadow.json    manifest (driver, shadows, interval, delay, config, seeds, status, progress)
    results.jsonl  one line per episode (driver outcome + per-controller latency/failures)
    episodes/episode_<i>_seed<s>/{config.json, events.jsonl, result.json, branches.jsonl}
    scores.json    takeover scores per controller (re-score: python -m benchmark.takeover <dir>)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from arena.cli import add_difficulty_args, config_from_args, controller_factory
from arena.dotenv import load_dotenv
from arena.environment import Environment
from arena.lockstep import LockstepRunner
from arena.recorder import JsonlRecorder, new_run_dir
from benchmark import takeover as tk


class Cancelled(Exception):
    pass


def _atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmark.shadow", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", default=tk.EXPLORER,
                    help="who acts in the episode (default: explorer, a neutral unscored state generator)")
    ap.add_argument("--shadows", default="simple_avoid,greedy,random",
                    help="comma-separated controllers that answer and take over in branches")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.1, help="world seconds between decision points")
    ap.add_argument("--delay", type=float, default=0.0, help="fixed world-time delay before an answer takes effect")
    ap.add_argument("--branch-every", type=float, default=1.0, help="world seconds between takeover branches")
    ap.add_argument("--takeover", type=float, default=3.0, help="world seconds each takeover branch lasts")
    ap.add_argument("--branch-workers", type=int, default=1,
                    help="takeovers run concurrently (use ~4-8 for network-bound controllers like jev)")
    ap.add_argument("--no-branches", action="store_true", help="skip takeover branches (answers/agreement only)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    add_difficulty_args(ap)
    args = ap.parse_args(argv)
    load_dotenv()

    names = [args.driver] + [s.strip() for s in args.shadows.split(",") if s.strip() and s.strip() != args.driver]
    for n in names:
        if "+" in n:
            ap.error(f"{n}: added latency is meaningless in lockstep (use --delay for a world-time delay)")
    ns = SimpleNamespace(**{**vars(args), "latency_ms": 0.0, "real_latency": False})
    factories = {n: controller_factory(n, SimpleNamespace(**{**vars(ns), "controller": n}))
                 for n in names if n != tk.EXPLORER}
    scored = [n for n in names if n != tk.EXPLORER]
    if not scored:
        ap.error("no controllers to score")

    # Lockstep: world time is decision-driven; decision_hz expresses the pinned world rate.
    cfg = config_from_args(args).with_overrides(world_speed_scale=1.0, decision_hz=1.0 / args.interval, max_inflight=1)

    if args.run_dir:
        out = Path(args.run_dir)
        if out.exists() and any(out.iterdir()):
            ap.error(f"--run-dir {out} is not empty")
    else:
        out = new_run_dir(args.out, "shadow", args.driver)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "kind": "shadow", "version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "running",
        "pid": os.getpid(), "driver": args.driver, "shadows": names[1:], "interval_s": args.interval,
        "delay_s": args.delay, "observation_mode": args.observation_mode, "episodes": args.episodes, "base_seed": args.seed,
        "seeds": [args.seed + i for i in range(args.episodes)], "config": cfg.to_dict(),
        "branch_every_s": None if args.no_branches else args.branch_every,
        "takeover_s": None if args.no_branches else args.takeover, "argv": sys.argv[1:] if argv is None else argv,
        "progress": {"episodes_done": 0, "episodes_total": args.episodes, "stage": "running",
                     "branches_done": 0, "branch_starts": 0},
        "updated": time.time(),
    }

    def save():
        manifest["updated"] = time.time()
        _atomic(out / "shadow.json", manifest)

    def on_term(signum, frame):
        raise Cancelled()

    prev = signal.signal(signal.SIGTERM, on_term)
    save()
    t0 = time.perf_counter()
    s = None
    try:
        with open(out / "results.jsonl", "w") as rf:
            for i in range(args.episodes):
                seed = args.seed + i
                ctrls = {n: f() for n, f in factories.items()}
                if args.driver == tk.EXPLORER:
                    ctrls = {tk.EXPLORER: tk.ExplorerDriver(seed), **ctrls}
                rec = JsonlRecorder(out / "episodes" / f"episode_{i:04d}_seed{seed}", record_observations=False)
                try:
                    r = LockstepRunner(Environment(cfg, seed), (args.driver, ctrls[args.driver]),
                                       {n: ctrls[n] for n in names[1:]}, args.interval, args.delay, rec).run()
                finally:
                    rec.close()
                    for c in ctrls.values():
                        c.close()
                ep_dir = out / "episodes" / f"episode_{i:04d}_seed{seed}"
                if not args.no_branches:
                    manifest["progress"].update(stage="branching", branches_done=0, branch_starts=0)
                    save()
                    def on_branch(done: int, total: int) -> None:
                        manifest["progress"].update(branches_done=done, branch_starts=total)
                        if done % 5 == 0:
                            save()
                    bs = tk.collect_branches(
                        ep_dir, cfg, seed, r["action_changes"], r["ticks"], factories,
                        tk.ticks(args.interval), round(args.delay * 60), tk.ticks(args.branch_every),
                        tk.ticks(args.takeover), episode=ep_dir.name, on_branch=on_branch,
                        workers=args.branch_workers)
                    r["branches"] = len(bs)
                    manifest["progress"]["stage"] = "running"
                row = {k: v for k, v in r.items() if k != "action_changes"}
                row["episode_dir"] = f"episodes/episode_{i:04d}_seed{seed}"
                rf.write(json.dumps(row, separators=(",", ":")) + "\n")
                rf.flush()
                manifest["progress"]["episodes_done"] = i + 1
                save()
                if not args.quiet:
                    print(f"  ep {i:3d} seed {seed:5d} {r['reason']:<14} t={r['survival_time']:6.1f}s "
                          f"points={r['decision_points']} branches={r.get('branches', 0)}", flush=True)
                # keep scores.json current so partial runs are inspectable
                s = tk.score_run(out)
        if not args.quiet and s:
            tk.print_scores(s)
    except (Cancelled, KeyboardInterrupt):
        manifest["status"] = "cancelled"
        if manifest["progress"]["episodes_done"]:
            tk.score_run(out)
        save()
        print(f"cancelled; {manifest['progress']['episodes_done']} episodes kept in {out}", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, prev)
    manifest["status"] = "complete"
    manifest["progress"]["stage"] = "done"
    manifest["wall_time_s"] = time.perf_counter() - t0
    save()
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
