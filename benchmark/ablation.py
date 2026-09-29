"""Observation ablation on operationally solvable, paired episodes.

Question: when a controller fails, at which layer does it fail? The same
controller runs under three observation modes (``arena.observation_views``):

    raw       canonical state: the controller does geometry, prediction, policy
    relative  + exact ego-centric coordinate transforms
    physics   + linear closest-approach projection

and a *reference* controller establishes, per seed, that the episode is
operationally solvable: completable under the very same conditions.

Operationally solvable seed: the reference controller (default SimpleAvoid,
a feasibility baseline, not an oracle) succeeds on that seed in every one of
``--reference-repeats`` runs, with the same difficulty config, world speed,
decision Hz, max inflight, action space, episode duration, observation mode
and timing as the candidate, and with a latency matched to the candidate's
(``--match-latency``: fixed simulated latency in ms, or ``auto`` = the median
of a short, unscored probe of the candidate's real latency). Only those seeds
are evaluated, so a candidate failure is a failure in an environment shown
to be feasible. This says nothing about seeds the reference fails.

Evaluation is closed-loop outcome only (success, collision, timeout,
survival, targets). Reference actions are never ground truth and never reach
the candidate: reference and candidate run in separate episodes.

    python -m benchmark.ablation --controller jev --modes raw,relative,physics \\
        --episodes 30 --preset easy --world-speed 0.5 --reference simple_avoid --match-latency 190

One mode (``--modes raw``) is the plain paired-solvable benchmark.
``--timing lockstep`` repeats the experiment with latency taken out (the world
waits for every answer; ``--interval`` pins the decision rate), which
separates latency failures from decision failures.

Artifact: runs/<stamp>_ablation_<controller>/
    ablation.json   manifest: experiment_type, controller, reference, modes, timing,
                    candidate seeds, config, latency configuration, status, progress
    results.jsonl   one line per episode: role (reference/candidate), mode, seed, repeat, raw result
    summary.json    per mode: qualified seeds, reference and candidate aggregates, per-seed
                    pairs, latency match; plus the capability decomposition across modes
    episodes/<mode>/<role>/episode_<i>_seed<s>[_r<k>]/{config.json, events.jsonl, result.json}
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
from typing import Any, Callable, Optional

from arena.action import Action
from arena.cli import add_difficulty_args, config_from_args, controller_factory
from arena.dotenv import load_dotenv
from arena.environment import Environment
from arena.observation_views import MODES
from arena.recorder import JsonlRecorder, new_run_dir
from arena.stats import mean, percentile
from benchmark.metrics import aggregate
from benchmark.runner import run_episode
from benchmark.suite import parse_spec

VERSION = 1
LATENCY_MATCH_TOLERANCE = (0.75, 1.333)  # candidate p50 / reference latency outside this is flagged


class Cancelled(Exception):
    pass


def _atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def probe_latency_ms(factory: Callable, config, seed: int, n: int, timeout_s: float = 30.0) -> list[float]:
    """Wall latency of ``n`` sequential, unscored requests on real observations (nothing is executed)."""
    env = Environment(config, seed)
    ctrl = factory()
    out = []
    try:
        ctrl.reset(env.public_info())
        for i in range(n):
            t0 = time.perf_counter()
            ctrl.request(env.observe(), i)
            d = None
            while d is None and time.perf_counter() - t0 < timeout_s:
                d = ctrl.poll()
                if d is None:
                    time.sleep(0.0005)
            if d is not None and d.action is not None:
                out.append((time.perf_counter() - t0) * 1000 + d.extra_latency_s * 1000)
            for _ in range(6):  # a slightly different snapshot each time
                env.step(Action.STAY)
    finally:
        ctrl.close()
    return out


def run_one(factory: Callable, name: str, config, seed: int, ep_dir: Path, timing: str,
            interval_s: float, delay_s: float) -> dict[str, Any]:
    rec = JsonlRecorder(ep_dir, record_observations=False)
    if timing == "lockstep":
        from arena.lockstep import LockstepRunner

        ctrl = factory()
        try:
            return LockstepRunner(Environment(config, seed), (name, ctrl), None, interval_s, delay_s, rec).run()
        finally:
            ctrl.close()
            rec.close()
    return run_episode(factory, config, seed, recorder=rec)


def _slim(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k != "action_changes"}


def _lat(rows: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    p50 = [r["p50_latency_ms"] for r in rows if r.get("p50_latency_ms") is not None]
    if not p50:  # lockstep results: per-controller latency block
        p50 = [r["latency_ms"][r["controller"]]["p50"] for r in rows
               if isinstance(r.get("latency_ms"), dict) and r["latency_ms"].get(r["controller"], {}).get("p50") is not None]
    return {"p50_ms": percentile(p50, 50) if p50 else None, "mean_ms": mean([r["mean_decision_latency_ms"] for r in rows
                                                                           if r.get("mean_decision_latency_ms") is not None])}


def _paired_delta(a: dict[int, bool], b: dict[int, bool]) -> dict[str, Any]:
    """Success-rate change from a to b on the seeds both evaluated (same seeds, paired)."""
    seeds = sorted(set(a) & set(b))
    n = len(seeds)
    sa, sb = sum(a[s] for s in seeds), sum(b[s] for s in seeds)
    return {
        "seeds": n,
        "from_success": sa, "to_success": sb,
        "delta_pp": None if not n else 100.0 * (sb - sa) / n,
        "gained": sum(1 for s in seeds if b[s] and not a[s]),  # failed before, succeeds now
        "lost": sum(1 for s in seeds if a[s] and not b[s]),
    }


def build_summary(m: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes_out: dict[str, Any] = {}
    cand_success: dict[str, dict[int, bool]] = {}
    for mode in m["modes"]:
        ref = [r for r in rows if r["role"] == "reference" and r["mode"] == mode]
        cand = [r for r in rows if r["role"] == "candidate" and r["mode"] == mode]
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for r in ref:
            by_seed.setdefault(r["seed"], []).append(r)
        complete = {s: rs for s, rs in by_seed.items() if len(rs) == m["reference_repeats"]}
        qualified = sorted(s for s, rs in complete.items() if all(x["success"] for x in rs))
        q_rows = [r for r in cand if r["seed"] in qualified]
        u_rows = [r for r in cand if r["seed"] not in qualified]
        first_ref = [rs[0] for rs in complete.values()]
        cand_success[mode] = {r["seed"]: bool(r["success"]) for r in q_rows}
        lat = _lat(q_rows)
        ref_ms = m["latency"]["reference_ms"]
        ratio = (lat["p50_ms"] / ref_ms) if (lat["p50_ms"] and ref_ms) else None
        modes_out[mode] = {
            "candidate_seeds": len(complete),
            "qualified_seeds": qualified,
            "qualified": len(qualified),
            "reference_all": aggregate(first_ref) if first_ref else None,
            "reference_qualified": aggregate([rs[0] for s, rs in complete.items() if s in qualified]) if qualified else None,
            "candidate": aggregate(q_rows) if q_rows else None,
            "candidate_unqualified": aggregate(u_rows) if u_rows else None,
            "candidate_latency": lat,
            "reference_latency": _lat(first_ref),
            "latency_match": {
                "reference_ms": ref_ms, "candidate_p50_ms": lat["p50_ms"], "ratio": ratio,
                "within_tolerance": None if ratio is None else LATENCY_MATCH_TOLERANCE[0] <= ratio <= LATENCY_MATCH_TOLERANCE[1],
            },
            "pairs": [{"seed": s,
                       "reference": [x["reason"] for x in complete[s]],
                       "candidate": next((r["reason"] for r in q_rows if r["seed"] == s), None),
                       "candidate_survival": next((r["survival_time"] for r in q_rows if r["seed"] == s), None),
                       "candidate_targets": next((r["targets_collected"] for r in q_rows if r["seed"] == s), None)}
                      for s in qualified],
        }
    # Capability decomposition: consecutive modes on the seeds both evaluated, then the last mode vs
    # the reference on its qualified seeds (where the reference succeeds by construction).
    steps = []
    order = [md for md in MODES if md in m["modes"]]
    for a, b in zip(order, order[1:]):
        steps.append({"from": a, "to": b, **_paired_delta(cand_success[a], cand_success[b])})
    if order:
        last = order[-1]
        steps.append({"from": last, "to": "reference",
                      **_paired_delta(cand_success[last], {s: True for s in modes_out[last]["qualified_seeds"]})})
    return {"experiment_type": m["experiment_type"], "controller": m["controller"],
            "reference_controller": m["reference_controller"], "timing": m["timing"], "modes": modes_out,
            "decomposition": steps, "latency": m["latency"]}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmark.ablation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", default="jev", help="candidate controller spec (name or name+NNNms)")
    ap.add_argument("--modes", default="raw,relative,physics", help=f"subset of {','.join(MODES)}")
    ap.add_argument("--reference", default="simple_avoid", help="feasibility baseline (not an oracle)")
    ap.add_argument("--reference-repeats", type=int, default=1,
                    help="a seed qualifies only if the reference succeeds in all repeats")
    ap.add_argument("--match-latency", default="auto",
                    help="reference latency: fixed ms (simulated), or 'auto' = median of a probe of the candidate")
    ap.add_argument("--probe-requests", type=int, default=12, help="requests in the auto latency probe")
    ap.add_argument("--episodes", type=int, default=30, help="candidate seeds (seed .. seed+episodes-1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--also-unqualified", action="store_true",
                    help="also run the candidate on seeds the reference failed (reported separately)")
    ap.add_argument("--timing", choices=["realtime", "lockstep"], default="realtime")
    ap.add_argument("--interval", type=float, default=0.1, help="lockstep: world seconds between decisions")
    ap.add_argument("--delay", type=float, default=0.0, help="lockstep: fixed world-time delay for both")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    add_difficulty_args(ap)
    args = ap.parse_args(argv)
    load_dotenv()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    if not modes or any(md not in MODES for md in modes) or len(set(modes)) != len(modes):
        ap.error(f"--modes must be distinct values from {','.join(MODES)}")
    modes = [md for md in MODES if md in modes]
    try:
        cand_name, cand_extra = parse_spec(args.controller)
        ref_name, ref_extra = parse_spec(args.reference)
    except ValueError as e:
        ap.error(str(e))
    if ref_extra:
        ap.error("give the reference latency with --match-latency, not as +ms")
    if args.reference_repeats < 1:
        ap.error("--reference-repeats must be >= 1")
    lockstep = args.timing == "lockstep"
    if lockstep and cand_extra:
        ap.error("added latency is meaningless in lockstep timing")
    cfg = config_from_args(args)
    if lockstep:
        cfg = cfg.with_overrides(world_speed_scale=1.0, decision_hz=1.0 / args.interval, max_inflight=1)

    def factory(name: str, mode: str, latency_ms: float):
        ns = SimpleNamespace(**{**vars(args), "controller": name, "observation_mode": mode,
                                "latency_ms": latency_ms, "real_latency": False})
        return controller_factory(name, ns, latency_ms=latency_ms)

    # ---- latency configuration (realtime only: lockstep charges no latency to anyone)
    probe: list[float] = []
    if lockstep:
        ref_ms = 0.0
        how = "none (lockstep: latency costs no world time)"
    elif args.match_latency == "auto":
        probe = probe_latency_ms(factory(cand_name, modes[0], cand_extra), cfg, args.seed, args.probe_requests)
        if not probe:
            ap.error("latency probe got no answers from the candidate; give --match-latency explicitly")
        ref_ms = float(percentile(probe, 50))
        how = f"auto: median of {len(probe)} probe requests"
    else:
        try:
            ref_ms = float(args.match_latency)
        except ValueError:
            ap.error("--match-latency must be a number of ms or 'auto'")
        if ref_ms < 0:
            ap.error("--match-latency must be >= 0")
        how = "fixed"

    out = Path(args.run_dir) if args.run_dir else new_run_dir(args.out, "ablation", cand_name)
    if args.run_dir and out.exists() and any(out.iterdir()):
        ap.error(f"--run-dir {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + i for i in range(args.episodes)]
    total = len(modes) * len(seeds) * args.reference_repeats
    m: dict[str, Any] = {
        "kind": "ablation", "version": VERSION,
        "experiment_type": "observation_ablation" if len(modes) > 1 else "paired_solvable",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "running", "pid": os.getpid(),
        "controller": args.controller, "reference_controller": ref_name,
        "reference_repeats": args.reference_repeats, "modes": modes, "timing": args.timing,
        **({"interval_s": args.interval, "delay_s": args.delay} if lockstep else {}),
        "candidate_seeds": seeds, "config": cfg.to_dict(),
        "latency": {"reference_ms": ref_ms, "reference_kind": "simulated_fixed" if not lockstep else "none",
                    "how": how, "candidate_added_ms": cand_extra, "probe_ms": probe,
                    "jitter_matched": False},  # a fixed latency has no jitter; trace replay is future work
        "also_unqualified": args.also_unqualified,
        "argv": sys.argv[1:] if argv is None else argv,
        "progress": {"stage": "reference", "mode": None, "done": 0, "total": total},
        "updated": time.time(),
    }

    def save():
        m["updated"] = time.time()
        _atomic(out / "ablation.json", m)

    def on_term(signum, frame):
        raise Cancelled()

    prev = signal.signal(signal.SIGTERM, on_term)
    save()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    try:
        with open(out / "results.jsonl", "w") as rf:
            def emit(row):
                rows.append(row)
                rf.write(json.dumps(row, separators=(",", ":")) + "\n")
                rf.flush()

            for mode in modes:
                m["progress"].update(stage="reference", mode=mode)
                save()
                ref_f = factory(ref_name, mode, ref_ms)
                qualified = []
                for i, seed in enumerate(seeds):
                    ok = True
                    for k in range(args.reference_repeats):
                        rel = Path("episodes") / mode / "reference" / (
                            f"episode_{i:04d}_seed{seed}" + (f"_r{k}" if args.reference_repeats > 1 else ""))
                        r = run_one(ref_f, ref_name, cfg, seed, out / rel, args.timing, args.interval, args.delay)
                        emit({**_slim(r), "role": "reference", "mode": mode, "seed": seed, "repeat": k,
                              "controller_spec": ref_name, "latency_config_ms": ref_ms, "episode_dir": str(rel)})
                        ok = ok and bool(r["success"])
                        m["progress"]["done"] += 1
                    if ok:
                        qualified.append(seed)
                    save()
                say(f"[{mode}] reference {ref_name} @ {ref_ms:.0f} ms qualified {len(qualified)} / {len(seeds)} seeds")

                todo = seeds if args.also_unqualified else qualified
                m["progress"].update(stage="candidate", mode=mode, cand_done=0, cand_total=len(todo))
                save()
                cand_f = factory(cand_name, mode, cand_extra)
                for seed in todo:
                    i = seeds.index(seed)
                    rel = Path("episodes") / mode / "candidate" / f"episode_{i:04d}_seed{seed}"
                    r = run_one(cand_f, args.controller, cfg, seed, out / rel, args.timing, args.interval, args.delay)
                    emit({**_slim(r), "role": "candidate", "mode": mode, "seed": seed, "repeat": 0,
                          "controller_spec": args.controller, "qualified": seed in qualified,
                          "latency_config_ms": cand_extra, "episode_dir": str(rel)})
                    m["progress"]["cand_done"] += 1
                    save()
                _atomic(out / "summary.json", build_summary(m, rows))
    except (Cancelled, KeyboardInterrupt):
        m["status"] = "cancelled"
        _atomic(out / "summary.json", build_summary(m, rows))
        save()
        print(f"cancelled; partial results kept in {out}", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, prev)

    s = build_summary(m, rows)
    _atomic(out / "summary.json", s)
    m["status"] = "complete"
    m["progress"]["stage"] = "done"
    m["wall_time_s"] = time.perf_counter() - t0
    save()
    if not args.quiet:
        print_report(s)
        print(f"\nsaved to {out}")
    return 0


def print_report(s: dict[str, Any]) -> None:
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    print(f"\nObservation ablation · {s['controller']} · reference {s['reference_controller']} · {s['timing']}")
    for mode, v in s["modes"].items():
        c = v["candidate"]
        print(f"\n{mode.upper()}")
        print(f"  reference qualified seeds  {v['qualified']} / {v['candidate_seeds']}")
        if c:
            lo, hi = c["success_rate_ci95"]
            print(f"  success                    {c['successes']} / {c['episodes']}  ({f(c['success_rate'])}, 95% CI {f(lo)}-{f(hi)})")
            print(f"  avg survival               {f(c['mean_survival_time'], 1)} s")
            print(f"  targets / episode          {f(c['mean_targets'], 1)}")
            print(f"  failure reasons            {c['failure_reasons']}")
        lm = v["latency_match"]
        print(f"  latency p50                {f(v['candidate_latency']['p50_ms'], 0)} ms (reference {f(lm['reference_ms'], 0)} ms"
              + ("" if lm["within_tolerance"] in (None, True) else ", MISMATCH") + ")")
    print("\nCapability decomposition (paired seeds, success-rate change)")
    for st in s["decomposition"]:
        delta = "-" if st["delta_pp"] is None else f"{st['delta_pp']:+.1f} pp"
        print(f"  {st['from']:>8} -> {st['to']:<9} {delta}"
              f"  on {st['seeds']} seeds (gained {st['gained']}, lost {st['lost']})")


if __name__ == "__main__":
    raise SystemExit(main())
