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
import math
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from arena.action import Action
from arena.cli import add_difficulty_args, config_from_args, controller_factory
from arena.difficulty import DifficultyConfig
from arena.dotenv import load_dotenv
from arena.environment import Environment
from arena.observation_views import MODES
from arena.recorder import JsonlRecorder, new_run_dir
from arena.stats import mean, percentile
from benchmark.metrics import aggregate
from benchmark.suite import parse_spec

VERSION = 1
TICK_MS = 1000.0 / 60.0  # one physics tick of controller (wall-clock) time
LATENCY_MATCH_TOLERANCE = (0.75, 1.333)  # candidate p50 / reference latency outside this is flagged


class Cancelled(Exception):
    pass


MODE_BLURB = {"raw": "canonical state only", "relative": "+ ego-relative coordinates",
              "physics": "+ linear closest-approach projection"}


def fmt_speed(cfg, lockstep: bool) -> str:
    return "lockstep" if lockstep else f"{cfg.world_speed_scale:g}x world speed · {cfg.decision_hz:g} Hz"


class Aborted(Exception):
    """The candidate's service refuses requests (auth, billing): stop instead of recording no-op episodes."""


def _atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def probe_latency_ms(factory: Callable, config, seed: int, n: int, timeout_s: float = 30.0,
                     errors: Optional[list[str]] = None) -> list[float]:
    """Wall latency of ``n`` sequential, unscored requests on real observations (nothing is executed).

    One extra warm-up request goes first and is discarded (connection setup is not decision latency).
    Failed requests are not latency; their errors are appended to ``errors``.
    """
    env = Environment(config, seed)
    ctrl = factory()
    out = []
    try:
        ctrl.reset(env.public_info())
        for i in range(n + 1):
            t0 = time.perf_counter()
            ctrl.request(env.observe(), i)
            d = None
            while d is None and time.perf_counter() - t0 < timeout_s:
                d = ctrl.poll()
                if d is None:
                    time.sleep(0.0005)
            if d is not None and d.action is not None:
                if i > 0:
                    out.append((time.perf_counter() - t0) * 1000 + d.extra_latency_s * 1000)
            elif errors is not None:
                errors.append(str(d.error if d is not None else "timeout"))
                if is_fatal(errors[-1]):
                    break
            for _ in range(6):  # a slightly different snapshot each time
                env.step(Action.STAY)
    finally:
        ctrl.close()
    return out


def run_one(factory: Callable, name: str, config, seed: int, ep_dir: Path, timing: str,
            interval_s: float, delay_s: float, view=None) -> dict[str, Any]:
    """One episode in the unchanged measurement runners. With a live ``view`` the environment is an
    ObservedEnvironment (identical dynamics, read-only hook) and decision events are also forwarded."""
    from arena.runner import EpisodeRunner

    rec = JsonlRecorder(ep_dir, record_observations=False)
    if view is not None:
        from arena.live_view import ObservedEnvironment, TeeRecorder

        env = ObservedEnvironment(config, seed, view)
        rec = TeeRecorder(rec, view)
    else:
        env = Environment(config, seed)
    ctrl = factory()
    t0 = time.perf_counter()
    try:
        if timing == "lockstep":
            from arena.lockstep import LockstepRunner

            r = LockstepRunner(env, (name, ctrl), None, interval_s, delay_s, rec).run()
        else:
            r = EpisodeRunner(env, ctrl, recorder=rec).run()
    finally:
        ctrl.close()
        rec.close()
    r["wall_s"] = time.perf_counter() - t0
    return r


def _slim(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k != "action_changes"}


# Errors that mean the service will not answer at all (auth, billing, access): stop the run
# instead of recording episodes in which the controller never acted.
FATAL_ERROR_MARKERS = ("HTTP 401", "HTTP 402", "HTTP 403", "Insufficient credits", "No API token")
DEFAULT_MAX_FAILURE_RATE = 0.05


def is_fatal(error: Optional[str]) -> bool:
    return bool(error) and any(k.lower() in error.lower() for k in FATAL_ERROR_MARKERS)


def episode_validity(row: dict[str, Any], run_dir: Optional[Path] = None,
                     max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE) -> dict[str, Any]:
    """Did the controller actually control this episode? Failed decisions leave the previous action
    running, so an episode full of transport/API errors measures the service, not the controller."""
    if "failed_decisions" in row:  # real-time runner
        failed, applied = int(row.get("failed_decisions") or 0), int(row.get("decision_count") or 0)
    else:  # lockstep runner
        failed = int((row.get("failed_answers") or {}).get(row.get("controller"), 0))
        applied = max(0, int(row.get("decision_points") or 0) - failed)
    first_error = None
    ev = (Path(run_dir) / row["episode_dir"] / "events.jsonl") if run_dir and row.get("episode_dir") else None
    if failed and ev and ev.is_file():
        with open(ev) as f:
            for line in f:
                e = json.loads(line)
                err = e.get("error") if e.get("type") == "decision_failed" else None
                if e.get("type") == "lockstep_decision":
                    err = ((e.get("answers") or {}).get(row.get("controller")) or {}).get("error")
                if err:
                    first_error = str(err)[:300]
                    break
    total = failed + applied
    rate = failed / total if total else 0.0
    fatal = is_fatal(first_error)
    return {"failed_decisions": failed, "applied_decisions": applied, "failure_rate": rate,
            "first_error": first_error, "fatal_error": fatal,
            "valid": (rate <= max_failure_rate) and not fatal and (applied > 0 or total == 0)}


def latest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The last attempt per (role, mode, seed, repeat): --resume appends retries, never rewrites."""
    out: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        out[(r["role"], r["mode"], r["seed"], r.get("repeat", 0))] = r
    return list(out.values())


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


def build_summary(m: dict[str, Any], rows: list[dict[str, Any]], run_dir: Optional[Path] = None) -> dict[str, Any]:
    max_fail = m.get("max_failure_rate", DEFAULT_MAX_FAILURE_RATE)
    rows = latest(rows)
    for r in rows:
        if "valid" not in r:  # runs recorded before validity existed: derive it from the logs
            r.update(episode_validity(r, run_dir, max_fail))
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
        q_all = [r for r in cand if r["seed"] in qualified]
        q_rows = [r for r in q_all if r["valid"]]
        invalid = [r for r in q_all if not r["valid"]]
        u_rows = [r for r in cand if r["seed"] not in qualified and r["valid"]]
        n_failed = sum(r["failed_decisions"] for r in q_all)
        n_total = n_failed + sum(r["applied_decisions"] for r in q_all)
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
            "candidate": aggregate(q_rows) if q_rows else None,  # valid episodes only
            "evaluated": len(q_rows),
            "invalid_seeds": sorted(r["seed"] for r in invalid),
            "invalid_errors": sorted({(r["first_error"] or "")[:160] for r in invalid if r["first_error"]}),
            "decision_failure_rate": (n_failed / n_total) if n_total else None,
            "candidate_unqualified": aggregate(u_rows) if u_rows else None,
            "candidate_latency": lat,
            "reference_latency": _lat(first_ref),
            "latency_match": {
                "reference_ms": ref_ms, "candidate_p50_ms": lat["p50_ms"], "ratio": ratio,
                "within_tolerance": None if ratio is None else LATENCY_MATCH_TOLERANCE[0] <= ratio <= LATENCY_MATCH_TOLERANCE[1],
            },
            "pairs": [{"seed": s,
                       "reference": [x["reason"] for x in complete[s]],
                       "candidate_valid": next((r["valid"] for r in q_all if r["seed"] == s), None),
                       "candidate_failure_rate": next((r["failure_rate"] for r in q_all if r["seed"] == s), None),
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
            "decomposition": steps, "latency": m["latency"], "max_failure_rate": max_fail,
            "status": m.get("status"), "abort_reason": m.get("abort_reason"),
            "valid": all(not v["invalid_seeds"] and v["evaluated"] == v["qualified"] for v in modes_out.values())}


def estimate(rows, modes, seeds, cfg, repeats, lockstep, limit, max_fail, run_dir, probe_ms) -> dict[str, Any]:
    """What this invocation will run, and roughly how long it takes (before anything is spent)."""
    rows = latest(rows)
    todo = 0
    exact = True
    for mode in modes:
        refs = {(r["seed"], r.get("repeat", 0)): r for r in rows if r["role"] == "reference" and r["mode"] == mode}
        n = 0
        for seed in seeds:
            rs = [refs.get((seed, k)) for k in range(repeats)]
            if any(x is None for x in rs):
                exact = False  # qualification not known yet: count as if it qualifies (upper bound)
            elif not all(x["success"] for x in rs):
                continue
            c = next((r for r in reversed(rows) if r["role"] == "candidate" and r["mode"] == mode and r["seed"] == seed), None)
            if c is not None and c.get("valid", episode_validity(c, run_dir, max_fail)["valid"]):
                continue
            n += 1
        todo += min(n, limit) if limit is not None else n
    scale = 1.0 if lockstep else cfg.world_speed_scale
    max_wall = cfg.max_duration / scale
    prior = [r for r in rows if r["role"] == "candidate" and r.get("valid", True) and r.get("failed_decisions", 0) == 0]
    walls = [r["wall_s"] if r.get("wall_s") else r["survival_time"] / scale for r in prior]
    typical = mean(walls) if walls else None
    lat = percentile(probe_ms, 50) if probe_ms else None
    rate = min(cfg.decision_hz, 1000.0 / lat * cfg.max_inflight) if (lat and not lockstep) else cfg.decision_hz
    if lockstep and lat:
        rate = 1000.0 / lat
    return {"episodes": todo, "exact": exact, "typical_wall_s": typical, "max_wall_s": max_wall,
            "typical_total_s": None if typical is None else typical * todo, "max_total_s": max_wall * todo,
            "requests_per_s": rate, "latency_ms": lat, "from_prior": len(walls)}


def print_plan(p: dict[str, Any], controller: str) -> None:
    from arena.live_view import _dur

    n = p["episodes"]
    print(f"\nPlan: {'' if p['exact'] else 'up to '}{n} {controller} episode(s) (reference episodes are fast and not counted)")
    if n:
        typ = p["typical_total_s"]
        print(f"  time: at most {_dur(p['max_total_s'])} (every episode survives {_dur(p['max_wall_s'])})"
              + (f"; about {_dur(typ)} at the {_dur(p['typical_wall_s'])} per episode seen in {p['from_prior']} earlier episodes"
                 if typ else ""))
        lo = p["requests_per_s"] * (p["typical_total_s"] or p["max_total_s"])
        print(f"  requests: about {lo:,.0f}" + (f" (at {p['latency_ms']:.0f} ms each)" if p["latency_ms"] else "")
              + (", at most " + f"{p['requests_per_s'] * p['max_total_s']:,.0f}" if p["typical_total_s"] else ""))
        print("  tip: --limit 2 runs two episodes per mode first; continue with --resume <run dir>")


def _board(m: dict[str, Any], rows: list[dict[str, Any]], qualified: dict[str, list[int]], limit) -> list[dict[str, Any]]:
    board = []
    lr = latest(rows)
    for mode in m["modes"]:
        c = [r for r in lr if r["role"] == "candidate" and r["mode"] == mode]
        valid = [r for r in c if r.get("valid")]
        q = qualified.get(mode)
        board.append({"label": mode.upper(), "done": len(valid), "total": len(q) if q is not None else len(m["candidate_seeds"]),
                      "success": sum(1 for r in valid if r["success"]),
                      "collision": sum(1 for r in valid if r["reason"] == "collision"),
                      "timeout": sum(1 for r in valid if r["reason"] == "target_timeout"),
                      "invalid": sum(1 for r in c if not r.get("valid"))})
    return board


def _read_rows(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / "results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def rescore(run_dir: Path) -> dict[str, Any]:
    """Rebuild summary.json from the raw rows and logs (e.g. to apply validity to an older run)."""
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "ablation.json").read_text())
    s = build_summary(m, _read_rows(run_dir), run_dir)
    _atomic(run_dir / "summary.json", s)
    return s


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
    ap.add_argument("--max-failure-rate", type=float, default=DEFAULT_MAX_FAILURE_RATE,
                    help="a candidate episode with a larger share of failed decisions is invalid (excluded)")
    ap.add_argument("--resume", metavar="RUN_DIR", default=None,
                    help="continue a run: same manifest; runs missing episodes and re-runs invalid candidate ones")
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most N candidate episodes per mode now (continue the rest later with --resume)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--watch", dest="watch", action="store_true", default=None,
                   help="open the live viewer (default when a display is available)")
    g.add_argument("--no-watch", dest="watch", action="store_false", help="run without the live viewer")
    ap.add_argument("--pause", type=float, default=1.5,
                    help="with the viewer: seconds to show each candidate outcome before the next episode")
    ap.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation before starting")
    ap.add_argument("--rescore", metavar="RUN_DIR", default=None, help="only rebuild summary.json of a run")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    add_difficulty_args(ap)
    args = ap.parse_args(argv)
    load_dotenv()

    if args.rescore:
        s = rescore(Path(args.rescore))
        if not args.quiet:
            print_report(s)
        return 0

    rows: list[dict[str, Any]] = []
    if args.resume:
        out = Path(args.resume)
        if not (out / "ablation.json").is_file():
            ap.error(f"{out} is not an ablation run")
        m = json.loads((out / "ablation.json").read_text())
        rows = _read_rows(out)
        modes, seeds, lockstep = m["modes"], m["candidate_seeds"], m["timing"] == "lockstep"
        cfg = DifficultyConfig.from_dict(m["config"])
        cand_name, cand_extra = parse_spec(m["controller"])
        ref_name, ref_ms = m["reference_controller"], m["latency"]["reference_ms"]
        args.timing, args.interval, args.delay = m["timing"], m.get("interval_s", 0.1), m.get("delay_s", 0.0)
        args.controller, args.reference_repeats = m["controller"], m["reference_repeats"]
        args.also_unqualified = m.get("also_unqualified", False)
        m.setdefault("max_failure_rate", args.max_failure_rate)
        m.update(status="running", pid=os.getpid(), abort_reason=None)
        m.setdefault("resumes", []).append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                            "argv": sys.argv[1:] if argv is None else argv})
    else:
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
        seeds = [args.seed + i for i in range(args.episodes)]

    def factory(name: str, mode: str, latency_ms: float):
        ns = SimpleNamespace(**{**vars(args), "controller": name, "observation_mode": mode,
                                "latency_ms": latency_ms, "real_latency": False})
        return controller_factory(name, ns, latency_ms=latency_ms)

    # ---- preflight + latency configuration (lockstep charges no latency to anyone)
    probe: list[float] = []
    probe_errors: list[str] = []
    # Always a short preflight: a service that refuses (no token, no credits) stops the run before it starts.
    probe_n = args.probe_requests if (not args.resume and not lockstep and args.match_latency == "auto") else 2
    probe = probe_latency_ms(factory(cand_name, modes[0], cand_extra), cfg, seeds[0], probe_n, errors=probe_errors)
    if not probe:
        ap.error("the candidate answered none of the preflight requests"
                 + (f": {probe_errors[-1][:300]}" if probe_errors else ""))
    plan = estimate(rows, modes, seeds, cfg, args.reference_repeats, lockstep, args.limit,
                    m.get("max_failure_rate", args.max_failure_rate) if args.resume else args.max_failure_rate,
                    Path(args.resume) if args.resume else None, probe)
    print_plan(plan, args.controller)
    if not args.yes and sys.stdin.isatty():
        if input("Start? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not started")
            return 1
    if not args.resume:
        if lockstep:
            ref_ms, how = 0.0, "none (lockstep: latency costs no world time)"
        elif args.match_latency == "auto":
            ref_ms = float(percentile(probe, 50))
            how = f"auto: median of {len(probe)} probe requests after 1 warm-up"
        else:
            try:
                ref_ms = float(args.match_latency)
            except ValueError:
                ap.error("--match-latency must be a number of ms or 'auto'")
            if ref_ms < 0:
                ap.error("--match-latency must be >= 0")
            how = "fixed"
        if not lockstep and ref_ms > 0:
            # A decision takes effect ceil(L / tick) ticks later. A latency near a tick boundary would let
            # microseconds of host jitter flip the delay by a tick and change the reference's outcome, so the
            # simulated latency is centred in its tick: the tick count a real latency of L (plus compute) gets.
            nominal = ref_ms
            n_ticks = math.floor(nominal / TICK_MS + 1e-9) + 1  # what L plus any compute time yields
            ref_ms = (n_ticks - 0.5) * TICK_MS
            how += f"; {nominal:.1f} ms -> {n_ticks} ticks, simulated as {ref_ms:.1f} ms (centre of the tick)"
        out = Path(args.run_dir) if args.run_dir else new_run_dir(args.out, "ablation", cand_name)
        if args.run_dir and out.exists() and any(out.iterdir()):
            ap.error(f"--run-dir {out} is not empty")
        out.mkdir(parents=True, exist_ok=True)
        m = {
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
            "also_unqualified": args.also_unqualified, "max_failure_rate": args.max_failure_rate,
            "argv": sys.argv[1:] if argv is None else argv,
        }
    total = len(modes) * len(seeds) * args.reference_repeats
    m["progress"] = {"stage": "reference", "mode": None, "done": 0, "total": total}

    def save():
        m["updated"] = time.time()
        _atomic(out / "ablation.json", m)

    def on_term(signum, frame):
        raise Cancelled()

    prev = signal.signal(signal.SIGTERM, on_term)
    save()
    t0 = time.perf_counter()
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    max_fail = m["max_failure_rate"]

    # ---- live viewer (default when a display is available); it only receives messages
    from arena.live_view import LiveView, StopRequested, display_available

    watch = display_available() if args.watch is None else args.watch
    view = None
    if watch:
        try:
            view = LiveView(title=f"Decision Arena · ablation · {args.controller}")
        except Exception as e:  # no window: the experiment itself is unaffected
            say(f"(live viewer unavailable: {e})")
    elif args.watch is None:
        say("(no display found: running without the live viewer; see the dashboard's Ablation tab)")
    qualified_by_mode: dict[str, list[int]] = {}
    cand_walls: list[float] = []
    ran_now = {md: 0 for md in modes}
    ref_label = f"{ref_name}" + (f" @ {ref_ms:.0f} ms" if not lockstep else " (lockstep)")
    tick_s = cfg.world_speed_scale / 60.0

    def push_plan(stage_text: str, remaining: Optional[int] = None) -> None:
        if view is None:
            return
        b = _board(m, rows, qualified_by_mode, args.limit)
        done = sum(x["done"] + x["invalid"] for x in b)
        total = sum(x["total"] for x in b)
        avg = mean(cand_walls) if cand_walls else None
        view.send({"type": "plan", "title": f"Observation ablation · {args.controller} vs {ref_label}",
                   "stage_text": stage_text, "done": done, "total": total, "board": b,
                   "eta_s": (avg * remaining) if (avg and remaining is not None) else None,
                   "notes": [f"{cfg.obstacle_count} obstacles · {fmt_speed(cfg, lockstep)} · {len(seeds)} seeds",
                             "only reference-qualified seeds are run for the candidate"]})

    def start_episode(title: str, subtitle: str, color, reference: Optional[dict[str, Any]] = None,
                      ref_title: str = "", ref_subtitle: str = "") -> None:
        if view is not None:
            view.send({"type": "episode_start", "config": cfg.to_dict(), "tick_s": tick_s, "title": title,
                       "subtitle": subtitle, "color": color, "reference": reference, "ref_title": ref_title,
                       "ref_subtitle": ref_subtitle})

    def reference_replay(mode: str, seed: int) -> Optional[dict[str, Any]]:
        r = next((x for x in reversed(rows) if x["role"] == "reference" and x["mode"] == mode
                  and x["seed"] == seed and x.get("repeat", 0) == 0), None)
        if r is None:
            return None
        try:
            res = json.loads((out / r["episode_dir"] / "result.json").read_text())
        except (OSError, ValueError):
            return None
        return {"config": cfg.to_dict(), "seed": seed, "action_changes": res["action_changes"], "ticks": res["ticks"],
                "result": {"reason": res["reason"]}}

    try:
        with open(out / "results.jsonl", "a") as rf:
            def emit(row):
                rows.append(row)
                rf.write(json.dumps(row, separators=(",", ":")) + "\n")
                rf.flush()

            def have(role, mode, seed, k=0):
                return next((r for r in reversed(rows) if (r["role"], r["mode"], r["seed"], r.get("repeat", 0))
                             == (role, mode, seed, k)), None)

            for mode in modes:
                m["progress"].update(stage="reference", mode=mode)
                save()
                ref_f = factory(ref_name, mode, ref_ms)
                qualified = []
                for i, seed in enumerate(seeds):
                    ok = True
                    for k in range(args.reference_repeats):
                        r = have("reference", mode, seed, k)
                        if r is None:
                            rel = Path("episodes") / mode / "reference" / (
                                f"episode_{i:04d}_seed{seed}" + (f"_r{k}" if args.reference_repeats > 1 else ""))
                            push_plan(f"{mode.upper()} · qualifying seeds with {ref_label}: seed {seed} "
                                      f"({i + 1}/{len(seeds)}) · fast-forward, simulated latency")
                            start_episode(f"{ref_name} · {mode.upper()} · seed {seed}", "qualification run (fast-forward)",
                                          (25, 158, 112))
                            r = _slim(run_one(ref_f, ref_name, cfg, seed, out / rel, args.timing, args.interval, args.delay,
                                              view))
                            r = {**r, "role": "reference", "mode": mode, "seed": seed, "repeat": k,
                                 "controller_spec": ref_name, "latency_config_ms": ref_ms, "episode_dir": str(rel)}
                            r.update(episode_validity(r, out, max_fail))
                            emit(r)
                        ok = ok and bool(r["success"])
                        m["progress"]["done"] += 1
                    if ok:
                        qualified.append(seed)
                    save()
                say(f"[{mode}] reference {ref_name} @ {ref_ms:.0f} ms qualified {len(qualified)} / {len(seeds)} seeds")
                qualified_by_mode[mode] = qualified

                todo = seeds if args.also_unqualified else qualified
                m["progress"].update(stage="candidate", mode=mode, cand_done=0, cand_total=len(todo))
                save()
                cand_f = factory(cand_name, mode, cand_extra)
                for seed in todo:
                    prior = have("candidate", mode, seed)
                    if prior is not None and prior.get("valid", episode_validity(prior, out, max_fail)["valid"]):
                        m["progress"]["cand_done"] += 1
                        continue
                    if args.limit is not None and ran_now[mode] >= args.limit:
                        continue  # left for a later --resume
                    ran_now[mode] += 1
                    attempt = 0 if prior is None else prior.get("attempt", 0) + 1
                    i = seeds.index(seed)
                    remaining = sum(1 for sd in todo if not (have("candidate", mode, sd) or {}).get("valid"))
                    push_plan(f"{mode.upper()} · {args.controller} on seed {seed} · live", remaining)
                    ref_replay = reference_replay(mode, seed)
                    start_episode(f"{args.controller} · {mode.upper()} · seed {seed}",
                                  f"live · {MODE_BLURB.get(mode, '')}", (57, 135, 229), ref_replay,
                                  f"{ref_label} · same seed", "its recorded run, replayed at the same moment")
                    rel = Path("episodes") / mode / "candidate" / (
                        f"episode_{i:04d}_seed{seed}" + (f"_a{attempt}" if attempt else ""))
                    r = _slim(run_one(cand_f, args.controller, cfg, seed, out / rel, args.timing, args.interval, args.delay,
                                      view))
                    cand_walls.append(r["wall_s"])
                    r = {**r, "role": "candidate", "mode": mode, "seed": seed, "repeat": 0, "attempt": attempt,
                         "controller_spec": args.controller, "qualified": seed in qualified,
                         "latency_config_ms": cand_extra, "episode_dir": str(rel)}
                    r.update(episode_validity(r, out, max_fail))
                    emit(r)
                    m["progress"]["cand_done"] += 1
                    save()
                    say(f"  [{mode}] seed {seed}: {r['reason']} at {r['survival_time']:.1f} s, {r['targets_collected']} targets"
                        + ("" if r["valid"] else f" · INVALID ({r['failure_rate']:.0%} failed decisions)"))
                    if view is not None:
                        view.send({"type": "episode_end", "result": {"reason": r["reason"]}})
                        push_plan(f"{mode.upper()} · seed {seed}: {r['reason']}", remaining - 1)
                    if r["fatal_error"]:
                        raise Aborted(r["first_error"])
                    if view is not None and args.pause > 0:
                        end = time.time() + args.pause  # let the outcome be seen; not part of any measurement
                        while time.time() < end and not view.closed():
                            time.sleep(0.05)
                    if view is not None and view.closed():
                        raise StopRequested()
                _atomic(out / "summary.json", build_summary(m, rows, out))
    except Aborted as e:
        m["status"] = "aborted"
        m["abort_reason"] = str(e)
        save()
        _atomic(out / "summary.json", build_summary(m, rows, out))
        print(f"aborted: the candidate's service refused requests ({str(e)[:200]}).\n"
              f"Nothing after this point was recorded. Fix it, then continue with:\n"
              f"  python -m benchmark.ablation --resume {out}", flush=True)
        return 3
    except (Cancelled, KeyboardInterrupt, StopRequested):
        m["status"] = "cancelled"
        save()
        _atomic(out / "summary.json", build_summary(m, rows, out))
        print(f"cancelled; partial results kept in {out} (continue with --resume {out})", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, prev)
        if view is not None:
            push_plan(f"{m.get('status', 'done')} · results in {out}")
            view.finish()

    left = any(args.limit is not None and ran_now[md] >= args.limit for md in modes)
    m["status"] = "partial" if left else "complete"
    m["progress"]["stage"] = "done"
    m["wall_time_s"] = m.get("wall_time_s", 0.0) + time.perf_counter() - t0
    save()
    s = build_summary(m, rows, out)
    _atomic(out / "summary.json", s)
    if not args.quiet:
        print_report(s)
        print(f"\nsaved to {out}")
        if left:
            print(f"--limit reached; continue with: python -m benchmark.ablation --resume {out}")
        if view is not None and not view.closed():
            print("the live viewer stays open with the final state; close it to exit")
    return 0


def print_report(s: dict[str, Any]) -> None:
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    print(f"\nObservation ablation · {s['controller']} · reference {s['reference_controller']} · {s['timing']}")
    if s.get("status") == "aborted":
        print(f"  ABORTED: {str(s.get('abort_reason'))[:200]}")
    if not s.get("valid", True):
        print("  INCOMPLETE: some qualified seeds have no valid candidate episode (see 'evaluated' per mode)")
    for mode, v in s["modes"].items():
        c = v["candidate"]
        print(f"\n{mode.upper()}")
        print(f"  reference qualified seeds  {v['qualified']} / {v['candidate_seeds']}")
        fr = v.get("decision_failure_rate")
        print(f"  valid candidate episodes   {v.get('evaluated', '-')} / {v['qualified']}"
              + (f"  (invalid seeds {v['invalid_seeds']})" if v.get("invalid_seeds") else "")
              + (f"  · failed decisions {fr:.0%}" if fr else ""))
        for err in v.get("invalid_errors", [])[:2]:
            print(f"    error: {err[:140]}")
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
