"""Matched snapshot inspector: an offline forensic tool, not a benchmark.

Rebuilds the exact observation a controller received at one of its decision
requests (deterministic replay from config + seed + logged actions, plus the
own-loop ``control`` feedback reconstructed from the logged decisions), shows
it in every observation mode, and asks other controllers what *they* would
answer to that same snapshot. Nothing is executed: no environment steps
beyond the replay, no outcome, no score. Different answers are not errors;
a dynamic task usually has several workable actions, so this is for looking,
never for grading (use closed-loop benchmarks for that).

Controllers with memory are primed with the controller's previous request
snapshot first (SimpleAvoid infers its decision interval from tick spacing).

    python -m benchmark.snapshot runs/<run>/episodes/.../episode_0003_seed3 --auto
    python -m benchmark.snapshot <episode dir> --tick 734 --controllers simple_avoid,greedy[,jev] --mode physics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.observation_views import MODES, view
from arena.replay import actions_per_tick

LOCAL_CONTROLLERS = ("simple_avoid", "greedy")


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def requests(ep_dir: Path) -> list[dict[str, Any]]:
    """The controller's own decision requests: tick, answer, latency, confidence (from the event log)."""
    out = []
    for e in _read_jsonl(Path(ep_dir) / "events.jsonl"):
        t = e.get("type")
        if t == "decision":
            out.append({"tick": e["request_tick"], "applied_tick": e["applied_tick"], "action": e["action"],
                        "latency_ms": e["latency_ms"], "confidence": (e.get("meta") or {}).get("confidence"),
                        "probabilities": (e.get("meta") or {}).get("probabilities")})
        elif t in ("decision_failed", "decision_superseded", "decision_dropped"):
            out.append({"tick": e.get("request_tick"), "applied_tick": None, "action": None,
                        "latency_ms": e.get("latency_ms"), "status": t})
        elif t == "lockstep_decision":
            cfg = json.loads((Path(ep_dir) / "config.json").read_text())
            a = (e.get("answers") or {}).get(cfg.get("controller")) or {}
            out.append({"tick": e["tick"], "applied_tick": e["effective_tick"] if a.get("action") else None,
                        "action": a.get("action"), "latency_ms": a.get("latency_ms"),
                        "confidence": a.get("confidence"), "probabilities": a.get("probabilities")})
    return sorted((r for r in out if r["tick"] is not None), key=lambda r: r["tick"])


def key_snapshots(ep_dir: Path, window_s: float = 2.0, limit: int = 8) -> list[int]:
    """Request ticks in the last ``window_s`` world seconds before a collision (or the end)."""
    ep_dir = Path(ep_dir)
    result = json.loads((ep_dir / "result.json").read_text())
    cfg = json.loads((ep_dir / "config.json").read_text())
    tick_s = cfg["config"]["world_speed_scale"] / 60.0
    end = int(result["ticks"])
    ticks = [r["tick"] for r in requests(ep_dir) if (end - r["tick"]) * tick_s <= window_s]
    return ticks[-limit:]


def _control_at(reqs: list[dict[str, Any]], tick: int, world_scale: float) -> dict[str, Any]:
    applied = [r for r in reqs if r.get("applied_tick") is not None and r["applied_tick"] <= tick]
    if not applied:
        return {"applied_request_tick": None, "applied_latency_s": None, "applied_latency_world_s": None}
    last = max(applied, key=lambda r: r["applied_tick"])
    lat = last["latency_ms"] / 1000.0
    return {"applied_request_tick": last["tick"], "applied_latency_s": lat, "applied_latency_world_s": lat * world_scale}


def rebuild(ep_dir: Path, ticks: list[int]):
    """Observations (canonical) at the given ticks, exactly as the controller received them."""
    ep_dir = Path(ep_dir)
    cfg = json.loads((ep_dir / "config.json").read_text())
    result = json.loads((ep_dir / "result.json").read_text())
    config = DifficultyConfig.from_dict(cfg["config"])
    total = int(result["ticks"])
    actions = list(actions_per_tick(result["action_changes"], total))
    reqs = requests(ep_dir)
    env = Environment(config, cfg["seed"])
    want = sorted(set(t for t in ticks if 0 <= t <= total))
    out = {}
    for tick in range(total + 1):
        if tick in want:
            out[tick] = env.observe(_control_at(reqs, tick, config.world_speed_scale))
            if tick == want[-1]:
                break
        if tick < total and not env.done:
            env.step(actions[tick])
    return out, env.public_info(), cfg, reqs


def inspect(ep_dir: Path, tick: int, controllers: tuple[str, ...] = LOCAL_CONTROLLERS, mode: str = "raw",
            factory_args: Optional[Any] = None) -> dict[str, Any]:
    from types import SimpleNamespace

    from arena.cli import controller_factory

    ep_dir = Path(ep_dir)
    reqs = requests(ep_dir)
    own = next((r for r in reqs if r["tick"] == tick), None)
    prev = max((r["tick"] for r in reqs if r["tick"] < tick), default=None)
    obs, info, cfg, _ = rebuild(ep_dir, [t for t in (prev, tick) if t is not None])
    if tick not in obs:
        raise ValueError(f"tick {tick} is outside the episode")
    answers = {}
    for name in controllers:
        ns = factory_args or SimpleNamespace(controller_seed=12345, sleep_ms=0, endpoint=None, latency_ms=0.0,
                                             real_latency=False)
        ns = SimpleNamespace(**{**vars(ns), "observation_mode": mode})
        ctrl = controller_factory(name, ns, latency_ms=0.0)()
        try:
            ctrl.reset(info)
            seq = ([prev] if prev is not None else []) + [tick]
            d = None
            for i, t in enumerate(seq):
                ctrl.request(obs[t], i)
                d = _wait(ctrl)
            answers[name] = {"action": None if d is None or d.action is None else d.action.value,
                             "primed": prev is not None,
                             **({"confidence": d.meta["confidence"]} if d and d.meta and "confidence" in d.meta else {}),
                             **({"error": d.error} if d and d.error else {})}
        finally:
            ctrl.close()
    o = obs[tick]
    return {
        "episode": ep_dir.name, "controller": cfg.get("controller"), "seed": cfg["seed"], "tick": tick,
        "t": o.timestamp, "observation_mode": mode,
        "logged": own,  # what the episode's controller actually answered to this snapshot
        "views": {m: view(o, m).to_dict() for m in MODES},
        "answers": answers,
        "note": "Inspection only. Several actions can be workable; no answer here is ground truth.",
    }


def _wait(ctrl, timeout_s: float = 30.0):
    import time

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        d = ctrl.poll()
        if d is not None:
            return d
        time.sleep(0.0005)
    return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tick", type=int, action="append", help="request tick (repeatable)")
    g.add_argument("--auto", action="store_true", help="the requests in the last 2 world s of the episode")
    ap.add_argument("--controllers", default=",".join(LOCAL_CONTROLLERS),
                    help="who to ask (jev makes real API calls; needs a token)")
    ap.add_argument("--mode", choices=MODES, default="raw", help="observation mode the controllers receive")
    ap.add_argument("--json", action="store_true", help="print full JSON (including all three views)")
    args = ap.parse_args(argv)
    from arena.dotenv import load_dotenv

    load_dotenv()
    ticks = key_snapshots(Path(args.episode_dir)) if args.auto else args.tick
    names = tuple(c.strip() for c in args.controllers.split(",") if c.strip())
    for t in ticks:
        r = inspect(Path(args.episode_dir), t, names, args.mode)
        if args.json:
            print(json.dumps(r, indent=2))
            continue
        lg = r["logged"] or {}
        conf = f"  confidence {lg['confidence']:.2f}" if lg.get("confidence") is not None else ""
        print(f"\nsnapshot t={r['t']:.2f}s (tick {t})")
        print(f"  {str(r['controller']) + ' (logged)':<24} {lg.get('action') or lg.get('status') or '-'}{conf}")
        for n, a in r["answers"].items():
            c = f"  confidence {a['confidence']:.2f}" if a.get("confidence") is not None else ""
            print(f"  {n:<24} {a['action'] or a.get('error')}{c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
