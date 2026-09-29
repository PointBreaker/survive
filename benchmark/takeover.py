"""Decision-quality scoring by counterfactual takeover branches (shadow mode).

Why branches: grading single actions against a look-ahead oracle turned out
to be dominated by noise (one 0.1 s action rarely decides anything, so a
random policy scored close to a competent one). What matters is whether a
policy, given control at a moment, gets itself out of trouble. So:

1. The driver plays an episode in lockstep; shadows answer the same
   observations (their answers are logged for agreement analysis).
2. Every ``branch_every`` world seconds along the driver's trajectory the
   exact world state is copied, and *each* controller takes over from that
   identical copy for ``takeover`` world seconds, in lockstep, at the same
   decision interval and delay. It is graded on its own consequences only.
3. The branch states are real states the benchmark produces, including the
   dangerous moments right before the driver's own collision.

Per controller (all components are always reported):
  survival            fraction of branches without a collision (Wilson 95% CI)
  contested survival  survival over *contested* branches: at least one
                      controller collided and at least one survived, i.e. the
                      choice decided the outcome (all-collide branches are
                      reported separately as unavoidable-for-everyone)
  progress            targets collected + approach (fraction of the initial
                      target distance closed when nothing was collected),
                      normalised per branch across controllers to 0..1
  min clearance       median over branches of the smallest surface gap to an
                      obstacle (negative = collided)
  takeover score      100 * (0.7 * survival + 0.3 * progress): safety first
  head-to-head        per pair: branches where one survived and the other did not

Driver bias: a deterministic controller taking over from its *own*
trajectory simply repeats it, so it survives nearly every branch by
construction. The default driver is therefore ``explorer``, a neutral state
generator that is never scored: SimpleAvoid interrupted by seeded random
action bursts, which reaches both calm and dangerous states and lives long
enough to yield many branches. If a real controller drives, its own score is
flagged ``self_continuation`` and should not be ranked against the others.
Branch states differ between drivers and seeds, so compare controllers
*within* a run (same states), not scores across runs.

    python -m benchmark.takeover runs/<shadow run>     # re-score existing branches
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from arena.action import ALL_ACTIONS, Action
from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.lockstep import TICKS_PER_WORLD_S, run_takeover
from arena.replay import actions_per_tick
from arena.stats import mean, percentile
from benchmark.metrics import wilson_interval
from controllers.base import Controller, SyncController
from controllers.simple_avoid import SimpleAvoidController

SURVIVAL_WEIGHT = 0.7
PROGRESS_WEIGHT = 0.3
EXPLORER = "explorer"


class ExplorerDriver(SyncController):
    """Neutral state generator for takeover branches (never scored).

    SimpleAvoid, but at each decision with probability ``burst_p`` it starts a
    burst of 3-8 random actions. Seeded per episode; public observation only.
    """

    name = EXPLORER

    def __init__(self, seed: int = 0, burst_p: float = 0.06):
        super().__init__()
        self._seed = seed
        self._burst_p = burst_p
        self._base = SimpleAvoidController()

    def reset(self, info) -> None:
        super().reset(info)
        self._base.reset(info)
        self._rng = random.Random(f"explorer-{self._seed}")
        self._left = 0
        self._burst = Action.STAY

    def decide(self, observation) -> Action:
        self._base.request(observation, None)
        d = self._base.poll()
        base = d.action if d and d.action else Action.STAY
        if self._left <= 0 and self._rng.random() < self._burst_p:
            self._left = self._rng.randint(3, 8)
            self._burst = self._rng.choice(ALL_ACTIONS)
        if self._left > 0:
            self._left -= 1
            return self._burst
        return base


def collect_branches(
    ep_dir: Path,
    config: DifficultyConfig,
    seed: int,
    action_changes: list,
    total_ticks: int,
    factories: dict[str, Callable[[], Controller]],
    interval_ticks: int,
    delay_ticks: int,
    every_ticks: int,
    takeover_ticks: int,
    frames_every: int = 6,
    episode: Optional[str] = None,
    on_branch: Optional[Callable[[int, int], None]] = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Replay the driver's episode, branch every ``every_ticks``; writes branches.jsonl.

    Each takeover gets a fresh controller from its factory, so takeovers are
    independent and may run concurrently (``workers`` > 1 helps network-bound
    controllers such as Jev; rule-based ones are CPU-bound either way).
    """
    actions = list(actions_per_tick(action_changes, total_ticks))
    start_set = set(range(every_ticks, total_ticks, every_ticks))
    env = Environment(config, seed)
    snaps = []
    for tick in range(total_ticks):
        if env.done:
            break
        if tick in start_set:
            snaps.append((tick, actions[tick - 1], copy.deepcopy(env)))
        env.step(actions[tick])

    def one(state: Environment, name: str, in_effect: Action) -> dict[str, Any]:
        c = factories[name]()
        try:
            return run_takeover(copy.deepcopy(state), c, interval_ticks, delay_ticks, takeover_ticks,
                                frames_every=frames_every, initial_action=in_effect)
        finally:
            c.close()

    branches = []
    with open(Path(ep_dir) / "branches.jsonl", "w") as f, ThreadPoolExecutor(max(1, workers)) as pool:
        futures = [{n: pool.submit(one, st, n, act) for n in factories} for _, act, st in snaps]
        for (tick, act, st), fut in zip(snaps, futures):
            b = {
                "episode": episode, "seed": seed, "tick": tick, "t": round(st.world_time, 3),
                "player": {"x": round(st.player.x, 1), "y": round(st.player.y, 1)},
                "target": {"x": round(st.target.x, 1), "y": round(st.target.y, 1), "r": st.target.radius},
                "driver_action": act.value,
                "outcomes": {n: fu.result() for n, fu in fut.items()},
            }
            branches.append(b)
            f.write(json.dumps(b, separators=(",", ":")) + "\n")
            f.flush()
            if on_branch:
                on_branch(len(branches), len(snaps))
    return branches


def _progress(o: dict[str, Any]) -> float:
    return o["targets"] + o["approach"]


def score_branches(branches: list[dict[str, Any]], controllers: list[str]) -> dict[str, Any]:
    per: dict[str, dict[str, list]] = {c: {"surv": [], "cont": [], "prog_q": [], "targets": [], "approach": [],
                                            "clear": [], "lat": [], "failed": []} for c in controllers}
    contested = unavoidable = 0
    h2h = {a: {b: 0 for b in controllers if b != a} for a in controllers}
    for br in branches:
        outs = {c: br["outcomes"][c] for c in controllers if c in br["outcomes"]}
        if not outs:
            continue
        coll = {c: o["collided"] for c, o in outs.items()}
        is_contested = any(coll.values()) and not all(coll.values())
        contested += is_contested
        unavoidable += all(coll.values())
        progs = {c: _progress(o) for c, o in outs.items()}
        lo, hi = min(progs.values()), max(progs.values())
        for c, o in outs.items():
            p = per[c]
            p["surv"].append(not o["collided"])
            if is_contested:
                p["cont"].append(not o["collided"])
            p["prog_q"].append(1.0 if hi - lo < 1e-9 else (progs[c] - lo) / (hi - lo))
            p["targets"].append(o["targets"])
            p["approach"].append(o["approach"])
            p["clear"].append(o["min_clearance"])
            if o.get("latency_ms_p50") is not None:
                p["lat"].append(o["latency_ms_p50"])
            p["failed"].append(o.get("failed_answers", 0))
        for a in outs:
            for b in outs:
                if a != b and not coll[a] and coll[b]:
                    h2h[a][b] += 1
    out: dict[str, Any] = {}
    for c, p in per.items():
        n = len(p["surv"])
        s = sum(p["surv"])
        survival = s / n if n else None
        prog = mean(p["prog_q"])
        out[c] = {
            "branches": n,
            "survived": s,
            "survival": survival,
            "survival_ci95": list(wilson_interval(s, n)) if n else None,
            "contested": len(p["cont"]),
            "contested_survival": (sum(p["cont"]) / len(p["cont"])) if p["cont"] else None,
            "progress_quality": prog,
            "targets_per_branch": mean(p["targets"]),
            "approach": mean(p["approach"]),
            "min_clearance_p50": percentile(p["clear"], 50),
            "latency_ms_p50": percentile(p["lat"], 50),
            "failed_answers": sum(p["failed"]),
            "score": None if survival is None or prog is None
            else 100 * (SURVIVAL_WEIGHT * survival + PROGRESS_WEIGHT * prog),
        }
    return {"controllers": out, "branches": len(branches), "contested": contested,
            "unavoidable": unavoidable, "head_to_head": h2h}


def agreement(decisions: list[dict[str, Any]], controllers: list[str]) -> dict[str, Any]:
    """Share of decision points where two controllers chose the same action."""
    pairs: dict[str, Optional[float]] = {}
    for i, a in enumerate(controllers):
        for b in controllers[i + 1:]:
            both = [d for d in decisions if (d["answers"].get(a) or {}).get("action")
                    and (d["answers"].get(b) or {}).get("action")]
            same = sum(1 for d in both if d["answers"][a]["action"] == d["answers"][b]["action"])
            pairs[f"{a}|{b}"] = same / len(both) if both else None
    return {"pairs": pairs, "decision_points": len(decisions)}


def scored_controllers(manifest: dict[str, Any]) -> list[str]:
    d = manifest["driver"]
    return ([] if d == EXPLORER else [d]) + list(manifest["shadows"])


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def score_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "shadow.json").read_text())
    controllers = scored_controllers(m)
    branches: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    episodes = []
    for ep in sorted((run_dir / "episodes").glob("episode_*")):
        bs = _read_jsonl(ep / "branches.jsonl")
        branches.extend(bs)
        decisions.extend(e for e in _read_jsonl(ep / "events.jsonl") if e.get("type") == "lockstep_decision")
        episodes.append({"episode": ep.name, "branches": len(bs)})
    scores = {
        "version": 3, "method": "takeover", "driver": m["driver"], "controllers_order": controllers,
        "self_continuation": m["driver"] if m["driver"] in controllers else None,
        "interval_s": m["interval_s"], "delay_s": m["delay_s"],
        "takeover_s": m.get("takeover_s"), "branch_every_s": m.get("branch_every_s"),
        "weights": {"survival": SURVIVAL_WEIGHT, "progress": PROGRESS_WEIGHT},
        "episodes": episodes, **score_branches(branches, controllers),
        "agreement": agreement(decisions, controllers),
    }
    tmp = run_dir / "scores.json.tmp"
    tmp.write_text(json.dumps(scores, indent=2))
    os.replace(tmp, run_dir / "scores.json")
    return scores


def print_scores(s: dict[str, Any]) -> None:
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    print(f"{s['branches']} takeover branches ({s['contested']} contested, {s['unavoidable']} unavoidable for all)")
    rows = sorted(s["controllers"].items(), key=lambda kv: -(kv[1]["score"] or -1))
    for c, v in rows:
        c = c + ("*" if c == s.get("self_continuation") else "")
        print(f"  {c:<16} score {f(v['score'], 1):>5}  survival {f(v['survival'])}  contested {f(v['contested_survival'])}"
              f"  progress {f(v['progress_quality'])}  targets/branch {f(v['targets_per_branch'])}"
              f"  clearance p50 {f(v['min_clearance_p50'], 1)}")
    if s.get("self_continuation"):
        print(f"  * {s['self_continuation']} drove: its branches continue its own trajectory (biased, don't rank it)")


def ticks(seconds: float) -> int:
    return max(1, round(seconds * TICKS_PER_WORLD_S))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    print_scores(score_run(Path(args.run_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
