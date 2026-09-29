"""Lockstep timing, shadow runs and takeover-branch scoring."""
import copy
import json

import pytest

from arena.action import Action
from arena.difficulty import PRESETS
from arena.environment import Environment
from arena.lockstep import LockstepRunner, run_takeover
from arena.replay import replay
from benchmark import shadow as shadow_mod
from benchmark import suite as suite_mod
from benchmark import takeover as tk
from controllers.greedy import GreedyController
from controllers.random import RandomController
from controllers.simple_avoid import SimpleAvoidController
from controllers.sleep import SleepController


def cfg(preset="medium", **kw):
    return PRESETS[preset].with_overrides(world_speed_scale=1.0, max_inflight=1, **kw)


@pytest.mark.parametrize("delay", [0.0, 0.25])
def test_lockstep_replays_exactly(delay):
    env = Environment(cfg(max_duration=12), 3)
    r = LockstepRunner(env, ("sa", SimpleAvoidController()), {"g": GreedyController()}, 0.1, delay).run()
    e = replay(cfg(max_duration=12), 3, r["action_changes"], r["ticks"])
    assert (e.tick, e.score, e.outcome.reason) == (r["ticks"], r["targets_collected"], r["reason"])
    ticks = [t for t, _ in r["action_changes"]]
    assert len(ticks) == len(set(ticks)), "one entry per tick (replay sorts entries)"


def test_slow_controller_costs_no_world_time():
    fast = LockstepRunner(Environment(cfg(max_duration=2), 1), ("s", SleepController(0)), {}, 0.1).run()
    slow = LockstepRunner(Environment(cfg(max_duration=2), 1), ("s", SleepController(40)), {}, 0.1).run()
    assert fast["action_changes"] == slow["action_changes"]
    assert slow["survival_time"] == fast["survival_time"]
    assert slow["latency_ms"]["s"]["p50"] >= 40 > fast["latency_ms"]["s"]["p50"]


def test_shadows_see_identical_observations_but_never_act():
    seen = {}

    class Spy(SimpleAvoidController):
        def __init__(self, key):
            super().__init__()
            self.key = key

        def decide(self, obs):
            seen.setdefault(self.key, []).append(json.dumps(obs.to_dict(), sort_keys=True))
            return super().decide(obs)

    solo = LockstepRunner(Environment(cfg(max_duration=4), 2), ("d", Spy("a")), {}, 0.1).run()
    seen.clear()
    both = LockstepRunner(Environment(cfg(max_duration=4), 2), ("d", Spy("a")),
                          {"r": RandomController(), "s": Spy("b")}, 0.1).run()
    assert solo["action_changes"] == both["action_changes"]  # shadows do not influence the world
    assert seen["a"] == seen["b"]


def test_delay_shifts_effect_by_whole_ticks():
    r = LockstepRunner(Environment(cfg(max_duration=3), 0), ("sa", SimpleAvoidController()), {}, 0.1, 0.2).run()
    assert all(t == 0 or t >= 12 for t, _ in r["action_changes"])


def test_takeover_is_deterministic_and_leaves_source_untouched():
    env = Environment(cfg("hard"), 1)
    for _ in range(60):
        env.step(Action.STAY)
    before = (env.tick, env.player.x, env.player.y)
    a = run_takeover(copy.deepcopy(env), SimpleAvoidController(), 6, 0, 180, frames_every=6)
    b = run_takeover(copy.deepcopy(env), SimpleAvoidController(), 6, 0, 180, frames_every=6)
    strip = lambda r: {k: v for k, v in r.items() if k != "latency_ms_p50"}  # noqa: E731 (wall clock)
    assert strip(a) == strip(b) and (env.tick, env.player.x, env.player.y) == before
    assert a["ticks"] <= 180 and len(a["frames"]) >= 2 and a["radii"]


def test_scoring_definitions():
    o = lambda c, t=0, ap=0.0, cl=10.0: {"collided": c, "targets": t, "approach": ap, "min_clearance": cl}  # noqa
    branches = [
        {"outcomes": {"a": o(False, 1), "b": o(True, 0, 0.5), "c": o(False, 0, -0.2)}},  # contested
        {"outcomes": {"a": o(True), "b": o(True), "c": o(True)}},  # unavoidable for all
        {"outcomes": {"a": o(False, 0, 0.3), "b": o(False, 0, 0.3), "c": o(False, 0, 0.3)}},  # tie
    ]
    s = tk.score_branches(branches, ["a", "b", "c"])
    assert (s["branches"], s["contested"], s["unavoidable"]) == (3, 1, 1)
    a, b = s["controllers"]["a"], s["controllers"]["b"]
    assert a["survival"] == pytest.approx(2 / 3) and a["contested_survival"] == 1.0
    assert b["contested_survival"] == 0.0 and s["head_to_head"]["a"]["b"] == 1 and s["head_to_head"]["b"]["a"] == 0
    # progress normalised per branch: a best in branch 1, all equal in 2 and 3
    assert a["progress_quality"] == pytest.approx(1.0)
    assert a["score"] == pytest.approx(100 * (0.7 * 2 / 3 + 0.3 * 1.0))


def test_agreement():
    d = [{"answers": {"a": {"action": "N"}, "b": {"action": "N"}}},
         {"answers": {"a": {"action": "N"}, "b": {"action": "S"}}},
         {"answers": {"a": {"action": None}, "b": {"action": "S"}}}]
    assert tk.agreement(d, ["a", "b"])["pairs"]["a|b"] == 0.5


def test_shadow_run_ranks_known_groups(tmp_path):
    """Validity check: a competent avoider must out-survive greedy and random from the same states."""
    rc = shadow_mod.main(["--shadows", "simple_avoid,greedy,random", "--episodes", "3", "--preset", "hard",
                          "--max-duration", "20", "--run-dir", str(tmp_path / "run"), "--quiet"])
    assert rc == 0
    run = tmp_path / "run"
    m = json.loads((run / "shadow.json").read_text())
    assert m["status"] == "complete" and m["driver"] == "explorer"
    s = json.loads((run / "scores.json").read_text())
    c = s["controllers"]
    assert "explorer" not in c and s["self_continuation"] is None and s["branches"] >= 20
    assert c["simple_avoid"]["survival"] > c["greedy"]["survival"]
    assert c["simple_avoid"]["survival"] > c["random"]["survival"]
    assert c["greedy"]["progress_quality"] > c["random"]["progress_quality"]
    assert set(s["agreement"]["pairs"]) == {"simple_avoid|greedy", "simple_avoid|random", "greedy|random"}
    # rescoring from disk is identical
    assert tk.score_run(run)["controllers"] == c


def test_shadow_rejects_added_latency(tmp_path):
    with pytest.raises(SystemExit):
        shadow_mod.main(["--shadows", "greedy+100ms", "--run-dir", str(tmp_path / "x")])


def test_suite_lockstep_delay_sweep(tmp_path):
    rc = suite_mod.main(["--timing", "lockstep", "--interval", "0.1", "--param", "decision_delay_ms",
                         "--levels", "0,300", "--controllers", "simple_avoid", "--episodes", "2",
                         "--max-duration", "10", "--run-dir", str(tmp_path / "s"), "--quiet"])
    assert rc == 0
    m = json.loads((tmp_path / "s" / "suite.json").read_text())
    assert m["timing"] == "lockstep" and m["config"]["world_speed_scale"] == 1.0
    rows = [json.loads(l) for l in (tmp_path / "s" / "results.jsonl").read_text().splitlines()]
    assert {r["delay_s"] for r in rows} == {0.0, 0.3} and all(r["mode"] == "lockstep" for r in rows)


@pytest.mark.parametrize("args", [
    ["--timing", "lockstep", "--param", "world_speed_scale", "--levels", "1"],
    ["--param", "decision_delay_ms", "--levels", "0"],
    ["--timing", "lockstep", "--param", "obstacle_count", "--levels", "5", "--controllers", "greedy+100ms"],
])
def test_suite_rejects_meaningless_combinations(tmp_path, args):
    with pytest.raises(SystemExit):
        suite_mod.main(args + ["--run-dir", str(tmp_path / "x")])
