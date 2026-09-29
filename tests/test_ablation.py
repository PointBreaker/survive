"""Paired-solvable observation ablation: fairness and integrity."""
import json
from pathlib import Path

import pytest

from arena.action import Action
from arena.observation_views import PHYSICS_FIELDS, RELATIVE_OBSTACLE_FIELDS
from benchmark import ablation
from controllers.base import SyncController

RAW_OBSTACLE = {"id", "x", "y", "vx", "vy", "radius"}


def run(tmp_path, *extra, name="run"):
    d = tmp_path / name
    rc = ablation.main(["--episodes", "6", "--preset", "easy", "--max-duration", "8", "--run-dir", str(d),
                        "--quiet", *extra])
    assert rc == 0
    rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines()]
    return d, json.loads((d / "ablation.json").read_text()), json.loads((d / "summary.json").read_text()), rows


def test_paired_seed_identity(tmp_path):
    d, m, s, rows = run(tmp_path, "--controller", "greedy+150ms", "--match-latency", "150")
    assert m["experiment_type"] == "observation_ablation" and m["modes"] == ["raw", "relative", "physics"]
    for mode, v in s["modes"].items():
        ref = {r["seed"]: r for r in rows if r["role"] == "reference" and r["mode"] == mode}
        cand = [r for r in rows if r["role"] == "candidate" and r["mode"] == mode]
        assert sorted(ref) == m["candidate_seeds"]
        assert v["qualified_seeds"] == sorted(s_ for s_, r in ref.items() if r["success"])
        assert sorted(r["seed"] for r in cand) == v["qualified_seeds"]  # only qualified seeds evaluated
        for r in cand:  # same seed, same config, same duration as its reference episode
            rc = json.loads((d / ref[r["seed"]]["episode_dir"] / "config.json").read_text())
            cc = json.loads((d / r["episode_dir"] / "config.json").read_text())
            assert rc["seed"] == cc["seed"] == r["seed"] and rc["config"] == cc["config"] == m["config"]
    # a controller that ignores the extra fields must be mode-invariant
    assert all(st["delta_pp"] == 0 for st in s["decomposition"] if st["to"] != "reference")


def test_latency_matched_reference(tmp_path):
    d, m, s, rows = run(tmp_path, "--controller", "greedy+200ms", "--match-latency", "200", "--modes", "raw")
    assert m["experiment_type"] == "paired_solvable"
    ref = [r for r in rows if r["role"] == "reference"]
    tick = 1000 / 60
    assert m["latency"]["reference_ms"] == pytest.approx(12.5 * tick)  # 200 ms (+compute) -> 13 ticks, centred
    # what matters: reference and candidate decisions take effect after the same number of ticks
    cand = [r for r in rows if r["role"] == "candidate"]
    ref_ticks = {round(r["mean_delay_ticks"], 6) for r in ref if r["mean_delay_ticks"] is not None}
    cand_ticks = {round(r["mean_delay_ticks"], 6) for r in cand if r["mean_delay_ticks"] is not None}
    assert ref_ticks == cand_ticks == {13.0}
    lm = s["modes"]["raw"]["latency_match"]
    assert lm["within_tolerance"] is True
    # auto: probe the candidate
    _, m2, _, _ = run(tmp_path, "--controller", "greedy+120ms", "--modes", "raw", name="auto")
    assert m2["latency"]["how"].startswith("auto") and m2["latency"]["reference_ms"] == pytest.approx(7.5 * tick)


class Spy(SyncController):
    name = "spy"
    seen: list = []

    def decide(self, obs):
        Spy.seen.append(obs.to_dict())
        return Action.STAY


def test_reference_action_not_exposed(tmp_path, monkeypatch):
    import controllers

    real = controllers.make_controller
    monkeypatch.setattr(controllers, "make_controller", lambda n, **kw: Spy() if n == "greedy" else real(n, **kw))
    Spy.seen = []
    run(tmp_path, "--controller", "greedy", "--match-latency", "0", "--modes", "raw,physics")
    assert Spy.seen
    top = {"timestamp", "tick", "player", "target", "obstacles", "arena", "score", "control"}
    for o in Spy.seen:
        mode = o.get("observation_mode", "raw")
        assert set(o) - top <= {"observation_mode", "prediction_horizon_s"}
        assert set(o["control"]) == {"applied_request_tick", "applied_latency_s", "applied_latency_world_s"}
        allowed = RAW_OBSTACLE | (set(RELATIVE_OBSTACLE_FIELDS) | set(PHYSICS_FIELDS) if mode == "physics" else set())
        for ob in o["obstacles"]:
            assert set(ob) <= allowed
        blob = json.dumps(o).lower()
        assert "simple_avoid" not in blob and "reference" not in blob
    assert {o.get("observation_mode", "raw") for o in Spy.seen} == {"raw", "physics"}


def test_lockstep_ablation_takes_latency_out(tmp_path):
    _, m, s, rows = run(tmp_path, "--controller", "greedy", "--timing", "lockstep", "--modes", "raw")
    assert m["latency"]["reference_ms"] == 0.0 and m["config"]["world_speed_scale"] == 1.0
    assert all(r["mode"] == "raw" and r.get("delay_s") == 0.0 for r in rows)


@pytest.mark.parametrize("args", [["--modes", "raw,bogus"], ["--reference", "simple_avoid+100ms"],
                                  ["--controller", "greedy+100ms", "--timing", "lockstep"],
                                  ["--match-latency", "fast"]])
def test_rejects_bad_arguments(tmp_path, args):
    with pytest.raises(SystemExit):
        ablation.main(args + ["--run-dir", str(tmp_path / "x"), "--controller", "greedy"] if "--controller" not in args
                      else args + ["--run-dir", str(tmp_path / "x")])


class Flaky(SyncController):
    """Answers like Greedy until a shared budget runs out, then fails like a service out of credits."""
    name = "flaky"
    budget = [0]

    def __init__(self):
        super().__init__()
        from controllers.greedy import GreedyController
        self.inner = GreedyController()

    def reset(self, info):
        super().reset(info)
        self.inner.reset(info)

    def decide(self, obs):
        from controllers.base import Decision
        if Flaky.budget[0] <= 0:
            return Decision.failed('ProtocolError: HTTP 402: b\'{"error":{"message":"Insufficient credits."}}\'')
        Flaky.budget[0] -= 1
        self.inner.request(obs, None)
        return self.inner.poll().action


def test_fatal_service_error_aborts_and_resume_completes(tmp_path, monkeypatch):
    import controllers

    real = controllers.make_controller
    monkeypatch.setattr(controllers, "make_controller", lambda n, **kw: Flaky() if n == "greedy" else real(n, **kw))
    Flaky.budget[0] = 400  # preflight + a few episodes, then "out of credits"
    d = tmp_path / "run"
    args = ["--controller", "greedy", "--match-latency", "0", "--modes", "raw,physics", "--episodes", "6",
            "--preset", "easy", "--max-duration", "8", "--run-dir", str(d), "--quiet"]
    assert ablation.main(args) == 3
    m = json.loads((d / "ablation.json").read_text())
    assert m["status"] == "aborted" and "402" in m["abort_reason"]
    s = json.loads((d / "summary.json").read_text())
    assert s["valid"] is False
    bad = [r for r in map(json.loads, (d / "results.jsonl").read_text().splitlines()) if not r["valid"]]
    assert len(bad) == 1 and bad[0]["fatal_error"]  # stopped right after the first refused episode
    # preflight refuses to start while the service is still refusing
    with pytest.raises(SystemExit):
        ablation.main(["--resume", str(d), "--quiet"])
    Flaky.budget[0] = 10**9  # credits topped up
    assert ablation.main(["--resume", str(d), "--quiet"]) == 0
    s = json.loads((d / "summary.json").read_text())
    rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines()]
    assert s["valid"] is True and json.loads((d / "ablation.json").read_text())["status"] == "complete"
    retried = [r for r in rows if r.get("attempt", 0) > 0]
    assert retried and all(r["valid"] and r["episode_dir"].endswith("_a1") for r in retried)
    # reference episodes were not re-run
    refs = [r for r in rows if r["role"] == "reference"]
    assert len(refs) == len({(r["mode"], r["seed"]) for r in refs})
    for mode, v in s["modes"].items():
        assert v["evaluated"] == v["qualified"] and not v["invalid_seeds"]


def test_rescore_excludes_invalid_episodes(tmp_path):
    d, m, s, rows = run(tmp_path, "--controller", "greedy", "--match-latency", "0", "--modes", "raw")
    # simulate an old artifact: strip validity, then make one candidate episode mostly failed
    cand = [r for r in rows if r["role"] == "candidate"]
    for r in rows:
        for k in ("valid", "failure_rate", "fatal_error", "first_error", "applied_decisions"):
            r.pop(k, None)
    cand[0]["failed_decisions"] = cand[0]["decision_count"] * 3
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s2 = ablation.rescore(d)
    v = s2["modes"]["raw"]
    assert v["invalid_seeds"] == [cand[0]["seed"]] and v["evaluated"] == v["qualified"] - 1
    assert v["candidate"]["episodes"] == v["qualified"] - 1 and s2["valid"] is False


def test_observed_environment_is_bit_identical_and_forwards_events():
    """The live viewer's hook must not change a single tick."""
    import threading

    from arena.difficulty import PRESETS
    from arena.environment import Environment
    from arena.live_view import ObservedEnvironment, TeeRecorder
    from arena.recorder import NullRecorder
    from arena.runner import EpisodeRunner
    from controllers.base import LatencyWrapper
    from controllers.simple_avoid import SimpleAvoidController

    class FakeView:
        def __init__(self):
            self.msgs = []
            self.closed_evt = threading.Event()

        def frame(self, env, force=False):
            self.msgs.append(("frame", env.tick))

        def send(self, msg, droppable=False):
            self.msgs.append(("msg", msg))

    cfg = PRESETS["medium"].with_overrides(max_duration=12)
    plain = EpisodeRunner(Environment(cfg, 4), LatencyWrapper(SimpleAvoidController(), 150)).run()
    v = FakeView()
    watched = EpisodeRunner(ObservedEnvironment(cfg, 4, v), LatencyWrapper(SimpleAvoidController(), 150),
                            recorder=TeeRecorder(NullRecorder(), v)).run()
    assert plain["action_changes"] == watched["action_changes"] and plain["reason"] == watched["reason"]
    assert any(m[0] == "frame" for m in v.msgs)
    assert any(m[0] == "msg" and m[1]["event"]["type"] == "decision" for m in v.msgs)
    assert all("observation" not in (m[1].get("event") or {}) for m in v.msgs if m[0] == "msg")


def test_closed_viewer_stops_the_episode():
    import threading

    from arena.difficulty import PRESETS
    from arena.live_view import ObservedEnvironment, StopRequested
    from arena.runner import EpisodeRunner
    from controllers.simple_avoid import SimpleAvoidController

    class V:
        closed_evt = threading.Event()

        def frame(self, env, force=False):
            if env.tick == 30:
                self.closed_evt.set()

    with pytest.raises(StopRequested):
        EpisodeRunner(ObservedEnvironment(PRESETS["easy"], 0, V()), SimpleAvoidController()).run()


def test_reference_latency_is_centred_in_its_tick(tmp_path):
    _, m, _, rows = run(tmp_path, "--controller", "greedy", "--match-latency", "266.5", "--modes", "raw")
    tick = 1000 / 60
    assert m["latency"]["reference_ms"] == pytest.approx(15.5 * tick)  # 266.5 ms -> 16 ticks, centred
    refs = [r for r in rows if r["role"] == "reference"]
    assert all(r["mean_delay_ticks"] == pytest.approx(16.0) for r in refs if r["mean_delay_ticks"] is not None)


def test_limit_runs_a_few_now_and_resume_finishes(tmp_path):
    d, m, s, rows = run(tmp_path, "--controller", "greedy", "--match-latency", "0", "--modes", "raw,physics",
                        "--limit", "1")
    assert m["status"] == "partial"
    assert all(v["evaluated"] == 1 for v in s["modes"].values())
    assert ablation.main(["--resume", str(d), "--quiet"]) == 0
    s2 = json.loads((d / "summary.json").read_text())
    assert s2["valid"] and all(v["evaluated"] == v["qualified"] for v in s2["modes"].values())


def test_plan_estimate_counts_only_remaining_work(tmp_path):
    from arena.difficulty import PRESETS

    d, m, s, rows = run(tmp_path, "--controller", "greedy", "--match-latency", "0", "--modes", "raw")
    cfg = PRESETS["easy"].with_overrides(max_duration=8, world_speed_scale=0.5)
    full = ablation.estimate(rows, ["raw"], m["candidate_seeds"], cfg, 1, False, None, 0.05, d, [250.0])
    assert full["episodes"] == 0 and full["exact"]
    fresh = ablation.estimate([], ["raw", "physics"], list(range(10)), cfg, 1, False, 2, 0.05, None, [250.0])
    assert fresh["episodes"] == 4 and not fresh["exact"] and fresh["max_total_s"] == pytest.approx(4 * 16.0)
