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
    assert m["experiment_type"] == "paired_solvable" and m["latency"]["reference_ms"] == 200.0
    ref = [r for r in rows if r["role"] == "reference"]
    assert all(r["latency_config_ms"] == 200.0 for r in ref)
    assert all(abs(r["p50_latency_ms"] - 200.0) < 5 for r in ref if r["p50_latency_ms"] is not None)
    lm = s["modes"]["raw"]["latency_match"]
    assert lm["within_tolerance"] is True and abs(lm["ratio"] - 1) < 0.05
    # auto: probe the candidate
    _, m2, _, _ = run(tmp_path, "--controller", "greedy+120ms", "--modes", "raw", name="auto")
    assert m2["latency"]["how"].startswith("auto") and abs(m2["latency"]["reference_ms"] - 120) < 5


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
