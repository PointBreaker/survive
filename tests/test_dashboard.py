"""Benchmark suite artifact + read-only dashboard API."""
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from arena.dashboard_api import ApiError, DashboardAPI, make_server
from benchmark import suite as suite_mod
from benchmark.metrics import threshold_summary


@pytest.fixture(scope="module")
def suite_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    rc = suite_mod.main(["--controllers", "simple_avoid,greedy,simple_avoid+200ms", "--levels", "1,4",
                         "--episodes", "2", "--max-duration", "4", "--out", str(root), "--quiet"])
    assert rc == 0
    (d,) = [p for p in root.iterdir() if p.is_dir()]
    return root, d


def test_suite_artifact_is_complete_and_self_describing(suite_run):
    root, d = suite_run
    m = json.loads((d / "suite.json").read_text())
    assert m["status"] == "complete" and m["param"] == "world_speed_scale" and m["levels"] == [1.0, 4.0]
    assert m["controllers"] == ["simple_avoid", "greedy", "simple_avoid+200ms"] and m["seeds"] == [0, 1]
    results = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines()]
    assert len(results) == 3 * 2 * 2
    for r in results:
        ep = d / r["episode_dir"]
        assert (ep / "events.jsonl").is_file() and (ep / "result.json").is_file()
        assert "action_changes" not in r  # kept in the episode's result.json only
    # paired seeds: every controller/level saw seeds 0 and 1
    assert {(r["controller_spec"], r["level"], r["seed"]) for r in results} == {
        (c, lv, s) for c in m["controllers"] for lv in (1.0, 4.0) for s in (0, 1)}
    # the latency-matched spec really ran with added latency
    lat = [r["mean_decision_latency_ms"] for r in results if r["controller_spec"] == "simple_avoid+200ms"]
    assert all(v >= 200 for v in lat)
    summary = json.loads((d / "summary.json").read_text())
    for spec in m["controllers"]:
        d50 = summary["controllers"][spec]["thresholds"]["D50"]
        assert d50["kind"] in ("interpolated", "above_range", "below_range")


def test_threshold_summary_reports_how_it_was_measured():
    assert threshold_summary([(1, 1.0), (2, 0.6), (4, 0.2)], 0.5)["kind"] == "interpolated"
    assert threshold_summary([(1, 1.0), (2, 0.9)], 0.5) == {"value": 2, "kind": "above_range", "p": 0.5}
    assert threshold_summary([(1, 0.2), (2, 0.0)], 0.5) == {"value": 1, "kind": "below_range", "p": 0.5}
    assert threshold_summary([], 0.5)["kind"] == "not_measured"


def test_api_aggregates_match_raw_artifacts(suite_run):
    root, d = suite_run
    api = DashboardAPI(root)
    assert [s["id"] for s in api.suites()] == [d.name]
    detail = api.suite(d.name)
    results = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines()]
    for spec, a in detail["analysis"].items():
        rs = [r for r in results if r["controller_spec"] == spec]
        assert a["episodes"] == len(rs)
        assert a["accounting"]["applied"] == sum(r["decision_count"] for r in rs)
        assert a["accounting"]["requests"] == sum(r["request_count"] for r in rs)
        assert a["failures"]["episodes_failed"] == sum(1 for r in rs if not r["success"])
        # every received answer has a logged latency
        n_answers = 0
        for r in rs:
            for line in (d / r["episode_dir"] / "events.jsonl").read_text().splitlines():
                if json.loads(line)["type"] in ("decision", "decision_failed", "decision_superseded", "decision_dropped"):
                    n_answers += 1
        assert a["latency"]["pooled"]["n"] == n_answers
        assert sum(a["actions"].values()) == a["accounting"]["applied"]
    slow = detail["analysis"]["simple_avoid+200ms"]["latency"]["pooled"]
    assert slow["p50"] >= 200 and slow["over_period"] == 1.0  # 200 ms > 100 ms period
    assert detail["decision_period_ms"] == 100.0


def test_api_compare_and_verified_replay(suite_run):
    root, d = suite_run
    api = DashboardAPI(root)
    cmp = api.compare(d.name, "4", "1")
    assert {e["controller_spec"] for e in cmp["episodes"]} == {"simple_avoid", "greedy", "simple_avoid+200ms"}
    ep = cmp["episodes"][0]
    rep = api.replay(d.name, ep["episode_dir"])
    assert rep["verified"] is True
    assert rep["frames"][0][0] == 0 and rep["frames"][-1][0] == rep["ticks"]
    assert rep["result"]["reason"] == ep["reason"]


def test_api_rejects_bad_paths(suite_run):
    root, d = suite_run
    api = DashboardAPI(root)
    for bad in ("../etc", "", ".hidden", "a/b"):
        with pytest.raises(ApiError):
            api.suite(bad)
    with pytest.raises(ApiError):
        api.replay(d.name, "../../../etc/passwd")
    with pytest.raises(ApiError):
        api.route("/api/nope", {})


def test_http_server_serves_json_and_build_hint(suite_run, tmp_path):
    root, d = suite_run
    srv = make_server(root, port=0, static_dir=tmp_path / "missing")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        suites = json.loads(urllib.request.urlopen(base + "/api/suites").read())
        assert suites[0]["id"] == d.name
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "npm run build" in page
        try:
            urllib.request.urlopen(base + "/api/suites/does-not-exist")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_dashboard_api_does_not_import_simulation_writers():
    """The API may read and re-simulate; it must not run benchmarks."""
    import ast

    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath("arena", "dashboard_api.py").read_text())
    imported = set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            names |= {a.name for a in node.names}
    assert not imported & {"arena.runner", "benchmark.runner", "benchmark.suite", "benchmark.adaptive",
                            "arena.lockstep", "benchmark.shadow", "benchmark.takeover"}
    assert not names & {"run_episodes", "EpisodeRunner", "find_frontier"}
