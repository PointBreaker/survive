"""Phase 3: starting benchmark suites from the dashboard."""
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from arena.dashboard_api import DashboardAPI, classify_run, effective_status, make_server
from arena.dashboard_jobs import JobError, JobManager, build_argv
from benchmark import suite as suite_mod

BENCH = ["jev", "simple_avoid", "greedy", "random"]


def argv_of(spec, token=False):
    return build_argv(spec, BENCH, token)


def test_build_argv_maps_request_to_suite_cli():
    a = argv_of({"controllers": ["simple_avoid", "greedy", "simple_avoid+300ms"], "param": "world_speed_scale",
                 "levels": [4, 0.25, 1], "episodes": 5, "preset": "hard", "max_duration": 30, "max_inflight": 3,
                 "seed": 7, "deadline_ms": None})
    assert a[:10] == ["--controllers", "simple_avoid,greedy,simple_avoid+300ms", "--param", "world_speed_scale",
                      "--levels", "0.25,1,4", "--episodes", "5", "--preset", "hard"]
    assert a[a.index("--seed") + 1] == "7" and a[a.index("--max-inflight") + 1] == "3"
    assert "--deadline-ms" not in a


@pytest.mark.parametrize("spec, msg", [
    ({"controllers": [], "levels": [1]}, "controllers"),
    ({"controllers": ["simple_avoid; rm -rf /"], "levels": [1]}, "unknown controller"),
    ({"controllers": ["human"], "levels": [1]}, "unknown controller"),
    ({"controllers": ["jev"], "levels": [1]}, "OPENROUTER_API_KEY"),
    ({"controllers": ["greedy"], "levels": [100]}, "between"),
    ({"controllers": ["greedy"], "param": "obstacle_count", "levels": [2.5]}, "integer"),
    ({"controllers": ["greedy"], "levels": [1], "param": "physics_hz"}, "param"),
    ({"controllers": ["greedy"], "levels": [1], "episodes": 100000}, "between"),
    ({"controllers": ["greedy", "random", "simple_avoid", "simple_avoid+100ms"], "levels": list(range(1, 13)),
      "episodes": 500}, "limit"),  # 4 x 12 x 500 = 24000 > 20000
    ({"controllers": ["greedy"], "levels": [1], "preset": "nightmare"}, "preset"),
    ({"controllers": ["greedy"], "levels": [1], "max_inflight": 99}, "max_inflight"),
])
def test_build_argv_rejects_bad_requests(spec, msg):
    with pytest.raises(JobError) as e:
        argv_of(spec)
    assert msg in str(e.value) and e.value.status == 400


def test_jev_allowed_with_token():
    assert "jev" in argv_of({"controllers": ["jev"], "levels": [1]}, token=True)[1]


def wait(pred, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.2)
    raise AssertionError("timed out")


def test_job_runs_suite_to_completion(tmp_path):
    jm = JobManager(tmp_path)
    j = jm.start({"controllers": ["greedy", "random"], "levels": [1, 2], "episodes": 2, "max_duration": 3},
                 BENCH, False)
    assert j["status"] == "running" and j["command"].startswith("python -m benchmark.suite --controllers greedy,random")
    done = wait(lambda: (lambda d: d if d["status"] != "running" else None)(jm.get(j["id"])))
    assert done["status"] == "succeeded", done["log_tail"]
    assert done["suite_status"] == "complete"
    assert done["progress"]["episodes_done"] == done["progress"]["episodes_total"] == 8
    m = json.loads((tmp_path / done["suite_id"] / "suite.json").read_text())
    assert m["status"] == "complete" and m["progress"]["points_done"] == 4


def test_one_job_at_a_time_and_cancel_keeps_completed_points(tmp_path):
    jm = JobManager(tmp_path)
    long_spec = {"controllers": ["simple_avoid"], "levels": [0.25, 0.5], "episodes": 40, "max_duration": 600}
    j = jm.start(long_spec, BENCH, False)
    try:
        with pytest.raises(JobError) as e:
            jm.start({"controllers": ["greedy"], "levels": [1], "episodes": 1}, BENCH, False)
        assert e.value.status == 409
        wait(lambda: (jm.get(j["id"])["progress"] or {}).get("episodes_done", 0) >= 1)
    finally:
        jm.cancel(j["id"])
    done = wait(lambda: (lambda d: d if d["status"] not in ("running", "cancelling") else None)(jm.get(j["id"])))
    assert done["status"] == "cancelled"
    m = json.loads((tmp_path / done["suite_id"] / "suite.json").read_text())
    assert m["status"] == "cancelled" and m["progress"]["current"] is None
    assert (tmp_path / done["suite_id"] / "summary.json").is_file()
    # a new job may start once the old one has ended
    j2 = jm.start({"controllers": ["greedy"], "levels": [1], "episodes": 1, "max_duration": 2}, BENCH, False)
    wait(lambda: jm.get(j2["id"])["status"] == "succeeded")


def test_dead_writer_is_reported_as_interrupted(tmp_path):
    d = tmp_path / "2026-01-01_000000_suite_world_speed_scale"
    d.mkdir()
    dead_pid = 2 ** 22 + 12345
    m = {"kind": "suite", "status": "running", "pid": dead_pid, "param": "world_speed_scale", "levels": [1],
         "controllers": ["greedy"], "episodes_per_point": 1}
    (d / "suite.json").write_text(json.dumps(m))
    assert effective_status(m) == "interrupted"
    assert classify_run(d)["status"] == "interrupted"
    assert effective_status({**m, "pid": os.getpid()}) == "running"


def test_suite_run_dir_must_be_empty(tmp_path):
    (tmp_path / "x").write_text("occupied")
    with pytest.raises(SystemExit):
        suite_mod.main(["--controllers", "greedy", "--levels", "1", "--episodes", "1", "--run-dir", str(tmp_path)])


def _post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    return urllib.request.urlopen(req)


def test_http_job_endpoints_and_csrf_guards(tmp_path):
    srv = make_server(tmp_path, port=0, static_dir=None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert json.loads(urllib.request.urlopen(base + "/api/meta").read())["jobs_enabled"] is True
        # form-encoded (no preflight) is refused
        req = urllib.request.Request(base + "/api/jobs", data=b"controllers=greedy", method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 415
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/api/jobs", {}, {"Origin": "https://evil.example"})
        assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/api/jobs", {"controllers": ["nope"], "levels": [1]})
        assert e.value.code == 400
        job = json.loads(_post(base + "/api/jobs", {"controllers": ["greedy"], "levels": [1], "episodes": 1,
                                                    "max_duration": 2}).read())
        wait(lambda: json.loads(urllib.request.urlopen(f"{base}/api/jobs/{job['id']}").read())["status"] == "succeeded")
        assert len(json.loads(urllib.request.urlopen(base + "/api/jobs").read())) == 1
    finally:
        srv.api.jobs.shutdown()
        srv.shutdown()
        srv.server_close()


def test_read_only_server_refuses_jobs(tmp_path):
    api = DashboardAPI(tmp_path, jobs_enabled=False)
    assert api.meta()["jobs_enabled"] is False
    from arena.dashboard_api import ApiError

    with pytest.raises(ApiError) as e:
        api.route_post("/api/jobs", {"controllers": ["greedy"], "levels": [1]})
    assert e.value.status == 403
    srv = make_server(tmp_path, host="0.0.0.0", port=0, static_dir=None)  # non-loopback -> jobs off by default
    try:
        assert srv.api.jobs is None
    finally:
        srv.server_close()


def test_build_argv_lockstep_and_shadow():
    from arena.dashboard_jobs import build_shadow_argv

    a = argv_of({"controllers": ["simple_avoid"], "timing": "lockstep", "interval": 0.1,
                 "param": "decision_delay_ms", "levels": [0, 200], "max_inflight": 3})
    assert a[a.index("--timing") + 1] == "lockstep" and "--max-inflight" not in a
    for bad in ({"controllers": ["greedy"], "timing": "lockstep", "param": "world_speed_scale", "levels": [1]},
                {"controllers": ["greedy"], "param": "decision_delay_ms", "levels": [0]},
                {"controllers": ["greedy+100ms"], "timing": "lockstep", "param": "obstacle_count", "levels": [5]}):
        with pytest.raises(JobError):
            argv_of(bad)
    s = build_shadow_argv({"shadows": ["simple_avoid", "random"], "episodes": 3, "branch_workers": 4}, BENCH, False)
    assert s[:4] == ["--driver", "explorer", "--shadows", "simple_avoid,random"] and "--branch-workers" in s
    with pytest.raises(JobError):
        build_shadow_argv({"shadows": ["jev"]}, BENCH, False)


def test_ablation_job_argv_and_run(tmp_path):
    from arena.dashboard_jobs import build_ablation_argv

    a = build_ablation_argv({"controller": "greedy+100ms", "episodes": 3, "match_latency": 100}, BENCH, False)
    assert a[:6] == ["--controller", "greedy+100ms", "--reference", "simple_avoid", "--modes", "raw,relative,physics"]
    assert a[a.index("--match-latency") + 1] == "100"
    for bad in ({"controller": "jev"}, {"controller": "greedy", "reference": "jev"},
                {"controller": "greedy", "modes": ["raw", "x"]}, {"controller": "greedy", "match_latency": "; ls"}):
        with pytest.raises(JobError):
            build_ablation_argv(bad, BENCH, False)
    jm = JobManager(tmp_path)
    j = jm.start({"kind": "ablation", "controller": "greedy+50ms", "episodes": 2, "max_duration": 3,
                  "match_latency": 50, "modes": ["raw", "physics"]}, BENCH, False)
    assert j["command"].startswith("python -m benchmark.ablation")
    done = wait(lambda: (lambda d: d if d["status"] != "running" else None)(jm.get(j["id"])))
    assert done["status"] == "succeeded", done["log_tail"]
    assert done["suite_status"] == "complete" and (tmp_path / done["suite_id"] / "summary.json").is_file()
