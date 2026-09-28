import json
import subprocess
import sys
from pathlib import Path

from benchmark import adaptive
from benchmark.metrics import aggregate, threshold_level

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_cli_runs_without_pygame(tmp_path):
    code = (
        "import sys; from arena.benchmark import main; "
        f"main(['--controller','greedy','--episodes','3','--quiet','--out',{str(tmp_path)!r},'--save-events']); "
        "print('PYGAME_LOADED' if 'pygame' in sys.modules else 'NO_PYGAME')"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "NO_PYGAME" in out.stdout
    assert "success rate" in out.stdout
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text())
    assert summary["aggregate"]["episodes"] == 3
    assert len(list((run_dirs[0] / "episodes").iterdir())) == 3


def _fake_result(success):
    return {"success": success, "reason": None if success else "collision", "targets_collected": 1,
            "survival_time": 1.0, "distance_travelled": 1.0, "mean_decision_latency_ms": 0.1,
            "p95_latency_ms": 0.1, "missed_slots": 0}


def test_adaptive_staircase_brackets_frontier(monkeypatch):
    true_frontier = 5.3

    def fake_run(make_ctrl, cfg, n, seed):
        return [_fake_result(cfg < true_frontier) for _ in range(n)]

    monkeypatch.setattr(adaptive, "run_episodes", fake_run)
    fr = adaptive.find_frontier("x", make_config=lambda lv: lv, make_controller=lambda lv: None,
                                start=1.0, episodes_per_level=4, refine_steps=6)
    assert fr.highest_pass < true_frontier <= fr.lowest_fail
    assert fr.lowest_fail - fr.highest_pass < 0.1
    tested = [lv["level"] for lv in fr.levels]
    assert tested[:4] == [1.0, 2.0, 4.0, 8.0]


def test_aggregate_and_threshold():
    agg = aggregate([_fake_result(True), _fake_result(False)])
    assert agg["success_rate"] == 0.5 and agg["failure_reasons"] == {"collision": 1}
    assert threshold_level([(1, 1.0), (2, 0.6), (4, 0.2)], 0.5) == 2 + (0.1 / 0.4) * 2
    assert threshold_level([(1, 0.3)], 0.5) is None
