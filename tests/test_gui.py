"""Inspector view model and replay viewer (headless, SDL dummy driver)."""
import json

from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.inspector import Inspector, TeeRecorder, event_tick, timeline_data
from arena.recorder import JsonlRecorder, NullRecorder
from arena.renderer import compact_json
from arena.runner import EpisodeRunner
from arena.viewer import Playback, run_viewer
from controllers.base import LatencyWrapper
from controllers.simple_avoid import SimpleAvoidController


def run_logged(tmp_path, latency_ms=150, seconds=6):
    rec = JsonlRecorder(tmp_path / "ep")
    ins = Inspector()
    env = Environment(DifficultyConfig(max_duration=seconds), 3)
    runner = EpisodeRunner(env, LatencyWrapper(SimpleAvoidController(), latency_ms), recorder=TeeRecorder(rec, ins))
    result = runner.run()
    rec.close()
    return result, ins, runner


def test_live_inspector_matches_runner_stats(tmp_path):
    result, ins, runner = run_logged(tmp_path)
    st = ins.stats()
    assert st["applied"] == result["decision_count"]
    assert st["missed"] == result["missed_slots"]
    assert all(r.delay_ticks >= 10 for r in ins.records)  # 150 ms -> >= 10 ticks
    assert ins.result is not None and ins.result["ticks"] == result["ticks"]


def test_inflight_is_tracked_until_resolved():
    ins = Inspector()
    ins.feed({"type": "request", "id": 0, "tick": 5, "observation": {"timestamp": 0.1}})
    assert ins.inflight()["id"] == 0
    ins.feed({"type": "decision", "id": 0, "request_tick": 5, "applied_tick": 15, "action": "N",
              "latency_ms": 160.0, "meta": {"confidence": 0.7}})
    assert ins.inflight() is None
    assert ins.last_applied().action == "N" and ins.last_applied().delay_ticks == 10


def test_playback_reconstructs_world_and_inspector_at_any_tick(tmp_path):
    result, _, runner = run_logged(tmp_path)
    pb = Playback(tmp_path / "ep")
    assert pb.total == result["ticks"]
    end = pb.env_at(pb.total)
    assert (end.player.x, end.player.y) == (runner.env.player.x, runner.env.player.y)
    mid = pb.total // 2
    x_mid = pb.env_at(mid).player.x
    pb.env_at(pb.total)
    assert pb.env_at(mid).player.x == x_mid  # seeking backwards is exact
    ins = pb.inspector_at(mid)
    assert all(r.resolve_tick <= mid for r in ins.records)
    ins_end = pb.inspector_at(pb.total)
    assert len(ins_end.records) == result["decision_count"] + result["failed_decisions"] + result["late_dropped"]
    tl = timeline_data(ins_end, pb.total, pb.total)
    assert len(tl["applied"]) == result["decision_count"]


def test_event_tick_ordering_is_monotonic_in_logs(tmp_path):
    run_logged(tmp_path)
    events = [json.loads(l) for l in (tmp_path / "ep" / "events.jsonl").read_text().splitlines()]
    ticks = [event_tick(e) for e in events]
    assert ticks == sorted(ticks)


def test_viewer_and_live_gui_render_headless(tmp_path):
    run_logged(tmp_path, seconds=3)
    shot = tmp_path / "v.png"
    run_viewer(tmp_path / "ep", start="collision", max_frames=5, screenshot=str(shot), panel="request", scale=0.5)
    assert shot.stat().st_size > 0
    from main import main as gui_main

    shot2 = tmp_path / "live.png"
    assert gui_main(["--controller", "simple_avoid", "--frames", "5", "--no-log", "--scale", "0.5",
                     "--screenshot", str(shot2)]) == 0
    assert shot2.stat().st_size > 0


def test_compact_json_tables_entities_and_collapses_rules():
    lines = compact_json({"state": {"rules": {"a": 1, "b": 2}, "observation": {
        "obstacles": [{"id": 0, "x": 1.23, "y": 2.0}, {"id": 1, "x": 3.0, "y": 4.56}]}}})
    text = "\n".join(lines)
    assert "...2 fields" in text and "(2 rows)" in text and "4.6" in text
