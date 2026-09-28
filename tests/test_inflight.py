"""Multiple requests in flight (max_inflight > 1)."""
import math
import time

from arena.action import Action
from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.inspector import Inspector
from arena.recorder import Recorder
from arena.replay import replay
from arena.runner import TICK, EpisodeRunner
from controllers.base import LatencyWrapper, ThreadedController
from controllers.remote import RemoteController
from controllers.simple_avoid import SimpleAvoidController
from controllers.sleep import SleepController
from remote.fake_server import FakeDecisionServer


class ListRecorder(Recorder):
    def __init__(self):
        self.events = []

    def event(self, data):
        self.events.append(data)

    def write_result(self, result):
        pass


def cfg(**kw):
    return DifficultyConfig(**{"obstacle_count": 3, "target_timeout": 1e9, "max_duration": 1e9, **kw})


def run_ticks(controller, n, **kw):
    rec = ListRecorder()
    runner = EpisodeRunner(Environment(cfg(**kw), 0), controller, recorder=rec)
    runner.start()
    for _ in range(n):
        runner.step_tick()
    controller.close()
    return runner, rec.events


def applied_ids(events):
    return [e["id"] for e in events if e["type"] == "decision"]


def test_more_requests_in_flight_means_more_decisions():
    r1, _ = run_ticks(SleepController(delay_ms=300), 120, max_inflight=1)
    r3, ev3 = run_ticks(SleepController(delay_ms=300), 120, max_inflight=3)
    assert r3.stats.applied >= 2 * r1.stats.applied
    # requests really overlapped
    reqs = [e["tick"] for e in ev3 if e["type"] == "request"]
    assert reqs[:3] == [0, 6, 12]


def test_older_answer_never_overwrites_newer():
    class Alternating(ThreadedController):
        name = "alternating"

        def decide(self, observation):
            slow = (observation.tick // 6) % 2 == 0
            time.sleep(0.4 if slow else 0.05)
            return Action.E if slow else Action.W

    runner, events = run_ticks(Alternating(), 150, max_inflight=4)
    ids = applied_ids(events)
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert runner.stats.superseded > 0
    sup = [e for e in events if e["type"] == "decision_superseded"]
    assert all(e["action"] == "E" for e in sup)  # the slow, stale ones


def test_simulated_latency_is_exact_with_pipelining():
    runner, events = run_ticks(LatencyWrapper(SimpleAvoidController(), 310), 240, max_inflight=4)
    delays = [e["applied_tick"] - e["request_tick"] for e in events if e["type"] == "decision"]
    assert delays and set(delays) == {math.ceil(0.310 / TICK + 1e-9)}
    assert runner.stats.superseded == 0
    assert runner.stats.applied > 30  # ~10/s instead of ~3/s


def test_pipelined_episode_replays_exactly():
    c = DifficultyConfig(max_duration=10, max_inflight=3)
    env = Environment(c, 4)
    result = EpisodeRunner(env, LatencyWrapper(SimpleAvoidController(), 250)).run()
    assert result["max_inflight"] == 3
    rep = replay(c, 4, result["action_changes"], max_ticks=result["ticks"])
    assert (rep.player.x, rep.player.y, rep.score) == (env.player.x, env.player.y, env.score)


def test_remote_controller_runs_concurrent_http_calls():
    srv = FakeDecisionServer(policy="greedy", latency_ms=200).start()
    try:
        r1, _ = run_ticks(RemoteController(endpoint=srv.url), 90, max_inflight=1)
        r3, ev3 = run_ticks(RemoteController(endpoint=srv.url), 90, max_inflight=3)
    finally:
        srv.stop()
    assert r1.stats.failed == 0 and r3.stats.failed == 0
    assert r3.stats.applied >= 2 * r1.stats.applied
    ids = applied_ids(ev3)
    assert ids == sorted(ids)


def test_inspector_tracks_several_open_requests():
    ins = Inspector()
    for i, t in enumerate((0, 6, 12)):
        ins.feed({"type": "request", "id": i, "tick": t, "observation": {"timestamp": t / 60}})
    assert [r["id"] for r in ins.inflight_all()] == [0, 1, 2]
    ins.feed({"type": "decision", "id": 1, "request_tick": 6, "applied_tick": 25, "action": "N", "latency_ms": 310})
    ins.feed({"type": "decision_superseded", "id": 0, "request_tick": 0, "tick": 26, "action": "E", "latency_ms": 420})
    assert [r["id"] for r in ins.inflight_all()] == [2]
    assert ins.stats()["superseded"] == 1 and ins.last_applied().id == 1
