"""Out-of-process controller path (JevController / RemoteController)."""
import json
import time

import pytest

from arena.action import Action
from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.observation import ArenaInfo, Observation
from arena.runner import TICK, EpisodeRunner
from controllers.base import LatencyWrapper
from controllers.remote import RemoteController
from controllers.simple_avoid import SimpleAvoidController
from remote import protocol
from remote.fake_server import FakeDecisionServer


@pytest.fixture
def server_factory():
    servers = []

    def make(**kw):
        srv = FakeDecisionServer(**kw).start()
        servers.append(srv)
        return srv

    yield make
    for s in servers:
        s.stop()


def cfg(**kw):
    return DifficultyConfig(**{"max_duration": 5.0, **kw})


def test_observation_and_info_roundtrip_through_json():
    env = Environment(DifficultyConfig(obstacle_count=30), 4)
    obs = env.observe()
    assert Observation.from_dict(json.loads(json.dumps(obs.to_dict()))) == obs
    info = env.public_info()
    assert ArenaInfo.from_dict(json.loads(json.dumps(info.to_dict()))) == info


def test_wire_payload_is_exactly_the_unified_observation(server_factory):
    srv = server_factory(policy="stay")
    env = Environment(cfg(), 1)
    ctrl = RemoteController(endpoint=srv.url)
    runner = EpisodeRunner(env, ctrl)
    runner.start()
    obs_sent = []
    real_request = ctrl.request
    ctrl.request = lambda o, rid=None: (obs_sent.append(o), real_request(o, rid))
    for _ in range(30):
        runner.step_tick()
    time.sleep(0.05)
    ctrl.close()
    assert obs_sent and srv.payloads
    body = srv.payloads[0]
    assert set(body) == {"protocol", "session", "request_id", "observation"}
    assert body["protocol"] == protocol.PROTOCOL
    assert body["observation"] == json.loads(json.dumps(obs_sent[0].to_dict()))


def test_remote_policy_reproduces_in_process_policy_exactly(server_factory):
    """Same observations -> same policy -> same actions, across a process boundary."""
    srv = server_factory(policy="simple_avoid", latency_ms=100)
    c = cfg(max_duration=4.0)
    remote = EpisodeRunner(Environment(c, 3), RemoteController(endpoint=srv.url)).run()
    local = EpisodeRunner(Environment(c, 3), LatencyWrapper(SimpleAvoidController(), 100)).run()
    assert remote["action_changes"] == local["action_changes"]
    assert remote["ticks"] == local["ticks"] and remote["reason"] == local["reason"]
    assert remote["mean_decision_latency_ms"] >= 100


def test_slow_service_does_not_pause_world(server_factory):
    srv = server_factory(policy="greedy", latency_ms=300)
    env = Environment(cfg(obstacle_count=5, target_timeout=1e9), 0)
    ctrl = RemoteController(endpoint=srv.url)
    runner = EpisodeRunner(env, ctrl)
    runner.start()
    x0 = [o.x for o in env.obstacles]
    for _ in range(15):  # 250 ms of world
        runner.step_tick()
    assert runner.current_action is Action.STAY
    assert [o.x for o in env.obstacles] != x0
    for _ in range(15):
        runner.step_tick()
    ctrl.close()
    assert runner.stats.applied == 1
    assert runner.action_changes[1][0] >= 0.3 / TICK


def test_injected_failures_keep_previous_action_without_fallback(server_factory):
    srv = server_factory(policy="greedy", fail_rate=1.0)
    env = Environment(cfg(obstacle_count=0), 0)
    ctrl = RemoteController(endpoint=srv.url)
    result = EpisodeRunner(env, ctrl).run()
    ctrl.close()
    assert result["decision_count"] == 0
    assert result["failed_decisions"] > 10
    assert result["action_changes"] == [(0, "STAY")]


def test_invalid_action_is_a_failed_decision(server_factory):
    srv = server_factory(policy="greedy", invalid_rate=1.0)
    ctrl = RemoteController(endpoint=srv.url)
    result = EpisodeRunner(Environment(cfg(), 0), ctrl).run()
    ctrl.close()
    assert result["decision_count"] == 0 and result["failed_decisions"] > 0


def test_service_dying_mid_episode_yields_failed_decisions(server_factory):
    srv = server_factory(policy="greedy")
    env = Environment(cfg(obstacle_count=0, target_timeout=1e9), 0)
    ctrl = RemoteController(endpoint=srv.url, timeout_s=0.5)
    runner = EpisodeRunner(env, ctrl)
    runner.start()
    for _ in range(60):
        runner.step_tick()
    applied = runner.stats.applied
    last = runner.current_action
    srv.stop()
    for _ in range(60):
        runner.step_tick()
    ctrl.close()
    assert applied > 0
    assert runner.stats.failed > 0
    assert runner.current_action is last  # nothing substituted


def test_protocol_rejects_bad_replies():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_decide_response({"action": "UP"}, 0)
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_decide_response({"action": "N", "request_id": 5}, 4)
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_decide_response(["N"], 0)
    assert protocol.parse_decide_response({"action": "N", "request_id": 4}, 4)[0] is Action.N


def test_bad_endpoint_rejected():
    with pytest.raises(ValueError):
        RemoteController(endpoint="localhost:8765")
