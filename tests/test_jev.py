"""JevController against a local stub of the OpenRouter decisions endpoint."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arena.action import Action
from arena.difficulty import DifficultyConfig
from arena.dotenv import load_dotenv
from arena.environment import Environment
from arena.runner import EpisodeRunner
from controllers import jev
from controllers.jev import ACTION_CRITERIA, INSTRUCTIONS, JevController, find_action


class Stub:
    """Records requests; replies with ``reply(body)``."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        stub = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, payload = stub.reply(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/api/alpha/decisions"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def stub_factory():
    stubs = []

    def make(reply):
        s = Stub(reply)
        stubs.append(s)
        return s

    yield make
    for s in stubs:
        s.stop()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "JEV_API_KEY", "JEV_MODEL", "JEV_ENDPOINT", "JEV_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)


def test_missing_token_is_a_clear_error():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        JevController()


def test_defaults_point_at_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "tok")
    c = JevController()
    assert c.endpoint == "https://openrouter.ai/api/alpha/decisions"
    assert c.model == "typesafe/jev-1.13"
    assert c._headers["Authorization"] == "Bearer tok"


def test_dotenv_loads_token_without_overriding(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nOPENROUTER_API_KEY="sk-or-123"\nexport JEV_MODEL=typesafe/other # note\nJEV_TIMEOUT_S=5\n')
    monkeypatch.setenv("JEV_TIMEOUT_S", "7")
    assert load_dotenv(f) == f
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-123"
    assert os.environ["JEV_MODEL"] == "typesafe/other"
    assert os.environ["JEV_TIMEOUT_S"] == "7"  # real environment wins


def test_request_is_the_unified_state_in_openrouter_shape(stub_factory):
    stub = stub_factory(lambda body: (200, {"answers": {"action": "NE"}}))
    env = Environment(DifficultyConfig(obstacle_count=15), 2)
    c = JevController(endpoint=stub.url, api_key="tok")
    c.reset(env.public_info())
    obs = env.observe()
    d = c.decide(obs)
    assert d.action is Action.NE
    req = stub.requests[0]
    assert req["path"] == "/api/alpha/decisions"
    assert req["headers"]["Authorization"] == "Bearer tok"
    body = req["body"]
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == "typesafe/jev-1.13"
    # state carries exactly the public rules + raw observation, nothing else
    assert body["state"] == json.loads(json.dumps({"rules": env.public_info().to_dict(),
                                                   "observation": obs.to_dict()}))
    q = body["questions"]
    assert list(q) == ["action"] and q["action"]["type"] == "choice"
    assert set(q["action"]["criteria"]) == {a.value for a in Action}


def test_prompt_text_explains_rules_only_no_strategy():
    text = (INSTRUCTIONS + " " + " ".join(ACTION_CRITERIA.values())).lower()
    for hint in ("recommend", "safe", "danger", "nearest", "closest", "threat", "best", "should",
                 "avoid", "escape", "risk", "prefer"):
        assert hint not in text, hint


@pytest.mark.parametrize("reply", [
    {"action": "SW"},
    {"answers": {"action": "SW"}},
    {"answers": {"action": {"value": "SW", "confidence": 0.9}}},
    {"decisions": {"action": {"choice": "sw"}}},
    {"results": [{"question": "action", "answer": "SW"}]},
    {"data": {"answers": {"action": ["SW"]}}},
])
def test_find_action_accepts_common_reply_layouts(reply):
    assert find_action(reply) is Action.SW


@pytest.mark.parametrize("reply", [{}, {"answers": {"action": "JUMP"}}, {"answers": {"team": "E"}}, ["N", "S"]])
def test_find_action_never_guesses(reply):
    assert find_action(reply) is None


def test_unparseable_reply_is_failed_decision_with_snippet(stub_factory):
    stub = stub_factory(lambda body: (200, {"weird": "format"}))
    c = JevController(endpoint=stub.url, api_key="tok")
    c.reset(Environment(DifficultyConfig(), 0).public_info())
    d = c.decide(Environment(DifficultyConfig(), 0).observe())
    assert d.action is None and "weird" in d.error


def test_http_error_is_failed_decision(stub_factory):
    stub = stub_factory(lambda body: (401, {"error": {"message": "bad key"}}))
    c = JevController(endpoint=stub.url, api_key="tok")
    c.reset(Environment(DifficultyConfig(), 0).public_info())
    d = c.decide(Environment(DifficultyConfig(), 0).observe())
    assert d.action is None and "401" in d.error


def test_episode_runs_end_to_end_with_stub(stub_factory):
    stub = stub_factory(lambda body: (200, {"answers": {"action": "E"}, "id": "x", "usage": {"tokens": 1}}))
    c = JevController(endpoint=stub.url, api_key="tok")
    result = EpisodeRunner(Environment(DifficultyConfig(max_duration=3), 1), c).run()
    c.close()
    assert result["decision_count"] > 0 and result["failed_decisions"] == 0
    assert ["E"] == sorted({a for _, a in result["action_changes"][1:]})
    assert all(r["body"]["state"]["rules"]["arena_width"] == 1200.0 for r in stub.requests)


def test_jev_check_cli_against_stub(stub_factory, monkeypatch, capsys):
    from arena import jev_check

    stub = stub_factory(lambda body: (200, {"answers": {"action": "N"}}))
    monkeypatch.setenv("OPENROUTER_API_KEY", "tok")
    monkeypatch.setenv("JEV_ENDPOINT", stub.url)
    assert jev_check.main(["-n", "2"]) == 0
    out = capsys.readouterr().out
    assert "action=N" in out and "2/2 decisions parsed" in out and "tok" not in out
