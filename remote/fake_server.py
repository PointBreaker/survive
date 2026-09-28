"""Local fake decision service implementing ``remote.protocol``.

Stands in for Jev (or any remote model) so the out-of-process path can be
validated before the real service exists. It runs a baseline policy
server-side and holds each reply until ``latency_ms`` (+ jitter) of real
wall time has passed since the request arrived.

    python -m remote.fake_server --port 8765 --policy simple_avoid --latency-ms 150

Prints ``LISTENING http://127.0.0.1:<port>`` once ready (port 0 = pick free).
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from arena.observation import ArenaInfo, Observation
from controllers.base import SyncController
from remote import protocol

POLICIES = ("stay", "greedy", "simple_avoid", "random")


def make_policy(name: str, seed: int) -> SyncController:
    from arena.action import Action
    from controllers.greedy import GreedyController
    from controllers.random import RandomController
    from controllers.simple_avoid import SimpleAvoidController

    if name == "stay":
        class Stay(SyncController):
            name = "stay"

            def decide(self, observation):
                return Action.STAY

        return Stay()
    return {
        "greedy": GreedyController,
        "simple_avoid": SimpleAvoidController,
        "random": lambda: RandomController(seed=seed),
    }[name]()


class _Session:
    def __init__(self, policy: SyncController):
        self.policy = policy
        self.lock = threading.Lock()


class FakeDecisionServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        policy: str = "simple_avoid",
        latency_ms: float = 0.0,
        jitter_ms: float = 0.0,
        fail_rate: float = 0.0,
        invalid_rate: float = 0.0,
        seed: int = 0,
    ):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.policy_name = policy
        self.latency_s = latency_ms / 1000.0
        self.jitter_s = jitter_ms / 1000.0
        self.fail_rate = fail_rate
        self.invalid_rate = invalid_rate
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()
        self._seed = seed
        self.sessions: dict[str, _Session] = {}
        self.payloads: deque = deque(maxlen=64)  # recent /decide bodies, for tests
        self.decide_count = 0
        self._conns: set = set()
        self._conns_lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), self._handler_class())
        self._httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def _draw(self) -> tuple[float, float, float]:
        with self._rng_lock:
            return self._rng.random(), self._rng.random(), self._rng.uniform(-1.0, 1.0)

    # ---------------------------------------------------------------- handlers
    def handle_reset(self, body: dict[str, Any]) -> dict[str, Any]:
        protocol.check_protocol(body)
        info = ArenaInfo.from_dict(body["info"])
        policy = make_policy(self.policy_name, self._seed)
        policy.reset(info)
        sid = uuid.uuid4().hex
        self.sessions[sid] = _Session(policy)
        return {"session": sid}

    def handle_decide(self, body: dict[str, Any], received: float) -> tuple[int, dict[str, Any]]:
        protocol.check_protocol(body)
        self.payloads.append(body)
        self.decide_count += 1
        session = self.sessions.get(body.get("session"))
        if session is None:
            return 400, {"error": "unknown session"}
        fail, invalid, jitter = self._draw()
        obs = Observation.from_dict(body["observation"])
        with session.lock:
            action = session.policy.decide(obs).value
        compute_ms = (time.perf_counter() - received) * 1000
        # Hold the reply until the configured latency has elapsed in total.
        release = received + max(0.0, self.latency_s + jitter * self.jitter_s)
        delay = release - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        if fail < self.fail_rate:
            return 500, {"error": "injected failure"}
        if invalid < self.invalid_rate:
            action = "JUMP"
        meta = {"server_ms": (time.perf_counter() - received) * 1000, "compute_ms": compute_ms}
        return 200, {"request_id": body.get("request_id"), "action": action, "meta": meta}

    def _handler_class(self) -> Callable[..., BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive
            disable_nagle_algorithm = True

            def setup(self) -> None:
                super().setup()
                with server._conns_lock:
                    server._conns.add(self.connection)

            def finish(self) -> None:
                with server._conns_lock:
                    server._conns.discard(self.connection)
                super().finish()

            def do_POST(self) -> None:
                received = time.perf_counter()
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    if self.path.endswith("/reset"):
                        status, reply = 200, server.handle_reset(body)
                    elif self.path.endswith("/decide"):
                        status, reply = server.handle_decide(body, received)
                    else:
                        status, reply = 404, {"error": "not found"}
                except (ValueError, KeyError, TypeError) as e:
                    status, reply = 400, {"error": f"{type(e).__name__}: {e}"}
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        return Handler

    # --------------------------------------------------------------- lifecycle
    def start(self) -> "FakeDecisionServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="fake-decision-server", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def stop(self) -> None:
        """Stop like a crashed service: stop accepting and sever live connections."""
        self._httpd.shutdown()
        self._httpd.server_close()
        with self._conns_lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--policy", choices=POLICIES, default="simple_avoid")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="total reply time per decision")
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="uniform +/- jitter on the latency")
    ap.add_argument("--fail-rate", type=float, default=0.0, help="probability of an HTTP 500")
    ap.add_argument("--invalid-rate", type=float, default=0.0, help="probability of an invalid action")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    srv = FakeDecisionServer(args.host, args.port, args.policy, args.latency_ms, args.jitter_ms,
                             args.fail_rate, args.invalid_rate, args.seed)
    print(f"LISTENING {srv.url}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
