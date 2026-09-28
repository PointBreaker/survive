"""Generic out-of-process controller: speaks ``remote.protocol`` over HTTP.

The HTTP round trip runs on a worker thread (``ThreadedController``), so a
slow service never stalls the simulation. Its whole wall-clock cost is
measured by the runner and charged in physics ticks.

Failure policy: any transport or protocol problem produces a failed
``Decision`` (action=None). The runner keeps the previous action. This
client never retries within a request and never picks an action itself.
"""
from __future__ import annotations

import http.client
import json
import socket
from typing import Any, Optional
from urllib.parse import urlsplit

from arena.action import Action
from arena.observation import ArenaInfo, Observation
from controllers.base import Decision, ThreadedController
import remote.protocol as protocol


class RemoteController(ThreadedController):
    name = "remote"
    # Paths appended to the endpoint's own path. reset_path=None: stateless
    # service, no reset call.
    reset_path: Optional[str] = "/reset"
    decide_path: str = "/decide"

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8765",
        timeout_s: float = 10.0,
        headers: Optional[dict[str, str]] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        if name:
            self.name = name
        u = urlsplit(endpoint)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"bad endpoint {endpoint!r}")
        self.endpoint = endpoint
        self._scheme = u.scheme
        self._host = u.hostname
        self._port = u.port
        self._base_path = u.path.rstrip("/")
        self.timeout_s = timeout_s
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._conn: Optional[http.client.HTTPConnection] = None
        self.session: Optional[str] = None
        self._request_id = 0

    # ---------------------------------------------------------- transport
    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            cls = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
            conn = cls(self._host, self._port, timeout=self.timeout_s)
            conn.connect()
            # Small JSON messages: disable Nagle so we do not pay delayed-ACK stalls.
            conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._conn = conn
        return self._conn

    def _drop_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        data = json.dumps(body, separators=(",", ":")).encode()
        try:
            conn = self._connection()
            conn.request("POST", self._base_path + path, body=data, headers=self._headers)
            resp = conn.getresponse()
            raw = resp.read()
        except (OSError, http.client.HTTPException):
            self._drop_connection()  # reconnect on the next request
            raise
        if resp.status != 200:
            raise protocol.ProtocolError(f"HTTP {resp.status}: {raw[:200]!r}")
        return json.loads(raw)

    # ------------------------------------------------------ encode/decode
    # Subclasses adapt to a different wire format by overriding these three.
    # They must preserve information content: serialize, don't summarize.
    def encode_reset(self, info: ArenaInfo) -> dict[str, Any]:
        return protocol.reset_request(info)

    def encode_decide(self, request_id: int, observation: Observation) -> dict[str, Any]:
        return protocol.decide_request(self.session, request_id, observation)

    def decode_decide(self, payload: Any, request_id: int) -> tuple[Action, Optional[dict]]:
        return protocol.parse_decide_response(payload, request_id)

    # --------------------------------------------------------- controller
    def reset(self, info: ArenaInfo) -> None:
        # Runs before the episode clock starts; failure here aborts the run.
        super().reset(info)
        self._request_id = 0
        if self.reset_path is None:
            return
        reply = self._post(self.reset_path, self.encode_reset(info))
        self.session = reply.get("session") if isinstance(reply, dict) else None

    def decide(self, observation: Observation) -> Decision:
        rid = self._request_id
        self._request_id += 1
        try:
            payload = self._post(self.decide_path, self.encode_decide(rid, observation))
            action, meta = self.decode_decide(payload, rid)
        except (OSError, http.client.HTTPException, ValueError) as e:
            return Decision.failed(f"{type(e).__name__}: {e}")
        return Decision(action, meta=meta)

    def close(self) -> None:
        super().close()
        self._drop_connection()
