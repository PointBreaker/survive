"""Decision Arena remote protocol, version ``decision-arena/v0``.

Plain JSON over HTTP. The payload carries exactly the same information as
the in-process interface: the ``ArenaInfo`` and ``Observation`` are sent as
their ``to_dict()`` form, unmodified. No summaries, no natural-language
rewording, no derived features. A remote model sees what every other
controller sees.

Endpoints (all POST, ``Content-Type: application/json``)::

    /reset   {"protocol": "decision-arena/v0", "info": ArenaInfo}
             -> {"session": "<opaque id>"}

    /decide  {"protocol": "decision-arena/v0", "session": "...",
              "request_id": int, "observation": Observation}
             -> {"request_id": int, "action": "NE", "meta": {...optional}}

``action`` must be one of ``info.actions``. Anything else (HTTP error,
timeout, bad JSON, unknown action, request_id mismatch) is a *failed
decision*: the runner logs it and keeps the previous action. The client
never invents an action on the service's behalf.

``meta`` is optional free-form diagnostics (e.g. ``server_ms``). It is
logged but never used for timing: latency is always measured by the
arena's own clock around the whole round trip.
"""
from __future__ import annotations

from typing import Any, Optional

from arena.action import Action
from arena.observation import ArenaInfo, Observation

PROTOCOL = "decision-arena/v0"
VALID_ACTIONS = frozenset(a.value for a in Action)


class ProtocolError(ValueError):
    pass


def reset_request(info: ArenaInfo) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "info": info.to_dict()}


def decide_request(session: Optional[str], request_id: int, obs: Observation) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "session": session, "request_id": request_id, "observation": obs.to_dict()}


def parse_decide_response(payload: Any, request_id: int) -> tuple[Action, Optional[dict]]:
    """Validate a /decide reply strictly. Raises ProtocolError on anything off."""
    if not isinstance(payload, dict):
        raise ProtocolError(f"reply is not a JSON object: {type(payload).__name__}")
    if "request_id" in payload and payload["request_id"] != request_id:
        raise ProtocolError(f"request_id mismatch: sent {request_id}, got {payload['request_id']}")
    action = payload.get("action")
    if action not in VALID_ACTIONS:
        raise ProtocolError(f"invalid action {action!r}")
    meta = payload.get("meta")
    return Action(action), meta if isinstance(meta, dict) else None


def check_protocol(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        got = payload.get("protocol") if isinstance(payload, dict) else None
        raise ProtocolError(f"expected protocol {PROTOCOL!r}, got {got!r}")
