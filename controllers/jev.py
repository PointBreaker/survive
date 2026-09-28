"""TypeSafe Jev controller via the OpenRouter decisions API.

Setup: put your token in ``.env`` at the repo root (see ``.env.example``)::

    OPENROUTER_API_KEY=sk-or-...

Then ``python main.py --controller jev`` or
``python -m arena.benchmark --controller jev``. Check connectivity first with
``python -m arena.jev_check``.

Each decision is one stateless POST::

    POST https://openrouter.ai/api/alpha/decisions
    {"model": "typesafe/jev-1.13",
     "state": {"rules": ArenaInfo, "observation": Observation},
     "questions": {"action": {"type": "choice", "instructions": ..., "criteria": {...9 actions}}}}

Fairness: ``state`` carries exactly what every controller gets, the public
rules (``ArenaInfo.to_dict()``, which in-process controllers receive at
reset) and the raw observation (``Observation.to_dict()``), unmodified. The
instructions and criteria only explain the game rules and what each action
means. They give no strategy, risk hints or recommended direction. The
full text is in ``INSTRUCTIONS`` / ``ACTION_CRITERIA`` below, open to audit.

Response parsing: we look for the answer to the ``action`` question under
common layouts (see ``find_action``). If none matches, the decision fails
(the previous action continues), and the error carries a snippet of the raw
reply for diagnosis. Nothing is substituted.

Configuration variables (arguments override them):
    OPENROUTER_API_KEY  required. Bearer token (JEV_API_KEY also accepted)
    JEV_MODEL           default typesafe/jev-1.13
    JEV_ENDPOINT        default https://openrouter.ai/api/alpha/decisions
    JEV_TIMEOUT_S       transport timeout, default 10
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from arena.action import Action
from arena.observation import ArenaInfo, Observation
from controllers.remote import RemoteController
from remote.protocol import ProtocolError

DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"

INSTRUCTIONS = (
    "You control the player in a real-time 2D top-down arena. `state.rules` holds the fixed rules "
    "and `state.observation` the current snapshot. Coordinates: x grows to the right, y grows "
    "downward; distances in world units, speeds in world units per world second, `timestamp` in "
    "world seconds. Every entity is a circle with the given radius. Obstacles move in straight "
    "lines and bounce off the arena edges. Touching any obstacle immediately ends the episode in "
    "failure. Touching the target scores one point and a new target appears; going longer than "
    "`rules.target_timeout` world seconds without scoring also ends the episode in failure. "
    "Surviving until `rules.max_duration` is success. The world keeps moving while you decide. "
    "Choose the player's thrust direction: it is applied as acceleration "
    "(`rules.player_acceleration`, with drag `rules.player_drag` and speed cap "
    "`rules.player_max_speed`) from the moment your answer arrives until your next answer arrives."
)

ACTION_CRITERIA: dict[str, str] = {
    "STAY": "No thrust; drag slows the player down.",
    "N": "Thrust toward decreasing y (up on screen), direction (0, -1).",
    "NE": "Thrust toward increasing x and decreasing y, direction (0.707, -0.707).",
    "E": "Thrust toward increasing x (right on screen), direction (1, 0).",
    "SE": "Thrust toward increasing x and increasing y, direction (0.707, 0.707).",
    "S": "Thrust toward increasing y (down on screen), direction (0, 1).",
    "SW": "Thrust toward decreasing x and increasing y, direction (-0.707, 0.707).",
    "W": "Thrust toward decreasing x (left on screen), direction (-1, 0).",
    "NW": "Thrust toward decreasing x and decreasing y, direction (-0.707, -0.707).",
}
assert set(ACTION_CRITERIA) == {a.value for a in Action}

_VALID = frozenset(ACTION_CRITERIA)
_VALUE_KEYS = ("value", "choice", "answer", "label", "selected", "result", "decision", "output")
_CONTAINER_KEYS = ("answers", "decisions", "results", "outputs", "output", "result", "data", "choices")


def _as_action(v: Any) -> Optional[Action]:
    if isinstance(v, str) and v.strip().upper() in _VALID:
        return Action(v.strip().upper())
    if isinstance(v, dict):
        for k in _VALUE_KEYS:
            a = _as_action(v.get(k))
            if a is not None:
                return a
    if isinstance(v, list) and len(v) == 1:
        return _as_action(v[0])
    return None


def find_action(payload: Any, question: str = "action", depth: int = 0) -> Optional[Action]:
    """Locate the answer to ``question`` in a decisions-API reply.

    Accepts ``{"action": "NE"}``, ``{"answers": {"action": "NE"}}``,
    ``{"decisions": {"action": {"value": "NE"}}}``, lists of such objects, and
    similar nestings. It only reads the answer the model gave. It never
    derives one.
    """
    if depth > 4:
        return None
    if isinstance(payload, dict):
        if question in payload:
            a = _as_action(payload[question])
            if a is not None:
                return a
        if payload.get("name") == question or payload.get("question") == question or payload.get("key") == question:
            a = _as_action(payload)
            if a is not None:
                return a
        for k in _CONTAINER_KEYS:
            if k in payload:
                a = find_action(payload[k], question, depth + 1)
                if a is not None:
                    return a
    elif isinstance(payload, list):
        for item in payload:
            a = find_action(item, question, depth + 1)
            if a is not None:
                return a
    return None


class JevController(RemoteController):
    name = "jev"
    reset_path = None  # stateless API: rules are sent with every request
    decide_path = ""   # POST to the endpoint URL itself

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ):
        endpoint = endpoint or os.environ.get("JEV_ENDPOINT") or DEFAULT_ENDPOINT
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JEV_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API token: set OPENROUTER_API_KEY in .env (copy .env.example) or the environment."
            )
        self.model = model or os.environ.get("JEV_MODEL") or DEFAULT_MODEL
        timeout = timeout_s if timeout_s is not None else float(os.environ.get("JEV_TIMEOUT_S") or 10)
        super().__init__(endpoint=endpoint, timeout_s=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self._rules: dict[str, Any] = {}

    def reset(self, info: ArenaInfo) -> None:
        super().reset(info)
        self._rules = info.to_dict()

    def encode_decide(self, request_id: int, observation: Observation) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": {"rules": self._rules, "observation": observation.to_dict()},
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": dict(ACTION_CRITERIA),
                }
            },
        }

    def decode_decide(self, payload: Any, request_id: int) -> tuple[Action, Optional[dict]]:
        action = find_action(payload)
        if action is None:
            snippet = json.dumps(payload)[:300]
            raise ProtocolError(f"no valid 'action' answer in reply: {snippet}")
        meta = {}
        if isinstance(payload, dict):
            for k in ("id", "model", "usage"):
                if k in payload:
                    meta[k] = payload[k]
        return action, meta or None
