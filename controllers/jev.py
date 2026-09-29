"""TypeSafe Jev controller (OpenRouter or TypeSafe first-party API).

Setup: put ONE token in ``.env`` at the repo root (see ``.env.example``)::

    OPENROUTER_API_KEY=sk-or-...     # -> https://openrouter.ai/api/alpha/decisions, typesafe/jev-1.13
    TYPESAFE_API_KEY=...             # -> https://api.typesafe.ai/v1/systemone, jev-latest

If both are set, OpenRouter is used unless JEV_ENDPOINT says otherwise.

Then ``python main.py --controller jev`` or
``python -m arena.benchmark --controller jev``. Check connectivity first with
``python -m arena.jev_check``.

Each decision is one stateless POST::

    POST https://openrouter.ai/api/alpha/decisions
    {"model": "typesafe/jev-1.13",
     "state": {"rules": ArenaInfo, "observation": Observation},
     "questions": {"action": {"type": "choice", "instructions": ..., "criteria": {...9 actions}}}}

Latency awareness: ``state.observation.control`` carries the runner-measured
latency of Jev's previous answer (the same own-loop feedback every
controller receives), and the instructions say what it means. It is a fact
about Jev's own timing, not advice about the world.

Fairness: ``state`` carries exactly what every controller gets, the public
rules (``ArenaInfo.to_dict()``, which in-process controllers receive at
reset) and the raw observation (``Observation.to_dict()``), unmodified. The
instructions and criteria only explain the game rules and what each action
means. They give no strategy, risk hints or recommended direction. The
full text is in ``INSTRUCTIONS`` / ``ACTION_CRITERIA`` below, open to audit.

Response: per the TypeSafe docs, a choice answer looks like::

    {"answers": {"action": {"type": "choice", "choice": "NE",
                            "probabilities": {"N": 0.05, "NE": 0.81, ...},
                            "confidence": 0.78}}}

We read ``answers.action.choice`` and log ``confidence`` and
``probabilities`` as decision metadata, so replays can show where Jev was
unsure. Other common layouts are accepted as a fallback (``find_action``).
If none matches, the decision fails (the previous action continues), and
the error carries a snippet of the raw reply. Nothing is substituted, and
the probabilities are never used to pick an action on Jev's behalf.

Configuration variables (arguments override them):
    OPENROUTER_API_KEY  or TYPESAFE_API_KEY (or JEV_API_KEY): bearer token
    JEV_MODEL           default depends on provider (see above)
    JEV_ENDPOINT        default depends on provider (see above)
    JEV_TIMEOUT_S       transport timeout, default 10

Published limits (TypeSafe docs, 2026-09): 70-500 ms per request (p50 around
200 ms via OpenRouter), 1,200 requests/minute. A single episode at <= 10 Hz
with one request in flight stays well under that.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from arena.action import Action
from arena.observation import ArenaInfo, Observation
from controllers.remote import RemoteController
from remote.protocol import ProtocolError

PROVIDERS = {
    # key variable: (endpoint, default model)
    "OPENROUTER_API_KEY": ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13"),
    "TYPESAFE_API_KEY": ("https://api.typesafe.ai/v1/systemone", "jev-latest"),
}
DEFAULT_ENDPOINT, DEFAULT_MODEL = PROVIDERS["OPENROUTER_API_KEY"]

INSTRUCTIONS = (
    "You control the player in a real-time 2D top-down arena. `state.rules` holds the fixed rules "
    "and `state.observation` the current snapshot. Coordinates: x grows to the right, y grows "
    "downward; distances in world units, speeds in world units per world second, `timestamp` in "
    "world seconds. Every entity is a circle with the given radius. Obstacles move in straight "
    "lines and bounce off the arena edges. Touching any obstacle immediately ends the episode in "
    "failure. Touching the target scores one point and a new target appears; going longer than "
    "`rules.target_timeout` world seconds without scoring also ends the episode in failure. "
    "Surviving until `rules.max_duration` is success. The world keeps moving while you decide: "
    "`state.observation.control.applied_latency_world_s` is how many world seconds your previous "
    "answer took to take effect after its snapshot, so the world will have moved on by roughly that "
    "much before this answer takes effect (null before your first answer has taken effect). "
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


def documented_answer(payload: Any, question: str = "action") -> Optional[dict]:
    """``payload["answers"][question]`` if it is a documented choice answer."""
    if isinstance(payload, dict) and isinstance(payload.get("answers"), dict):
        ans = payload["answers"].get(question)
        if isinstance(ans, dict) and "choice" in ans:
            return ans
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
        default_endpoint, default_model = DEFAULT_ENDPOINT, DEFAULT_MODEL
        if api_key is None:
            for var, (ep, mdl) in PROVIDERS.items():
                if os.environ.get(var):
                    api_key, default_endpoint, default_model = os.environ[var], ep, mdl
                    break
            else:
                api_key = os.environ.get("JEV_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API token: set OPENROUTER_API_KEY (or TYPESAFE_API_KEY) in .env "
                "(copy .env.example) or the environment."
            )
        endpoint = endpoint or os.environ.get("JEV_ENDPOINT") or default_endpoint
        self.model = model or os.environ.get("JEV_MODEL") or default_model
        timeout = timeout_s if timeout_s is not None else float(os.environ.get("JEV_TIMEOUT_S") or 10)
        super().__init__(endpoint=endpoint, timeout_s=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self._rules: dict[str, Any] = {}
        self.observation_mode = "raw"

    def set_observation_mode(self, mode: str) -> None:
        """Called by ``arena.observation_views.ObservationView``: explain the extra fields (formulas only)."""
        from arena.observation_views import FIELD_DEFINITIONS

        if mode not in FIELD_DEFINITIONS:
            raise ValueError(f"unknown observation mode {mode!r}")
        self.observation_mode = mode

    def instructions(self) -> str:
        from arena.observation_views import FIELD_DEFINITIONS

        return INSTRUCTIONS + FIELD_DEFINITIONS[self.observation_mode]

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
                    "instructions": self.instructions(),  # == INSTRUCTIONS in raw mode
                    "criteria": dict(ACTION_CRITERIA),
                }
            },
        }

    def decode_decide(self, payload: Any, request_id: int) -> tuple[Action, Optional[dict]]:
        meta: dict[str, Any] = {}
        ans = documented_answer(payload)
        if ans is not None:
            action = _as_action(ans.get("choice"))
            if action is None:
                raise ProtocolError(f"invalid choice {ans.get('choice')!r} in reply: {json.dumps(payload)[:300]}")
            for k in ("confidence", "probabilities"):
                if k in ans:
                    meta[k] = ans[k]
        else:
            action = find_action(payload)
            if action is None:
                raise ProtocolError(f"no valid 'action' answer in reply: {json.dumps(payload)[:300]}")
            meta["layout"] = "fallback"
        if isinstance(payload, dict):
            for k in ("id", "model", "usage"):
                if k in payload:
                    meta[k] = payload[k]
        return action, meta or None
