"""TypeSafe Jev controller (out-of-process skeleton).

Talks to a Jev decision service over HTTP using ``remote.protocol``
(``decision-arena/v0``). The unified Observation is serialized to JSON
as-is. Jev gets exactly the raw structured state every other controller
gets, no pre-digested hints.

Configuration (arguments override environment variables):
    JEV_ENDPOINT   base URL of the service   (default http://127.0.0.1:8765)
    JEV_API_KEY    optional bearer token (sent as a header, never logged)
    JEV_TIMEOUT_S  per-request transport timeout (default 10)

If the real Jev API uses a different wire format, override
``encode_reset`` / ``encode_decide`` / ``decode_decide``. Those hooks may
change the *encoding*, never the *information content*.

The transport timeout is not a benchmark deadline. Use
``decision_deadline_ms`` in DifficultyConfig for that; either way a late or
failed call just means the previous action keeps running.
"""
from __future__ import annotations

import os
from typing import Optional

from controllers.remote import RemoteController

DEFAULT_ENDPOINT = "http://127.0.0.1:8765"


class JevController(RemoteController):
    name = "jev"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ):
        endpoint = endpoint or os.environ.get("JEV_ENDPOINT", DEFAULT_ENDPOINT)
        api_key = api_key if api_key is not None else os.environ.get("JEV_API_KEY")
        timeout = timeout_s if timeout_s is not None else float(os.environ.get("JEV_TIMEOUT_S", "10"))
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(endpoint=endpoint, timeout_s=timeout, headers=headers)
