"""Controller interface.

Every controller (human, heuristic, search, LLM, Jev, ...) implements the
same asynchronous protocol:

    reset(info)                      once per episode, receives public ArenaInfo only
    request(observation, request_id) hand over a snapshot; must return promptly
    poll()                           -> Decision | None, non-blocking

The runner never pauses the world to wait for a controller. Whatever wall
time a controller spends is charged to it: a decision only takes effect
``ceil(latency / tick)`` physics ticks after its observation was captured,
and until then the previous action keeps being applied.

Up to ``info.max_inflight`` requests may be outstanding at once (default 1).
Each Decision echoes the ``request_id`` it answers. Answers may arrive out of
order; the runner never lets an older answer overwrite a newer one.

Helpers:
* ``SyncController``: implement ``decide(obs) -> Action``; computed inline
  in ``request``. The compute time is measured and charged as latency.
* ``ThreadedController``: implement ``decide``; runs on a pool of
  ``max_inflight`` worker threads, so blocking I/O (network APIs, sleeps)
  never stalls the simulation loop.
* ``LatencyWrapper``: add a fixed extra latency to any controller.
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Optional

from arena.action import Action
from arena.observation import ArenaInfo, Observation


@dataclass(frozen=True)
class Decision:
    # None means the request failed (network error, invalid reply, ...). The
    # runner then keeps the previous action; nobody substitutes a "safe" one.
    action: Optional[Action]
    # Simulated latency added on top of the measured one. It can only make a
    # controller slower. The measured part is timed by the runner's own clock
    # (request handed over -> decision received); controllers cannot report it.
    extra_latency_s: float = 0.0
    error: Optional[str] = None
    # Free-form diagnostics for the log (e.g. server-reported compute time).
    # Never used for timing or scoring.
    meta: Optional[dict] = None
    # Which request this answers. Set by the helper base classes.
    request_id: Optional[int] = None

    @classmethod
    def failed(cls, error: str, meta: Optional[dict] = None) -> "Decision":
        return cls(None, error=error, meta=meta)


def _coerce(result: "Action | Decision", request_id: Optional[int]) -> Decision:
    d = result if isinstance(result, Decision) else Decision(Action(result))
    return replace(d, request_id=request_id)


class Controller(ABC):
    name: str = "controller"

    def reset(self, info: ArenaInfo) -> None:
        self.info = info

    @abstractmethod
    def request(self, observation: Observation, request_id: Optional[int] = None) -> None:
        """Start a decision for ``observation``. Must not block on I/O."""

    @abstractmethod
    def poll(self) -> Optional[Decision]:
        """Return one finished decision (any outstanding request), or None."""

    def close(self) -> None:
        pass


class SyncController(Controller):
    """Computes inline. Compute time is measured and charged as latency."""

    def __init__(self) -> None:
        self._results: collections.deque[Decision] = collections.deque()

    @abstractmethod
    def decide(self, observation: Observation) -> Action:
        ...

    def request(self, observation: Observation, request_id: Optional[int] = None) -> None:
        self._results.append(_coerce(self.decide(observation), request_id))

    def poll(self) -> Optional[Decision]:
        return self._results.popleft() if self._results else None


class ThreadedController(Controller):
    """Runs ``decide`` on a pool of background worker threads."""

    def __init__(self) -> None:
        self._inbox: "queue.Queue[Optional[tuple[Observation, Optional[int]]]]" = queue.Queue()
        self._outbox: "queue.Queue[Decision]" = queue.Queue()
        self._workers: list[threading.Thread] = []

    @abstractmethod
    def decide(self, observation: Observation) -> "Action | Decision":
        """Return an Action, or a full Decision (e.g. ``Decision.failed``).

        With ``max_inflight > 1`` this runs concurrently on several threads.
        """
        ...

    def _run(self) -> None:
        while True:
            item = self._inbox.get()
            if item is None:
                return
            obs, rid = item
            try:
                d = _coerce(self.decide(obs), rid)
            except Exception as e:  # a crashing decide() is a failed request, not a dead worker
                d = Decision.failed(f"{type(e).__name__}: {e}", None)
                d = replace(d, request_id=rid)
            self._outbox.put(d)

    def reset(self, info: ArenaInfo) -> None:
        super().reset(info)
        self._workers = [w for w in self._workers if w.is_alive()]
        want = max(1, getattr(info, "max_inflight", 1))
        while len(self._workers) < want:
            w = threading.Thread(target=self._run, name=f"{self.name}-worker-{len(self._workers)}", daemon=True)
            w.start()
            self._workers.append(w)
        # Drop anything left over from a previous episode.
        while not self._outbox.empty():
            self._outbox.get_nowait()

    def request(self, observation: Observation, request_id: Optional[int] = None) -> None:
        self._inbox.put((observation, request_id))

    def poll(self) -> Optional[Decision]:
        try:
            return self._outbox.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        for w in self._workers:
            if w.is_alive():
                self._inbox.put(None)


class LatencyWrapper(Controller):
    """Adds ``delay_ms`` of latency to every decision of ``inner``.

    simulated=True: the delay is declared on the Decision and charged by the
    runner in simulation ticks; no real sleeping, so headless benchmarks
    stay fast. simulated=False: each decision is genuinely withheld until
    the wall-clock delay since its own request has elapsed.
    """

    def __init__(self, inner: Controller, delay_ms: float, simulated: bool = True):
        self.inner = inner
        self.delay_s = delay_ms / 1000.0
        self.simulated = simulated
        self.name = f"{inner.name}+{delay_ms:g}ms"
        self._held: list[Decision] = []
        self._release_at: dict[Optional[int], float] = {}

    def reset(self, info: ArenaInfo) -> None:
        super().reset(info)
        self._held = []
        self._release_at = {}
        self.inner.reset(info)

    def request(self, observation: Observation, request_id: Optional[int] = None) -> None:
        self._release_at[request_id] = time.perf_counter() + self.delay_s
        self.inner.request(observation, request_id)

    def poll(self) -> Optional[Decision]:
        while True:
            d = self.inner.poll()
            if d is None:
                break
            self._held.append(d)
        if not self._held:
            return None
        if self.simulated:
            d = self._held.pop(0)
            self._release_at.pop(d.request_id, None)
            return replace(d, extra_latency_s=d.extra_latency_s + self.delay_s)
        now = time.perf_counter()
        for i, d in enumerate(self._held):
            if now >= self._release_at.get(d.request_id, 0.0):
                self._release_at.pop(d.request_id, None)
                return self._held.pop(i)
        return None

    def close(self) -> None:
        self.inner.close()
