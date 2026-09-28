"""Controller interface.

Every controller (human, heuristic, search, LLM, Jev, ...) implements the
same asynchronous protocol:

    reset(info)          once per episode, receives public ArenaInfo only
    request(observation) hand over a snapshot; must return promptly
    poll()               -> Decision | None, non-blocking

The runner never pauses the world to wait for a controller. Whatever wall
time a controller spends is charged to it: a decision only takes effect
``ceil(latency / tick)`` physics ticks after its observation was captured,
and until then the previous action keeps being applied.

Helpers:
* ``SyncController``: implement ``decide(obs) -> Action``; computed inline
  in ``request``. The compute time is measured and charged as latency.
* ``ThreadedController``: implement ``decide``; runs on a worker thread, so
  blocking I/O (network APIs, sleeps) never stalls the simulation loop.
* ``LatencyWrapper``: add a fixed extra latency to any controller.
"""
from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from arena.action import Action
from arena.observation import ArenaInfo, Observation


@dataclass(frozen=True)
class Decision:
    action: Action
    # Simulated latency added on top of the measured one. It can only make a
    # controller slower. The measured part is timed by the runner's own clock
    # (request handed over -> decision received); controllers cannot report it.
    extra_latency_s: float = 0.0


class Controller(ABC):
    name: str = "controller"

    def reset(self, info: ArenaInfo) -> None:
        self.info = info

    @abstractmethod
    def request(self, observation: Observation) -> None:
        """Start a decision for ``observation``. Must not block on I/O."""

    @abstractmethod
    def poll(self) -> Optional[Decision]:
        """Return the finished decision for the last request, or None."""

    def close(self) -> None:
        pass


class SyncController(Controller):
    """Computes inline. Compute time is measured and charged as latency."""

    def __init__(self) -> None:
        self._result: Optional[Decision] = None

    @abstractmethod
    def decide(self, observation: Observation) -> Action:
        ...

    def request(self, observation: Observation) -> None:
        self._result = Decision(Action(self.decide(observation)))

    def poll(self) -> Optional[Decision]:
        r, self._result = self._result, None
        return r


class ThreadedController(Controller):
    """Runs ``decide`` on a background worker thread."""

    def __init__(self) -> None:
        self._inbox: "queue.Queue[Optional[Observation]]" = queue.Queue()
        self._outbox: "queue.Queue[Decision]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

    @abstractmethod
    def decide(self, observation: Observation) -> Action:
        ...

    def _run(self) -> None:
        while True:
            obs = self._inbox.get()
            if obs is None:
                return
            self._outbox.put(Decision(Action(self.decide(obs))))

    def reset(self, info: ArenaInfo) -> None:
        super().reset(info)
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name=f"{self.name}-worker", daemon=True)
            self._worker.start()
        # Drop anything left over from a previous episode.
        while not self._outbox.empty():
            self._outbox.get_nowait()

    def request(self, observation: Observation) -> None:
        self._inbox.put(observation)

    def poll(self) -> Optional[Decision]:
        try:
            return self._outbox.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._inbox.put(None)


class LatencyWrapper(Controller):
    """Adds ``delay_ms`` of latency to every decision of ``inner``.

    simulated=True: the delay is declared on the Decision and charged by the
    runner in simulation ticks; no real sleeping, so headless benchmarks
    stay fast. simulated=False: the decision is genuinely withheld until
    the wall-clock delay has elapsed.
    """

    def __init__(self, inner: Controller, delay_ms: float, simulated: bool = True):
        self.inner = inner
        self.delay_s = delay_ms / 1000.0
        self.simulated = simulated
        self.name = f"{inner.name}+{delay_ms:g}ms"
        self._held: Optional[Decision] = None
        self._release_at = 0.0

    def reset(self, info: ArenaInfo) -> None:
        super().reset(info)
        self._held = None
        self.inner.reset(info)

    def request(self, observation: Observation) -> None:
        self._release_at = time.perf_counter() + self.delay_s
        self.inner.request(observation)

    def poll(self) -> Optional[Decision]:
        if self._held is None:
            self._held = self.inner.poll()
        if self._held is None:
            return None
        d = self._held
        if self.simulated:
            self._held = None
            return Decision(d.action, d.extra_latency_s + self.delay_s)
        if time.perf_counter() < self._release_at:
            return None
        self._held = None
        return d

    def close(self) -> None:
        self.inner.close()
