"""Episode logging (JSONL) for exact replay and post-mortem analysis.

Layout::

    runs/<timestamp>_<controller>/
        config.json   difficulty config, seed, controller, public info
        events.jsonl  requests (with full observations), decisions, latencies,
                      skipped slots, targets, collisions, end
        result.json   raw episode metrics, including the per-tick action changes
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


class Recorder:
    def event(self, data: dict[str, Any]) -> None:
        raise NotImplementedError

    def write_result(self, result: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullRecorder(Recorder):
    def event(self, data: dict[str, Any]) -> None:
        pass

    def write_result(self, result: dict[str, Any]) -> None:
        pass


class JsonlRecorder(Recorder):
    def __init__(self, run_dir: Path, record_observations: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.record_observations = record_observations
        self._events = open(self.run_dir / "events.jsonl", "w", encoding="utf-8")

    def event(self, data: dict[str, Any]) -> None:
        if data.get("type") == "episode_start":
            with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in data.items() if k != "type"}, f, indent=2)
        if not self.record_observations and "observation" in data:
            data = {k: v for k, v in data.items() if k != "observation"}
        self._events.write(json.dumps(data, separators=(",", ":")) + "\n")

    def write_result(self, result: dict[str, Any]) -> None:
        self._events.flush()
        with open(self.run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def close(self) -> None:
        if not self._events.closed:
            self._events.close()


def new_run_dir(root: Path | str, controller: str, suffix: Optional[str] = None) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    name = f"{stamp}_{controller}" + (f"_{suffix}" if suffix else "")
    path = Path(root) / name
    i = 1
    while path.exists():
        path = Path(root) / f"{name}_{i}"
        i += 1
    return path
