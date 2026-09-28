"""Replay viewer: step through a logged episode with the full inspector.

    python -m arena.viewer runs/<bench>/episodes/episode_0000_seed0
    python -m arena.viewer runs/<bench>            # all episodes of a benchmark run
    python -m arena.viewer runs/<bench> --at collision

The world is re-simulated exactly from (config, seed, logged per-tick
actions). The inspector is fed the logged events up to the current tick. So
at every moment you see what the controller had been sent, what it
answered, how confident it was and how stale that answer was when it took
effect. No controller or API is called.

Keys: Space play/pause, Left/Right step 1 tick (Shift = 10), , / . previous /
next decision, [ / ] slower / faster, Home / End, C = 2 s before the
collision, PageUp / PageDown previous / next episode, J panel, G ghost,
P compass, F1 debug, click or drag the timeline to seek, Esc / Q quit.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import json
import sys
from pathlib import Path
from typing import Any, Optional

from arena.action import Action
from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.inspector import TICK_MS, Inspector, event_tick, timeline_data
from arena.replay import actions_per_tick

SNAPSHOT_EVERY = 60


def episode_dirs(path: Path) -> list[Path]:
    path = Path(path)
    if (path / "events.jsonl").is_file():
        return [path]
    eps = path / "episodes"
    if eps.is_dir():
        found = sorted(p for p in eps.iterdir() if (p / "events.jsonl").is_file())
        if found:
            return found
    raise SystemExit(f"no episode logs found under {path} (benchmark runs need --save-events)")


class Playback:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        cfg = json.loads((self.run_dir / "config.json").read_text())
        self.config = DifficultyConfig.from_dict(cfg["config"])
        self.seed = cfg["seed"]
        self.controller = cfg.get("controller", "?")
        self.events = [json.loads(l) for l in (self.run_dir / "events.jsonl").read_text().splitlines() if l]
        rp = self.run_dir / "result.json"
        result = json.loads(rp.read_text()) if rp.is_file() else None
        if result is None:  # interrupted run: rebuild action changes from events
            changes = [(0, "STAY")] + [(e["applied_tick"], e["action"]) for e in self.events if e["type"] == "decision"]
            ticks = max([int(event_tick(e)) for e in self.events if event_tick(e) != float("inf")] + [0]) + 1
            result = {"action_changes": changes, "ticks": ticks}
        self.result = result
        self.total = int(result["ticks"])
        self.actions = list(actions_per_tick(result["action_changes"], self.total))
        self.event_ticks = [event_tick(e) for e in self.events]
        self._snapshots: dict[int, Environment] = {0: Environment(self.config, self.seed)}
        self._env = copy.deepcopy(self._snapshots[0])
        self._ins = Inspector(history=10**9)
        self._ev_i = 0
        self.decision_ticks = sorted(
            {e["tick"] for e in self.events if e["type"] == "request"}
            | {int(t) for e, t in zip(self.events, self.event_ticks)
               if e["type"] in ("decision", "decision_failed", "decision_dropped", "decision_superseded")}
        )
        col = next((e for e in self.events if e["type"] == "collision"), None)
        self.collision_tick = col["tick"] if col else None

    def env_at(self, tick: int) -> Environment:
        tick = max(0, min(tick, self.total))
        if tick < self._env.tick or tick - self._env.tick > SNAPSHOT_EVERY:
            base = max(t for t in self._snapshots if t <= tick)
            self._env = copy.deepcopy(self._snapshots[base])
        while self._env.tick < tick and not self._env.done:
            self._env.step(self.actions[self._env.tick])
            if self._env.tick % SNAPSHOT_EVERY == 0 and self._env.tick not in self._snapshots:
                self._snapshots[self._env.tick] = copy.deepcopy(self._env)
        return self._env

    def inspector_at(self, tick: int) -> Inspector:
        if self._ev_i > 0 and self.event_ticks[self._ev_i - 1] > tick:
            self._ins, self._ev_i = Inspector(history=10**9), 0
        while self._ev_i < len(self.events) and self.event_ticks[self._ev_i] <= tick:
            self._ins.feed(self.events[self._ev_i])
            self._ev_i += 1
        return self._ins

    def action_at(self, tick: int) -> str:
        if not self.actions:
            return "STAY"
        return self.actions[min(tick, self.total - 1)].value


def run_viewer(path: Path | str, start: Optional[str] = None, max_frames: int = 0,
               screenshot: Optional[str] = None, panel: str = "decisions",
               window: Optional[tuple[int, int]] = None) -> None:
    from arena.app import run_app

    episode_dirs(Path(path))  # fail early with a clear message
    run_app(start="replay", replay_path=Path(path), replay_at=start, panel=panel, max_frames=max_frames,
            screenshot=screenshot, window=window)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="episode dir (with events.jsonl) or a benchmark run dir")
    ap.add_argument("--at", choices=("start", "collision"), default="start")
    ap.add_argument("--panel", choices=("decisions", "request", "response"), default="decisions")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    run_viewer(Path(args.path), start=args.at, max_frames=args.frames, screenshot=args.screenshot,
               panel=args.panel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
