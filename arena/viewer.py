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

SPEEDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
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
               if e["type"] in ("decision", "decision_failed", "decision_dropped")}
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


def run_viewer(path: Path | str, renderer=None, start: Optional[str] = None,
               max_frames: int = 0, screenshot: Optional[str] = None, panel: str = "decisions",
               scale: Optional[float] = None) -> None:
    import pygame

    from arena.renderer import PANEL_MODES, Renderer

    dirs = episode_dirs(Path(path))
    ep = 0
    pb = Playback(dirs[ep])
    own = renderer is None
    if own:
        renderer = Renderer(pb.config.arena_width, pb.config.arena_height, title="Decision Arena replay",
                            timeline=True, scale=scale)
    clock = pygame.time.Clock()
    tick = 0.0
    playing = True
    speed_i = SPEEDS.index(1.0)
    debug, ghost, compass = False, True, True
    panel_mode = PANEL_MODES.index(panel)
    dragging = False

    def jump_collision():
        if pb.collision_tick is not None:
            return float(max(0, pb.collision_tick - 120)), False
        return tick, playing

    if start == "collision":
        tick, playing = jump_collision()
    frames = 0

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                if own:
                    renderer.close()
                return
            if ev.type == pygame.KEYDOWN:
                shift = ev.mod & pygame.KMOD_SHIFT
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    if own:
                        renderer.close()
                    return
                elif ev.key == pygame.K_SPACE:
                    if tick >= pb.total:
                        tick = 0.0
                    playing = not playing
                elif ev.key == pygame.K_RIGHT:
                    tick, playing = min(pb.total, int(tick) + (10 if shift else 1)), False
                elif ev.key == pygame.K_LEFT:
                    tick, playing = max(0, int(tick) - (10 if shift else 1)), False
                elif ev.key == pygame.K_PERIOD:
                    i = bisect.bisect_right(pb.decision_ticks, int(tick))
                    if i < len(pb.decision_ticks):
                        tick, playing = pb.decision_ticks[i], False
                elif ev.key == pygame.K_COMMA:
                    i = bisect.bisect_left(pb.decision_ticks, int(tick)) - 1
                    if i >= 0:
                        tick, playing = pb.decision_ticks[i], False
                elif ev.key == pygame.K_RIGHTBRACKET:
                    speed_i = min(len(SPEEDS) - 1, speed_i + 1)
                elif ev.key == pygame.K_LEFTBRACKET:
                    speed_i = max(0, speed_i - 1)
                elif ev.key == pygame.K_HOME:
                    tick = 0.0
                elif ev.key == pygame.K_END:
                    tick, playing = float(pb.total), False
                elif ev.key == pygame.K_c:
                    tick, playing = jump_collision()
                elif ev.key in (pygame.K_PAGEDOWN, pygame.K_PAGEUP) and len(dirs) > 1:
                    ep = (ep + (1 if ev.key == pygame.K_PAGEDOWN else -1)) % len(dirs)
                    pb = Playback(dirs[ep])
                    tick, playing = 0.0, True
                elif ev.key == pygame.K_j:
                    panel_mode = (panel_mode + 1) % len(PANEL_MODES)
                elif ev.key == pygame.K_g:
                    ghost = not ghost
                elif ev.key == pygame.K_p:
                    compass = not compass
                elif ev.key in (pygame.K_F1, pygame.K_TAB):
                    debug = not debug
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and renderer.timeline_rect.collidepoint(ev.pos):
                dragging, playing = True, False
                tick = float(renderer.timeline_tick_at(ev.pos[0], pb.total))
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                tick = float(renderer.timeline_tick_at(ev.pos[0], pb.total))

        if playing:
            tick += SPEEDS[speed_i]
            if tick >= pb.total:
                tick, playing = float(pb.total), False
        t = int(tick)
        env = pb.env_at(t)
        ins = pb.inspector_at(t)
        last = ins.last_applied()
        lat = last.latency_ms if last else None
        req = ins.inflight() or (ins.requests.get(ins.records[-1].id) if ins.records else None)
        raw_req = {"observation": req["observation"]} if req else None
        raw_resp = ({k: v for k, v in ins.records[-1].__dict__.items() if v is not None} if ins.records else None)
        state = "PLAYING" if playing else "PAUSED"
        label = (f"{state} {SPEEDS[speed_i]:g}x   tick {t}/{pb.total}  ({t * TICK_MS / 1000:.2f}s controller time)"
                 f"   episode {ep + 1}/{len(dirs)}  {pb.run_dir.name}")
        header = [f"replay  {pb.controller}  seed {pb.seed}", "Space  ,/. decision  <-/-> tick  [] speed  C crash"]
        renderer.draw(env, pb.controller, pb.action_at(t), lat, debug, inspector=ins, ghost=ghost,
                      compass=compass, panel_mode=PANEL_MODES[panel_mode], header=header,
                      raw_request=raw_req, raw_response=raw_resp,
                      status=f"REPLAY {state} {SPEEDS[speed_i]:g}x",
                      timeline=timeline_data(ins, pb.total, t, label))
        clock.tick(60)
        frames += 1
        if max_frames and frames >= max_frames:
            if screenshot:
                pygame.image.save(renderer.screen, screenshot)
            if own:
                renderer.close()
            return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="episode dir (with events.jsonl) or a benchmark run dir")
    ap.add_argument("--at", choices=("start", "collision"), default="start")
    ap.add_argument("--panel", choices=("decisions", "request", "response"), default="decisions")
    ap.add_argument("--scale", type=float, default=None, help="arena zoom (default: fit screen)")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    run_viewer(Path(args.path), start=args.at, max_frames=args.frames, screenshot=args.screenshot,
               panel=args.panel, scale=args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
