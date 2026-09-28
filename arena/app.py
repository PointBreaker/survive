"""Decision Arena desktop console: one page, three columns.

    python main.py                      # open the console
    python main.py --controller jev     # open it with Jev selected and start right away

* Left: controller, parameters, run controls, Jev connection check, recent
  runs. Parameter changes restart the episode on the same seed, because they
  are rules every controller is told at the start (a running episode keeps
  running; switching controller waits for Start).
* Centre: the live arena, or the replay of a saved run (same column).
* Right: the inspector: requests in flight, responses, latency, raw JSON.

All of this is presentation. Episodes still run through the same
EpisodeRunner with the same rules as the headless benchmark.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from urllib.parse import urlsplit

import pygame

from arena.difficulty import PRESETS, DifficultyConfig
from arena.display import Display
from arena.dotenv import load_dotenv
from arena.inspector import TICK_MS, Inspector, TeeRecorder, timeline_data
from arena.recorder import JsonlRecorder, NullRecorder, new_run_dir
from arena.renderer import PANEL_MODES, Renderer
from arena.ui import (ACCENT, BAD, BG, BORDER, FAINT, MUTED, OK, SURFACE, SURFACE_2, TEXT, UI, WARN)

TICK = TICK_MS / 1000.0
SPEED_STEPS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

CONTROLLERS = [
    ("human", "Human", "You play: WASD / arrow keys"),
    ("simple_avoid", "SimpleAvoid", "Heuristic: attraction + repulsion"),
    ("greedy", "Greedy", "Chases the target, ignores obstacles"),
    ("random", "Random", "Random actions: the floor"),
    ("jev", "Jev", "TypeSafe Jev over the network"),
]


# ============================================================== settings
@dataclass
class Settings:
    controller: str = "simple_avoid"
    preset: str = "medium"
    obstacles: int = PRESETS["medium"].obstacle_count
    world_speed: float = 1.0
    decision_hz: float = 10.0
    max_inflight: int = 1
    latency_ms: float = 0.0
    max_duration: float = 60.0
    seed: int = 0
    save_logs: bool = True
    out: str = "runs"
    endpoint: Optional[str] = None
    real_latency: bool = False
    base: Optional[DifficultyConfig] = None  # exact config from the CLI, if any

    def config(self) -> DifficultyConfig:
        base = self.base or PRESETS[self.preset]
        return base.with_overrides(
            obstacle_count=self.obstacles,
            world_speed_scale=self.world_speed,
            decision_hz=self.decision_hz,
            max_inflight=self.max_inflight,
            max_duration=self.max_duration,
        )

    @classmethod
    def from_config(cls, cfg: DifficultyConfig, **kw) -> "Settings":
        return cls(obstacles=cfg.obstacle_count, world_speed=cfg.world_speed_scale, decision_hz=cfg.decision_hz,
                   max_inflight=cfg.max_inflight, max_duration=cfg.max_duration, base=cfg, **kw)

    def controller_factory(self):
        """Returns (factory, error message)."""
        from arena.cli import controller_factory

        ns = SimpleNamespace(controller_seed=12345, sleep_ms=500.0, endpoint=self.endpoint,
                             latency_ms=self.latency_ms, real_latency=self.real_latency)
        try:
            return controller_factory(self.controller, ns), None
        except SystemExit as e:
            return None, str(e).replace("error: ", "")


def jev_status() -> tuple[bool, str]:
    if os.environ.get("OPENROUTER_API_KEY"):
        return True, f"OpenRouter · {os.environ.get('JEV_MODEL') or 'typesafe/jev-1.13'}"
    if os.environ.get("TYPESAFE_API_KEY"):
        return True, f"TypeSafe API · {os.environ.get('JEV_MODEL') or 'jev-latest'}"
    if os.environ.get("JEV_API_KEY"):
        return True, "custom endpoint"
    return False, "no token: add OPENROUTER_API_KEY to .env"


def recent_runs(root: str, limit: int = 8) -> list[tuple[Path, str]]:
    base = Path(root)
    if not base.is_dir():
        return []
    items = []
    for d in base.iterdir():
        if (d / "events.jsonl").is_file():
            items.append((d.stat().st_mtime, d, d.name))
        elif (d / "episodes").is_dir():
            n = sum(1 for e in (d / "episodes").iterdir() if (e / "events.jsonl").is_file())
            if n:
                items.append((d.stat().st_mtime, d, f"{d.name}  ({n} episodes)"))
    items.sort(reverse=True)
    return [(d, label) for _, d, label in items[:limit]]


# ================================================================== app
class App:
    def __init__(self, display: Optional[Display] = None, fps_cap: int = 240):
        load_dotenv()
        self.display = display or Display()
        self.renderer = Renderer(self.display)
        self.ui = UI(self.renderer.t)
        self.clock = pygame.time.Clock()
        self.fps_cap = fps_cap
        self.scene: Any = None
        self.running = True
        self.settings = Settings()
        self.toast: Optional[tuple[str, float]] = None

    @property
    def t(self):
        return self.renderer.t

    def switch(self, scene) -> None:
        if self.scene is not None and hasattr(self.scene, "close"):
            self.scene.close()
        self.scene = scene

    def notify(self, msg: str, seconds: float = 3.0) -> None:
        self.toast = (msg, time.perf_counter() + seconds)

    def run(self, scene, max_frames: int = 0, screenshot: Optional[str] = None) -> None:
        self.scene = scene
        frames = 0
        while self.running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                    break
                if self.display.handle_resize(ev):
                    continue
                if hasattr(ev, "pos"):
                    ev.px = self.display.to_pixels(ev.pos)
                self.scene.handle(ev)
            if not self.running:
                break
            self.scene.update()
            mouse = self.display.to_pixels(pygame.mouse.get_pos())
            self.ui.t = self.renderer.t
            self.ui.begin(mouse)
            self.scene.draw()
            self._draw_toast()
            self.display.present()
            self.clock.tick(self.fps_cap)
            frames += 1
            if max_frames and frames >= max_frames:
                if screenshot:
                    self.display.save(screenshot)
                break
        if self.scene is not None and hasattr(self.scene, "close"):
            self.scene.close()
        self.display.close()

    def _draw_toast(self) -> None:
        if not self.toast:
            return
        msg, until = self.toast
        if time.perf_counter() > until:
            self.toast = None
            return
        t, surf = self.t, self.display.surface
        w = t.text_width(msg, 13) + t.u(32)
        r = pygame.Rect(0, 0, w, t.u(36))
        r.midbottom = (surf.get_width() // 2, surf.get_height() - t.u(60))
        pygame.draw.rect(surf, SURFACE_2, r, border_radius=t.u(8))
        pygame.draw.rect(surf, ACCENT, r, width=max(1, t.u(1)), border_radius=t.u(8))
        t.text(surf, msg, r.center, TEXT, 13, anchor="center")

    @property
    def fps(self) -> float:
        return self.clock.get_fps()



# ============================================================== helpers
def controller_header(ctrl) -> list[str]:
    inner = ctrl
    while hasattr(inner, "inner"):
        inner = inner.inner
    lines = []
    model = getattr(inner, "model", None)
    endpoint = getattr(inner, "endpoint", None)
    if model:
        lines.append(f"model     {model}")
    if endpoint:
        u = urlsplit(endpoint)
        lines.append(f"endpoint  {u.netloc}{u.path}")
    if inner is not ctrl:
        lines.append(f"wrapper   {ctrl.name}")
    return lines


def raw_traffic(ctrl, inspector: Inspector):
    """Wire bodies for remote controllers; the observation for in-process ones."""
    inner = ctrl
    while hasattr(inner, "inner"):
        inner = inner.inner
    req = getattr(inner, "last_request_body", None)
    resp = getattr(inner, "last_response", None)
    if req is None:
        r = inspector.inflight() or (inspector.requests.get(inspector.records[-1].id) if inspector.records else None)
        req = {"observation": r["observation"]} if r else None
    if resp is None and inspector.records:
        last = inspector.records[-1]
        resp = {k: v for k, v in last.__dict__.items() if v is not None}
    return req, resp



# ============================================================== console
class ConsoleScene:
    """One page: parameters (left) · arena (centre) · inspector (right).

    Live mode states: ``ready`` (world shown, not running), ``running``,
    ``finished``. Replay mode re-simulates a saved run in the centre column.
    """

    SIDEBAR_PT = 330
    REPLAY_SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(self, app: App, autostart: bool = False, panel: str = "decisions"):
        self.app = app
        self.s = app.settings
        self.debug, self.ghost, self.compass = False, True, True
        self.panel_mode = PANEL_MODES.index(panel)
        self.error: Optional[str] = None
        self.check: Optional[str] = None
        self.checking = False
        self.scroll = 0
        self.sidebar_content_h = 0
        self.layout = None
        self.mode = "live"
        self.runner = None
        self.rec = None
        self.pb = None
        self.runs = recent_runs(self.s.out)
        app.renderer.left_pt = self.SIDEBAR_PT
        app.renderer.timeline_pt = Renderer.TIMELINE_PT
        self.prepare(self.s.seed)
        if autostart:
            self.start()

    # ================================================================ live
    def prepare(self, seed: int) -> None:
        """Show a fresh world for (settings, seed) without running it."""
        from arena.environment import Environment

        self.stop_episode()
        self.seed = seed
        self.s.seed = seed
        self.cfg = self.s.config()
        self.env = Environment(self.cfg, seed)
        self.inspector = Inspector()
        self.state = "ready"
        self.result = None
        self.run_dir = None
        self.header: list[str] = []
        self.max_ticks = int(self.cfg.max_duration / self.cfg.world_speed_scale * 60)

    def start(self) -> None:
        from arena.environment import Environment
        from arena.runner import EpisodeRunner

        if self.mode != "live":
            self.exit_replay()
        if self.state != "ready":
            self.prepare(self.seed)
        factory, err = self.s.controller_factory()
        if err:
            self.error = err
            return
        self.error = None
        ctrl = factory()
        self.run_dir = new_run_dir(self.s.out, ctrl.name, f"seed{self.seed}") if self.s.save_logs else None
        inner = JsonlRecorder(self.run_dir) if self.run_dir else NullRecorder()
        self.inspector = Inspector()
        self.rec = TeeRecorder(inner, self.inspector)
        self.env = Environment(self.cfg, self.seed)
        self.runner = EpisodeRunner(self.env, ctrl, recorder=self.rec, realtime=True)
        self.runner.start()
        self.header = controller_header(ctrl)
        self.state = "running"

    def stop_episode(self) -> None:
        if self.runner is not None:
            if self.state == "running":
                self.runner.finish()
            self.runner.controller.close()
            self.rec.close()
        self.runner = None
        self.rec = None

    def restart(self, seed: Optional[int] = None) -> None:
        self.prepare(self.seed if seed is None else seed)
        self.start()

    def settings_changed(self, keep_running: bool = True) -> None:
        was_running = self.mode == "live" and self.state == "running"
        self.prepare(self.seed)
        if was_running and keep_running:
            self.start()

    def close(self) -> None:
        self.stop_episode()
        self.app.renderer.left_pt = 0

    # ============================================================== replay
    def open_replay(self, path) -> None:
        from arena.viewer import Playback, episode_dirs

        try:
            dirs = episode_dirs(Path(path))
        except SystemExit as e:
            self.error = str(e)
            return
        self.stop_episode()
        self.state = "ready"
        self.mode = "replay"
        self.dirs, self.ep = dirs, 0
        self.pb = Playback(dirs[0])
        self.rtick, self.playing, self.rspeed = 0.0, True, 1.0
        self.last_time = time.perf_counter()
        self.dragging = False
        self.app.renderer.timeline_pt = 92

    def exit_replay(self) -> None:
        self.mode = "live"
        self.pb = None
        self.app.renderer.timeline_pt = Renderer.TIMELINE_PT
        self.prepare(self.seed)

    def replay_episode(self, i: int) -> None:
        from arena.viewer import Playback

        self.ep = i % len(self.dirs)
        self.pb = Playback(self.dirs[self.ep])
        self.rtick, self.playing = 0.0, True

    def jump_collision(self) -> None:
        if self.pb and self.pb.collision_tick is not None:
            self.rtick, self.playing = float(max(0, self.pb.collision_tick - 120)), False

    def step_decision(self, direction: int) -> None:
        import bisect

        dt = self.pb.decision_ticks
        if direction > 0:
            i = bisect.bisect_right(dt, int(self.rtick))
            if i < len(dt):
                self.rtick = float(dt[i])
        else:
            i = bisect.bisect_left(dt, int(self.rtick)) - 1
            if i >= 0:
                self.rtick = float(dt[i])
        self.playing = False

    def toggle_play(self) -> None:
        if self.rtick >= self.pb.total:
            self.rtick = 0.0
        self.playing = not self.playing

    # =============================================================== input
    def handle(self, ev) -> None:
        if ev.type == pygame.KEYDOWN:
            self.on_key(ev.key, ev.mod & pygame.KMOD_SHIFT)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            h = self.app.ui.hit(ev.px)
            if h:
                self.on(*h)
            elif self.mode == "replay" and self.layout is not None and self.layout.timeline.collidepoint(ev.px):
                self.dragging, self.playing = True, False
                self.rtick = float(self.app.renderer.timeline_tick_at(self.layout, ev.px[0], self.pb.total))
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.mode == "replay":
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.mode == "replay" and self.dragging and self.layout:
            self.rtick = float(self.app.renderer.timeline_tick_at(self.layout, ev.px[0], self.pb.total))
        elif ev.type == pygame.MOUSEWHEEL and self.layout is not None and self.layout.sidebar is not None:
            if self.layout.sidebar.collidepoint(self.app.ui.mouse):
                max_scroll = max(0, self.sidebar_content_h - self.layout.sidebar.h)
                self.scroll = max(0, min(max_scroll, self.scroll - ev.y * self.app.t.u(40)))

    def on_key(self, k, shift) -> None:
        if k == pygame.K_j:
            self.panel_mode = (self.panel_mode + 1) % len(PANEL_MODES)
        elif k == pygame.K_g:
            self.ghost = not self.ghost
        elif k == pygame.K_p:
            self.compass = not self.compass
        elif k in (pygame.K_F1, pygame.K_TAB):
            self.debug = not self.debug
        elif self.mode == "replay":
            if k == pygame.K_ESCAPE:
                self.exit_replay()
            elif k == pygame.K_SPACE:
                self.toggle_play()
            elif k == pygame.K_RIGHT:
                self.rtick, self.playing = min(self.pb.total, int(self.rtick) + (10 if shift else 1)), False
            elif k == pygame.K_LEFT:
                self.rtick, self.playing = max(0, int(self.rtick) - (10 if shift else 1)), False
            elif k == pygame.K_PERIOD:
                self.step_decision(+1)
            elif k == pygame.K_COMMA:
                self.step_decision(-1)
            elif k == pygame.K_c:
                self.jump_collision()
            elif k in (pygame.K_PAGEDOWN, pygame.K_PAGEUP) and len(self.dirs) > 1:
                self.replay_episode(self.ep + (1 if k == pygame.K_PAGEDOWN else -1))
        else:
            if k in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.state == "running":
                    self.prepare(self.seed)
                else:
                    self.start()
            elif k == pygame.K_r:
                self.restart()
            elif k == pygame.K_n:
                self.restart(self.seed + 1)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.step_speed(-1)
            elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.step_speed(+1)
            elif k == pygame.K_v and self.state == "finished" and self.run_dir:
                self.open_replay(self.run_dir)
            elif k == pygame.K_ESCAPE and self.state == "running":
                self.prepare(self.seed)

    def step_speed(self, direction: int) -> None:
        steps = list(SPEED_STEPS)
        i = min(range(len(steps)), key=lambda j: abs(steps[j] - self.s.world_speed))
        i = max(0, min(len(steps) - 1, i + direction))
        if steps[i] != self.s.world_speed:
            self.s.world_speed = steps[i]
            self.settings_changed()

    def on(self, key: str, v: Any) -> None:
        s = self.s
        if key == "controller":
            s.controller = v
            if v == "human":
                s.decision_hz, s.max_inflight = 60.0, 1
            elif s.decision_hz == 60.0:
                s.decision_hz = 10.0
            self.check, self.error = None, None
            if self.mode == "replay":
                self.exit_replay()
            else:
                self.settings_changed(keep_running=False)
            return
        param_keys = {"preset", "obstacles", "obstacles_step", "speed", "hz", "inflight", "latency", "duration",
                      "seed", "logs"}
        if key in param_keys:
            if key == "preset":
                s.preset, s.base = v, None
                s.obstacles = PRESETS[v].obstacle_count
            elif key == "obstacles_step":
                s.obstacles = max(0, min(400, s.obstacles + v * (1 if s.obstacles < 20 else 5)))
            elif key == "obstacles":
                s.obstacles = v
            elif key == "speed":
                s.world_speed = v
            elif key == "hz":
                s.decision_hz = v
            elif key == "inflight":
                s.max_inflight = v
            elif key == "latency":
                s.latency_ms = v
            elif key == "duration":
                s.max_duration = v
            elif key == "seed":
                self.seed = max(0, self.seed + v)
            elif key == "logs":
                s.save_logs = v
            if self.mode == "replay":
                self.exit_replay()
            else:
                self.settings_changed()
            return
        if key == "start":
            self.start()
        elif key == "stop":
            self.prepare(self.seed)
        elif key == "restart":
            self.restart()
        elif key == "next":
            self.restart(self.seed + 1)
        elif key == "check":
            self.run_check()
        elif key == "open_run":
            self.open_replay(v)
        elif key == "replay_last" and self.run_dir:
            self.open_replay(self.run_dir)
        elif key == "exit_replay":
            self.exit_replay()
        elif key == "play":
            self.toggle_play()
        elif key == "dec":
            self.step_decision(v)
        elif key == "crash":
            self.jump_collision()
        elif key == "rspeed":
            self.rspeed = v
        elif key == "ep":
            self.replay_episode(self.ep + v)
        elif key == "quit":
            self.app.running = False

    def run_check(self) -> None:
        """One real Jev call in the background: latency + parsed answer."""
        if self.checking:
            return
        self.checking, self.check = True, "calling Jev…"
        cfg = self.cfg

        def work():
            try:
                from arena.environment import Environment
                from controllers.jev import JevController

                c = JevController(endpoint=self.s.endpoint)
                env = Environment(cfg, 0)
                c.reset(env.public_info())
                t0 = time.perf_counter()
                d = c.decide(env.observe())
                ms = (time.perf_counter() - t0) * 1000
                c.close()
                if d.action is None:
                    self.check = f"failed after {ms:.0f} ms: {d.error}"
                else:
                    conf = (d.meta or {}).get("confidence")
                    cs = f", confidence {conf:.2f}" if isinstance(conf, (int, float)) else ""
                    self.check = f"OK · {ms:.0f} ms · answered {d.action.value}{cs}"
            except Exception as e:
                self.check = f"error: {e}"
            self.checking = False

        threading.Thread(target=work, daemon=True).start()

    # ============================================================== update
    def update(self) -> None:
        if self.mode == "replay":
            now = time.perf_counter()
            dt = min(0.1, now - self.last_time)
            self.last_time = now
            if self.playing:
                self.rtick += self.rspeed * dt * 60.0
                if self.rtick >= self.pb.total:
                    self.rtick, self.playing = float(self.pb.total), False
            return
        if self.state != "running" or self.runner is None:
            return
        self.runner.advance_realtime()
        if self.runner.env.done:
            self.result = self.runner.finish()
            self.rec.close()
            self.runner.controller.close()
            self.state = "finished"
            self.runs = recent_runs(self.s.out)

    # ================================================================ draw
    def draw(self) -> None:
        if self.mode == "replay":
            L = self.draw_replay_centre()
        else:
            L = self.draw_live_centre()
        self.layout = L
        self.draw_sidebar(L)

    def draw_live_centre(self):
        app, t, ui = self.app, self.app.t, self.app.ui
        env = self.env
        r = self.runner
        alpha = 0.0
        if self.state == "running" and r is not None:
            alpha = max(0.0, min(1.0, (time.perf_counter() - r.wall_origin) / TICK - env.tick))
        lat = None if (r is None or r.last_latency_s is None) else r.last_latency_s * 1000
        action = r.current_action.value if r is not None else "STAY"
        name = r.controller.name if r is not None else self.s.controller
        req, resp = raw_traffic(r.controller, self.inspector) if r is not None else (None, None)
        card = None
        if self.state == "finished" and self.result:
            res = self.result
            card = [f"survived {res['survival_time']:.1f} s   ·   {res['targets_collected']} targets",
                    f"{res['decision_count']} decisions   ·   mean latency "
                    f"{(res['mean_decision_latency_ms'] or 0):.0f} ms",
                    f"seed {res['seed']}   ·   world {self.cfg.world_speed_scale:g}×"]
        label = {"ready": "ready · press Start or Space", "running": f"live · tick {env.tick}",
                 "finished": f"finished · tick {env.tick}"}[self.state]
        L = app.renderer.draw(
            env, name, action, lat, alpha=alpha, debug=self.debug, inspector=self.inspector, ghost=self.ghost,
            compass=self.compass, panel_mode=PANEL_MODES[self.panel_mode], header=self.header,
            raw_request=req, raw_response=resp, fps=app.fps, card_lines=card,
            timeline=timeline_data(self.inspector, self.max_ticks, env.tick,
                                   label + " · timeline spans the full episode length"),
        )
        surf = app.display.surface
        if self.state == "ready":
            box = pygame.Rect(0, 0, t.u(360), t.u(96))
            box.center = L.arena.center
            pygame.draw.rect(surf, SURFACE, box, border_radius=t.u(12))
            pygame.draw.rect(surf, BORDER, box, width=max(1, t.u(1)), border_radius=t.u(12))
            nm = dict((k, n) for k, n, _ in CONTROLLERS).get(self.s.controller, self.s.controller)
            t.text(surf, f"Ready: {nm} · seed {self.seed}", (box.centerx, box.y + t.u(16)), TEXT, 14, bold=True,
                   anchor="midtop")
            b = pygame.Rect(0, 0, t.u(150), t.u(36))
            b.midbottom = (box.centerx, box.bottom - t.u(14))
            ui.button(surf, b, "Start  (Space)", "start", kind="primary", size=13)
        if L.card is not None:
            gap = t.u(6)
            labels = [("replay_last", "Replay  V"), ("restart", "Restart  R"), ("next", "Next seed  N")]
            if not self.run_dir:
                labels = labels[1:]
            bw = (L.card.w - t.u(24) - gap * (len(labels) - 1)) // len(labels)
            bx = L.card.x + t.u(12)
            for i, (key, lab) in enumerate(labels):
                ui.button(surf, pygame.Rect(bx, L.card_buttons_y, bw, t.u(36)), lab, key,
                          kind="primary" if i == 0 else "normal", size=12.5)
                bx += bw + gap
        return L

    def draw_replay_centre(self):
        app, t, ui = self.app, self.app.t, self.app.ui
        pb = self.pb
        tk = int(self.rtick)
        env = pb.env_at(tk)
        ins = pb.inspector_at(tk)
        last = ins.last_applied()
        req = ins.inflight() or (ins.requests.get(ins.records[-1].id) if ins.records else None)
        raw_req = {"observation": req["observation"]} if req else None
        raw_resp = {k: v for k, v in ins.records[-1].__dict__.items() if v is not None} if ins.records else None
        st = "playing" if self.playing else "paused"
        label = (f"replay · {st} {self.rspeed:g}× · tick {tk}/{pb.total} ({tk * TICK:.2f} s) · "
                 f"episode {self.ep + 1}/{len(self.dirs)} · {pb.run_dir.name}")
        L = app.renderer.draw(
            env, f"replay · {pb.controller}", pb.action_at(tk), last.latency_ms if last else None,
            alpha=(self.rtick - tk) if self.playing else 0.0, debug=self.debug, inspector=ins, ghost=self.ghost,
            compass=self.compass, panel_mode=PANEL_MODES[self.panel_mode], header=[f"seed {pb.seed}"],
            raw_request=raw_req, raw_response=raw_resp, timeline=timeline_data(ins, pb.total, tk, label),
            fps=app.fps)
        surf = app.display.surface
        bh, gap = t.u(30), t.u(6)
        y = L.timeline.y + t.u(10)
        x = L.timeline.x + t.u(12)
        items = [("‹ Exit replay", "exit_replay", None, 104, "normal"), ("|‹", "dec", -1, 36, "normal"),
                 ("Pause" if self.playing else "Play", "play", None, 60, "primary"), ("›|", "dec", +1, 36, "normal"),
                 ("crash", "crash", None, 56, "normal")]
        if len(self.dirs) > 1:
            items += [("‹ ep", "ep", -1, 46, "normal"), ("ep ›", "ep", +1, 46, "normal")]
        for lab, key, v, w, kind in items:
            rect = pygame.Rect(x, y, t.u(w), bh)
            if rect.right > L.timeline.right - t.u(8):
                break
            ui.button(surf, rect, lab, key, v, kind=kind, size=12.5)
            x = rect.right + gap
        x += t.u(8)
        for v in self.REPLAY_SPEEDS:
            rect = pygame.Rect(x, y, t.u(42), bh)
            if rect.right > L.timeline.right - t.u(8):
                break
            ui.button(surf, rect, f"{v:g}×", "rspeed", v, selected=(v == self.rspeed), size=12)
            x = rect.right + t.u(4)
        return L

    # ------------------------------------------------------------- sidebar
    def draw_sidebar(self, L) -> None:
        app, t, ui, s = self.app, self.app.t, self.app.ui, self.s
        surf = app.display.surface
        sb = L.sidebar
        pygame.draw.rect(surf, SURFACE, sb)
        pygame.draw.line(surf, BORDER, sb.topright, sb.bottomright, max(1, t.u(1)))
        prev_clip = surf.get_clip()
        surf.set_clip(sb)
        x = sb.x + t.u(18)
        w = sb.w - t.u(36)
        y0 = t.u(16) - self.scroll
        y = y0

        t.text(surf, "Decision Arena", (x, y), TEXT, 19, bold=True)
        y += t.u(34)

        def section(title):
            nonlocal y
            t.text(surf, title, (x, y), FAINT, 11, bold=True)
            y += t.u(20)

        def chips(options, selected, key, min_w=38):
            nonlocal y
            y = ui.chips(surf, x, y, options, selected, key, min_w=min_w, h=26, size=12) + t.u(12)

        def label(text, right=None):
            nonlocal y
            t.text(surf, text, (x, y), MUTED, 12)
            if right:
                t.text(surf, right, (x + w, y), TEXT, 12, bold=True, anchor="topright")
            y += t.u(18)

        # ---- controller
        section("CONTROLLER")
        ok, jev_msg = jev_status()
        cw = (w - t.u(8)) // 2
        for i, (key, title, _) in enumerate(CONTROLLERS):
            r = pygame.Rect(x + (i % 2) * (cw + t.u(8)), y + (i // 2) * t.u(36), cw, t.u(30))
            ui.button(surf, r, title, "controller", key, selected=(s.controller == key), size=12.5)
        y += t.u(36) * ((len(CONTROLLERS) + 1) // 2) + t.u(2)
        desc = dict((k, d) for k, _, d in CONTROLLERS)[s.controller]
        t.text(surf, jev_msg if s.controller == "jev" else desc, (x, y), WARN if (s.controller == "jev" and not ok)
               else MUTED, 11.5)
        y += t.u(20)
        if s.controller == "jev":
            b = pygame.Rect(x, y, t.u(140), t.u(28))
            ui.button(surf, b, "Check connection", "check", enabled=ok and not self.checking, size=12)
            y += t.u(34)
            if self.check:
                col = OK if self.check.startswith("OK") else (MUTED if self.checking else BAD)
                y = self._wrap(surf, self.check, x, y, w, col, 11.5) + t.u(6)
        y += t.u(8)

        # ---- run controls
        if self.mode == "live":
            if self.state == "running":
                ui.button(surf, pygame.Rect(x, y, t.u(96), t.u(38)), "Stop", "stop", size=13.5)
            else:
                ui.button(surf, pygame.Rect(x, y, t.u(96), t.u(38)), "Start", "start", kind="primary", size=13.5)
            ui.button(surf, pygame.Rect(x + t.u(102), y, t.u(84), t.u(38)), "Restart", "restart", size=12.5)
            ui.button(surf, pygame.Rect(x + t.u(192), y, w - t.u(192), t.u(38)), "Next seed", "next", size=12.5)
        else:
            ui.button(surf, pygame.Rect(x, y, w, t.u(38)), "‹ Back to live", "exit_replay", size=13)
        y += t.u(44)
        if self.error:
            y = self._wrap(surf, self.error, x, y, w, BAD, 11.5) + t.u(6)
        t.text(surf, "Parameter changes restart the episode (same seed).", (x, y), FAINT, 10.5)
        y += t.u(22)

        # ---- parameters
        section("PARAMETERS")
        label("Difficulty preset")
        chips([(p, p) for p in ("easy", "medium", "hard")], s.preset if s.base is None else None, "preset", 64)
        t.text(surf, "Obstacles", (x, y + t.u(5)), MUTED, 12)
        ui.stepper(surf, x + w - t.u(28 * 2 + 50 + 8), y, str(s.obstacles), "obstacles_step", w=50, h=26)
        y += t.u(32)
        chips([(n, str(n)) for n in (5, 10, 20, 40, 80)], s.obstacles, "obstacles", 36)
        label("World speed  (world time vs real time)")
        chips([(v, f"{v:g}×") for v in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)], s.world_speed, "speed", 36)
        label("Decision rate (max)")
        chips([(v, f"{v:g}") for v in (2.0, 5.0, 10.0, 20.0, 60.0)], s.decision_hz, "hz", 40)
        label("Requests in flight")
        chips([(v, str(v)) for v in (1, 2, 3, 4)], s.max_inflight, "inflight", 40)
        label("Added latency (simulated)")
        chips([(v, f"{v:g}" if v else "0") for v in (0.0, 100.0, 200.0, 300.0, 500.0)], s.latency_ms, "latency", 40)
        label("Episode length (world s)")
        chips([(v, f"{v:g}") for v in (20.0, 60.0, 120.0)], s.max_duration, "duration", 44)
        t.text(surf, "Seed", (x, y + t.u(5)), MUTED, 12)
        ui.stepper(surf, x + w - t.u(28 * 2 + 50 + 8), y, str(self.seed), "seed", w=50, h=26)
        y += t.u(36)
        ui.toggle(surf, x, y, "save run logs", s.save_logs, "logs")
        y += t.u(34)

        # ---- recent runs
        section("RECENT RUNS")
        if not self.runs:
            t.text(surf, "none yet", (x, y), FAINT, 11.5)
            y += t.u(20)
        for path, lab in self.runs:
            r = pygame.Rect(x - t.u(6), y - t.u(3), w + t.u(12), t.u(22))
            hover = r.collidepoint(ui.mouse) and sb.collidepoint(ui.mouse)
            if hover:
                pygame.draw.rect(surf, SURFACE_2, r, border_radius=t.u(4))
            short = lab if len(lab) < 40 else lab[:38] + "…"
            t.text(surf, short, (x, y), ACCENT if hover else MUTED, 11, mono=True)
            if sb.colliderect(r):
                ui.hits.append((r.clip(sb), "open_run", path, True))
            y += t.u(22)
        y += t.u(10)
        t.text(surf, "keys: Space start/stop · R · N · -/= speed · J · G · P", (x, y), FAINT, 10.5)
        y += t.u(24)
        self.sidebar_content_h = y - y0 + t.u(16)
        surf.set_clip(prev_clip)

    def _wrap(self, surf, text, x, y, width, color, size) -> int:
        t = self.app.t
        line = ""
        for word in str(text).split():
            trial = (line + " " + word).strip()
            if t.text_width(trial, size) > width and line:
                t.text(surf, line, (x, y), color, size)
                y += t.u(size + 5)
                line = word
            else:
                line = trial
        if line:
            t.text(surf, line, (x, y), color, size)
            y += t.u(size + 5)
        return y


# =================================================================== entry
def run_app(settings: Optional[Settings] = None, autostart: bool = False, replay_path=None, replay_at=None,
            panel: str = "decisions", max_frames: int = 0, screenshot: Optional[str] = None,
            window: Optional[tuple[int, int]] = None) -> int:
    app = App(Display(window))
    if settings is not None:
        app.settings = settings
    scene = ConsoleScene(app, autostart=autostart and replay_path is None, panel=panel)
    if replay_path is not None:
        scene.open_replay(replay_path)
        if replay_at == "collision":
            scene.jump_collision()
    app.run(scene, max_frames=max_frames, screenshot=screenshot)
    return 0
