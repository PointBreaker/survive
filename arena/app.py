"""Decision Arena desktop app: launcher menu, live play, replay, in one window.

    python main.py                      # opens the launcher
    python main.py --controller jev     # skips straight into a game (Esc -> launcher)

Scenes:
* MenuScene: pick a controller (Human / SimpleAvoid / Greedy / Random / Jev)
  and click to set the parameters; browse and replay recent runs.
* PlayScene: the live arena with the inspector panel. World speed can be
  changed with the HUD buttons; that restarts the episode on the same seed,
  because world speed is a rule every controller is told at the start.
* ReplayScene: exact re-simulation of a logged episode with transport
  controls.

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


# ================================================================= menu
class MenuScene:
    def __init__(self, app: App, error: Optional[str] = None):
        self.app = app
        self.s = app.settings
        self.error = error
        self.check: Optional[str] = None
        self.checking = False
        self.runs = recent_runs(self.s.out)

    # -------------------------------------------------------------- input
    def handle(self, ev) -> None:
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.start()
            elif ev.key == pygame.K_ESCAPE:
                self.app.running = False
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            h = self.app.ui.hit(ev.px)
            if h:
                self.on(*h)

    def on(self, key: str, v: Any) -> None:
        s = self.s
        self.error = None
        if key == "controller":
            s.controller = v
            if v == "human":
                s.decision_hz, s.max_inflight = 60.0, 1
            elif s.decision_hz == 60.0:
                s.decision_hz = 10.0
            self.check = None
        elif key == "preset":
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
            s.seed = max(0, s.seed + v)
        elif key == "logs":
            s.save_logs = v
        elif key == "start":
            self.start()
        elif key == "check":
            self.run_check()
        elif key == "replay":
            self.app.switch(ReplayScene(self.app, v, back=lambda: MenuScene(self.app)))
        elif key == "quit":
            self.app.running = False

    def start(self) -> None:
        factory, err = self.s.controller_factory()
        if err:
            self.error = err
            return
        self.app.switch(PlayScene(self.app, self.s, factory))

    def run_check(self) -> None:
        """One real Jev call in the background: latency + parsed answer."""
        if self.checking:
            return
        self.checking, self.check = True, "calling Jev…"

        def work():
            try:
                from arena.environment import Environment
                from controllers.jev import JevController

                c = JevController(endpoint=self.s.endpoint)
                env = Environment(self.s.config(), 0)
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

    def update(self) -> None:
        pass

    # --------------------------------------------------------------- draw
    def draw(self) -> None:
        app, t, ui, s = self.app, self.app.t, self.app.ui, self.s
        surf = app.display.surface
        W, H = surf.get_size()
        surf.fill(BG)
        content_w = min(W - t.u(64), t.u(1240))
        x0 = (W - content_w) // 2
        y = t.u(36)
        t.text(surf, "Decision Arena", (x0, y), TEXT, 30, bold=True)
        t.text(surf, "real-time closed-loop decision benchmark", (x0, y + t.u(40)), MUTED, 14)
        y += t.u(84)

        left_w = int(content_w * 0.44)
        rx = x0 + left_w + t.u(40)
        right_w = content_w - left_w - t.u(40)

        # ---- controllers
        t.text(surf, "CONTROLLER", (x0, y), FAINT, 11.5, bold=True)
        cy = y + t.u(22)
        cw = (left_w - t.u(12)) // 2
        ch = t.u(64)
        ok, jev_msg = jev_status()
        for i, (key, title, desc) in enumerate(CONTROLLERS):
            r = pygame.Rect(x0 + (i % 2) * (cw + t.u(12)), cy + (i // 2) * (ch + t.u(12)), cw, ch)
            self._card(surf, r, key, title, jev_msg if key == "jev" else desc,
                       warn=(key == "jev" and not ok))
        cy += 3 * (ch + t.u(12))
        if s.controller == "jev":
            b = pygame.Rect(x0, cy, t.u(150), t.u(30))
            ui.button(surf, b, "Check connection", "check", enabled=ok and not self.checking, size=12.5)
            if self.check:
                col = OK if self.check.startswith("OK") else (MUTED if self.checking else BAD)
                self._wrap(surf, self.check, (b.right + t.u(12), b.y + t.u(7)), left_w - b.w - t.u(12), col, 12)
            cy += t.u(44)

        # ---- recent runs
        t.text(surf, "RECENT RUNS  (click to replay)", (x0, cy + t.u(8)), FAINT, 11.5, bold=True)
        ry = cy + t.u(30)
        if not self.runs:
            t.text(surf, "no saved runs yet", (x0, ry), FAINT, 12.5)
        for path, label in self.runs:
            if ry > H - t.u(60):
                break
            r = pygame.Rect(x0, ry, left_w, t.u(26))
            hover = r.collidepoint(ui.mouse)
            if hover:
                pygame.draw.rect(surf, SURFACE_2, r, border_radius=t.u(5))
            t.text(surf, label, (r.x + t.u(8), r.centery), ACCENT if hover else MUTED, 12, mono=True, anchor="midleft")
            ui.hits.append((r, "replay", path, True))
            ry += t.u(28)

        # ---- parameters
        t.text(surf, "PARAMETERS", (rx, y), FAINT, 11.5, bold=True)
        py = y + t.u(24)
        lx = rx
        cx = rx + t.u(130)
        row = t.u(44)

        def label(text, sub=None):
            t.text(surf, text, (lx, py + t.u(6)), TEXT, 13.5)
            if sub:
                t.text(surf, sub, (lx, py + t.u(24)), FAINT, 11.5)

        label("Difficulty", "preset")
        ui.chips(surf, cx, py, [(p, p) for p in ("easy", "medium", "hard")], s.preset if s.base is None else None,
                 "preset", min_w=70)
        py += row
        label("Obstacles")
        end = ui.stepper(surf, cx, py, f"{s.obstacles}", "obstacles_step", w=50)
        ui.chips(surf, cx + t.u(130), py, [(n, str(n)) for n in (5, 10, 20, 40, 80)], s.obstacles, "obstacles",
                 min_w=38)
        py += row
        label("World speed", "world time vs real time")
        ui.chips(surf, cx, py, [(v, f"{v:g}×") for v in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)], s.world_speed, "speed",
                 min_w=46)
        py += row
        label("Decision rate", "max requests/s")
        ui.chips(surf, cx, py, [(v, f"{v:g} Hz") for v in (2.0, 5.0, 10.0, 20.0, 60.0)], s.decision_hz, "hz",
                 min_w=52)
        py += row
        label("In flight", "concurrent requests")
        ui.chips(surf, cx, py, [(v, str(v)) for v in (1, 2, 3, 4)], s.max_inflight, "inflight", min_w=40)
        py += row
        label("Added latency", "simulated, on top")
        ui.chips(surf, cx, py, [(v, f"{v:g} ms" if v else "none") for v in (0.0, 100.0, 200.0, 300.0, 500.0)],
                 s.latency_ms, "latency", min_w=52)
        py += row
        label("Episode length", "world seconds")
        ui.chips(surf, cx, py, [(v, f"{v:g} s") for v in (20.0, 60.0, 120.0)], s.max_duration, "duration", min_w=52)
        py += row
        label("Seed")
        ui.stepper(surf, cx, py, str(s.seed), "seed", w=60)
        py += row
        ui.toggle(surf, cx, py + t.u(3), "save run logs (for replay)", s.save_logs, "logs")
        py += row + t.u(8)

        start = pygame.Rect(cx, py, t.u(240), t.u(52))
        name = dict((k, n) for k, n, _ in CONTROLLERS)[s.controller]
        ui.button(surf, start, f"Start  ·  {name}", "start", kind="primary", size=15, hint="press Enter")
        ui.button(surf, pygame.Rect(start.right + t.u(12), py, t.u(90), t.u(52)), "Quit", "quit", size=13)
        py += t.u(66)
        if self.error:
            self._wrap(surf, self.error, (cx, py), right_w - t.u(130), BAD, 12.5)
        elif s.controller == "human":
            t.text(surf, "Human input is read at 60 Hz. Move with WASD or the arrow keys.", (cx, py), MUTED, 12)
        elif s.controller == "jev" and s.max_inflight == 1:
            t.text(surf, "Tip: with ~300 ms latency, 3 requests in flight keeps decisions near 10/s.", (cx, py),
                   MUTED, 12)

    def _card(self, surf, r, key, title, desc, warn=False) -> None:
        t, ui = self.app.t, self.app.ui
        sel = self.s.controller == key
        hover = r.collidepoint(ui.mouse)
        pygame.draw.rect(surf, (30, 44, 66) if sel else (SURFACE_2 if hover else SURFACE), r, border_radius=t.u(10))
        pygame.draw.rect(surf, ACCENT if sel else BORDER, r, width=max(1, t.u(2 if sel else 1)),
                         border_radius=t.u(10))
        t.text(surf, title, (r.x + t.u(14), r.y + t.u(12)), TEXT, 15, bold=True)
        t.text(surf, desc, (r.x + t.u(14), r.y + t.u(36)), WARN if warn else MUTED, 11.5)
        ui.hits.append((r, "controller", key, True))

    def _wrap(self, surf, text, pos, width, color, size) -> None:
        t = self.app.t
        x, y = pos
        line = ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if t.text_width(trial, size) > width and line:
                t.text(surf, line, (x, y), color, size)
                y += t.u(size + 5)
                line = word
            else:
                line = trial
        if line:
            t.text(surf, line, (x, y), color, size)


# ================================================================= play
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


class PlayScene:
    def __init__(self, app: App, settings: Settings, factory, panel: str = "decisions"):
        self.app = app
        self.s = settings
        self.factory = factory
        self.cfg = settings.config()
        self.debug, self.ghost, self.compass = False, True, True
        self.panel_mode = PANEL_MODES.index(panel)
        self.runner = None
        app.renderer.timeline_pt = 92
        self.new_episode(settings.seed)

    def new_episode(self, seed: int) -> None:
        from arena.environment import Environment
        from arena.runner import EpisodeRunner

        self.close()
        self.seed = seed
        ctrl = self.factory()
        self.run_dir = new_run_dir(self.s.out, ctrl.name, f"seed{seed}") if self.s.save_logs else None
        inner = JsonlRecorder(self.run_dir) if self.run_dir else NullRecorder()
        self.inspector = Inspector()
        self.rec = TeeRecorder(inner, self.inspector)
        self.runner = EpisodeRunner(Environment(self.cfg, seed), ctrl, recorder=self.rec, realtime=True)
        self.runner.start()
        self.header = controller_header(ctrl)
        self.finished = False
        self.result = None
        self.max_ticks = int(self.cfg.max_duration / self.cfg.world_speed_scale * 60)

    def close(self) -> None:
        if self.runner is None:
            return
        if not self.finished:
            self.runner.finish()
        self.runner.controller.close()
        self.rec.close()
        self.runner = None

    # -------------------------------------------------------------- input
    def handle(self, ev) -> None:
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.to_menu()
            elif k == pygame.K_r:
                self.new_episode(self.seed)
            elif k == pygame.K_n:
                self.new_episode(self.seed + 1)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.change_speed(-1)
            elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.change_speed(+1)
            elif k == pygame.K_j:
                self.panel_mode = (self.panel_mode + 1) % len(PANEL_MODES)
            elif k == pygame.K_g:
                self.ghost = not self.ghost
            elif k == pygame.K_p:
                self.compass = not self.compass
            elif k in (pygame.K_F1, pygame.K_TAB):
                self.debug = not self.debug
            elif k == pygame.K_v and self.finished and self.run_dir:
                self.replay()
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            h = self.app.ui.hit(ev.px)
            if not h:
                return
            key, v = h
            if key == "menu":
                self.to_menu()
            elif key == "restart":
                self.new_episode(self.seed)
            elif key == "next":
                self.new_episode(self.seed + 1)
            elif key == "speed":
                self.change_speed(v)
            elif key == "replay":
                self.replay()

    def change_speed(self, direction: int) -> None:
        cur = self.cfg.world_speed_scale
        steps = list(SPEED_STEPS)
        i = min(range(len(steps)), key=lambda j: abs(steps[j] - cur))
        i = max(0, min(len(steps) - 1, i + direction))
        if steps[i] == cur:
            return
        self.s.world_speed = steps[i]
        self.cfg = replace(self.cfg, world_speed_scale=steps[i])
        self.new_episode(self.seed)
        self.app.notify(f"World speed {steps[i]:g}×: episode restarted on seed {self.seed} "
                        f"(controllers are told the speed at the start)")

    def to_menu(self) -> None:
        self.s.seed = self.seed
        self.app.switch(MenuScene(self.app))

    def replay(self) -> None:
        run_dir, s, factory = self.run_dir, self.s, self.factory
        self.app.switch(ReplayScene(self.app, run_dir, back=lambda: PlayScene(self.app, s, factory)))

    # ------------------------------------------------------------- update
    def update(self) -> None:
        if self.finished or self.runner is None:
            return
        self.runner.advance_realtime()
        if self.runner.env.done:
            self.result = self.runner.finish()
            self.rec.close()
            self.finished = True

    # --------------------------------------------------------------- draw
    def draw(self) -> None:
        app, t, ui = self.app, self.app.t, self.app.ui
        r = self.runner
        env = r.env
        alpha = 0.0
        if not self.finished:
            alpha = max(0.0, min(1.0, (time.perf_counter() - r.wall_origin) / TICK - env.tick))
        lat = None if r.last_latency_s is None else r.last_latency_s * 1000
        req, resp = raw_traffic(r.controller, self.inspector)
        card = None
        if self.finished and self.result:
            res = self.result
            card = [f"survived {res['survival_time']:.1f} s   ·   {res['targets_collected']} targets",
                    f"{res['decision_count']} decisions   ·   mean latency "
                    f"{(res['mean_decision_latency_ms'] or 0):.0f} ms",
                    f"seed {res['seed']}   ·   world {self.cfg.world_speed_scale:g}×"]
        L = app.renderer.draw(
            env, r.controller.name, r.current_action.value, lat, alpha=alpha, debug=self.debug,
            inspector=self.inspector, ghost=self.ghost, compass=self.compass,
            panel_mode=PANEL_MODES[self.panel_mode], header=self.header, raw_request=req, raw_response=resp,
            timeline=timeline_data(self.inspector, self.max_ticks, env.tick,
                                   f"live · tick {env.tick} · timeline spans the full episode length"),
            fps=app.fps, card_lines=card,
        )
        surf = app.display.surface
        # Controls row above the timeline track.
        bh, gap = t.u(30), t.u(6)
        y = L.timeline.y + t.u(10)
        x = L.timeline.x + t.u(12)
        for key, label, w in (("menu", "‹ Menu", 74), ("restart", "Restart  R", 92), ("next", "Next seed  N", 106)):
            rect = pygame.Rect(x, y, t.u(w), bh)
            ui.button(surf, rect, label, key, size=12.5)
            x = rect.right + gap
        x += t.u(14)
        lab = t.text(surf, "world speed", (x, y + bh // 2), MUTED, 12, anchor="midleft")
        x = lab.right + t.u(8)
        ui.button(surf, pygame.Rect(x, y, bh, bh), "−", "speed", -1, size=15)
        val = pygame.Rect(x + bh + t.u(4), y, t.u(52), bh)
        pygame.draw.rect(surf, SURFACE_2, val, border_radius=t.u(6))
        t.text(surf, f"{self.cfg.world_speed_scale:g}×", val.center, TEXT, 13, bold=True, anchor="center")
        ui.button(surf, pygame.Rect(val.right + t.u(4), y, bh, bh), "+", "speed", +1, size=15)
        # End-of-episode card buttons
        if L.card is not None:
            labels = [("replay", "Replay  V"), ("restart", "Restart  R"), ("next", "Next seed  N"), ("menu", "Menu")]
            if not self.run_dir:
                labels = labels[1:]
            bw = (L.card.w - t.u(24) - gap * (len(labels) - 1)) // len(labels)
            bx = L.card.x + t.u(12)
            for i, (key, label) in enumerate(labels):
                ui.button(surf, pygame.Rect(bx, L.card_buttons_y, bw, t.u(36)), label, key,
                          kind="primary" if i == 0 else "normal", size=12.5)
                bx += bw + gap


# =============================================================== replay
class ReplayScene:
    SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(self, app: App, path, back=None, start: Optional[str] = None, panel: str = "decisions"):
        from arena.viewer import Playback, episode_dirs

        self.app = app
        self.back = back
        self.dirs = episode_dirs(Path(path))
        self.ep = 0
        self._Playback = Playback
        self.pb = Playback(self.dirs[0])
        self.tick = 0.0
        self.playing = True
        self.speed = 1.0
        self.debug, self.ghost, self.compass = False, True, True
        self.panel_mode = PANEL_MODES.index(panel)
        self.dragging = False
        self.layout = None
        self.last_time = time.perf_counter()
        app.renderer.timeline_pt = 92
        if start == "collision":
            self.jump_collision()

    def jump_collision(self) -> None:
        if self.pb.collision_tick is not None:
            self.tick, self.playing = float(max(0, self.pb.collision_tick - 120)), False

    def set_episode(self, i: int) -> None:
        self.ep = i % len(self.dirs)
        self.pb = self._Playback(self.dirs[self.ep])
        self.tick, self.playing = 0.0, True

    def close(self) -> None:
        self.app.renderer.timeline_pt = Renderer.TIMELINE_PT

    def go_back(self) -> None:
        self.app.switch(self.back() if self.back else MenuScene(self.app))

    def step_decision(self, direction: int) -> None:
        import bisect

        dt = self.pb.decision_ticks
        if direction > 0:
            i = bisect.bisect_right(dt, int(self.tick))
            if i < len(dt):
                self.tick = float(dt[i])
        else:
            i = bisect.bisect_left(dt, int(self.tick)) - 1
            if i >= 0:
                self.tick = float(dt[i])
        self.playing = False

    def handle(self, ev) -> None:
        if ev.type == pygame.KEYDOWN:
            k, shift = ev.key, ev.mod & pygame.KMOD_SHIFT
            if k in (pygame.K_ESCAPE, pygame.K_q):
                self.go_back()
            elif k == pygame.K_SPACE:
                self.toggle_play()
            elif k == pygame.K_RIGHT:
                self.tick, self.playing = min(self.pb.total, int(self.tick) + (10 if shift else 1)), False
            elif k == pygame.K_LEFT:
                self.tick, self.playing = max(0, int(self.tick) - (10 if shift else 1)), False
            elif k == pygame.K_PERIOD:
                self.step_decision(+1)
            elif k == pygame.K_COMMA:
                self.step_decision(-1)
            elif k == pygame.K_RIGHTBRACKET:
                self.speed = self.SPEEDS[min(len(self.SPEEDS) - 1, self.SPEEDS.index(self.speed) + 1)]
            elif k == pygame.K_LEFTBRACKET:
                self.speed = self.SPEEDS[max(0, self.SPEEDS.index(self.speed) - 1)]
            elif k == pygame.K_HOME:
                self.tick = 0.0
            elif k == pygame.K_END:
                self.tick, self.playing = float(self.pb.total), False
            elif k == pygame.K_c:
                self.jump_collision()
            elif k in (pygame.K_PAGEDOWN, pygame.K_PAGEUP) and len(self.dirs) > 1:
                self.set_episode(self.ep + (1 if k == pygame.K_PAGEDOWN else -1))
            elif k == pygame.K_j:
                self.panel_mode = (self.panel_mode + 1) % len(PANEL_MODES)
            elif k == pygame.K_g:
                self.ghost = not self.ghost
            elif k == pygame.K_p:
                self.compass = not self.compass
            elif k in (pygame.K_F1, pygame.K_TAB):
                self.debug = not self.debug
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            h = self.app.ui.hit(ev.px)
            if h:
                key, v = h
                if key == "back":
                    self.go_back()
                elif key == "play":
                    self.toggle_play()
                elif key == "dec":
                    self.step_decision(v)
                elif key == "crash":
                    self.jump_collision()
                elif key == "rspeed":
                    self.speed = v
                elif key == "ep":
                    self.set_episode(self.ep + v)
            elif self.layout is not None and self.layout.timeline.collidepoint(ev.px):
                self.dragging, self.playing = True, False
                self.tick = float(self.app.renderer.timeline_tick_at(self.layout, ev.px[0], self.pb.total))
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging and self.layout is not None:
            self.tick = float(self.app.renderer.timeline_tick_at(self.layout, ev.px[0], self.pb.total))

    def toggle_play(self) -> None:
        if self.tick >= self.pb.total:
            self.tick = 0.0
        self.playing = not self.playing

    def update(self) -> None:
        now = time.perf_counter()
        dt = min(0.1, now - self.last_time)
        self.last_time = now
        if self.playing:
            self.tick += self.speed * dt * 60.0  # frame-rate independent playback
            if self.tick >= self.pb.total:
                self.tick, self.playing = float(self.pb.total), False

    def draw(self) -> None:
        app, t, ui = self.app, self.app.t, self.app.ui
        pb = self.pb
        tk = int(self.tick)
        env = pb.env_at(tk)
        ins = pb.inspector_at(tk)
        last = ins.last_applied()
        req = ins.inflight() or (ins.requests.get(ins.records[-1].id) if ins.records else None)
        raw_req = {"observation": req["observation"]} if req else None
        raw_resp = {k: v for k, v in ins.records[-1].__dict__.items() if v is not None} if ins.records else None
        state = "playing" if self.playing else "paused"
        label = (f"replay · {state} {self.speed:g}× · tick {tk}/{pb.total} ({tk * TICK:.2f} s controller time) · "
                 f"episode {self.ep + 1}/{len(self.dirs)} · {pb.run_dir.name}")
        alpha = (self.tick - tk) if self.playing else 0.0
        self.layout = L = app.renderer.draw(
            env, f"replay · {pb.controller}", pb.action_at(tk), last.latency_ms if last else None, alpha=alpha,
            debug=self.debug, inspector=ins, ghost=self.ghost, compass=self.compass,
            panel_mode=PANEL_MODES[self.panel_mode], header=[f"seed {pb.seed}", ",/. decision  ←/→ tick  C crash"],
            raw_request=raw_req, raw_response=raw_resp, timeline=timeline_data(ins, pb.total, tk, label),
            fps=app.fps, hud_reserve=t.u(70 + (180 if len(self.dirs) > 1 else 0)))
        surf = app.display.surface
        bh, gap = t.u(30), t.u(6)
        # HUD (right): back + episode switching
        y = L.hud.y + (L.hud.h - bh) // 2
        x = L.hud.right - t.u(12)

        def btn(label, key, v=None, w=40, selected=False, kind="normal", row_y=None):
            nonlocal x
            r = pygame.Rect(x - t.u(w), row_y if row_y is not None else y, t.u(w), bh)
            ui.button(surf, r, label, key, v, selected=selected, kind=kind, size=12.5)
            x = r.x - gap

        btn("‹ Back", "back", w=70)
        if len(self.dirs) > 1:
            btn("next ep ›", "ep", +1, w=84)
            btn("‹ prev ep", "ep", -1, w=84)
        # Timeline (left): transport controls
        ty = L.timeline.y + t.u(10)
        items = [("|‹", "dec", -1, 36, "normal"), ("Pause" if self.playing else "Play", "play", None, 60, "primary"),
                 ("›|", "dec", +1, 36, "normal"), ("crash", "crash", None, 56, "normal")]
        cx = L.timeline.x + t.u(12)
        for label, key, v, w, kind in items:
            r = pygame.Rect(cx, ty, t.u(w), bh)
            ui.button(surf, r, label, key, v, kind=kind, size=12.5)
            cx = r.right + gap
        cx += t.u(8)
        for v in self.SPEEDS:
            r = pygame.Rect(cx, ty, t.u(42), bh)
            ui.button(surf, r, f"{v:g}×", "rspeed", v, selected=(v == self.speed), size=12)
            cx = r.right + t.u(4)


# =================================================================== entry
def run_app(settings: Optional[Settings] = None, start: str = "menu", replay_path=None, replay_at=None,
            panel: str = "decisions", max_frames: int = 0, screenshot: Optional[str] = None,
            window: Optional[tuple[int, int]] = None) -> int:
    app = App(Display(window))
    if settings is not None:
        app.settings = settings
    if start == "play":
        factory, err = app.settings.controller_factory()
        scene = PlayScene(app, app.settings, factory, panel=panel) if factory else MenuScene(app, error=err)
    elif start == "replay":
        scene = ReplayScene(app, replay_path, start=replay_at, panel=panel)
    else:
        scene = MenuScene(app)
    app.run(scene, max_frames=max_frames, screenshot=screenshot)
    return 0
