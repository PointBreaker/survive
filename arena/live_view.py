"""Live viewer for headless experiments: watch what is being measured.

The experiment keeps running in the headless measurement runner. The viewer
lives in its own process and only *receives* small state messages through a
bounded queue (``put_nowait``; frames are dropped rather than ever blocking),
so drawing can never slow the control loop or add measured latency. It never
sends anything back except "the window was closed" (a stop request).

    main process                          viewer process (pygame window)
    ObservedEnvironment.step() ─ frame ─▶ left arena: the candidate, live
    TeeRecorder.event()        ─ event ─▶ requests / responses feed
    experiment loop            ─ plan  ─▶ scoreboard, progress, ETA
                                          right arena: the reference's recorded
                                          run on the same seed, re-simulated at
                                          the same tick (deterministic replay)

Closing the window (or pressing Q) asks the experiment to stop; it records the
run as cancelled and it can be continued with ``--resume``.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import sys
import time
from collections import deque
from typing import Any, Optional

from arena.environment import Environment
from arena.recorder import Recorder

FORWARD_EVENTS = ("request", "decision", "decision_failed", "decision_superseded", "decision_dropped",
                  "target_collected", "collision")


def display_available() -> bool:
    """Best guess whether a window can open (no window on a headless Linux box or in CI)."""
    if os.environ.get("SDL_VIDEODRIVER") in ("dummy", "offscreen"):
        return False
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class StopRequested(Exception):
    """The viewer window was closed: stop the experiment (resumable)."""


# ============================================================ producer side
class LiveView:
    """Owns the viewer process and the one-way message queue."""

    def __init__(self, title: str = "Decision Arena · live", start: bool = True, fps: float = 60.0):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=600)
        self.closed_evt = ctx.Event()
        self.min_dt = 1.0 / fps
        self._last = 0.0
        self.proc = None
        if start:
            self.proc = ctx.Process(target=viewer_main, args=(self.q, self.closed_evt, title), daemon=False)
            self.proc.start()

    def send(self, msg: dict[str, Any], droppable: bool = False) -> None:
        try:
            if droppable:
                self.q.put_nowait(msg)
            else:
                self.q.put(msg, timeout=0.05)
        except (queue.Full, ValueError, OSError):
            pass  # never let the viewer slow or break the experiment

    def closed(self) -> bool:
        return self.closed_evt.is_set() or (self.proc is not None and not self.proc.is_alive())

    def frame(self, env: Environment, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last < self.min_dt:
            return
        self._last = now
        self.send({"type": "frame", **state_of(env)}, droppable=not force)

    def finish(self) -> None:
        """Tell the viewer the experiment ended; the window stays open until the user closes it."""
        self.send({"type": "finished"})


def state_of(env: Environment) -> dict[str, Any]:
    p, t = env.player, env.target
    return {
        "tick": env.tick, "t": env.world_time, "score": env.score, "done": env.done,
        "reason": env.outcome.reason if env.done else None,
        "player": [p.x, p.y, p.vx, p.vy, p.radius],
        "target": [t.x, t.y, t.radius],
        "obstacles": [[o.id, o.x, o.y, o.vx, o.vy, o.radius] for o in env.obstacles],
    }


class ObservedEnvironment(Environment):
    """The unchanged environment, plus a read-only hook after every step (for the viewer).

    ``step`` calls the real ``Environment.step`` first and only reads state
    afterwards, so dynamics are bit-identical (tested). The hook also turns a
    closed viewer window into ``StopRequested``.
    """

    def __init__(self, config, seed: int, view: LiveView):
        super().__init__(config, seed)
        self._view = view

    def step(self, action):
        events = super().step(action)
        self._view.frame(self, force=self.done)
        if self._view.closed_evt.is_set():
            raise StopRequested()
        return events


class TeeRecorder(Recorder):
    """Writes to the real recorder and forwards decision events to the viewer."""

    def __init__(self, inner: Recorder, view: LiveView):
        self.inner = inner
        self.view = view

    def event(self, data: dict[str, Any]) -> None:
        self.inner.event(data)
        if data.get("type") in FORWARD_EVENTS:
            msg = {k: v for k, v in data.items() if k != "observation"}
            if "meta" in msg and isinstance(msg["meta"], dict):
                msg["meta"] = {k: msg["meta"][k] for k in ("confidence",) if k in msg["meta"]}
            self.view.send({"type": "event", "event": msg}, droppable=True)

    def write_result(self, result: dict[str, Any]) -> None:
        self.inner.write_result(result)

    def close(self) -> None:
        self.inner.close()


# ============================================================ viewer process
def viewer_main(q, closed_evt, title: str, max_frames: Optional[int] = None) -> None:  # pragma: no cover - UI
    # ARENA_LIVE_SNAPSHOT=<path.png> saves the frame every ~0.5 s (for tests and screenshots);
    # ARENA_LIVE_MAX_FRAMES closes the window after that many frames.
    env_max = os.environ.get("ARENA_LIVE_MAX_FRAMES")
    try:
        _Viewer(q, closed_evt, title).loop(int(env_max) if env_max else max_frames)
    finally:
        closed_evt.set()


class _Viewer:
    FEED = 16

    def __init__(self, q, closed_evt, title: str):
        import pygame

        from arena.display import Display
        from arena.renderer import Renderer

        self.pg = pygame
        self.q, self.closed_evt = q, closed_evt
        self.display = Display(title=title)
        self.r = Renderer(self.display, show_panel=False, show_timeline=False)
        self.plan: dict[str, Any] = {}
        self.episode: Optional[dict[str, Any]] = None  # current episode info
        self.frame: Optional[dict[str, Any]] = None
        self.feed: deque = deque(maxlen=self.FEED)
        self.ref_env: Optional[Environment] = None
        self.ref_actions: list = []
        self.last_action: Optional[str] = None
        self.finished = False
        self.started = time.time()

    # ---------------------------------------------------------------- input
    def drain(self) -> None:
        for _ in range(2000):
            try:
                m = self.q.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return
            k = m.get("type")
            if k == "frame":
                self.frame = m
            elif k == "plan":
                self.plan = m
            elif k == "episode_start":
                self.episode = m
                self.frame = None
                self.feed.clear()
                self.last_action = None
                self._setup_reference(m.get("reference"))
            elif k == "episode_end":
                if self.episode is not None:
                    self.episode["result"] = m.get("result")
            elif k == "event":
                self._on_event(m["event"])
            elif k == "finished":
                self.finished = True

    def _setup_reference(self, ref: Optional[dict[str, Any]]) -> None:
        self.ref_env = None
        if not ref:
            return
        from arena.difficulty import DifficultyConfig
        from arena.replay import actions_per_tick

        cfg = DifficultyConfig.from_dict(ref["config"])
        self.ref_env = Environment(cfg, ref["seed"])
        self.ref_actions = list(actions_per_tick(ref["action_changes"], int(ref["ticks"])))
        self.ref_result = ref.get("result", {})

    def _on_event(self, e: dict[str, Any]) -> None:
        t = e.get("type")
        if t == "decision":
            self.last_action = e.get("action")
            self.feed.appendleft(("ok", e.get("request_tick"), e.get("action"), (e.get("meta") or {}).get("confidence"),
                                  e.get("latency_ms")))
        elif t == "decision_failed":
            self.feed.appendleft(("fail", e.get("request_tick"), str(e.get("error"))[:60], None, e.get("latency_ms")))
        elif t in ("decision_superseded", "decision_dropped"):
            self.feed.appendleft(("skip", e.get("request_tick"), t.replace("decision_", ""), None, e.get("latency_ms")))

    # ----------------------------------------------------------------- loop
    def loop(self, max_frames: Optional[int]) -> None:
        pg = self.pg
        n = 0
        while True:
            for ev in pg.event.get():
                if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key in (pg.K_q, pg.K_ESCAPE)):
                    self.closed_evt.set()
                    self.display.close()
                    return
                self.display.handle_resize(ev)
            self.drain()
            self._advance_reference()
            self.draw()
            self.display.present()
            n += 1
            snap = os.environ.get("ARENA_LIVE_SNAPSHOT")
            if snap and n % 30 == 0:
                try:
                    self.display.save(snap.replace("{n}", str(n)))
                except Exception:
                    pass
            if max_frames is not None and n >= max_frames:
                return
            if self.display.backend == "classic":
                time.sleep(1 / 60)

    def _advance_reference(self) -> None:
        env = self.ref_env
        if env is None or self.frame is None:
            return
        target = self.frame["tick"]
        while env.tick < target and not env.done and env.tick < len(self.ref_actions):
            env.step(self.ref_actions[env.tick])

    # ----------------------------------------------------------------- draw
    def draw(self) -> None:
        from arena.ui import BG, BORDER, FAINT, MUTED, SURFACE, TEXT

        pg, r = self.pg, self.r
        t = r.t
        if t.d != self.display.density:
            from arena.ui import Theme

            r.t = t = Theme(self.display.density)
        surf = self.display.surface
        W, H = self.display.size
        surf.fill(BG)
        u = t.u
        top = u(64)
        panel_w = min(u(400), W // 3)
        # header
        self._header(surf, pg.Rect(0, 0, W, top))
        # arenas
        area = pg.Rect(u(12), top, W - panel_w - u(24), H - top - u(12))
        ep = self.episode or {}
        cfg = ep.get("config")
        if cfg:
            ww, wh = cfg["arena_width"], cfg["arena_height"]
            two = self.ref_env is not None
            cols = 2 if two else 1
            gap = u(14)
            cw = (area.w - gap * (cols - 1)) // cols
            label_h = u(46)
            s = max(0.05, min(cw / ww, (area.h - label_h) / wh))
            aw, ah = int(ww * s), int(wh * s)
            y = area.y + label_h + max(0, (area.h - label_h - ah) // 2)
            left = pg.Rect(area.x + (cw - aw) // 2, y, aw, ah)
            self._arena(surf, left, s, ww, wh, self._frame_state(), ep.get("color", (64, 200, 255)),
                        self.last_action, ep.get("title", ""), ep.get("subtitle", ""),
                        (ep.get("result") or {}).get("reason") or (self.frame or {}).get("reason"))
            if two:
                right = pg.Rect(area.x + cw + gap + (cw - aw) // 2, y, aw, ah)
                env = self.ref_env
                st = state_of(env)
                self._arena(surf, right, s, ww, wh, st, (25, 158, 112), None, ep.get("ref_title", "Reference"),
                            ep.get("ref_subtitle", "recorded run, same seed, same moment"),
                            st["reason"])
        else:
            t.text(surf, "waiting for the experiment…", area.center, MUTED, 15, anchor="center")
        # panel
        panel = pg.Rect(W - panel_w, top, panel_w, H - top)
        pg.draw.rect(surf, SURFACE, panel)
        pg.draw.line(surf, BORDER, panel.topleft, panel.bottomleft)
        self._panel(surf, panel)
        t.text(surf, "close the window or press Q to stop (the run can be resumed)", (u(14), H - u(8)), FAINT, 10.5,
               anchor="bottomleft")
        _ = TEXT

    def _frame_state(self) -> Optional[dict[str, Any]]:
        return self.frame

    def _header(self, surf, rect) -> None:
        from arena.ui import ACCENT, BORDER, MUTED, SURFACE_2, TEXT

        t, u, pg = self.r.t, self.r.t.u, self.pg
        p = self.plan
        t.text(surf, p.get("title", "Decision Arena"), (u(16), u(12)), TEXT, 16, bold=True)
        stage = p.get("stage_text", "")
        if self.finished:
            stage = "finished · results saved · close this window when done"
        t.text(surf, stage, (u(16), u(36)), MUTED, 12)
        # progress bar
        done, total = p.get("done", 0), max(1, p.get("total", 1))
        bw = min(u(360), rect.w // 3)
        bx = rect.w - bw - u(16)
        pg.draw.rect(surf, SURFACE_2, (bx, u(18), bw, u(8)), border_radius=u(4))
        pg.draw.rect(surf, ACCENT, (bx, u(18), int(bw * min(1.0, done / total)), u(8)), border_radius=u(4))
        el = time.time() - self.started
        eta = p.get("eta_s")
        txt = f"{done} / {p.get('total', 0)} episodes · elapsed {_dur(el)}" + (f" · ≈ {_dur(eta)} left" if eta else "")
        t.text(surf, txt, (bx + bw, u(34)), MUTED, 11.5, anchor="topright")
        pg.draw.line(surf, BORDER, (0, rect.bottom - 1), (rect.w, rect.bottom - 1))

    def _arena(self, surf, rect, s, ww, wh, st, color, action, title, subtitle, reason) -> None:
        from arena.action import Action
        from arena.ui import ARENA_BG, BAD, BORDER, MUTED, OBSTACLE, OK, TARGET, TEXT, WARN

        pg, r = self.pg, self.r
        t, u = r.t, r.t.u
        t.text(surf, title, (rect.x, rect.y - u(40)), TEXT, 14, bold=True)
        t.text(surf, subtitle, (rect.x, rect.y - u(20)), MUTED, 11.5)
        surf.blit(r._grid(_L(rect, s), ww, wh), rect.topleft)
        pg.draw.rect(surf, BORDER, rect, width=max(1, u(1)))
        if not st:
            return
        clip = surf.get_clip()
        surf.set_clip(rect)
        P = lambda x, y: (int(rect.x + x * s), int(rect.y + y * s))  # noqa: E731
        tx, ty, tr = st["target"]
        r._disc(surf, TARGET, P(tx, ty), tr * s)
        r._disc(surf, ARENA_BG, P(tx, ty), tr * s * 0.45)
        for _id, x, y, _vx, _vy, rad in st["obstacles"]:
            r._disc(surf, OBSTACLE, P(x, y), rad * s)
        px, py, _pvx, _pvy, pr = st["player"]
        pc = P(px, py)
        r._disc(surf, color, pc, pr * s)
        if action and action != "STAY":
            dx, dy = Action(action).direction
            a = (int(pc[0] + dx * (pr + 4) * s), int(pc[1] + dy * (pr + 4) * s))
            b = (int(pc[0] + dx * (pr + 22) * s), int(pc[1] + dy * (pr + 22) * s))
            pg.draw.line(surf, TEXT, a, b, max(2, u(2)))
            ang = math.atan2(dy, dx)
            for sg in (-0.55, 0.55):
                pg.draw.line(surf, TEXT, b, (int(b[0] - u(7) * math.cos(ang + sg)), int(b[1] - u(7) * math.sin(ang + sg))),
                             max(2, u(2)))
        surf.set_clip(clip)
        # status line
        status = f"t {st['t']:5.1f} s   score {st['score']}"
        t.text(surf, status, (rect.right, rect.y - u(20)), MUTED, 11.5, mono=True, anchor="topright")
        if reason:
            col = {"collision": BAD, "target_timeout": WARN}.get(reason, OK)
            label = {"collision": "COLLISION", "target_timeout": "TARGET TIMEOUT", "max_duration": "SURVIVED",
                     "target_goal": "GOAL"}.get(reason, reason)
            t.text(surf, label, (rect.centerx, rect.y + u(14)), col, 18, bold=True, anchor="midtop")

    def _panel(self, surf, rect) -> None:
        from arena.ui import BAD, BORDER, FAINT, MUTED, OK, TEXT, WARN

        t, u, pg = self.r.t, self.r.t.u, self.pg
        x, y, w = rect.x + u(16), rect.y + u(14), rect.w - u(32)
        t.text(surf, "Requests / responses", (x, y), TEXT, 13, bold=True)
        y += u(24)
        tick_s = (self.episode or {}).get("tick_s", 1 / 60)
        if not self.feed:
            t.text(surf, "–", (x, y), FAINT, 12)
            y += u(20)
        for kind, rt, what, conf, lat in list(self.feed)[:12]:
            ts = f"{(rt or 0) * tick_s:6.2f}s"
            t.text(surf, ts, (x, y), FAINT, 11.5, mono=True)
            col = TEXT if kind == "ok" else BAD if kind == "fail" else WARN
            t.text(surf, str(what), (x + u(70), y), col, 11.5, mono=True, bold=kind == "ok")
            right = []
            if conf is not None:
                right.append(f"conf {conf:.2f}")
            if lat is not None:
                right.append(f"{lat:.0f} ms")
            if kind == "ok":
                t.text(surf, "  ".join(right), (x + w, y), MUTED, 11, mono=True, anchor="topright")
            y += u(19)
        y += u(10)
        pg.draw.line(surf, BORDER, (x, y), (x + w, y))
        y += u(14)
        t.text(surf, "Scoreboard", (x, y), TEXT, 13, bold=True)
        y += u(24)
        board = self.plan.get("board", [])
        cols = [("", 0), ("done", 0.30), ("ok", 0.46), ("crash", 0.59), ("timeout", 0.72)]
        for name, fx in cols[1:]:
            t.text(surf, name, (x + int(w * fx), y), FAINT, 11)
        t.text(surf, "invalid", (x + w, y), FAINT, 11, anchor="topright")
        y += u(18)
        for row in board:
            t.text(surf, row["label"], (x, y), TEXT, 12, bold=True)
            vals = [f"{row['done']}/{row['total']}", row["success"], row["collision"], row["timeout"]]
            for (name, fx), v in zip(cols[1:], vals):
                col = OK if name == "ok" and row["success"] else TEXT
                t.text(surf, str(v), (x + int(w * fx), y), col, 12, mono=True)
            t.text(surf, str(row.get("invalid", 0)), (x + w, y), WARN if row.get("invalid") else FAINT, 12, mono=True,
                   anchor="topright")
            y += u(20)
        for line in self.plan.get("notes", []):
            y += u(6)
            t.text(surf, line, (x, y), MUTED, 11)
            y += u(14)


class _L:
    """Minimal layout stand-in for Renderer._grid (arena rect + scale)."""

    def __init__(self, rect, scale):
        self.arena = rect
        self.scale = scale


def _dur(s: Optional[float]) -> str:
    if s is None or not math.isfinite(s) or s < 0:
        return "–"
    s = int(s)
    if s < 60:
        return f"{s}s"
    m = s // 60
    return f"{m}m {s % 60:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"
