"""pygame renderer. Visualization only: it reads environment state and the
inspector's event-derived view model for drawing, and never feeds anything
back to controllers.

Layout::

    +--------------------------------------------+------------------+
    | HUD                                        |                  |
    +--------------------------------------------+   inspector      |
    | arena (scaled to fit the screen)           |   panel          |
    |   ghost = snapshot the model is deciding on|                  |
    +--------------------------------------------+                  |
    | timeline (replay viewer only)              |                  |
    +--------------------------------------------+------------------+
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence

import pygame

from arena.action import Action
from arena.environment import Environment
from arena.inspector import TICK_MS, DecisionRecord, Inspector

BG = (18, 20, 26)
GRID = (28, 31, 40)
PANEL_BG = (13, 14, 19)
RULE = (40, 44, 56)
PLAYER = (80, 200, 255)
OBSTACLE = (235, 90, 80)
TARGET = (120, 230, 120)
GHOST = (150, 150, 190)
TEXT = (220, 224, 232)
DIM = (130, 136, 150)
VEL = (255, 220, 90)
OK = (120, 230, 120)
BAD = (240, 95, 85)
WARN = (245, 170, 70)
ACCENT = (110, 170, 255)
STATUS_COLOR = {"applied": TEXT, "failed": BAD, "dropped": WARN, "superseded": DIM}

COMPASS_ORDER = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "STAY")
PANEL_MODES = ("decisions", "request", "response")


def _fmt_num(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, dict):
        return {k: _fmt_num(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_fmt_num(x) for x in v]
    return v


COLLAPSED_KEYS = {"rules": "constant for the whole episode"}


def compact_json(obj: Any, indent: int = 0) -> list[str]:
    """Readable JSON lines: entity lists become aligned tables."""
    pad = "  " * indent
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if k in COLLAPSED_KEYS and isinstance(v, dict):
                out.append(f'{pad}"{k}": {{...{len(v)} fields}}  ({COLLAPSED_KEYS[k]})')
            elif isinstance(v, dict) and len(json.dumps(v)) > 60:
                out.append(f'{pad}"{k}": {{')
                out.extend(compact_json(v, indent + 1))
                out.append(pad + "}")
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                keys = list(v[0])
                numeric = all(isinstance(x, dict) and list(x) == keys and
                              all(isinstance(x[c], (int, float)) for c in keys) for x in v)
                if numeric:  # aligned table, one row per entity
                    out.append(f'{pad}"{k}": ({len(v)} rows)')
                    names = ["r" if c == "radius" else c for c in keys]
                    out.append(pad + "  " + "".join(f"{n:>8}" for n in names))
                    for x in v:
                        out.append(pad + "  " + "".join(
                            f"{x[c]:>8d}" if isinstance(x[c], int) else f"{x[c]:>8.1f}" for c in keys))
                else:
                    out.append(f'{pad}"{k}": [  ({len(v)})')
                    out.extend(pad + "  " + json.dumps(_fmt_num(x), separators=(",", ":")) for x in v)
                    out.append(pad + "]")
            else:
                out.append(f'{pad}"{k}": {json.dumps(_fmt_num(v))}')
        return out
    return [pad + json.dumps(_fmt_num(obj))]


class Renderer:
    HUD_H = 56
    PANEL_W = 460
    TIMELINE_H = 46

    def __init__(
        self,
        width: float,
        height: float,
        title: str = "Decision Arena",
        panel: bool = True,
        timeline: bool = False,
        scale: Optional[float] = None,
    ):
        pygame.init()
        self.ww, self.wh = int(width), int(height)
        self.panel = panel
        self.timeline = timeline
        panel_w = self.PANEL_W if panel else 0
        tl_h = self.TIMELINE_H if timeline else 0
        if scale is None:
            info = pygame.display.Info()
            sw, sh = info.current_w or 0, info.current_h or 0
            scale = 1.0
            if sw > 0 and sh > 0:
                scale = min(1.0, (sw * 0.96 - panel_w) / self.ww, (sh * 0.88 - self.HUD_H - tl_h) / self.wh)
            scale = max(0.4, scale)
        self.scale = scale
        self.aw, self.ah = int(self.ww * scale), int(self.wh * scale)
        self.W = self.aw + panel_w
        self.H = max(self.HUD_H + self.ah + tl_h, 700 if panel else 0)
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption(title)
        self.world = pygame.Surface((self.ww, self.wh))
        self.f_small = pygame.font.SysFont("monospace", 13)
        self.f = pygame.font.SysFont("monospace", 15)
        self.f_bold = pygame.font.SysFont("monospace", 15, bold=True)
        self.f_big = pygame.font.SysFont("monospace", 30, bold=True)
        self.f_huge = pygame.font.SysFont("monospace", 34, bold=True)
        self._grid = self._make_grid()
        self.timeline_rect = pygame.Rect(0, self.HUD_H + self.ah, self.aw, tl_h)

    def _make_grid(self) -> pygame.Surface:
        s = pygame.Surface((self.ww, self.wh))
        s.fill(BG)
        for x in range(0, self.ww, 50):
            pygame.draw.line(s, GRID, (x, 0), (x, self.wh))
        for y in range(0, self.wh, 50):
            pygame.draw.line(s, GRID, (0, y), (self.ww, y))
        return s

    def _text(self, surf, s: str, pos, color=TEXT, font=None) -> int:
        font = font or self.f
        img = font.render(s, True, color)
        surf.blit(img, pos)
        return img.get_height()

    # ================================================================= frame
    def draw(
        self,
        env: Environment,
        controller_name: str,
        action: str,
        latency_ms: Optional[float],
        debug: bool = False,
        inspector: Optional[Inspector] = None,
        ghost: bool = True,
        compass: bool = True,
        panel_mode: str = "decisions",
        header: Sequence[str] = (),
        raw_request: Any = None,
        raw_response: Any = None,
        status: str = "",
        timeline: Optional[dict[str, Any]] = None,
    ) -> None:
        scr = self.screen
        scr.fill(PANEL_BG)
        self._draw_world(env, action, debug, inspector if ghost else None,
                         inspector.last_applied() if (inspector and compass) else None)
        if self.scale != 1.0:
            scr.blit(pygame.transform.smoothscale(self.world, (self.aw, self.ah)), (0, self.HUD_H))
        else:
            scr.blit(self.world, (0, self.HUD_H))
        self._draw_hud(env, controller_name, action, latency_ms, debug, status)
        if self.timeline and timeline:
            self._draw_timeline(timeline)
        if self.panel:
            self._draw_panel(env, inspector, panel_mode, header, raw_request, raw_response)
        pygame.display.flip()

    # ================================================================= arena
    def _draw_world(self, env, action, debug, inspector, last_applied) -> None:
        w = self.world
        w.blit(self._grid, (0, 0))
        t = env.target
        pygame.draw.circle(w, TARGET, (int(t.x), int(t.y)), int(t.radius))
        pygame.draw.circle(w, BG, (int(t.x), int(t.y)), max(1, int(t.radius * 0.45)))

        p = env.player
        # Ghost: the snapshot the controller is currently deciding on.
        req = inspector.inflight() if inspector else None
        if req and req.get("observation"):
            obs = req["observation"]
            for o in obs["obstacles"]:
                pygame.draw.circle(w, GHOST, (int(o["x"]), int(o["y"])), int(o["radius"]), 1)
            gp = obs["player"]
            pygame.draw.circle(w, GHOST, (int(gp["x"]), int(gp["y"])), int(gp["radius"]), 2)
            pygame.draw.line(w, GHOST, (int(gp["x"]), int(gp["y"])), (int(p.x), int(p.y)), 1)
            age = (env.tick - req["tick"]) * TICK_MS
            self._text(w, f"snapshot -{age:.0f}ms", (int(gp["x"]) + 14, int(gp["y"]) - 22), GHOST, self.f_small)

        for o in env.obstacles:
            c = (int(o.x), int(o.y))
            pygame.draw.circle(w, OBSTACLE, c, int(o.radius))
            if debug:
                pygame.draw.line(w, VEL, c, (int(o.x + o.vx * 0.5), int(o.y + o.vy * 0.5)), 1)
                img = self.f_small.render(str(o.id), True, TEXT)
                w.blit(img, img.get_rect(center=c))

        pc = (int(p.x), int(p.y))
        # Probability compass of the decision currently in force.
        probs = (last_applied.meta or {}).get("probabilities") if last_applied else None
        if isinstance(probs, dict):
            for name, prob in probs.items():
                if name in Action.__members__ and name != "STAY" and isinstance(prob, (int, float)):
                    dx, dy = Action(name).direction
                    L = 18 + 70 * float(prob)
                    col = ACCENT if name == last_applied.action else (70, 90, 130)
                    pygame.draw.line(w, col, pc, (int(p.x + dx * L), int(p.y + dy * L)), 3)
            stay = probs.get("STAY")
            if isinstance(stay, (int, float)) and stay > 0.02:
                pygame.draw.circle(w, (70, 90, 130), pc, int(p.radius + 4 + 20 * stay), 2)

        pygame.draw.circle(w, PLAYER, pc, int(p.radius))
        if action and action != "STAY":
            dx, dy = Action(action).direction
            tip = (p.x + dx * (p.radius + 16), p.y + dy * (p.radius + 16))
            pygame.draw.line(w, TEXT, pc, (int(tip[0]), int(tip[1])), 2)
            ang = math.atan2(dy, dx)
            for s in (-0.5, 0.5):
                pygame.draw.line(w, TEXT, (int(tip[0]), int(tip[1])),
                                 (int(tip[0] - 7 * math.cos(ang + s)), int(tip[1] - 7 * math.sin(ang + s))), 2)
        if debug:
            pygame.draw.line(w, VEL, pc, (int(p.x + p.vx * 0.5), int(p.y + p.vy * 0.5)), 2)

        if env.done:
            o = env.outcome
            msg = "SURVIVED" if o.success else f"FAILED: {o.reason}"
            img = self.f_huge.render(msg, True, OK if o.success else BAD)
            w.blit(img, img.get_rect(center=(self.ww // 2, self.wh // 2 - 20)))

    def _draw_hud(self, env, controller_name, action, latency_ms, debug, status) -> None:
        scr = self.screen
        cfg = env.config
        pygame.draw.rect(scr, (10, 11, 15), (0, 0, self.aw, self.HUD_H))
        lat = "-" if latency_ms is None else f"{latency_ms:.0f}ms"
        since = env.world_time - env.last_target_time
        l1 = (f"{controller_name}  score {env.score}  t {env.world_time:5.2f}s  "
              f"world {cfg.world_speed_scale:g}x  obst {len(env.obstacles)}  {cfg.decision_hz:g}Hz")
        l2 = (f"action {action:<4} latency {lat:<6} seed {env.seed}  timer {since:4.1f}/{cfg.target_timeout:g}s"
              + ("  [debug]" if debug else ""))
        self._text(scr, l1, (10, 8))
        self._text(scr, l2, (10, 30), DIM)
        if status:  # overlay in the arena's top-right corner, never over HUD text
            img = self.f_bold.render(status, True, WARN)
            box = img.get_rect(topright=(self.aw - 8, self.HUD_H + 8)).inflate(12, 6)
            pygame.draw.rect(scr, (10, 11, 15), box)
            scr.blit(img, img.get_rect(center=box.center))

    # ============================================================== timeline
    def _draw_timeline(self, tl: dict[str, Any]) -> None:
        scr = self.screen
        r = self.timeline_rect
        pygame.draw.rect(scr, (10, 11, 15), r)
        total = max(1, tl["total_ticks"])
        x0, x1 = r.x + 10, r.right - 10

        def X(tick):
            return int(x0 + (x1 - x0) * min(max(tick, 0), total) / total)

        y = r.y + 8
        pygame.draw.line(scr, RULE, (x0, y + 14), (x1, y + 14), 1)
        for t in tl.get("requests", ()):
            pygame.draw.line(scr, (70, 76, 92), (X(t), y + 10), (X(t), y + 18))
        for t in tl.get("applied", ()):
            pygame.draw.line(scr, ACCENT, (X(t), y + 4), (X(t), y + 12))
        for t in tl.get("failed", ()):
            pygame.draw.line(scr, BAD, (X(t), y + 2), (X(t), y + 24), 2)
        for t in tl.get("targets", ()):
            pygame.draw.circle(scr, OK, (X(t), y + 22), 3)
        ct = tl.get("collision_tick")
        if ct is not None:
            cx = X(ct)
            pygame.draw.line(scr, BAD, (cx - 5, y + 9), (cx + 5, y + 19), 2)
            pygame.draw.line(scr, BAD, (cx - 5, y + 19), (cx + 5, y + 9), 2)
        cx = X(tl["current"])
        pygame.draw.line(scr, TEXT, (cx, r.y + 2), (cx, r.bottom - 12), 2)
        self._text(scr, tl.get("label", ""), (x0, r.bottom - 16), DIM, self.f_small)

    def timeline_tick_at(self, x: int, total_ticks: int) -> int:
        r = self.timeline_rect
        x0, x1 = r.x + 10, r.right - 10
        return int(round((min(max(x, x0), x1) - x0) / max(1, x1 - x0) * total_ticks))

    # ================================================================= panel
    def _draw_panel(self, env, ins: Optional[Inspector], mode, header, raw_request, raw_response) -> None:
        scr = self.screen
        x0 = self.aw
        pygame.draw.rect(scr, PANEL_BG, (x0, 0, self.PANEL_W, self.H))
        pygame.draw.line(scr, RULE, (x0, 0), (x0, self.H))
        x = x0 + 14
        y = 10
        tabs = "  ".join(f"[{m}]" if m == mode else m for m in PANEL_MODES)
        y += self._text(scr, f"INSPECTOR  {tabs}", (x, y), ACCENT, self.f_bold) + 2
        for h in header:
            y += self._text(scr, h, (x, y), DIM, self.f_small)
        y += 6
        if ins is None:
            self._text(scr, "(no inspector)", (x, y), DIM)
            return
        y = self._section_inflight(env, ins, x, y)
        if mode == "decisions":
            y = self._section_decision(ins, x, y)
            y = self._section_stats(env, ins, x, y)
            self._section_recent(ins, x, y)
        else:
            payload = raw_request if mode == "request" else raw_response
            title = "last request body" if mode == "request" else "last response"
            self._section_json(title, payload, x, y)
        self._text(scr, "J panel  G ghost  P compass  F1 debug", (x, self.H - 20), DIM, self.f_small)

    def _rule(self, x, y) -> int:
        pygame.draw.line(self.screen, RULE, (x, y), (x + self.PANEL_W - 28, y))
        return y + 8

    def _section_inflight(self, env, ins: Inspector, x, y) -> int:
        scr = self.screen
        y = self._rule(x, y)
        reqs = ins.inflight_all()
        period_ms = 1000.0 / env.config.decision_hz
        cap = env.config.max_inflight
        if reqs:
            self._text(scr, f"IN FLIGHT  {len(reqs)}/{cap}", (x, y), WARN, self.f_bold)
            y += 20
            waits = [(r, (env.tick - r["tick"]) * TICK_MS) for r in reqs]
            scale_ms = max(500.0, max(w for _, w in waits) * 1.1)
            bw = self.PANEL_W - 28 - 120
            for r, waited in waits[:5]:
                self._text(scr, f"#{r['id']:<4}", (x, y - 2), TEXT, self.f_small)
                bx = x + 50
                pygame.draw.rect(scr, (30, 32, 40), (bx, y, bw, 9))
                pygame.draw.rect(scr, WARN, (bx, y, int(bw * min(1, waited / scale_ms)), 9))
                px = bx + int(bw * period_ms / scale_ms)
                pygame.draw.line(scr, TEXT, (px, y - 2), (px, y + 11), 1)
                self._text(scr, f"{waited:5.0f} ms", (bx + bw + 8, y - 2), WARN, self.f_small)
                y += 14
            self._text(scr, f"ghost = oldest (#{reqs[0]['id']})   | = period {period_ms:.0f}ms",
                       (x, y), DIM, self.f_small)
            y += 16
        else:
            self._text(scr, f"idle  (0/{cap} requests in flight)", (x, y), DIM, self.f_bold)
            y += 22
        return y + 4

    def _section_decision(self, ins: Inspector, x, y) -> int:
        scr = self.screen
        y = self._rule(x, y)
        last = ins.records[-1] if ins.records else None
        if last is None:
            self._text(scr, "no decision yet", (x, y), DIM)
            return y + 24
        color = STATUS_COLOR[last.status]
        self._text(scr, f"LAST RESPONSE  #{last.id}  {last.status}", (x, y), DIM, self.f_bold)
        y += 20
        label = last.action or "—"
        self._text(scr, label, (x, y), color, self.f_big)
        meta = last.meta or {}
        conf = meta.get("confidence")
        right = x + 110
        self._text(scr, f"latency {last.latency_ms:6.0f} ms", (right, y), TEXT)
        self._text(scr, f"applied +{last.delay_ticks} ticks ({last.delay_ticks * TICK_MS:.0f}ms after snapshot)",
                   (right, y + 17), DIM, self.f_small)
        if isinstance(conf, (int, float)):
            self._text(scr, f"confidence {conf:.2f}", (right, y + 31), ACCENT, self.f_small)
        y += 50
        if last.error:
            for i in range(0, min(len(last.error), 180), 58):
                y += self._text(scr, last.error[i:i + 58], (x, y), BAD, self.f_small)
            y += 4
        probs = meta.get("probabilities")
        if isinstance(probs, dict) and probs:
            bw = self.PANEL_W - 130
            for name in COMPASS_ORDER:
                pv = probs.get(name)
                if not isinstance(pv, (int, float)):
                    continue
                chosen = name == last.action
                self._text(scr, f"{name:>4}", (x, y), TEXT if chosen else DIM, self.f_small)
                pygame.draw.rect(scr, (30, 32, 40), (x + 44, y + 3, bw, 9))
                pygame.draw.rect(scr, ACCENT if chosen else (70, 90, 130), (x + 44, y + 3, int(bw * pv), 9))
                self._text(scr, f"{pv:.2f}", (x + 50 + bw, y), TEXT if chosen else DIM, self.f_small)
                y += 15
        elif last.status == "applied":
            self._text(scr, "(controller reports no probabilities)", (x, y), DIM, self.f_small)
            y += 16
        return y + 6

    def _section_stats(self, env, ins: Inspector, x, y) -> int:
        scr = self.screen
        y = self._rule(x, y)
        st = ins.stats()
        secs = max(env.tick / 60.0, 1e-9)
        mean_ms = "-" if st["mean_ms"] is None else f"{st['mean_ms']:.0f}"
        p95 = "-" if st["p95_ms"] is None else f"{st['p95_ms']:.0f}"
        self._text(scr, f"applied {st['applied']}  failed {st['failed']}  dropped {st['dropped']}  "
                        f"superseded {st['superseded']}", (x, y), TEXT, self.f_small)
        y += 16
        self._text(scr, f"missed slots {st['missed']}  latency mean {mean_ms}ms p95 {p95}ms", (x, y), TEXT,
                   self.f_small)
        y += 16
        self._text(scr, f"rate {st['applied'] / secs:.1f} decisions/s  (max {env.config.decision_hz:g}Hz, "
                        f"{env.config.max_inflight} in flight)", (x, y), TEXT, self.f_small)
        y += 18
        recent = st["recent_ms"]
        h = 40
        bw = self.PANEL_W - 28
        pygame.draw.rect(scr, (22, 24, 30), (x, y, bw, h))
        if recent:
            top = max(max(recent) * 1.1, 1000.0 / env.config.decision_hz * 1.5)
            period_y = y + h - int(h * (1000.0 / env.config.decision_hz) / top)
            for i in range(0, bw, 6):
                pygame.draw.line(scr, (80, 80, 90), (x + i, period_y), (x + i + 3, period_y))
            step = bw / max(1, 60)
            for i, v in enumerate(recent):
                bh = int(h * v / top)
                pygame.draw.rect(scr, WARN, (x + int(i * step), y + h - bh, max(2, int(step) - 1), bh))
            self._text(scr, f"{top:.0f}ms", (x + bw - 50, y), DIM, self.f_small)
        y += h + 2
        self._text(scr, "latency per decision (dashed = decision period)", (x, y), DIM, self.f_small)
        return y + 20

    def _section_recent(self, ins: Inspector, x, y) -> int:
        scr = self.screen
        y = self._rule(x, y)
        self._text(scr, "  id   t(s)  action  conf  latency  delay", (x, y), DIM, self.f_small)
        y += 16
        for r in reversed(ins.records):
            if y > self.H - 40:
                break
            conf = (r.meta or {}).get("confidence")
            cs = f"{conf:.2f}" if isinstance(conf, (int, float)) else "  - "
            color = STATUS_COLOR[r.status]
            act = r.action or r.status.upper()
            self._text(scr, f"{r.id:4d} {r.request_tick / 60:6.2f}  {act:<6} {cs}  {r.latency_ms:5.0f}ms  "
                            f"+{r.delay_ticks}t", (x, y), color, self.f_small)
            y += 15
        return y

    def _section_json(self, title, payload, x, y) -> int:
        scr = self.screen
        y = self._rule(x, y)
        self._text(scr, title + "  (floats rounded for display)", (x, y), DIM, self.f_small)
        y += 18
        if payload is None:
            self._text(scr, "(nothing yet)", (x, y), DIM)
            return y + 20
        max_chars = max(20, (self.PANEL_W - 28) // self.f_small.size("0")[0])
        lines = []
        for line in compact_json(payload):
            while len(line) > max_chars:  # wrap, never silently cut
                lines.append(line[:max_chars])
                line = "      " + line[max_chars:]
            lines.append(line)
        for line in lines:
            if y > self.H - 40:
                self._text(scr, "...", (x, y), DIM, self.f_small)
                break
            self._text(scr, line, (x, y), TEXT, self.f_small)
            y += 14
        return y

    def close(self) -> None:
        pygame.quit()
