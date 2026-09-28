"""pygame renderer. Visualization only: it reads environment state and the
inspector's event-derived view model for drawing, and never feeds anything
back to controllers.

Everything is drawn directly at the display's pixel resolution (no surface
rescaling), with antialiased shapes. Moving objects are interpolated between
60 Hz physics ticks using their velocity, so motion stays smooth at any
monitor refresh rate. That interpolation is visual only; the simulation is
untouched.

Layout (recomputed every frame from the window size)::

    +------------------------------------------+----------------+
    | HUD (stats left, scene buttons right)    |                |
    +------------------------------------------+   inspector    |
    | arena (fit, centered)                    |   panel        |
    +------------------------------------------+                |
    | timeline                                 |                |
    +------------------------------------------+----------------+
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import pygame

from arena.action import Action
from arena.environment import Environment
from arena.inspector import TICK_MS, Inspector
from arena.ui import (ACCENT, ACCENT_DIM, ARENA_BG, BAD, BG, BORDER, FAINT, GHOST, GRID, MUTED, OBSTACLE, OK,
                      PLAYER, SURFACE, SURFACE_2, TARGET, TEXT, WARN, Theme)

STATUS_COLOR = {"applied": TEXT, "failed": BAD, "dropped": WARN, "superseded": FAINT}
COMPASS_ORDER = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "STAY")
PANEL_MODES = ("decisions", "request", "response")
VEL = (255, 214, 102)


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


@dataclass
class Layout:
    W: int
    H: int
    hud: pygame.Rect
    arena_area: pygame.Rect
    arena: pygame.Rect  # exact world rectangle in pixels
    scale: float  # pixels per world unit
    timeline: pygame.Rect
    panel: pygame.Rect
    hud_controls_x: int  # scene buttons may use hud from here to the right
    sidebar: Optional[pygame.Rect] = None  # left column, drawn by the scene
    card: Optional[pygame.Rect] = None  # end-of-episode card (if drawn)
    card_buttons_y: int = 0


class Renderer:
    HUD_PT = 60
    PANEL_PT = 440
    TIMELINE_PT = 48

    def __init__(self, display, show_panel: bool = True, show_timeline: bool = True):
        self.display = display
        self.show_panel = show_panel
        self.show_timeline = show_timeline
        self.t = Theme(display.density)
        self._grid_cache: tuple[tuple, Optional[pygame.Surface]] = ((), None)
        self._sprites: dict[tuple, pygame.Surface] = {}
        self.timeline_reserve = 0
        self.timeline_pt = self.TIMELINE_PT  # scenes may make it taller to host controls above the track
        self.left_pt = 0  # width of a scene-drawn left sidebar

    # ================================================================ layout
    def layout(self, world_w: float, world_h: float) -> Layout:
        t = self.t
        if t.d != self.display.density:
            self.t = t = Theme(self.display.density)
        W, H = self.display.size
        left = min(t.u(self.left_pt), W // 3)
        panel_w = t.u(self.PANEL_PT) if self.show_panel else 0
        panel_w = min(panel_w, (W - left) // 2)
        cw = W - panel_w - left  # centre column
        hud = pygame.Rect(left, 0, cw, t.u(self.HUD_PT))
        tl_h = t.u(self.timeline_pt) if self.show_timeline else 0
        area = pygame.Rect(left, hud.bottom, cw, H - hud.bottom - tl_h)
        m = t.u(10)
        s = max(0.05, min((area.w - 2 * m) / world_w, (area.h - 2 * m) / world_h))
        aw, ah = int(world_w * s), int(world_h * s)
        arena = pygame.Rect(area.x + (area.w - aw) // 2, area.y + (area.h - ah) // 2, aw, ah)
        timeline = pygame.Rect(left, area.bottom, cw, tl_h)
        panel = pygame.Rect(W - panel_w, 0, panel_w, H)
        sidebar = pygame.Rect(0, 0, left, H)
        return Layout(W, H, hud, area, arena, s, timeline, panel, hud_controls_x=hud.x + int(hud.w * 0.52),
                      sidebar=sidebar)

    # ================================================================= frame
    def draw(
        self,
        env: Environment,
        controller_name: str,
        action: str,
        latency_ms: Optional[float],
        *,
        alpha: float = 0.0,
        debug: bool = False,
        inspector: Optional[Inspector] = None,
        ghost: bool = True,
        compass: bool = True,
        panel_mode: str = "decisions",
        header: Sequence[str] = (),
        raw_request: Any = None,
        raw_response: Any = None,
        timeline: Optional[dict[str, Any]] = None,
        fps: Optional[float] = None,
        card_lines: Optional[Sequence[str]] = None,
        hud_reserve: int = 0,
        timeline_reserve: int = 0,
    ) -> Layout:
        surf = self.display.surface
        cfg = env.config
        L = self.layout(cfg.arena_width, cfg.arena_height)
        surf.fill(BG)
        self._draw_world(surf, L, env, action, alpha, debug, inspector if ghost else None,
                         inspector.last_applied() if (inspector and compass) else None)
        self.timeline_reserve = timeline_reserve
        self._draw_hud(surf, L, env, controller_name, action, latency_ms, fps, hud_reserve)
        if self.show_timeline and timeline:
            self._draw_timeline(surf, L, timeline)
        if self.show_panel and L.panel.w > 0:
            self._draw_panel(surf, L, env, inspector, panel_mode, header, raw_request, raw_response)
        if env.done and card_lines is not None:
            self._draw_card(surf, L, env, card_lines)
        return L

    # ================================================================= arena
    def _grid(self, L: Layout, world_w: float, world_h: float) -> pygame.Surface:
        key = (L.arena.size, L.scale)
        if self._grid_cache[0] != key:
            # Opaque source: blitting it is a plain copy, no per-pixel blending.
            g = pygame.Surface(L.arena.size, 0, 32)
            g.fill(ARENA_BG)
            step = 50
            for i in range(0, int(world_w) + 1, step):
                x = int(i * L.scale)
                pygame.draw.line(g, GRID, (x, 0), (x, L.arena.h))
            for i in range(0, int(world_h) + 1, step):
                y = int(i * L.scale)
                pygame.draw.line(g, GRID, (0, y), (L.arena.w, y))
            self._grid_cache = (key, g)
        return self._grid_cache[1]

    def _sprite(self, color, r: int, filled: bool) -> pygame.Surface:
        """Antialiased circle on a transparent sprite, cached by (color, radius).

        Blitting (alpha-blending) the sprite mixes edge pixels with whatever
        is underneath and keeps the frame opaque. gfxdraw's AA shapes write
        partial alpha instead of blending, which left speckled edges.
        """
        key = (color, r, filled)
        spr = self._sprites.get(key)
        if spr is None:
            if len(self._sprites) > 2000:
                self._sprites.clear()
            # Supersample 4x and downsample: exact edge coverage, no holes.
            k = 4
            size = 2 * r + 3
            big = pygame.Surface((size * k, size * k), pygame.SRCALPHA, 32)
            centre = ((r + 1.5) * k, (r + 1.5) * k)
            width = 0 if filled else max(1, int(round(1.3 * k * max(1.0, self.t.d))))
            pygame.draw.circle(big, color, centre, r * k, width)
            spr = pygame.transform.smoothscale(big, (size, size))
            self._sprites[key] = spr
        return spr

    def _disc(self, surf, color, c, r) -> None:
        r = max(1, int(round(r)))
        surf.blit(self._sprite(color, r, True), (c[0] - r - 1, c[1] - r - 1))

    def _ring(self, surf, color, c, r) -> None:
        r = max(1, int(round(r)))
        surf.blit(self._sprite(color, r, False), (c[0] - r - 1, c[1] - r - 1))

    def _draw_world(self, surf, L: Layout, env, action, alpha, debug, inspector, last_applied) -> None:
        t, cfg = self.t, env.config
        s, ax, ay = L.scale, L.arena.x, L.arena.y
        surf.blit(self._grid(L, cfg.arena_width, cfg.arena_height), L.arena.topleft)
        pygame.draw.rect(surf, BORDER, L.arena, width=max(1, t.u(1)))
        prev_clip = surf.get_clip()
        surf.set_clip(L.arena)
        # Visual interpolation within the current physics tick.
        lead = 0.0 if env.done else alpha * cfg.world_speed_scale / 60.0

        def P(x, y, vx=0.0, vy=0.0):
            return int(ax + (x + vx * lead) * s), int(ay + (y + vy * lead) * s)

        tg = env.target
        tc = P(tg.x, tg.y)
        self._disc(surf, TARGET, tc, tg.radius * s)
        self._disc(surf, ARENA_BG, tc, tg.radius * s * 0.45)

        p = env.player
        pc = P(p.x, p.y, p.vx, p.vy)
        # Ghost: the (oldest) snapshot the controller is currently deciding on.
        req = inspector.inflight() if inspector else None
        if req and req.get("observation") and (env.tick - req["tick"]) * TICK_MS >= 50:
            obs = req["observation"]
            for o in obs["obstacles"]:
                self._ring(surf, GHOST, P(o["x"], o["y"]), o["radius"] * s)
            gp = obs["player"]
            gc = P(gp["x"], gp["y"])
            self._ring(surf, GHOST, gc, gp["radius"] * s)
            pygame.draw.aaline(surf, GHOST, gc, pc)
            age = (env.tick - req["tick"]) * TICK_MS
            t.text(surf, f"snapshot −{age:.0f} ms", (gc[0] + t.u(12), gc[1] - t.u(20)), GHOST, 10.5)

        for o in env.obstacles:
            c = P(o.x, o.y, o.vx, o.vy)
            self._disc(surf, OBSTACLE, c, o.radius * s)
            if debug:
                pygame.draw.aaline(surf, VEL, c, (c[0] + int(o.vx * 0.5 * s), c[1] + int(o.vy * 0.5 * s)))
                t.text(surf, str(o.id), c, (20, 20, 24), 10, bold=True, anchor="center")

        # Probability compass of the decision currently in force.
        probs = (last_applied.meta or {}).get("probabilities") if last_applied else None
        if isinstance(probs, dict):
            for name, prob in probs.items():
                if name in Action.__members__ and name != "STAY" and isinstance(prob, (int, float)):
                    dx, dy = Action(name).direction
                    Lg = (p.radius + 6 + 70 * float(prob)) * s
                    col = ACCENT if name == last_applied.action else ACCENT_DIM
                    pygame.draw.line(surf, col, pc, (int(pc[0] + dx * Lg), int(pc[1] + dy * Lg)), max(2, t.u(3)))
            stay = probs.get("STAY")
            if isinstance(stay, (int, float)) and stay > 0.02:
                self._ring(surf, ACCENT_DIM, pc, (p.radius + 4 + 20 * stay) * s)

        self._disc(surf, PLAYER, pc, p.radius * s)
        if action and action != "STAY":
            dx, dy = Action(action).direction
            r0 = (p.radius + 4) * s
            r1 = (p.radius + 20) * s
            a = (int(pc[0] + dx * r0), int(pc[1] + dy * r0))
            b = (int(pc[0] + dx * r1), int(pc[1] + dy * r1))
            pygame.draw.line(surf, TEXT, a, b, max(2, t.u(2)))
            ang = math.atan2(dy, dx)
            h = t.u(7)
            for sgn in (-0.55, 0.55):
                pygame.draw.line(surf, TEXT, b, (int(b[0] - h * math.cos(ang + sgn)),
                                                 int(b[1] - h * math.sin(ang + sgn))), max(2, t.u(2)))
        if debug:
            pygame.draw.aaline(surf, VEL, pc, (pc[0] + int(p.vx * 0.5 * s), pc[1] + int(p.vy * 0.5 * s)))
        surf.set_clip(prev_clip)

    # =================================================================== HUD
    def _draw_hud(self, surf, L: Layout, env, controller_name, action, latency_ms, fps, reserve=0) -> None:
        t, cfg = self.t, env.config
        pygame.draw.rect(surf, SURFACE, L.hud)
        pygame.draw.line(surf, BORDER, L.hud.bottomleft, L.hud.bottomright, max(1, t.u(1)))
        prev_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(L.hud.x, 0, max(0, L.hud.w - reserve - t.u(12)), L.hud.h))
        x, y = L.hud.x + t.u(16), t.u(9)
        r = t.text(surf, controller_name, (x, y), TEXT, 16, bold=True)
        since = env.world_time - env.last_target_time
        timer_col = WARN if since > cfg.target_timeout * 0.7 else MUTED
        t.text(surf, f"score {env.score}", (r.right + t.u(16), y + t.u(2)), OK, 14, bold=True)
        tx = r.right + t.u(16) + t.text_width(f"score {env.score}", 14, bold=True) + t.u(14)
        t.text(surf, f"{env.world_time:5.1f} s", (tx, y + t.u(2)), TEXT, 14, mono=True)
        t.text(surf, f"target timer {since:4.1f}/{cfg.target_timeout:g}s",
               (tx + t.text_width("000.0 s", 14, mono=True) + t.u(14), y + t.u(3)), timer_col, 12.5)
        lat = "–" if latency_ms is None else f"{latency_ms:.0f} ms"
        parts = [f"world {cfg.world_speed_scale:g}×", f"{len(env.obstacles)} obstacles", f"{cfg.decision_hz:g} Hz",
                 f"{cfg.max_inflight} in flight", f"action {action}", f"latency {lat}", f"seed {env.seed}"]
        if fps is not None:
            parts.append(f"{fps:.0f} fps")
        t.text(surf, "   ".join(parts), (x, y + t.u(26)), MUTED, 12)
        surf.set_clip(prev_clip)

    # ============================================================== timeline
    def _draw_timeline(self, surf, L: Layout, tl: dict[str, Any]) -> None:
        t = self.t
        r = L.timeline
        pygame.draw.rect(surf, SURFACE, r)
        pygame.draw.line(surf, BORDER, r.topleft, r.topright, max(1, t.u(1)))
        total = max(1, tl["total_ticks"])
        x0, x1 = r.x + t.u(16) + self.timeline_reserve, r.right - t.u(16)

        def X(tick):
            return int(x0 + (x1 - x0) * min(max(tick, 0), total) / total)

        top = r.bottom - t.u(self.TIMELINE_PT)
        yb = top + t.u(16)
        pygame.draw.line(surf, BORDER, (x0, yb), (x1, yb), max(1, t.u(2)))
        for tk in tl.get("requests", ()):
            pygame.draw.line(surf, FAINT, (X(tk), yb - t.u(3)), (X(tk), yb + t.u(3)))
        for tk in tl.get("applied", ()):
            pygame.draw.line(surf, ACCENT, (X(tk), yb - t.u(9)), (X(tk), yb - t.u(3)))
        for tk in tl.get("failed", ()):
            pygame.draw.line(surf, BAD, (X(tk), yb - t.u(10)), (X(tk), yb + t.u(10)), max(1, t.u(2)))
        for tk in tl.get("targets", ()):
            self._disc(surf, OK, (X(tk), yb + t.u(7)), t.u(3))
        ct = tl.get("collision_tick")
        if ct is not None:
            cx, h = X(ct), t.u(5)
            pygame.draw.line(surf, BAD, (cx - h, yb - h), (cx + h, yb + h), max(2, t.u(2)))
            pygame.draw.line(surf, BAD, (cx - h, yb + h), (cx + h, yb - h), max(2, t.u(2)))
        cx = X(tl["current"])
        pygame.draw.line(surf, TEXT, (cx, top + t.u(4)), (cx, yb + t.u(10)), max(2, t.u(2)))
        prev = surf.get_clip()
        surf.set_clip(pygame.Rect(x0, r.y, x1 - x0, r.h))
        t.text(surf, tl.get("label", ""), (x0, r.bottom - t.u(6)), MUTED, 11, anchor="bottomleft")
        surf.set_clip(prev)

    def timeline_tick_at(self, L: Layout, x: int, total_ticks: int) -> int:
        r = L.timeline
        x0, x1 = r.x + self.t.u(16) + self.timeline_reserve, r.right - self.t.u(16)
        return int(round((min(max(x, x0), x1) - x0) / max(1, x1 - x0) * total_ticks))

    # ================================================================== card
    def _draw_card(self, surf, L: Layout, env, lines: Sequence[str]) -> None:
        t = self.t
        o = env.outcome
        w, h = t.u(420), t.u(118 + 20 * len(lines) + 56)
        card = pygame.Rect(0, 0, w, h)
        card.center = L.arena.center
        shade = pygame.Surface(L.arena.size, pygame.SRCALPHA)
        shade.fill((0, 0, 0, 110))
        surf.blit(shade, L.arena.topleft)
        pygame.draw.rect(surf, SURFACE, card, border_radius=t.u(12))
        pygame.draw.rect(surf, BORDER, card, width=max(1, t.u(1)), border_radius=t.u(12))
        title = "Survived" if o.success else {"collision": "Collision", "target_timeout": "Target timeout"}.get(
            o.reason, str(o.reason))
        t.text(surf, title, (card.centerx, card.y + t.u(22)), OK if o.success else BAD, 26, bold=True,
               anchor="midtop")
        y = card.y + t.u(70)
        for line in lines:
            t.text(surf, line, (card.centerx, y), MUTED, 13, anchor="midtop")
            y += t.u(20)
        L.card = card
        L.card_buttons_y = card.bottom - t.u(52)

    # ================================================================= panel
    def _draw_panel(self, surf, L: Layout, env, ins: Optional[Inspector], mode, header, raw_request,
                    raw_response) -> None:
        t = self.t
        P = L.panel
        pygame.draw.rect(surf, SURFACE, P)
        pygame.draw.line(surf, BORDER, P.topleft, P.bottomleft, max(1, t.u(1)))
        x = P.x + t.u(18)
        w = P.w - t.u(36)
        y = t.u(14)
        t.text(surf, "Inspector", (x, y), TEXT, 16, bold=True)
        tx = x + t.text_width("Inspector", 16, bold=True) + t.u(14)
        for m in PANEL_MODES:
            col = ACCENT if m == mode else FAINT
            r = t.text(surf, m, (tx, y + t.u(3)), col, 12.5, bold=m == mode)
            if m == mode:
                pygame.draw.line(surf, ACCENT, (r.x, r.bottom + t.u(2)), (r.right, r.bottom + t.u(2)), t.u(2))
            tx = r.right + t.u(12)
        y += t.u(26)
        for h in header:
            t.text(surf, h, (x, y), MUTED, 11.5, mono=True)
            y += t.u(16)
        y += t.u(6)
        if ins is None:
            t.text(surf, "(no inspector)", (x, y), FAINT)
            return
        y = self._section_inflight(surf, env, ins, x, y, w)
        bottom = P.bottom - t.u(30)
        if mode == "decisions":
            y = self._section_decision(surf, ins, x, y, w)
            y = self._section_stats(surf, env, ins, x, y, w)
            self._section_recent(surf, ins, x, y, bottom)
        else:
            payload = raw_request if mode == "request" else raw_response
            title = "last request body" if mode == "request" else "last response"
            self._section_json(surf, title, payload, x, y, w, bottom)
        t.text(surf, "J panel   G ghost   P compass   F1 debug", (x, P.bottom - t.u(10)), FAINT, 11,
               anchor="bottomleft")

    def _rule(self, surf, x, y, w) -> int:
        pygame.draw.line(surf, BORDER, (x, y), (x + w, y), max(1, self.t.u(1)))
        return y + self.t.u(10)

    def _bar(self, surf, x, y, w, h, frac, color) -> None:
        t = self.t
        pygame.draw.rect(surf, SURFACE_2, (x, y, w, h), border_radius=t.u(3))
        if frac > 0:
            pygame.draw.rect(surf, color, (x, y, max(t.u(2), int(w * min(1.0, frac))), h), border_radius=t.u(3))

    def _section_inflight(self, surf, env, ins: Inspector, x, y, w) -> int:
        t = self.t
        y = self._rule(surf, x, y, w)
        reqs = ins.inflight_all()
        period_ms = 1000.0 / env.config.decision_hz
        cap = env.config.max_inflight
        if reqs:
            t.text(surf, f"In flight  {len(reqs)}/{cap}", (x, y), WARN, 13, bold=True)
            y += t.u(22)
            waits = [(r, (env.tick - r["tick"]) * TICK_MS) for r in reqs]
            scale_ms = max(500.0, max(wt for _, wt in waits) * 1.1)
            bw = w - t.u(120)
            for r, waited in waits[:5]:
                t.text(surf, f"#{r['id']}", (x, y - t.u(1)), MUTED, 11.5, mono=True)
                bx = x + t.u(52)
                self._bar(surf, bx, y + t.u(2), bw, t.u(9), waited / scale_ms, WARN)
                px = bx + int(bw * period_ms / scale_ms)
                pygame.draw.line(surf, TEXT, (px, y - t.u(1)), (px, y + t.u(13)), max(1, t.u(1)))
                t.text(surf, f"{waited:4.0f} ms", (x + w, y - t.u(1)), WARN, 11.5, mono=True, anchor="topright")
                y += t.u(17)
            t.text(surf, f"ghost = oldest snapshot   │ = decision period {period_ms:.0f} ms", (x, y), FAINT, 11)
            y += t.u(18)
        else:
            t.text(surf, f"Idle   0/{cap} requests in flight", (x, y), FAINT, 13, bold=True)
            y += t.u(24)
        return y + t.u(4)

    def _section_decision(self, surf, ins: Inspector, x, y, w) -> int:
        t = self.t
        y = self._rule(surf, x, y, w)
        last = ins.records[-1] if ins.records else None
        if last is None:
            t.text(surf, "No response yet", (x, y), FAINT, 13)
            return y + t.u(26)
        t.text(surf, f"Last response  #{last.id}", (x, y), MUTED, 12.5, bold=True)
        t.text(surf, last.status, (x + w, y), STATUS_COLOR[last.status], 12.5, bold=True, anchor="topright")
        y += t.u(22)
        t.text(surf, last.action or "—", (x, y), STATUS_COLOR[last.status], 30, bold=True)
        right = x + t.u(96)
        meta = last.meta or {}
        t.text(surf, f"latency {last.latency_ms:.0f} ms", (right, y + t.u(1)), TEXT, 13)
        t.text(surf, f"took effect {last.delay_ticks} ticks ({last.delay_ticks * TICK_MS:.0f} ms) after snapshot",
               (right, y + t.u(19)), MUTED, 11)
        conf = meta.get("confidence")
        if isinstance(conf, (int, float)):
            t.text(surf, f"confidence {conf:.2f}", (right, y + t.u(34)), ACCENT, 11.5, bold=True)
        y += t.u(54)
        if last.error:
            for i in range(0, min(len(last.error), 200), 56):
                t.text(surf, last.error[i:i + 56], (x, y), BAD, 11, mono=True)
                y += t.u(15)
            y += t.u(4)
        probs = meta.get("probabilities")
        if isinstance(probs, dict) and probs:
            bw = w - t.u(90)
            for name in COMPASS_ORDER:
                pv = probs.get(name)
                if not isinstance(pv, (int, float)):
                    continue
                chosen = name == last.action
                t.text(surf, name, (x + t.u(34), y), TEXT if chosen else MUTED, 11.5, bold=chosen, anchor="topright")
                self._bar(surf, x + t.u(42), y + t.u(3), bw, t.u(9), pv, ACCENT if chosen else ACCENT_DIM)
                t.text(surf, f"{pv:.2f}", (x + w, y), TEXT if chosen else MUTED, 11.5, mono=True, anchor="topright")
                y += t.u(16)
        elif last.status == "applied":
            t.text(surf, "controller reports no probabilities", (x, y), FAINT, 11)
            y += t.u(16)
        return y + t.u(6)

    def _section_stats(self, surf, env, ins: Inspector, x, y, w) -> int:
        t = self.t
        y = self._rule(surf, x, y, w)
        st = ins.stats()
        secs = max(env.tick / 60.0, 1e-9)
        mean_ms = "–" if st["mean_ms"] is None else f"{st['mean_ms']:.0f}"
        p95 = "–" if st["p95_ms"] is None else f"{st['p95_ms']:.0f}"
        cells = [("applied", st["applied"], TEXT), ("failed", st["failed"], BAD if st["failed"] else TEXT),
                 ("superseded", st["superseded"], TEXT), ("missed", st["missed"], TEXT)]
        cw = w // len(cells)
        for i, (label, val, col) in enumerate(cells):
            t.text(surf, str(val), (x + i * cw, y), col, 16, bold=True, mono=True)
            t.text(surf, label, (x + i * cw, y + t.u(21)), FAINT, 11)
        y += t.u(40)
        t.text(surf, f"latency mean {mean_ms} ms · p95 {p95} ms", (x, y), TEXT, 12)
        t.text(surf, f"{st['applied'] / secs:.1f}/s of {env.config.decision_hz:g} Hz", (x + w, y), MUTED, 12,
               anchor="topright")
        y += t.u(20)
        recent = st["recent_ms"]
        h = t.u(38)
        pygame.draw.rect(surf, SURFACE_2, (x, y, w, h), border_radius=t.u(4))
        if recent:
            top = max(max(recent) * 1.1, 1000.0 / env.config.decision_hz * 1.5)
            py = y + h - int(h * (1000.0 / env.config.decision_hz) / top)
            for i in range(0, w, t.u(6)):
                pygame.draw.line(surf, FAINT, (x + i, py), (x + i + t.u(3), py))
            step = w / 60
            for i, v in enumerate(recent):
                bh = max(t.u(2), int((h - t.u(2)) * v / top))
                pygame.draw.rect(surf, WARN, (x + int(i * step), y + h - bh, max(t.u(2), int(step) - t.u(1)), bh))
            t.text(surf, f"{top:.0f} ms", (x + w - t.u(4), y + t.u(2)), MUTED, 10, anchor="topright")
        y += h + t.u(4)
        t.text(surf, "latency per decision   (dashed = decision period)", (x, y), FAINT, 10.5)
        return y + t.u(22)

    def _section_recent(self, surf, ins: Inspector, x, y, bottom) -> int:
        t = self.t
        y = self._rule(surf, x, y, self.display.size[0] - x - t.u(18))
        cols = [("id", 0), ("t (s)", 44), ("action", 104), ("conf", 170), ("latency", 222), ("delay", 300)]
        for label, cx in cols:
            t.text(surf, label, (x + t.u(cx), y), FAINT, 11)
        y += t.u(18)
        for r in reversed(ins.records):
            if y > bottom - t.u(16):
                break
            conf = (r.meta or {}).get("confidence")
            color = STATUS_COLOR[r.status]
            vals = [f"{r.id}", f"{r.request_tick / 60:.2f}", r.action or r.status,
                    f"{conf:.2f}" if isinstance(conf, (int, float)) else "–", f"{r.latency_ms:.0f} ms",
                    f"+{r.delay_ticks}t"]
            for (label, cx), v in zip(cols, vals):
                t.text(surf, v, (x + t.u(cx), y), color, 11.5, mono=True)
            y += t.u(17)
        return y

    def _section_json(self, surf, title, payload, x, y, w, bottom) -> int:
        t = self.t
        y = self._rule(surf, x, y, w)
        t.text(surf, title + "   (floats rounded for display)", (x, y), MUTED, 11.5)
        y += t.u(20)
        if payload is None:
            t.text(surf, "(nothing yet)", (x, y), FAINT, 12)
            return y + t.u(20)
        cw = max(1, t.text_width("0", 11, mono=True))
        max_chars = max(20, w // cw)
        lines = []
        for line in compact_json(payload):
            while len(line) > max_chars:  # wrap, never silently cut
                lines.append(line[:max_chars])
                line = "      " + line[max_chars:]
            lines.append(line)
        lh = t.u(15)
        for line in lines:
            if y > bottom - lh:
                t.text(surf, "…", (x, y), FAINT, 11, mono=True)
                break
            t.text(surf, line, (x, y), TEXT, 11, mono=True)
            y += lh
        return y
