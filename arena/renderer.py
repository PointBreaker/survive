"""pygame renderer. Visualization only: it reads environment state for
drawing and never feeds anything back to controllers."""
from __future__ import annotations

import math
from typing import Optional

import pygame

from arena.environment import Environment

BG = (18, 20, 26)
GRID = (28, 31, 40)
PLAYER = (80, 200, 255)
OBSTACLE = (235, 90, 80)
TARGET = (120, 230, 120)
TEXT = (220, 224, 232)
DIM = (140, 146, 160)
VEL = (255, 220, 90)


class Renderer:
    HUD_HEIGHT = 56

    def __init__(self, width: float, height: float, title: str = "Decision Arena"):
        pygame.init()
        self.w, self.h = int(width), int(height)
        self.screen = pygame.display.set_mode((self.w, self.h + self.HUD_HEIGHT))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("monospace", 16)
        self.big = pygame.font.SysFont("monospace", 34, bold=True)
        self._grid = self._make_grid()

    def _make_grid(self) -> pygame.Surface:
        s = pygame.Surface((self.w, self.h))
        s.fill(BG)
        for x in range(0, self.w, 50):
            pygame.draw.line(s, GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, 50):
            pygame.draw.line(s, GRID, (0, y), (self.w, y))
        return s

    def draw(
        self,
        env: Environment,
        controller_name: str,
        action: str,
        latency_ms: Optional[float],
        debug: bool = False,
    ) -> None:
        scr = self.screen
        oy = self.HUD_HEIGHT
        scr.fill(BG)
        scr.blit(self._grid, (0, oy))

        t = env.target
        pygame.draw.circle(scr, TARGET, (int(t.x), int(t.y + oy)), int(t.radius))
        pygame.draw.circle(scr, BG, (int(t.x), int(t.y + oy)), max(1, int(t.radius * 0.45)))

        for o in env.obstacles:
            c = (int(o.x), int(o.y + oy))
            pygame.draw.circle(scr, OBSTACLE, c, int(o.radius))
            if debug:
                pygame.draw.line(scr, VEL, c, (int(o.x + o.vx * 0.5), int(o.y + oy + o.vy * 0.5)), 1)
                label = self.font.render(str(o.id), True, TEXT)
                scr.blit(label, label.get_rect(center=c))

        p = env.player
        pc = (int(p.x), int(p.y + oy))
        pygame.draw.circle(scr, PLAYER, pc, int(p.radius))
        if debug:
            pygame.draw.line(scr, VEL, pc, (int(p.x + p.vx * 0.5), int(p.y + oy + p.vy * 0.5)), 2)

        cfg = env.config
        lat = "-" if latency_ms is None else f"{latency_ms:.1f}ms"
        line1 = (
            f"{controller_name:<14} score {env.score:<3} time {env.world_time:6.2f}s  "
            f"world {cfg.world_speed_scale:g}x  obstacles {len(env.obstacles)}  decision {cfg.decision_hz:g}Hz"
        )
        since = env.world_time - env.last_target_time
        line2 = (
            f"action {action:<5} latency {lat:<9} seed {env.seed:<6} "
            f"target timer {since:4.1f}/{cfg.target_timeout:g}s  speed {math.hypot(p.vx, p.vy):5.1f}"
            + ("  [debug]" if debug else "")
        )
        pygame.draw.rect(scr, (10, 11, 15), (0, 0, self.w, oy))
        scr.blit(self.font.render(line1, True, TEXT), (10, 8))
        scr.blit(self.font.render(line2, True, DIM), (10, 30))

        if env.done:
            o = env.outcome
            msg = "SURVIVED" if o.success else f"FAILED: {o.reason}"
            surf = self.big.render(msg, True, TARGET if o.success else OBSTACLE)
            scr.blit(surf, surf.get_rect(center=(self.w // 2, oy + self.h // 2 - 20)))
            hint = self.font.render("R: restart (next seed)   Esc: quit", True, TEXT)
            scr.blit(hint, hint.get_rect(center=(self.w // 2, oy + self.h // 2 + 20)))

        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
