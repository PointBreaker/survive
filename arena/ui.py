"""Theme + a tiny immediate-mode widget toolkit for pygame.

All sizes are given in *points* and multiplied by the display density, so
text and shapes are drawn at the screen's real pixel resolution.

Usage per frame: ``ui.begin(mouse_px)``, draw widgets (each registers its hit
rectangle), then on a click ``ui.hit(pos_px)`` returns ``(key, value)``.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pygame

# Palette: high contrast on a near-black background.
BG = (12, 14, 18)
SURFACE = (20, 23, 29)
SURFACE_2 = (28, 32, 40)
BORDER = (44, 49, 60)
ARENA_BG = (16, 19, 24)
GRID = (26, 30, 38)
TEXT = (236, 238, 242)
MUTED = (160, 167, 180)
FAINT = (105, 112, 126)
ACCENT = (90, 170, 255)
ACCENT_DIM = (52, 92, 150)
PLAYER = (64, 200, 255)
OBSTACLE = (255, 94, 84)
TARGET = (74, 222, 128)
GHOST = (170, 176, 215)
WARN = (255, 181, 51)
BAD = (255, 99, 90)
OK = (74, 222, 128)

UI_FONTS = "Helvetica Neue,Segoe UI,SF Pro Text,Inter,Arial,DejaVu Sans,sans-serif"
MONO_FONTS = "Menlo,SF Mono,Consolas,JetBrains Mono,DejaVu Sans Mono,monospace"


class Theme:
    def __init__(self, density: float = 1.0):
        self.d = density
        self._cache: dict[tuple, pygame.font.Font] = {}
        self._glyphs: dict[tuple, pygame.Surface] = {}  # rendered text, most panel text repeats per frame

    def u(self, v: float) -> int:
        return int(round(v * self.d))

    def font(self, size: float, bold: bool = False, mono: bool = False) -> pygame.font.Font:
        key = (size, bold, mono, self.d)
        f = self._cache.get(key)
        if f is None:
            f = pygame.font.SysFont(MONO_FONTS if mono else UI_FONTS, self.u(size), bold=bold)
            self._cache[key] = f
        return f

    def text(self, surf: pygame.Surface, s: str, pos: tuple[int, int], color=TEXT, size: float = 13,
             bold: bool = False, mono: bool = False, anchor: str = "topleft") -> pygame.Rect:
        key = (s, color, size, bold, mono)
        img = self._glyphs.get(key)
        if img is None:
            if len(self._glyphs) > 4000:
                self._glyphs.clear()
            img = self._glyphs[key] = self.font(size, bold, mono).render(s, True, color)
        r = img.get_rect(**{anchor: pos})
        surf.blit(img, r)
        return r

    def text_width(self, s: str, size: float = 13, bold: bool = False, mono: bool = False) -> int:
        return self.font(size, bold, mono).size(s)[0]


class UI:
    def __init__(self, theme: Theme):
        self.t = theme
        self.hits: list[tuple[pygame.Rect, str, Any, bool]] = []
        self.mouse = (-1, -1)

    def begin(self, mouse_px: tuple[int, int]) -> None:
        self.hits = []
        self.mouse = mouse_px

    def hit(self, pos_px: tuple[int, int]) -> Optional[tuple[str, Any]]:
        for rect, key, value, enabled in reversed(self.hits):
            if enabled and rect.collidepoint(pos_px):
                return key, value
        return None

    def hovering(self) -> bool:
        return any(e and r.collidepoint(self.mouse) for r, _, _, e in self.hits)

    # ------------------------------------------------------------ widgets
    def button(self, surf, rect: pygame.Rect, label: str, key: str, value: Any = None, kind: str = "normal",
               enabled: bool = True, selected: bool = False, size: float = 13, hint: str = "") -> pygame.Rect:
        t = self.t
        hover = enabled and rect.collidepoint(self.mouse)
        if kind == "primary":
            bg = (110, 185, 255) if hover else ACCENT
            fg = (8, 16, 28)
            border = None
        elif selected:
            bg, fg, border = ACCENT_DIM, TEXT, ACCENT
        else:
            bg = SURFACE_2 if not hover else (38, 43, 54)
            fg = TEXT if enabled else FAINT
            border = BORDER
        pygame.draw.rect(surf, bg, rect, border_radius=t.u(7))
        if border:
            pygame.draw.rect(surf, border, rect, width=max(1, t.u(1)), border_radius=t.u(7))
        if hint:
            t.text(surf, label, (rect.centerx, rect.centery - t.u(7)), fg, size, bold=kind == "primary",
                   anchor="center")
            t.text(surf, hint, (rect.centerx, rect.centery + t.u(9)), MUTED if kind != "primary" else fg, 11,
                   anchor="center")
        else:
            t.text(surf, label, rect.center, fg, size, bold=kind == "primary", anchor="center")
        self.hits.append((rect, key, value, enabled))
        return rect

    def chips(self, surf, x: int, y: int, options: Sequence[tuple[Any, str]], selected: Any, key: str,
              min_w: float = 44, h: float = 28, size: float = 12.5) -> int:
        """A row of mutually exclusive pills. Returns the bottom y."""
        t = self.t
        gap = t.u(6)
        for value, label in options:
            w = max(t.u(min_w), t.text_width(label, size) + t.u(20))
            r = pygame.Rect(x, y, w, t.u(h))
            self.button(surf, r, label, key, value, selected=(value == selected), size=size)
            x += w + gap
        return y + t.u(h)

    def stepper(self, surf, x: int, y: int, value_label: str, key: str, w: float = 64, h: float = 28) -> int:
        t = self.t
        b = t.u(h)
        self.button(surf, pygame.Rect(x, y, b, b), "−", key, -1, size=15)
        box = pygame.Rect(x + b + t.u(4), y, t.u(w), b)
        pygame.draw.rect(surf, SURFACE, box, border_radius=t.u(6))
        pygame.draw.rect(surf, BORDER, box, width=max(1, t.u(1)), border_radius=t.u(6))
        t.text(surf, value_label, box.center, TEXT, 13, mono=True, anchor="center")
        self.button(surf, pygame.Rect(box.right + t.u(4), y, b, b), "+", key, +1, size=15)
        return y + b

    def toggle(self, surf, x: int, y: int, label: str, on: bool, key: str) -> int:
        t = self.t
        w, h = t.u(38), t.u(22)
        r = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, ACCENT if on else SURFACE_2, r, border_radius=h // 2)
        pygame.draw.rect(surf, BORDER, r, width=max(1, t.u(1)), border_radius=h // 2)
        knob_x = r.right - h // 2 if on else r.left + h // 2
        pygame.draw.circle(surf, TEXT, (knob_x, r.centery), h // 2 - t.u(3))
        t.text(surf, label, (r.right + t.u(10), r.centery), TEXT, 13, anchor="midleft")
        full = pygame.Rect(x, y, w + t.u(10) + t.text_width(label, 13), h)
        self.hits.append((full, key, not on, True))
        return y + h
