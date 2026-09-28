"""Window + frame presentation.

Preferred backend: an SDL2 window created with ``allow_highdpi``, so on Retina
and other HiDPI screens we draw at the real pixel density instead of letting
the OS upscale a half-resolution image (which is what made the old UI look
hazy). Every frame is drawn into ``surface`` at full pixel resolution, uploaded
to a streaming texture and presented with vsync.

Fallback: the classic ``pygame.display.set_mode`` window (density 1).

Coordinates: ``surface`` is in *pixels*. ``density`` = pixels per window
point; mouse events arrive in points, so convert with ``to_pixels``.
"""
from __future__ import annotations

import os
from typing import Optional

import pygame


def _init_pygame() -> None:
    # Only what we need: no audio (avoids noisy audio-device probing).
    pygame.display.init()
    pygame.font.init()


def default_window_size() -> tuple[int, int]:
    _init_pygame()
    try:
        sizes = pygame.display.get_desktop_sizes()
        sw, sh = sizes[0]
    except Exception:
        sw, sh = 1440, 900
    return max(960, int(sw * 0.9)), max(640, int(sh * 0.85))


class Display:
    def __init__(self, size: Optional[tuple[int, int]] = None, title: str = "Decision Arena",
                 hidpi: bool = True, vsync: bool = True):
        _init_pygame()
        self.title = title
        self.points = tuple(size or default_window_size())
        self.backend = "classic"
        self.density = 1.0
        self._texture = None
        if hidpi and os.environ.get("ARENA_CLASSIC_WINDOW") != "1":
            try:
                self._open_sdl2(vsync)
            except Exception:
                self.backend = "classic"
        if self.backend == "classic":
            self.surface = pygame.display.set_mode(self.points, pygame.RESIZABLE)
            pygame.display.set_caption(title)

    # ---------------------------------------------------------------- sdl2
    def _open_sdl2(self, vsync: bool) -> None:
        from pygame._sdl2.video import Renderer, Window

        self.window = Window(self.title, size=self.points, allow_highdpi=True, resizable=True)
        try:
            self.renderer = Renderer(self.window, vsync=vsync)
        except Exception:
            self.renderer = Renderer(self.window)
        self.backend = "sdl2"
        self._fit_surface()

    def _fit_surface(self) -> None:
        self.points = tuple(self.window.size)
        vp = self.renderer.get_viewport()
        self.density = max(1.0, vp.w / max(1, self.points[0]))
        size = (max(1, vp.w), max(1, vp.h))
        # ARGB8888 matches the streaming texture, so uploads skip a per-frame
        # pixel-format conversion (about half the upload cost).
        self.surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        self._texture = None

    # ------------------------------------------------------------- public
    @property
    def size(self) -> tuple[int, int]:
        return self.surface.get_size()

    def to_pixels(self, pos: tuple[int, int]) -> tuple[int, int]:
        return int(pos[0] * self.density), int(pos[1] * self.density)

    def handle_resize(self, event: pygame.event.Event) -> bool:
        """Call for every event; returns True if the drawable size changed."""
        if event.type == pygame.VIDEORESIZE or event.type == getattr(pygame, "WINDOWSIZECHANGED", -1):
            if self.backend == "sdl2":
                # The viewport follows the window only after a present.
                self.renderer.set_viewport(None)
                self._fit_surface()
            else:
                self.surface = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE) \
                    if event.type == pygame.VIDEORESIZE else pygame.display.get_surface()
            return True
        return False

    def present(self) -> None:
        if self.backend == "sdl2":
            from pygame._sdl2.video import Texture

            if self._texture is None or (self._texture.width, self._texture.height) != self.surface.get_size():
                self._texture = Texture(self.renderer, self.surface.get_size(), streaming=True)
            self._texture.update(self.surface)
            self.renderer.clear()
            self._texture.draw()
            self.renderer.present()
        else:
            pygame.display.flip()

    def save(self, path: str) -> None:
        # Save what the screen shows: RGB only (the frame is fully opaque anyway).
        img = pygame.Surface(self.surface.get_size(), 0, 32)
        img.blit(self.surface, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        pygame.image.save(img, path)

    def set_title(self, title: str) -> None:
        if self.backend == "sdl2":
            self.window.title = title
        else:
            pygame.display.set_caption(title)

    def close(self) -> None:
        if self.backend == "sdl2":
            self.window.destroy()
        pygame.display.quit()
