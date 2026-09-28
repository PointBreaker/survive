"""Keyboard controller (WASD / arrow keys, 8 directions).

The human's "observation" is the rendered screen, which draws the same
objective state that other controllers receive as data.
"""
from __future__ import annotations

from arena.action import Action
from arena.observation import Observation
from controllers.base import SyncController


class HumanController(SyncController):
    name = "human"

    def decide(self, observation: Observation) -> Action:
        import pygame  # imported lazily so headless runs never load pygame

        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        dy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
        return Action.from_vector(dx, dy)
