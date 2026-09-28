"""Controllers. They may only import the public interface:
``arena.action`` and ``arena.observation`` (plus stdlib and each other)."""
from __future__ import annotations

from typing import Callable

from controllers.base import Controller


def _registry() -> dict[str, Callable[..., Controller]]:
    from controllers.greedy import GreedyController
    from controllers.human import HumanController
    from controllers.random import RandomController
    from controllers.simple_avoid import SimpleAvoidController
    from controllers.sleep import SleepController

    return {
        "human": HumanController,
        "random": RandomController,
        "greedy": GreedyController,
        "simple_avoid": SimpleAvoidController,
        "sleep": SleepController,
    }


CONTROLLER_NAMES = ("human", "random", "greedy", "simple_avoid", "sleep")


def make_controller(name: str, **kwargs) -> Controller:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"unknown controller {name!r}; choose from {sorted(reg)}")
    return reg[name](**kwargs)
