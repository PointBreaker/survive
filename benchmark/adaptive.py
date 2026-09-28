"""Adaptive difficulty: staircase search for a controller's failure frontier.

Kept completely outside the game logic. It only builds configs/controllers
for a level, runs batches through ``benchmark.runner`` and reads success.

Procedure (for a parameter where *larger is harder*):
1. Geometric growth from ``start``: x1, x2, x4, ... while the batch passes.
2. On the first failing level, bisect between the last pass and first fail.
The same seeds are used at every level, so levels are paired comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from arena.difficulty import DifficultyConfig
from benchmark.metrics import aggregate, threshold_level
from benchmark.runner import ControllerFactory, run_episodes


@dataclass
class FrontierResult:
    param: str
    pass_rate: float
    highest_pass: Optional[float]
    lowest_fail: Optional[float]
    levels: list[dict[str, Any]] = field(default_factory=list)

    @property
    def curve(self) -> list[tuple[float, float]]:
        return sorted((lv["level"], lv["success_rate"]) for lv in self.levels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "param": self.param,
            "pass_rate": self.pass_rate,
            "highest_pass": self.highest_pass,
            "lowest_fail": self.lowest_fail,
            "interpolated_threshold": threshold_level(self.curve, self.pass_rate),
            "levels": self.levels,
        }


def find_frontier(
    param: str,
    make_config: Callable[[float], DifficultyConfig],
    make_controller: Callable[[float], ControllerFactory],
    start: float = 1.0,
    episodes_per_level: int = 10,
    pass_rate: float = 0.5,
    max_level: float = 64.0,
    refine_steps: int = 4,
    integer: bool = False,
    base_seed: int = 0,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> FrontierResult:
    out = FrontierResult(param, pass_rate, None, None)
    tested: dict[float, float] = {}

    def evaluate(level: float) -> bool:
        if level in tested:
            return tested[level] >= pass_rate
        results = run_episodes(make_controller(level), make_config(level), episodes_per_level, base_seed)
        agg = aggregate(results)
        rec = {"level": level, **agg}
        out.levels.append(rec)
        tested[level] = agg["success_rate"]
        if progress:
            progress(rec)
        return agg["success_rate"] >= pass_rate

    norm = (lambda x: float(max(1, round(x)))) if integer else (lambda x: x)

    level = norm(start)
    while level <= max_level:
        if evaluate(level):
            out.highest_pass = level
            level = norm(level * 2)
        else:
            out.lowest_fail = level
            break
    if out.lowest_fail is None:
        return out  # never failed within max_level
    if out.highest_pass is None:
        # Failed at the start level: search downward for a passing level.
        level = out.lowest_fail
        while True:
            nxt = norm(level / 2)
            if nxt == level or nxt < (1 if integer else 1e-3):
                return out
            level = nxt
            if evaluate(level):
                out.highest_pass = level
                break
            out.lowest_fail = level

    for _ in range(refine_steps):
        mid = norm((out.highest_pass + out.lowest_fail) / 2)
        if mid in (out.highest_pass, out.lowest_fail):
            break
        if evaluate(mid):
            out.highest_pass = mid
        else:
            out.lowest_fail = mid
    return out
