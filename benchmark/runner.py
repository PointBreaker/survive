"""Batch episode execution (headless)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.recorder import JsonlRecorder, Recorder
from arena.runner import EpisodeRunner
from controllers.base import Controller

ControllerFactory = Callable[[], Controller]


def run_episode(
    make_controller: ControllerFactory,
    config: DifficultyConfig,
    seed: int,
    recorder: Optional[Recorder] = None,
    realtime: bool = False,
) -> dict[str, Any]:
    controller = make_controller()
    try:
        env = Environment(config, seed)
        return EpisodeRunner(env, controller, recorder=recorder, realtime=realtime).run()
    finally:
        controller.close()
        if recorder is not None:
            recorder.close()


def run_episodes(
    make_controller: ControllerFactory,
    config: DifficultyConfig,
    episodes: int,
    base_seed: int = 0,
    events_dir: Optional[Path] = None,
    record_observations: bool = True,
    realtime: bool = False,
    progress: Optional[Callable[[int, dict[str, Any]], None]] = None,
) -> list[dict[str, Any]]:
    """Episode i uses seed ``base_seed + i``: identical worlds across controllers."""
    results = []
    for i in range(episodes):
        seed = base_seed + i
        rec = None
        if events_dir is not None:
            rec = JsonlRecorder(Path(events_dir) / f"episode_{i:04d}_seed{seed}", record_observations)
        r = run_episode(make_controller, config, seed, recorder=rec, realtime=realtime)
        results.append(r)
        if progress:
            progress(i, r)
    return results
