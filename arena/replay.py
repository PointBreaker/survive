"""Deterministic replay from (config, seed, per-tick action changes).

The environment is a pure function of these three things, so a logged
episode can be re-simulated exactly, and any decision can be inspected
against the exact world state it was made in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from arena.action import Action
from arena.difficulty import DifficultyConfig
from arena.environment import Environment


def actions_per_tick(action_changes: Sequence[Sequence], max_ticks: int) -> Iterator[Action]:
    changes = sorted((int(t), Action(a)) for t, a in action_changes)
    current = Action.STAY
    i = 0
    for tick in range(max_ticks):
        while i < len(changes) and changes[i][0] <= tick:
            current = changes[i][1]
            i += 1
        yield current


def replay(
    config: DifficultyConfig,
    seed: int,
    action_changes: Iterable[Sequence],
    max_ticks: Optional[int] = None,
) -> Environment:
    env = Environment(config, seed)
    limit = max_ticks if max_ticks is not None else 10**9
    for action in actions_per_tick(list(action_changes), limit):
        if env.done:
            break
        env.step(action)
    return env


def replay_run_dir(run_dir: Path | str) -> Environment:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    config = DifficultyConfig.from_dict(cfg["config"])
    return replay(config, cfg["seed"], result["action_changes"], max_ticks=result["ticks"])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Re-simulate a logged run and compare with its result.")
    ap.add_argument("run_dir")
    args = ap.parse_args()
    env = replay_run_dir(args.run_dir)
    result = json.loads((Path(args.run_dir) / "result.json").read_text())
    same = (
        env.tick == result["ticks"]
        and env.score == result["targets_collected"]
        and env.outcome.reason == result["reason"]
    )
    print(f"replayed ticks={env.tick} score={env.score} outcome={env.outcome.reason}")
    print(f"logged   ticks={result['ticks']} score={result['targets_collected']} outcome={result['reason']}")
    print("MATCH" if same else "MISMATCH")


if __name__ == "__main__":
    main()
