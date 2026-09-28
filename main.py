"""Interactive Decision Arena (pygame window, world paced to the wall clock).

    python main.py --controller human
    python main.py --controller simple_avoid --obstacles 30 --world-speed 2
    python main.py --controller simple_avoid --latency-ms 150 --real-latency

Keys: WASD / arrows move (human), F1 or Tab toggles debug overlay,
R restarts with the next seed, Esc quits.
"""
from __future__ import annotations

import argparse
import sys

from arena.cli import add_difficulty_args, config_from_args, controller_factory
from arena.recorder import JsonlRecorder, NullRecorder, new_run_dir

INTERACTIVE_CONTROLLERS = ("human", "random", "greedy", "simple_avoid", "sleep", "jev", "remote")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=INTERACTIVE_CONTROLLERS, default="human")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-log", action="store_true", help="do not write run logs")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)  # smoke tests: quit after N frames
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    add_difficulty_args(ap)
    args = ap.parse_args(argv)

    # Humans need a responsive input loop; other controllers use the config default.
    human_hz = 60.0 if args.controller == "human" and args.decision_hz is None else None
    cfg = config_from_args(args, decision_hz=human_hz)
    make_ctrl = controller_factory(args.controller, args)

    import pygame

    from arena.environment import Environment
    from arena.renderer import Renderer
    from arena.runner import EpisodeRunner

    renderer = Renderer(cfg.arena_width, cfg.arena_height)
    clock = pygame.time.Clock()
    debug = args.debug
    seed = args.seed
    frames = 0

    def new_episode(seed: int):
        ctrl = make_ctrl()
        rec = NullRecorder() if args.no_log else JsonlRecorder(new_run_dir(args.out, ctrl.name, f"seed{seed}"))
        runner = EpisodeRunner(Environment(cfg, seed), ctrl, recorder=rec, realtime=True)
        runner.start()
        return runner, rec

    runner, rec = new_episode(seed)
    finished = False
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in (pygame.K_F1, pygame.K_TAB):
                    debug = not debug
                elif ev.key == pygame.K_r:
                    if not finished:
                        runner.finish()
                    runner.controller.close()
                    rec.close()
                    seed += 1
                    runner, rec = new_episode(seed)
                    finished = False

        if not finished:
            runner.advance_realtime()
            if runner.env.done:
                result = runner.finish()
                rec.close()
                finished = True
                print(
                    f"[{result['controller']}] seed={result['seed']} success={result['success']} "
                    f"reason={result['reason']} time={result['survival_time']:.2f}s "
                    f"targets={result['targets_collected']} decisions={result['decision_count']} "
                    f"mean_latency={result['mean_decision_latency_ms'] or 0:.2f}ms"
                )

        lat = None if runner.last_latency_s is None else runner.last_latency_s * 1000
        renderer.draw(runner.env, runner.controller.name, runner.current_action.value, lat, debug)
        clock.tick(60)
        frames += 1
        if args.frames and frames >= args.frames:
            if args.screenshot:
                pygame.image.save(renderer.screen, args.screenshot)
            running = False

    if not finished:
        runner.finish()
        rec.close()
    runner.controller.close()
    renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
