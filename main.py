"""Interactive Decision Arena (pygame window, world paced to the wall clock).

    python main.py --controller human
    python main.py --controller simple_avoid --obstacles 30 --world-speed 2
    python main.py --controller simple_avoid --latency-ms 150 --real-latency

The right-hand inspector panel shows every request/response live. The faded
"ghost" in the arena is the snapshot the controller is currently deciding
on; the gap between ghost and reality is the latency you are paying.

Keys: WASD / arrows move (human), J cycles panel (decisions / request JSON /
response JSON), G ghost, P probability compass, F1 or Tab debug overlay,
R restart with next seed, V replay the finished episode, Esc quit.
"""
from __future__ import annotations

import argparse
import sys

from arena.dotenv import load_dotenv
from arena.cli import add_difficulty_args, config_from_args, controller_factory
from arena.recorder import JsonlRecorder, NullRecorder, new_run_dir
from urllib.parse import urlsplit

INTERACTIVE_CONTROLLERS = ("human", "random", "greedy", "simple_avoid", "sleep", "jev", "remote")


def controller_header(ctrl) -> list[str]:
    inner = ctrl
    while hasattr(inner, "inner"):
        inner = inner.inner
    lines = []
    model = getattr(inner, "model", None)
    endpoint = getattr(inner, "endpoint", None)
    if model:
        lines.append(f"model    {model}")
    if endpoint:
        lines.append(f"endpoint {urlsplit(endpoint).netloc}{urlsplit(endpoint).path}")
    if inner is not ctrl:
        lines.append(f"wrapper  {ctrl.name}")
    return lines


def raw_traffic(ctrl, inspector):
    """Wire bodies for remote controllers; the observation for in-process ones."""
    inner = ctrl
    while hasattr(inner, "inner"):
        inner = inner.inner
    req = getattr(inner, "last_request_body", None)
    resp = getattr(inner, "last_response", None)
    if req is None:
        r = inspector.inflight() or (inspector.requests.get(inspector.records[-1].id) if inspector.records else None)
        req = {"observation": r["observation"]} if r else None
    if resp is None and inspector.records:
        last = inspector.records[-1]
        resp = {k: v for k, v in last.__dict__.items() if v is not None}
    return req, resp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=INTERACTIVE_CONTROLLERS, default="human")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-log", action="store_true", help="do not write run logs")
    ap.add_argument("--no-panel", action="store_true", help="hide the inspector panel")
    ap.add_argument("--panel", choices=("decisions", "request", "response"), default="decisions",
                    help="initial inspector panel (J cycles)")
    ap.add_argument("--scale", type=float, default=None, help="arena zoom (default: fit screen)")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)  # smoke tests: quit after N frames
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    add_difficulty_args(ap)
    args = ap.parse_args(argv)
    load_dotenv()  # e.g. OPENROUTER_API_KEY for --controller jev

    # Humans need a responsive input loop; other controllers use the config default.
    human_hz = 60.0 if args.controller == "human" and args.decision_hz is None else None
    cfg = config_from_args(args, decision_hz=human_hz)
    make_ctrl = controller_factory(args.controller, args)

    import pygame

    from arena.environment import Environment
    from arena.inspector import Inspector, TeeRecorder, timeline_data
    from arena.renderer import PANEL_MODES, Renderer
    from arena.runner import EpisodeRunner

    renderer = Renderer(cfg.arena_width, cfg.arena_height, panel=not args.no_panel, timeline=True, scale=args.scale)
    max_ticks = int(cfg.max_duration / cfg.world_speed_scale * 60)
    clock = pygame.time.Clock()
    debug = args.debug
    ghost, compass = True, True
    panel_mode = PANEL_MODES.index(args.panel)
    seed = args.seed
    frames = 0
    run_dir = None

    def new_episode(seed: int):
        nonlocal run_dir
        ctrl = make_ctrl()
        run_dir = None if args.no_log else new_run_dir(args.out, ctrl.name, f"seed{seed}")
        inner = NullRecorder() if run_dir is None else JsonlRecorder(run_dir)
        inspector = Inspector()
        rec = TeeRecorder(inner, inspector)
        runner = EpisodeRunner(Environment(cfg, seed), ctrl, recorder=rec, realtime=True)
        runner.start()
        return runner, rec, inspector

    runner, rec, inspector = new_episode(seed)
    header = controller_header(runner.controller)
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
                elif ev.key == pygame.K_j:
                    panel_mode = (panel_mode + 1) % len(PANEL_MODES)
                elif ev.key == pygame.K_g:
                    ghost = not ghost
                elif ev.key == pygame.K_p:
                    compass = not compass
                elif ev.key == pygame.K_v and finished and run_dir is not None:
                    from arena.viewer import run_viewer

                    run_viewer(run_dir, renderer=renderer)
                elif ev.key == pygame.K_r:
                    if not finished:
                        runner.finish()
                    runner.controller.close()
                    rec.close()
                    seed += 1
                    runner, rec, inspector = new_episode(seed)
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
                if run_dir is not None:
                    print(f"  replay: python -m arena.viewer {run_dir}   (or press V)")

        lat = None if runner.last_latency_s is None else runner.last_latency_s * 1000
        req, resp = raw_traffic(runner.controller, inspector)
        status = ("R restart" + ("   V replay" if run_dir else "")) if finished else ""
        renderer.draw(runner.env, runner.controller.name, runner.current_action.value, lat, debug,
                      inspector=inspector, ghost=ghost, compass=compass, panel_mode=PANEL_MODES[panel_mode],
                      header=header, raw_request=req, raw_response=resp, status=status,
                      timeline=timeline_data(inspector, max_ticks, runner.env.tick,
                                             f"LIVE  tick {runner.env.tick}  (timeline = max_duration)"))
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
