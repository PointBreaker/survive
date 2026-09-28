"""Single-page console, driven by synthetic clicks and keys (SDL dummy driver)."""
import pygame
import pytest

from arena.app import App, ConsoleScene, Settings
from arena.display import Display


@pytest.fixture
def app(tmp_path, monkeypatch):
    for k in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "JEV_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    a = App(Display((1500, 900)))
    a.settings = Settings(out=str(tmp_path / "runs"))
    a.scene = ConsoleScene(a)
    yield a
    a.scene.close()
    a.display.close()


def frame(app):
    app.scene.update()
    app.ui.t = app.renderer.t
    app.ui.begin((-1, -1))
    app.scene.draw()


def click(app, key, value=None):
    frame(app)
    for rect, k, v, enabled in app.ui.hits:
        if k == key and (value is None or v == value) and enabled:
            ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center)
            ev.px = rect.center
            app.scene.handle(ev)
            return
    raise AssertionError(f"no widget {key}={value}")


def press(app, k):
    app.scene.handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0))


def finish_episode(scene):
    while scene.state == "running":
        scene.runner.step_tick()  # drive without waiting for wall time
        scene.update()


def test_opens_ready_with_all_three_columns(app):
    frame(app)
    L = app.scene.layout
    assert app.scene.state == "ready" and app.scene.runner is None
    assert L.sidebar.w > 0 and L.panel.w > 0 and L.sidebar.right <= L.arena.x and L.arena.right <= L.panel.x


def test_configure_by_clicking_then_start(app):
    sc = app.scene
    click(app, "controller", "greedy")
    click(app, "speed", 2.0)
    click(app, "obstacles", 20)
    click(app, "inflight", 3)
    click(app, "duration", 20.0)
    click(app, "seed", +1)
    assert sc.state == "ready"  # nothing runs until Start
    click(app, "start")
    cfg = sc.runner.env.config
    assert sc.state == "running" and sc.runner.controller.name == "greedy"
    assert (cfg.world_speed_scale, cfg.obstacle_count, cfg.max_inflight, cfg.max_duration) == (2.0, 20, 3, 20.0)
    assert sc.runner.env.seed == 1


def test_param_change_while_running_restarts_same_seed_and_keeps_running(app):
    sc = app.scene
    click(app, "start")
    for _ in range(5):
        sc.runner.step_tick()
    click(app, "speed", 4.0)
    assert sc.state == "running" and sc.runner.env.tick == 0
    assert sc.runner.env.config.world_speed_scale == 4.0 and sc.runner.env.seed == 0
    press(app, pygame.K_MINUS)
    assert sc.runner.env.config.world_speed_scale == 3.0


def test_switching_controller_waits_for_start(app):
    sc = app.scene
    click(app, "start")
    click(app, "controller", "random")
    assert sc.state == "ready" and sc.runner is None
    click(app, "controller", "human")
    assert app.settings.decision_hz == 60.0


def test_jev_without_token_shows_error(app):
    click(app, "controller", "jev")
    click(app, "start")
    assert app.scene.state == "ready" and "OPENROUTER_API_KEY" in app.scene.error


def test_stop_restart_next_seed(app):
    sc = app.scene
    press(app, pygame.K_SPACE)
    assert sc.state == "running"
    click(app, "stop")
    assert sc.state == "ready"
    click(app, "restart")
    assert sc.state == "running" and sc.runner.env.seed == 0
    click(app, "next")
    assert sc.runner.env.seed == 1


def test_finished_episode_replays_in_the_centre_column(app):
    sc = app.scene
    click(app, "controller", "greedy")
    click(app, "duration", 20.0)
    click(app, "start")
    finish_episode(sc)
    assert sc.state == "finished" and sc.runs  # recent runs refreshed
    click(app, "replay_last")
    assert sc.mode == "replay"
    frame(app)
    assert app.scene.layout.sidebar.w > 0  # still one page
    click(app, "play")
    assert sc.playing is False
    click(app, "dec", +1)
    assert sc.rtick > 0
    click(app, "exit_replay")
    assert sc.mode == "live" and sc.state == "ready"


def test_recent_run_click_opens_replay(app):
    sc = app.scene
    click(app, "controller", "greedy")
    click(app, "duration", 20.0)
    click(app, "start")
    finish_episode(sc)
    click(app, "open_run")
    assert sc.mode == "replay"
    click(app, "speed", 0.5)  # changing a parameter returns to live
    assert sc.mode == "live"
