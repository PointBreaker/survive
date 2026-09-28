"""Launcher / play / replay scenes, driven by synthetic clicks (SDL dummy driver)."""
import pygame
import pytest

from arena.app import App, MenuScene, PlayScene, ReplayScene, Settings
from arena.display import Display


@pytest.fixture
def app(tmp_path, monkeypatch):
    for k in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "JEV_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    a = App(Display((1400, 860)))
    a.settings = Settings(out=str(tmp_path / "runs"))
    yield a
    if a.scene is not None and hasattr(a.scene, "close"):
        a.scene.close()
    a.display.close()


def frame(app):
    app.scene.update()
    app.ui.t = app.renderer.t
    app.ui.begin((-1, -1))
    app.scene.draw()


def click(app, key, value=None):
    """Click the centre of the widget registered as (key, value)."""
    frame(app)
    for rect, k, v, enabled in app.ui.hits:
        if k == key and (value is None or v == value) and enabled:
            ev = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center)
            ev.px = rect.center
            app.scene.handle(ev)
            return
    raise AssertionError(f"no widget {key}={value}")


def key(app, k):
    app.scene.handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0))


def test_menu_click_configures_and_starts_game(app):
    app.scene = MenuScene(app)
    click(app, "controller", "greedy")
    click(app, "speed", 2.0)
    click(app, "obstacles", 20)
    click(app, "inflight", 3)
    click(app, "duration", 20.0)
    click(app, "seed", +1)
    click(app, "start")
    assert isinstance(app.scene, PlayScene)
    cfg = app.scene.runner.env.config
    assert (cfg.world_speed_scale, cfg.obstacle_count, cfg.max_inflight, cfg.max_duration) == (2.0, 20, 3, 20.0)
    assert app.scene.runner.env.seed == 1 and app.scene.runner.controller.name == "greedy"


def test_selecting_human_uses_60hz_input(app):
    app.scene = MenuScene(app)
    click(app, "controller", "human")
    assert app.settings.decision_hz == 60.0 and app.settings.max_inflight == 1


def test_jev_without_token_shows_error_and_stays_in_menu(app):
    app.scene = MenuScene(app)
    click(app, "controller", "jev")
    click(app, "start")
    assert isinstance(app.scene, MenuScene)
    assert "OPENROUTER_API_KEY" in app.scene.error


def test_world_speed_buttons_restart_same_seed_with_new_speed(app):
    app.settings.controller = "simple_avoid"
    factory, err = app.settings.controller_factory()
    app.scene = PlayScene(app, app.settings, factory)
    for _ in range(3):
        frame(app)
    seed = app.scene.seed
    click(app, "speed", +1)
    env = app.scene.runner.env
    assert env.config.world_speed_scale == 1.5 and env.seed == seed and env.tick == 0
    key(app, pygame.K_MINUS)
    assert app.scene.runner.env.config.world_speed_scale == 1.0
    key(app, pygame.K_n)
    assert app.scene.runner.env.seed == seed + 1
    key(app, pygame.K_ESCAPE)
    assert isinstance(app.scene, MenuScene)


def test_finished_game_offers_replay_of_the_logged_episode(app):
    app.settings = Settings(controller="greedy", max_duration=1.0, out=app.settings.out)
    factory, _ = app.settings.controller_factory()
    app.scene = PlayScene(app, app.settings, factory)
    scene = app.scene
    while not scene.finished:
        scene.runner.step_tick()  # drive the episode without waiting for wall time
        scene.update()
    click(app, "replay")
    assert isinstance(app.scene, ReplayScene)
    rp = app.scene
    click(app, "play")
    assert rp.playing is False
    click(app, "dec", +1)
    assert rp.tick > 0
    click(app, "rspeed", 0.25)
    assert rp.speed == 0.25
    click(app, "back")
    assert isinstance(app.scene, PlayScene)


def test_menu_lists_recent_runs_and_opens_replay(app):
    app.settings = Settings(controller="greedy", max_duration=0.5, out=app.settings.out)
    factory, _ = app.settings.controller_factory()
    play = PlayScene(app, app.settings, factory)
    while not play.finished:
        play.runner.step_tick()
        play.update()
    play.close()
    app.scene = MenuScene(app)
    assert app.scene.runs
    click(app, "replay")
    assert isinstance(app.scene, ReplayScene)
