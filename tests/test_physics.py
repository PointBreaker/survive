import math

from arena import physics
from arena.action import Action
from arena.entities import Obstacle
from tests.helpers import empty_env, park_target_far


def speed(env):
    return math.hypot(env.player.vx, env.player.vy)


def test_action_goes_through_acceleration_not_teleport():
    env = empty_env()
    park_target_far(env)
    x0 = env.player.x
    env.step(Action.E)
    # One tick of thrust produces a small velocity, not max speed.
    assert 0 < env.player.vx < env.config.player_max_speed * 0.2
    assert env.player.x - x0 < 1.0
    for _ in range(120):
        env.step(Action.E)
    assert abs(speed(env) - env.config.player_max_speed) < 1e-6


def test_inertia_direction_does_not_flip_instantly():
    env = empty_env()
    park_target_far(env)
    for _ in range(60):
        env.step(Action.E)
    env.step(Action.W)
    assert env.player.vx > 0  # still moving east after one tick of reverse thrust


def test_drag_slows_player_when_staying():
    env = empty_env()
    park_target_far(env)
    for _ in range(60):
        env.step(Action.E)
    v0 = speed(env)
    for _ in range(30):
        env.step(Action.STAY)
    assert 0 < speed(env) < v0


def test_diagonal_is_not_faster():
    env = empty_env()
    park_target_far(env)
    for _ in range(200):
        env.step(Action.SE)
    assert speed(env) <= env.config.player_max_speed + 1e-9


def test_walls_clamp_player():
    env = empty_env()
    park_target_far(env)
    env.target.x, env.target.y = 30, 700  # keep target out of the path
    for _ in range(600):
        env.step(Action.E)
    assert env.player.x == env.config.arena_width - env.player.radius
    assert env.player.vx == 0.0
    assert not env.done


def test_obstacle_bounces_off_wall():
    ob = Obstacle(0, 1190.0, 400.0, 300.0, 0.0, 10.0)
    physics.move_obstacle(ob, 0.1, 1200, 800)
    assert ob.vx < 0 and ob.x <= 1190.0


def test_world_speed_scales_world_time_per_tick():
    for scale in (1.0, 2.0, 4.0, 8.0):
        env = empty_env(world_speed_scale=scale)
        park_target_far(env)
        for _ in range(60):
            env.step(Action.STAY)
        assert math.isclose(env.world_time, scale, rel_tol=1e-9)


def test_action_quantization():
    assert Action.from_vector(1, 0) is Action.E
    assert Action.from_vector(0, -1) is Action.N
    assert Action.from_vector(-1, 1) is Action.SW
    assert Action.from_vector(0, 0) is Action.STAY
