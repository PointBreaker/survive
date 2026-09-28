from arena import physics
from arena.action import Action
from tests.helpers import empty_env, park_target_far, place_obstacle


def test_circles_overlap_boundary():
    assert physics.circles_overlap(0, 0, 10, 19.9, 0, 10)
    assert not physics.circles_overlap(0, 0, 10, 20.0, 0, 10)
    assert not physics.circles_overlap(0, 0, 10, 30, 0, 10)


def test_collision_ends_episode_with_obstacle_id():
    env = empty_env()
    park_target_far(env)
    p = env.player
    place_obstacle(env, p.x + 100, p.y, vx=-200, r=20, oid=7)
    for _ in range(120):
        env.step(Action.STAY)
        if env.done:
            break
    assert env.done and not env.outcome.success
    assert env.outcome.reason == "collision"
    assert env.outcome.collision_obstacle_id == 7


def test_no_safety_override_when_driving_into_obstacle():
    env = empty_env()
    park_target_far(env)
    p = env.player
    place_obstacle(env, p.x + 150, p.y, r=25)
    for _ in range(240):
        env.step(Action.E)
        if env.done:
            break
    assert env.outcome.reason == "collision"


def test_target_collection_scores_and_respawns():
    env = empty_env()
    p = env.player
    env.target.x, env.target.y = p.x + 60, p.y
    for _ in range(120):
        env.step(Action.E)
        if env.score:
            break
    assert env.score == 1
    assert (env.target.x, env.target.y) != (p.x + 60, p.y)


def test_target_timeout_fails_episode():
    env = empty_env(target_timeout=2.0)
    park_target_far(env)
    for _ in range(200):
        env.step(Action.STAY)
    assert env.done and env.outcome.reason == "target_timeout" and not env.outcome.success
