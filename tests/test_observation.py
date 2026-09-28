import math
import numbers

from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.observation import FORBIDDEN_KEYS, Observation
from arena.action import Action


def walk(value, path="obs"):
    if isinstance(value, dict):
        for k, v in value.items():
            yield path, k, v
            yield from walk(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")


def make_env(seed=3, **kw):
    return Environment(DifficultyConfig(obstacle_count=30, **kw), seed)


def test_observation_has_no_forbidden_keys_and_only_primitives():
    env = make_env()
    for _ in range(30):
        env.step(Action.NE)
    d = env.observe().to_dict()
    keys = set(d)
    for path, k, v in walk(d):
        keys.add(k)
        if not isinstance(v, (dict, list, tuple)):
            assert isinstance(v, (numbers.Number, str)) or v is None, (path, k, type(v))
    assert not keys & FORBIDDEN_KEYS
    assert set(d) == {"timestamp", "tick", "player", "target", "obstacles", "arena", "score"}
    assert set(d["player"]) == {"x", "y", "vx", "vy", "radius"}
    assert set(d["target"]) == {"x", "y", "radius"}
    for o in d["obstacles"]:
        assert set(o) == {"id", "x", "y", "vx", "vy", "radius"}


def test_observation_is_current_state_not_future():
    env = make_env()
    for _ in range(10):
        env.step(Action.S)
    obs = env.observe()
    assert obs.timestamp == env.world_time and obs.tick == env.tick
    assert (obs.player["x"], obs.player["y"]) == (env.player.x, env.player.y)
    for o, ob in zip(obs.obstacles, env.obstacles):
        assert (o["x"], o["y"], o["vx"], o["vy"]) == (ob.x, ob.y, ob.vx, ob.vy)


def test_obstacles_listed_in_id_order_not_by_danger():
    env = make_env()
    ids = [o["id"] for o in env.observe().obstacles]
    assert ids == sorted(ids)


def test_observation_is_a_detached_snapshot():
    env = make_env()
    obs = env.observe()
    x0 = env.obstacles[0].x
    obs.obstacles[0]["x"] = -999.0
    obs.player["x"] = -999.0
    assert env.obstacles[0].x == x0 and env.player.x != -999.0
    before = env.observe().to_dict()
    snap = env.observe()
    for _ in range(10):
        env.step(Action.STAY)
    assert snap.to_dict() == before  # old snapshot did not follow the world


def test_scales_to_many_obstacles():
    env = Environment(DifficultyConfig(obstacle_count=160), 1)
    assert len(env.observe().obstacles) == len(env.obstacles) >= 150


def test_public_info_contains_no_seed():
    env = make_env(seed=424242)
    info = env.public_info().to_dict()
    assert "seed" not in info and 424242 not in info.values()
    assert not set(info) & FORBIDDEN_KEYS


def test_initial_state_keeps_safe_space():
    for seed in range(50):
        env = Environment(DifficultyConfig(obstacle_count=40), seed)
        p = env.player
        for o in env.obstacles:
            assert math.hypot(o.x - p.x, o.y - p.y) >= env.config.spawn_safe_radius + o.radius - 1e-6
        t = env.target
        for o in env.obstacles:
            assert math.hypot(o.x - t.x, o.y - t.y) > o.radius + t.radius
