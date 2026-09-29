"""Observation ablation views: RAW is untouched, derived modes are exact and hint-free."""
import ast
import copy
import json
import math
import random
from pathlib import Path

import pytest

from arena.action import Action
from arena.difficulty import DifficultyConfig, PRESETS
from arena.environment import Environment
from arena.observation import FORBIDDEN_KEYS, Observation
from arena.observation_views import (MODES, PHYSICS_FIELDS, POLICY_HINT_WORDS, PREDICTION_HORIZON_S,
                                     RELATIVE_OBSTACLE_FIELDS, ObservationView, view)
from arena.runner import EpisodeRunner
from controllers.base import SyncController
from controllers.jev import INSTRUCTIONS, JevController
from controllers.simple_avoid import SimpleAvoidController

ROOT = Path(__file__).resolve().parents[1]


def env_at(seed=4, ticks=40, **kw):
    env = Environment(DifficultyConfig(obstacle_count=20, **kw), seed)
    for i in range(ticks):
        env.step(Action.NE if i % 2 else Action.S)
    return env


def keys_of(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from keys_of(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from keys_of(v)


def test_raw_observation_unchanged():
    obs = env_at().observe()
    assert view(obs, "raw") is obs
    d = view(obs, "raw").to_dict()
    assert set(d) == {"timestamp", "tick", "player", "target", "obstacles", "arena", "score", "control"}
    assert all(set(o) == {"id", "x", "y", "vx", "vy", "radius"} for o in d["obstacles"])


def test_raw_mode_jev_request_is_byte_identical_to_before():
    j = JevController(api_key="x", endpoint="http://127.0.0.1:9")
    obs = env_at().observe()
    before = j.encode_decide(1, obs)
    ObservationView(j, "raw")
    assert j.encode_decide(1, obs) == before
    assert before["questions"]["action"]["instructions"] == INSTRUCTIONS
    assert before["state"]["observation"] == obs.to_dict()


def test_relative_features_exact():
    obs = env_at().observe()
    v = view(obs, "relative").to_dict()
    p = obs.player
    assert v["observation_mode"] == "relative" and "prediction_horizon_s" not in v
    assert v["target"]["dx"] == obs.target["x"] - p["x"] and v["target"]["dy"] == obs.target["y"] - p["y"]
    assert v["target"]["distance"] == math.hypot(v["target"]["dx"], v["target"]["dy"])
    assert v["player"]["wall_distance"] == {"left": p["x"], "right": obs.arena["width"] - p["x"],
                                           "top": p["y"], "bottom": obs.arena["height"] - p["y"]}
    for raw, o in zip(obs.obstacles, v["obstacles"]):
        assert {k: o[k] for k in raw} == raw  # raw fields untouched
        assert set(o) == set(raw) | set(RELATIVE_OBSTACLE_FIELDS)
        assert o["dx"] == raw["x"] - p["x"] and o["dy"] == raw["y"] - p["y"]
        assert o["dvx"] == raw["vx"] - p["vx"] and o["dvy"] == raw["vy"] - p["vy"]
        assert o["distance"] == math.hypot(o["dx"], o["dy"])
        assert o["gap"] == o["distance"] - raw["radius"] - p["radius"]
        assert o["relative_speed"] == math.hypot(o["dvx"], o["dvy"])
    assert [o["id"] for o in v["obstacles"]] == [o["id"] for o in obs.obstacles]  # id order, not by distance


def hand_obs(player, obstacles, target=None):
    return Observation(timestamp=0.0, tick=0, player=player, target=target or {"x": 500.0, "y": 500.0, "radius": 16.0},
                       obstacles=tuple(obstacles), arena={"width": 1200.0, "height": 800.0}, score=0)


def test_physics_projection_exact():
    p = {"x": 100.0, "y": 100.0, "vx": 0.0, "vy": 0.0, "radius": 10.0}
    obs = hand_obs(p, [
        # passes 30 units above the player, moving left at 50/s: closest at t = 200/50 = 4 -> clamped to 3
        {"id": 1, "x": 300.0, "y": 70.0, "vx": -50.0, "vy": 0.0, "radius": 5.0},
        # head-on at 100/s from 150 away: t* = 1.5, distance 0
        {"id": 2, "x": 250.0, "y": 100.0, "vx": -100.0, "vy": 0.0, "radius": 5.0},
        # moving away: t* = 0
        {"id": 3, "x": 150.0, "y": 100.0, "vx": 10.0, "vy": 0.0, "radius": 5.0},
        # relative velocity 0: t* = 0
        {"id": 4, "x": 100.0, "y": 300.0, "vx": 0.0, "vy": 0.0, "radius": 5.0},
    ])
    v = view(obs, "physics").to_dict()
    assert v["prediction_horizon_s"] == PREDICTION_HORIZON_S == 3.0
    o1, o2, o3, o4 = v["obstacles"]
    assert o1["time_to_closest_approach"] == 3.0 and o1["closest_approach_distance"] == pytest.approx(math.hypot(50, 30))
    assert o2["time_to_closest_approach"] == pytest.approx(1.5) and o2["closest_approach_distance"] == pytest.approx(0.0)
    assert o2["closest_approach_gap"] == pytest.approx(-15.0)
    assert o3["time_to_closest_approach"] == 0.0 and o3["closest_approach_distance"] == pytest.approx(50.0)
    assert o4["time_to_closest_approach"] == 0.0 and o4["closest_approach_distance"] == pytest.approx(200.0)
    # static target: relative velocity = -player velocity
    moving = view(hand_obs({**p, "vx": 100.0}, [], {"x": 400.0, "y": 140.0, "radius": 16.0}), "physics").to_dict()
    assert moving["target"]["time_to_closest_approach"] == pytest.approx(3.0)
    assert moving["target"]["closest_approach_distance"] == pytest.approx(40.0)
    for o in v["obstacles"]:
        assert set(PHYSICS_FIELDS) <= set(o)


def test_no_future_state_leak():
    env = env_at(spawn_rate=2.0)
    other = copy.deepcopy(env)
    other._rng = random.Random(987654)  # different future spawns, identical present
    for mode in MODES:
        assert view(env.observe(), mode).to_dict() == view(other.observe(), mode).to_dict()
        # and it is a function of the serialised snapshot alone
        assert view(Observation.from_dict(env.observe().to_dict()), mode).to_dict() == view(env.observe(), mode).to_dict()
    tree = ast.parse((ROOT / "arena" / "observation_views.py").read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not mods & {"arena.environment", "arena.physics", "arena.entities", "arena.replay", "random", "copy"}


@pytest.mark.parametrize("mode", ["relative", "physics"])
def test_no_policy_hint_words(mode):
    env = env_at(ticks=90)
    d = view(env.observe(), mode).to_dict()
    keys = set(keys_of(d))
    assert not keys & FORBIDDEN_KEYS
    for k in keys:
        assert not any(w in k.lower() for w in POLICY_HINT_WORDS), k
    from arena.observation_views import FIELD_DEFINITIONS
    text = FIELD_DEFINITIONS[mode].lower()
    for w in ("safe", "danger", "threat", "risk", "recommend", "avoid", "best", "should", "dodge"):
        assert w not in text, w


def test_environment_behaviour_identical_across_modes():
    """SimpleAvoid ignores the extra fields, so every mode must give the identical episode."""
    cfg = PRESETS["medium"].with_overrides(max_duration=15)
    runs = {}
    for mode in MODES:
        ctrl = ObservationView(SimpleAvoidController(), mode)
        r = EpisodeRunner(Environment(cfg, 5), ctrl).run()
        runs[mode] = (r["action_changes"], r["reason"], r["targets_collected"])
    assert runs["raw"] == runs["relative"] == runs["physics"]


def test_controller_receives_view_through_factory(monkeypatch):
    from types import SimpleNamespace

    from arena.cli import controller_factory

    seen = []

    class Spy(SyncController):
        name = "spy"

        def decide(self, obs):
            seen.append(obs.to_dict())
            return Action.STAY

    import controllers
    monkeypatch.setattr(controllers, "make_controller", lambda name, **kw: Spy())
    args = SimpleNamespace(controller_seed=1, sleep_ms=0, endpoint=None, latency_ms=0.0, real_latency=False,
                           observation_mode="physics")
    f = controller_factory("spy_test", args)
    EpisodeRunner(Environment(PRESETS["easy"].with_overrides(max_duration=1), 0), f()).run()
    assert seen and all(d["observation_mode"] == "physics" for d in seen)
    assert "closest_approach_gap" in json.dumps(seen[-1])
