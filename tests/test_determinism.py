import random

from arena.action import ALL_ACTIONS
from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.recorder import JsonlRecorder
from arena.replay import replay, replay_run_dir
from arena.runner import EpisodeRunner
from controllers.random import RandomController
from controllers.simple_avoid import SimpleAvoidController


def trajectory(seed, actions, cfg):
    env = Environment(cfg, seed)
    states = []
    for a in actions:
        if env.done:
            break
        env.step(a)
        states.append(
            (env.player.x, env.player.y, env.score, tuple((o.x, o.y, o.vx, o.vy) for o in env.obstacles))
        )
    return states, env.outcome


def test_same_seed_same_actions_same_world():
    rng = random.Random(9)
    actions = [rng.choice(ALL_ACTIONS) for _ in range(900)]
    cfg = DifficultyConfig(obstacle_count=25, spawn_rate=0.5, world_speed_scale=2.0)
    assert trajectory(11, actions, cfg) == trajectory(11, actions, cfg)


def test_different_seed_different_world():
    actions = [ALL_ACTIONS[0]] * 10
    cfg = DifficultyConfig()
    assert trajectory(1, actions, cfg)[0] != trajectory(2, actions, cfg)[0]


def test_runner_episode_replays_exactly(tmp_path):
    cfg = DifficultyConfig(obstacle_count=15, max_duration=20)
    rec = JsonlRecorder(tmp_path / "run")
    env = Environment(cfg, 5)
    result = EpisodeRunner(env, SimpleAvoidController(), recorder=rec).run()
    rec.close()

    replayed = replay(cfg, 5, result["action_changes"], max_ticks=result["ticks"])
    assert replayed.tick == env.tick
    assert (replayed.player.x, replayed.player.y) == (env.player.x, env.player.y)
    assert replayed.score == result["targets_collected"]
    assert replayed.outcome.reason == result["reason"]

    from_disk = replay_run_dir(tmp_path / "run")
    assert (from_disk.player.x, from_disk.player.y, from_disk.score) == (env.player.x, env.player.y, env.score)
    assert (tmp_path / "run" / "events.jsonl").stat().st_size > 0


def test_random_controller_rng_independent_of_env_seed():
    a, b = RandomController(seed=1), RandomController(seed=1)
    for ctrl in (a, b):
        ctrl.reset(Environment(DifficultyConfig(), 0).public_info())
    env1, env2 = Environment(DifficultyConfig(), 100), Environment(DifficultyConfig(), 200)
    seq1 = [a.decide(env1.observe()) for _ in range(20)]
    seq2 = [b.decide(env2.observe()) for _ in range(20)]
    assert seq1 == seq2
