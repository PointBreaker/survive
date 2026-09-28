from arena.action import Action
from arena.difficulty import DifficultyConfig
from arena.entities import Obstacle
from arena.environment import Environment


def empty_env(**overrides) -> Environment:
    cfg = DifficultyConfig(obstacle_count=0, target_timeout=1e9, max_duration=1e9).with_overrides(**overrides)
    return Environment(cfg, seed=0)


def place_obstacle(env: Environment, x, y, vx=0.0, vy=0.0, r=20.0, oid=None) -> Obstacle:
    ob = Obstacle(oid if oid is not None else len(env.obstacles), x, y, vx, vy, r)
    env.obstacles.append(ob)
    return ob


def park_target_far(env: Environment) -> None:
    env.target.x, env.target.y = 30.0, 30.0


def run_actions(env: Environment, actions):
    for a in actions:
        if env.done:
            break
        env.step(Action(a))
