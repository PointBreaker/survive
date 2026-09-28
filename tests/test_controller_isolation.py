import ast
import gc
from pathlib import Path

from arena.action import Action
from arena.difficulty import DifficultyConfig
from arena.entities import Obstacle, Player, Target
from arena.environment import Environment
from arena.observation import ArenaInfo, Observation
from arena.runner import EpisodeRunner
from controllers.base import SyncController

CONTROLLERS_DIR = Path(__file__).resolve().parents[1] / "controllers"
ALLOWED_ARENA_MODULES = {"arena.action", "arena.observation"}
REMOTE_DIR = Path(__file__).resolve().parents[1] / "remote"
FORBIDDEN_MODULES = {"arena.environment", "arena.physics", "arena.entities", "arena.runner",
                     "arena.replay", "arena.recorder", "benchmark", "gc", "inspect", "ctypes"}


def imported_modules(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_controllers_only_import_public_interface():
    for path in CONTROLLERS_DIR.glob("*.py"):
        for mod in imported_modules(path):
            if mod.startswith("arena"):
                assert mod in ALLOWED_ARENA_MODULES, f"{path.name} imports {mod}"
            if mod.startswith("remote"):
                assert mod == "remote.protocol", f"{path.name} imports {mod}"
            assert not any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_MODULES), \
                f"{path.name} imports {mod}"


def test_wire_protocol_only_uses_public_interface():
    for path in REMOTE_DIR.glob("*.py"):
        for mod in imported_modules(path):
            if mod.startswith("arena"):
                assert mod in ALLOWED_ARENA_MODULES, f"remote/{path.name} imports {mod}"


def test_controller_source_has_no_backdoor_names():
    for path in CONTROLLERS_DIR.glob("*.py"):
        src = path.read_text()
        for bad in ("_rng", "Environment", "env.", "get_referrers", "__dict__"):
            if bad == "_rng" and path.name == "random.py":
                continue  # the random controller's own RNG attribute
            assert bad not in src, f"{path.name} contains {bad!r}"


class Spy(SyncController):
    name = "spy"

    def __init__(self):
        super().__init__()
        self.reset_args = []
        self.obs = []

    def reset(self, info):
        super().reset(info)
        self.reset_args.append(info)

    def decide(self, observation):
        self.obs.append(observation)
        return Action.STAY


def test_controller_receives_only_public_types():
    spy = Spy()
    EpisodeRunner(Environment(DifficultyConfig(max_duration=3), 1), spy).run()
    assert all(type(i) is ArenaInfo for i in spy.reset_args)
    assert spy.obs and all(type(o) is Observation for o in spy.obs)


def test_observation_holds_no_reference_to_environment_objects():
    env = Environment(DifficultyConfig(obstacle_count=20), 2)
    obs = env.observe()
    internal = (Environment, Player, Obstacle, Target)
    seen, stack = set(), [obs, env.public_info()]
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        assert not isinstance(x, internal), type(x)
        assert type(x).__name__ != "Random"
        if isinstance(x, (type, type(len))):
            continue
        stack.extend(r for r in gc.get_referents(x) if not isinstance(r, type))


def test_environment_does_not_compute_advice():
    src = (Path(__file__).resolve().parents[1] / "arena" / "environment.py").read_text()
    for word in ("recommend", "best_action", "safe_direction", "escape", "danger", "time_to_collision"):
        assert word not in src
