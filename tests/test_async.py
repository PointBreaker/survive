import math
import time

from arena.action import Action
from arena.difficulty import DifficultyConfig
from arena.environment import Environment
from arena.observation import Observation
from arena.runner import TICK, EpisodeRunner
from controllers.base import LatencyWrapper, SyncController
from controllers.sleep import SleepController


def env_with_moving_obstacle(**kw):
    cfg = DifficultyConfig(obstacle_count=5, target_timeout=1e9, max_duration=1e9, **kw)
    return Environment(cfg, 0)


def decision_ticks(runner):
    return [t for t, _ in runner.action_changes[1:]]


def test_slow_threaded_controller_does_not_pause_world():
    env = env_with_moving_obstacle()
    ctrl = SleepController(delay_ms=500, action=Action.E)
    runner = EpisodeRunner(env, ctrl)
    runner.start()
    x0 = [o.x for o in env.obstacles]
    t_wall = time.perf_counter()
    for _ in range(25):  # < 500 ms worth of ticks
        runner.step_tick()
    # World advanced 25 ticks while the controller was still sleeping.
    assert env.tick == 25
    assert runner.current_action is Action.STAY
    assert [o.x for o in env.obstacles] != x0
    for _ in range(40):
        runner.step_tick()
    ctrl.close()
    first = decision_ticks(runner)[0]
    assert first >= math.ceil(0.5 / TICK)  # applied only after >= 500 ms of world time
    assert first <= math.ceil(0.5 / TICK) + 6
    assert runner.stats.missed_slots >= 1
    assert time.perf_counter() - t_wall >= 0.5


def test_blocking_inline_controller_is_charged_its_compute_time():
    class Slow(SyncController):
        name = "slow"

        def decide(self, observation: Observation) -> Action:
            time.sleep(0.2)
            return Action.W

    env = env_with_moving_obstacle()
    runner = EpisodeRunner(env, Slow())
    runner.start()
    x0 = [o.x for o in env.obstacles]
    for _ in range(12):
        runner.step_tick()
    assert runner.current_action is Action.STAY  # 200 ms not yet elapsed in world time
    assert [o.x for o in env.obstacles] != x0
    for _ in range(3):
        runner.step_tick()
    assert runner.current_action is Action.W
    # 200 ms == 12 ticks exactly; sleep always overshoots a little, so ceil -> 13.
    assert decision_ticks(runner)[0] in (13, 14)


def test_realtime_world_keeps_running_while_controller_sleeps():
    env = env_with_moving_obstacle()
    ctrl = SleepController(delay_ms=500)
    runner = EpisodeRunner(env, ctrl, realtime=True)
    runner.start()
    x0 = [o.x for o in env.obstacles]
    end = time.perf_counter() + 0.3
    while time.perf_counter() < end:
        runner.advance_realtime()
        time.sleep(0.005)
    ctrl.close()
    assert env.tick >= 15
    assert runner.current_action is Action.STAY
    assert [o.x for o in env.obstacles] != x0


def test_simulated_latency_is_charged_in_ticks():
    class Fast(SyncController):
        name = "fast"

        def decide(self, observation):
            return Action.N

    env = env_with_moving_obstacle()
    runner = EpisodeRunner(env, LatencyWrapper(Fast(), 150))
    runner.start()
    for _ in range(12):
        runner.step_tick()
    assert decision_ticks(runner)[0] == 10  # ceil(0.150 / (1/60)) = 9 -> +1 from tiny compute time
    assert runner.last_latency_s >= 0.150


def test_deadline_drops_late_decision_and_keeps_previous_action():
    env = env_with_moving_obstacle(decision_deadline_ms=100)
    ctrl = SleepController(delay_ms=200, action=Action.E)
    runner = EpisodeRunner(env, ctrl)
    runner.start()
    for _ in range(40):
        runner.step_tick()
    ctrl.close()
    assert runner.current_action is Action.STAY
    assert runner.stats.late_dropped >= 1
    assert runner.stats.applied == 0


def test_minimum_one_tick_delay_even_for_instant_controller():
    class Instant(SyncController):
        name = "instant"

        def decide(self, observation):
            return Action.S

    env = env_with_moving_obstacle()
    runner = EpisodeRunner(env, Instant())
    runner.start()
    runner.step_tick()
    assert runner.current_action is Action.STAY
    runner.step_tick()
    assert runner.current_action is Action.S


def test_controller_cannot_report_itself_faster():
    from controllers.base import Decision

    class Liar(SyncController):
        name = "liar"

        def decide(self, observation):
            time.sleep(0.1)
            return Action.N

        def poll(self):
            d = super().poll()
            return None if d is None else Decision(d.action, extra_latency_s=-10.0)

    env = env_with_moving_obstacle()
    runner = EpisodeRunner(env, Liar())
    runner.start()
    for _ in range(10):
        runner.step_tick()
    assert runner.last_latency_s >= 0.1
    assert decision_ticks(runner)[0] >= 6
