# Decision Arena v0

A real-time, closed-loop benchmark for decision models. It asks one question:
**as the environment gets more complex, faster, and decision latency becomes
real, at what point does a controller start to fail?**

The benchmark doesn't depend on any particular model. Human, heuristic, search,
LLM and (later) TypeSafe Jev controllers all use the same `Observation → Action`
interface.

```
Environment ──Observation──▶ Controller ──Action──▶ Environment   (world never pauses)
```

The rule is **provide facts, not answers**. The environment gives positions,
velocities and radii. It never gives risk scores, predictions, safe directions
or recommended actions. It never saves the player, and it never waits for a
slow controller.

## Quick start

```bash
pip install -r requirements.txt

python main.py --controller human          # WASD / arrows, F1/Tab debug, R restart, Esc quit
python main.py --controller random         # dies fast
python main.py --controller greedy         # chases targets, ignores obstacles
python main.py --controller simple_avoid   # much better, not invincible

python -m arena.benchmark --controller simple_avoid --episodes 50
python -m arena.benchmark --controller simple_avoid --episodes 100 --obstacles 20 --world-speed 4
python -m arena.benchmark --controller simple_avoid --episodes 30 --sweep latency_ms=0,50,100,150,200
python -m arena.benchmark --controller simple_avoid --episodes 20 --adaptive world_speed_scale

python -m arena.replay runs/<run_dir>      # re-simulate a logged episode and verify it matches
python -m pytest
```

Useful flags (both `main.py` and the benchmark): `--preset easy|medium|hard`,
`--obstacles`, `--world-speed`, `--decision-hz`, `--deadline-ms`, `--spawn-rate`,
`--speed-min/max`, `--target-timeout`, `--max-duration`, `--seed`,
`--latency-ms N` (add N ms to every decision), `--real-latency` (withhold the
decision for real wall time instead of charging it in ticks).

## Layout

```
main.py                   interactive pygame mode (world paced to the wall clock)
arena/
  action.py               9 discrete actions (STAY + 8 compass directions)
  difficulty.py           DifficultyConfig: structured difficulty knobs + presets
  entities.py             internal mutable Player / Obstacle / Target
  physics.py              pure functions: thrust→accel→velocity→position, bounce, collision
  environment.py          world state, spawning, scoring, observation building
  observation.py          PUBLIC interface: Observation, ArenaInfo, FORBIDDEN_KEYS
  runner.py               closed loop: decision slots, async latency accounting
  recorder.py             JSONL run logs      replay.py   deterministic re-simulation
  renderer.py             pygame drawing (visual only)
  benchmark.py            headless CLI (`python -m arena.benchmark`)
controllers/
  base.py                 Controller protocol, SyncController, ThreadedController, LatencyWrapper
  human.py random.py greedy.py simple_avoid.py sleep.py
benchmark/
  runner.py               batch episodes (seed i = base_seed + i)
  metrics.py              aggregation, Wilson CI, threshold (D90/D50/D10) helper
  adaptive.py             staircase search for the failure frontier
tests/                    physics, collision, observation, determinism, isolation, async, headless
```

## Design

### What a controller sees
* `reset(info: ArenaInfo)`: the fixed public rules. Arena size, player radius,
  max speed, acceleration, drag, target radius, target timeout, max duration,
  physics Hz, decision Hz, world speed scale, deadline, action list. **No seed.**
* `request(obs: Observation)`: `timestamp` (world s), `tick`, `player{x,y,vx,vy,radius}`,
  `target{x,y,radius}`, `obstacles[{id,x,y,vx,vy,radius}]` in id order,
  `arena{width,height}`, `score`. It's built from fresh primitive values on
  every call, so it's a detached snapshot.

### What stays hidden
The RNG and seed, future obstacle spawns, the next target location, spawn
schedule and legality checks, internal entity objects, `last_target_time`
(the controller can work it out from timestamps and score), and the episode
outcome until it happens.

### Actions and dynamics
An action is a thrust direction. Each substep does
`v += dir·accel·dt; v *= (1 − drag·dt); clamp |v| ≤ max_speed; x += v·dt`.
Walls stop the player but don't hurt it. Obstacles bounce off walls and pass
through each other. Touching an obstacle ends the episode immediately.

### Timing
* Physics runs at **60 ticks per controller second**. Each tick advances the
  world by `world_speed_scale/60` world seconds, with ⌈scale⌉ substeps so
  integration quality doesn't change.
* `world_speed_scale` **dilates world time**. It never changes real latency.
  At 8×, a 150 ms controller still takes 150 real ms, but the world moves 8×
  further in that time. Task geometry stays the same; only time pressure grows.
* Decision slots come every `60/decision_hz` ticks, with at most one request
  in flight. A slot that comes while the controller is busy counts as
  *missed* and is served when the controller frees up. So `decision_hz` is a
  maximum rate.
* A decision with latency *L* takes effect at tick `request_tick + max(1, ⌈L/tick⌉)`.
  Until then the previous action stays in force. **L is measured by the runner's
  clock** (observation handed over → decision received). Controllers can't
  report their own speed. `extra_latency_s` can only add latency.
* `decision_deadline_ms`: late decisions are dropped and the previous action
  continues. There's no safe fallback.
* Headless pacing: the sim runs as fast as possible, *except* while a request
  is in flight. Then it never runs ahead of the wall clock since the request.
  Fast controllers run faster than real time, slow or remote ones are charged
  exactly what they cost, and results match real-time mode. (Checked:
  `--latency-ms 150` simulated and `--real-latency` give the same outcomes.)

### Episodes and metrics
An episode ends on collision (fail), target timeout (fail, which blocks the
"hide in a corner" exploit), `max_duration` (success), or the optional
`target_goal` (success). Raw per-episode fields: success, reason,
survival_time, targets_collected, distance_travelled, average_speed,
collision_time, collision_obstacle_id, decision/request/missed/late counts,
mean/p50/p95 latency, world_speed_scale, decision_hz, obstacle_count, and
per-tick `action_changes` for exact replay. There's no composite score.

## Baseline results (medium preset, 12 obstacles, 10 Hz, 1×)

| controller | success (50 ep) | avg targets | avg survival |
|---|---|---|---|
| random | 0.00 | 0.1 | 7.4 s |
| greedy | 0.00 | 2.2 | 6.3 s |
| simple_avoid | 0.96 | 14.4 | 59.1 s |

SimpleAvoid failure frontiers (interpolated from sweeps):
world speed D50 ≈ 3.0× · added latency D50 ≈ 108 ms · obstacle count D50 ≈ 33.

## Benchmark integrity audit (v0)
* The environment computes no best action, safe direction, risk, recommended
  speed or future path. Its only forward-looking code is the **spawn legality
  check** (`_violates_spawn_safety`). It's private, runs only when spawning,
  and makes sure new obstacles don't start in or head straight into the
  player's safety bubble. Controllers never see it.
* Controllers import only `arena.action` and `arena.observation`. This is
  enforced by an AST test. Tests also check that observations can't reach
  env/entity/RNG objects, and that controllers get only `ArenaInfo` and
  `Observation`.
* No action is changed after the controller decides. The runner sets the
  action only from controller decisions. There's no teleport, recovery or
  collision prevention.
* Latency can't be self-reported (this was found and fixed during the audit).
* The HUD shows a target timer to human players. A machine controller can
  work out the same value from `timestamp` and `score`.

## Known limitations
* Python can't truly sandbox an in-process controller. Isolation is by
  interface plus static checks. Out-of-process controllers (APIs) are
  isolated for real.
* In real-time UI mode, a threaded controller's answer is noticed at the next
  frame (up to ~16 ms of extra measured latency). Headless is the measurement
  mode.
* No oracle yet, so some high-difficulty episodes may be unwinnable.
  Generation only rules out obviously unfair starts.
* Obstacles are straight-line and bounce-only, with no obstacle-obstacle
  collisions. The action space is discrete.
