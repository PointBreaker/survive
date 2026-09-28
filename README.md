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
python -m arena.viewer runs/<bench_dir> --at collision   # step through saved episodes (needs --save-events)

# Jev via OpenRouter: cp .env.example .env, fill OPENROUTER_API_KEY, then
python -m arena.jev_check
python -m arena.benchmark --controller jev --episodes 10

# out-of-process path validated against a local fake service
python -m remote.fake_server --port 8765 --policy simple_avoid --latency-ms 150 &
python -m arena.benchmark --controller remote --endpoint http://127.0.0.1:8765 --episodes 10
python -m benchmark.remote_validation --latencies 50,65,100,150,200 --episodes 10
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
  jev_check.py            Jev/OpenRouter connectivity check
  dotenv.py               tiny .env loader (no dependency)
controllers/
  base.py                 Controller protocol, SyncController, ThreadedController, LatencyWrapper
  human.py random.py greedy.py simple_avoid.py sleep.py
  remote.py               generic HTTP client for remote.protocol (worker thread, keep-alive)
  jev.py                  JevController: OpenRouter decisions API (typesafe/jev-1.13), token from .env
remote/
  protocol.py             wire protocol decision-arena/v0 (JSON over HTTP)
  fake_server.py          local stand-in service: baseline policy + real latency/jitter/failures
benchmark/
  runner.py               batch episodes (seed i = base_seed + i)
  metrics.py              aggregation, Wilson CI, threshold (D90/D50/D10) helper
  adaptive.py             staircase search for the failure frontier
  remote_validation.py    remote (real latency) vs simulated latency, paired per seed
tests/                    physics, collision, observation, determinism, isolation, async, headless
```

## Watching decisions (GUI)

`main.py` has a live **inspector panel**, and `arena.viewer` replays saved
episodes with the same panel:

* **In flight**: the pending request and how long it has been waiting, with
  the decision period marked.
* **Last response**: the action, latency, how many ticks after its snapshot
  it took effect, confidence, and a probability bar per action (Jev).
* **Stats**: applied / failed / missed counts, mean and p95 latency, the
  effective decision rate, and a latency sparkline.
* **History**: the most recent decisions.
* **J** cycles to the **raw request body** (exactly what went over the wire)
  and the **raw response**.
* **In the arena**: a faded **ghost** of the snapshot the controller is
  currently deciding on (the gap to reality is the latency cost), the thrust
  arrow, and a **probability compass** around the player.

Viewer keys: Space play/pause, ←/→ step a tick (Shift = 10), `,`/`.` jump
between decisions, `[`/`]` speed from 0.05× to 4×, C jumps to 2 s before the
collision, PageUp/PageDown switches episode, and you can click or drag the
timeline. In `main.py`, press V after an episode ends to replay it. The world
is re-simulated exactly from the log and no controller or API is called, so
replays are free.

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

### Out-of-process controllers (Jev)

**Running Jev (TypeSafe Jev via OpenRouter or the first-party API):**

```bash
cp .env.example .env            # put ONE token in: OPENROUTER_API_KEY=sk-or-...  or  TYPESAFE_API_KEY=...
python -m arena.jev_check       # 3 real calls: latency, parsed action, raw reply
python main.py --controller jev
python -m arena.benchmark --controller jev --episodes 10
```

`.env` is gitignored. `OPENROUTER_API_KEY` uses
`https://openrouter.ai/api/alpha/decisions` with `typesafe/jev-1.13`.
`TYPESAFE_API_KEY` uses `https://api.typesafe.ai/v1/systemone` with `jev-latest`.
Optional overrides: `JEV_MODEL`, `JEV_ENDPOINT`, `JEV_TIMEOUT_S`. Published
figures: 70–500 ms per request (p50 around 200 ms on OpenRouter), 1,200 req/min.

Each decision is one stateless POST in the decisions-API shape:

```json
{"model": "typesafe/jev-1.13",
 "state": {"rules": <ArenaInfo.to_dict()>, "observation": <Observation.to_dict()>},
 "questions": {"action": {"type": "choice", "instructions": "...", "criteria": {"STAY": "...", "N": "...", ...}}}}
```

* `state` holds exactly what every controller gets: the public rules and
  the raw observation, unmodified. The instructions and criteria
  (`controllers/jev.py`: `INSTRUCTIONS`, `ACTION_CRITERIA`) only explain the
  rules and what each action does. A test makes sure they contain no strategy
  words (safe, danger, avoid, nearest, recommend, ...).
* The reply's documented form is
  `{"answers": {"action": {"type": "choice", "choice": "NE", "probabilities": {..}, "confidence": 0.78}}}`.
  The runner applies `choice`, and `confidence` and `probabilities` are logged
  per decision (for "where was Jev unsure" analysis). They never select an
  action. Other layouts are accepted as a fallback, and `jev_check` flags it
  when one is used. An unparseable reply is a failed decision with the raw
  snippet in the event log.
* The HTTP call runs on a worker thread, and the runner times the whole round
  trip. Transport and network are part of the latency.
* Any failure (HTTP error, timeout, bad JSON, unknown action, service down) is
  a **failed decision**. It's logged, and the previous action continues. No
  retry, no substitute action.

**Local validation path:** `RemoteController` speaks a simple
`/reset` + `/decide` protocol (`remote/protocol.py`) to
`remote/fake_server.py`. The fake server runs a baseline policy in its own
process with real injected latency. It uses the same HTTP client code as
`JevController` and is used by `benchmark.remote_validation`.

**Validation** (`benchmark.remote_validation`, SimpleAvoid served from a
separate process vs. in-process simulated latency, 10 paired seeds, 20 s
episodes):

| latency | remote success | sim success | remote / sim delay (ticks) | identical episodes |
|---|---|---|---|---|
| 50 ms | 1.00 | 1.00 | 4.00 / 4.00 | 10/10 |
| 65 ms | 1.00 | 1.00 | 4.40 / 4.00 | 6/10 |
| 100 ms | 0.90 | 0.90 | 7.00 / 7.00 | 10/10 |
| 150 ms | 0.50 | 0.50 | 10.00 / 10.00 | 10/10 |
| 200 ms | 0.10 | 0.10 | 13.05 / 13.00 | 8/10 |

Transport overhead is about 1.9 ms on loopback. When the effective delay in
ticks matches, the runs are bit-identical. The mismatches are expected:
65 ms is just under a 4-tick boundary (66.7 ms), so real overhead pushes
some decisions to 5 ticks. At 200 ms, occasional OS scheduling spikes
(>16 ms) cost an extra tick. Real overhead is charged, as it should be.

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
* The fake server holds one session per `/reset`. It's a test double, not a
  production server.
* Obstacles are straight-line and bounce-only, with no obstacle-obstacle
  collisions. The action space is discrete.
