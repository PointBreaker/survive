"""Start benchmark suites and shadow runs from the dashboard.

A job is exactly ``python -m benchmark.suite ...`` (or ``benchmark.shadow``)
run as a subprocess. That's
the same CLI, the same runner and the same semantics as a terminal run. This
module only validates a request, builds the argument list (never a shell
string), watches the process and reads the suite's own progress fields.

One job at a time: concurrent suites would compete for CPU and distort
wall-clock latency, which the benchmark charges to controllers.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from arena.difficulty import PRESETS
from arena.recorder import new_run_dir

ROOT = Path(__file__).resolve().parents[1]
SPEC_RE = re.compile(r"^([a-z_]+)(?:\+(\d{1,5})ms)?$")
MAX_TOTAL_EPISODES = 20000

# (min, max) per swept parameter; obstacle_count must be an integer.
LEVEL_BOUNDS = {
    "world_speed_scale": (0.05, 64.0),
    "obstacle_count": (0, 400),
    "latency_ms": (0, 10000),
    "obstacle_speed_max": (1, 2000),
    "spawn_rate": (0, 50),
    "decision_delay_ms": (0, 5000),
}
LOCKSTEP_ONLY = {"decision_delay_ms"}
REALTIME_ONLY = {"world_speed_scale", "latency_ms"}


class JobError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _num(spec: dict, key: str, lo: float, hi: float, integer: bool = False, default=None):
    v = spec.get(key, default)
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise JobError(400, f"{key} must be a number")
    if not (lo <= f <= hi) or (integer and f != int(f)):
        raise JobError(400, f"{key} must be {'an integer ' if integer else ''}between {lo:g} and {hi:g}")
    return int(f) if integer else f


def build_argv(spec: dict[str, Any], benchmarkable: list[str], jev_token: bool) -> list[str]:
    """Validate a job request and turn it into ``benchmark.suite`` arguments."""
    ctrls = spec.get("controllers")
    if not isinstance(ctrls, list) or not 1 <= len(ctrls) <= 8:
        raise JobError(400, "controllers: choose 1 to 8")
    clean = []
    for c in ctrls:
        m = SPEC_RE.match(str(c))
        if not m or m.group(1) not in benchmarkable:
            raise JobError(400, f"unknown controller {c!r}")
        if m.group(1) == "jev" and not jev_token:
            raise JobError(400, "Jev needs OPENROUTER_API_KEY (or TYPESAFE_API_KEY) in .env")
        if c not in clean:
            clean.append(str(c))

    timing = spec.get("timing", "realtime")
    if timing not in ("realtime", "lockstep"):
        raise JobError(400, "timing must be realtime or lockstep")
    param = spec.get("param", "world_speed_scale")
    if param not in LEVEL_BOUNDS:
        raise JobError(400, f"param must be one of {sorted(LEVEL_BOUNDS)}")
    if timing == "lockstep" and param in REALTIME_ONLY:
        raise JobError(400, f"{param} has no meaning in lockstep timing")
    if timing == "realtime" and param in LOCKSTEP_ONLY:
        raise JobError(400, f"{param} requires lockstep timing")
    if timing == "lockstep" and any("+" in c for c in clean):
        raise JobError(400, "added latency (+ms) is meaningless in lockstep; sweep decision_delay_ms instead")
    levels = spec.get("levels")
    if not isinstance(levels, list) or not 1 <= len(levels) <= 12:
        raise JobError(400, "levels: choose 1 to 12")
    lo, hi = LEVEL_BOUNDS[param]
    lv = sorted({_num({"level": x}, "level", lo, hi, integer=param == "obstacle_count") for x in levels})

    episodes = _num(spec, "episodes", 1, 500, integer=True, default=20)
    total = episodes * len(lv) * len(clean)
    if total > MAX_TOTAL_EPISODES:
        raise JobError(400, f"{total} episodes requested; the limit is {MAX_TOTAL_EPISODES}")

    preset = spec.get("preset", "medium")
    if preset not in PRESETS:
        raise JobError(400, f"preset must be one of {sorted(PRESETS)}")

    argv = ["--controllers", ",".join(clean), "--param", param, "--levels", ",".join(f"{v:g}" for v in lv),
            "--episodes", str(episodes), "--preset", preset]
    if timing == "lockstep":
        argv += ["--timing", "lockstep", "--interval", f"{_num(spec, 'interval', 0.02, 2, default=0.1):g}"]
        d = _num(spec, "delay", 0, 5, default=None)
        if d:
            argv += ["--delay", f"{d:g}"]
    optional = [
        ("seed", "--seed", 0, 10**9, True),
        ("obstacles", "--obstacles", 0, 400, True),
        ("decision_hz", "--decision-hz", 0.1, 120, False),
        ("max_duration", "--max-duration", 1, 600, False),
        ("max_inflight", "--max-inflight", 1, 8, True),
        ("deadline_ms", "--deadline-ms", 1, 60000, False),
        ("target_timeout", "--target-timeout", 1, 600, False),
    ]
    for key, flag, a, b, integer in optional:
        if key == "obstacles" and param == "obstacle_count":
            continue
        if timing == "lockstep" and key in ("decision_hz", "max_inflight", "deadline_ms"):
            continue
        v = _num(spec, key, a, b, integer)
        if v is not None:
            argv += [flag, f"{v:g}" if isinstance(v, float) else str(v)]
    return argv


def build_shadow_argv(spec: dict[str, Any], benchmarkable: list[str], jev_token: bool) -> list[str]:
    """Validate a shadow-run request and turn it into ``benchmark.shadow`` arguments."""
    shadows = spec.get("shadows")
    if not isinstance(shadows, list) or not 1 <= len(shadows) <= 6:
        raise JobError(400, "shadows: choose 1 to 6 controllers to score")
    clean: list[str] = []
    for c in shadows:
        c = str(c)
        if c not in benchmarkable:
            raise JobError(400, f"unknown controller {c!r}")
        if c == "jev" and not jev_token:
            raise JobError(400, "Jev needs OPENROUTER_API_KEY (or TYPESAFE_API_KEY) in .env")
        if c not in clean:
            clean.append(c)
    driver = str(spec.get("driver", "explorer"))
    if driver != "explorer" and driver not in benchmarkable:
        raise JobError(400, f"unknown driver {driver!r}")
    if driver == "jev" and not jev_token:
        raise JobError(400, "Jev needs OPENROUTER_API_KEY (or TYPESAFE_API_KEY) in .env")
    preset = spec.get("preset", "medium")
    if preset not in PRESETS:
        raise JobError(400, f"preset must be one of {sorted(PRESETS)}")
    episodes = _num(spec, "episodes", 1, 200, integer=True, default=10)
    interval = _num(spec, "interval", 0.02, 2, default=0.1)
    every = _num(spec, "branch_every", 0.25, 30, default=1.0)
    takeover = _num(spec, "takeover", 0.5, 10, default=3.0)
    argv = ["--driver", driver, "--shadows", ",".join(clean), "--episodes", str(episodes), "--preset", preset,
            "--interval", f"{interval:g}", "--branch-every", f"{every:g}", "--takeover", f"{takeover:g}"]
    for key, flag, a, b, integer in [("delay", "--delay", 0, 5, False), ("seed", "--seed", 0, 10**9, True),
                                     ("obstacles", "--obstacles", 0, 400, True),
                                     ("max_duration", "--max-duration", 1, 600, False),
                                     ("branch_workers", "--branch-workers", 1, 16, True)]:
        v = _num(spec, key, a, b, integer)
        if v is not None:
            argv += [flag, f"{v:g}" if isinstance(v, float) else str(v)]
    return argv


def build_ablation_argv(spec: dict[str, Any], benchmarkable: list[str], jev_token: bool) -> list[str]:
    """Validate an observation-ablation request and turn it into ``benchmark.ablation`` arguments."""
    c = str(spec.get("controller", ""))
    m = SPEC_RE.match(c)
    if not m or m.group(1) not in benchmarkable:
        raise JobError(400, f"unknown controller {c!r}")
    if m.group(1) == "jev" and not jev_token:
        raise JobError(400, "Jev needs OPENROUTER_API_KEY (or TYPESAFE_API_KEY) in .env")
    ref = str(spec.get("reference", "simple_avoid"))
    if ref not in benchmarkable or ref == "jev":
        raise JobError(400, f"reference must be a local controller, got {ref!r}")
    modes = spec.get("modes", ["raw", "relative", "physics"])
    if not isinstance(modes, list) or not modes or any(x not in ("raw", "relative", "physics") for x in modes):
        raise JobError(400, "modes: choose from raw, relative, physics")
    timing = spec.get("timing", "realtime")
    if timing not in ("realtime", "lockstep"):
        raise JobError(400, "timing must be realtime or lockstep")
    preset = spec.get("preset", "easy")
    if preset not in PRESETS:
        raise JobError(400, f"preset must be one of {sorted(PRESETS)}")
    episodes = _num(spec, "episodes", 1, 500, integer=True, default=30)
    argv = ["--controller", c, "--reference", ref, "--modes", ",".join(dict.fromkeys(modes)),
            "--episodes", str(episodes), "--preset", preset, "--timing", timing]
    if timing == "realtime":
        ml = spec.get("match_latency", "auto")
        if ml != "auto":
            ml = f"{_num({'match_latency': ml}, 'match_latency', 0, 10000):g}"
        argv += ["--match-latency", ml]
    else:
        argv += ["--interval", f"{_num(spec, 'interval', 0.02, 2, default=0.1):g}"]
    for key, flag, a, b, integer in [("seed", "--seed", 0, 10**9, True), ("world_speed", "--world-speed", 0.05, 64, False),
                                     ("reference_repeats", "--reference-repeats", 1, 5, True),
                                     ("obstacles", "--obstacles", 0, 400, True),
                                     ("max_duration", "--max-duration", 1, 600, False),
                                     ("decision_hz", "--decision-hz", 0.1, 120, False),
                                     ("max_inflight", "--max-inflight", 1, 8, True)]:
        if timing == "lockstep" and key in ("world_speed", "decision_hz", "max_inflight"):
            continue
        v = _num(spec, key, a, b, integer)
        if v is not None:
            argv += [flag, f"{v:g}" if isinstance(v, float) else str(v)]
    if spec.get("also_unqualified"):
        argv.append("--also-unqualified")
    return argv


@dataclass
class Job:
    id: str
    argv: list[str]
    run_dir: Path
    log_path: Path
    started: float
    proc: subprocess.Popen
    kind: str = "suite"  # "suite" | "shadow" | "ablation"
    ended: Optional[float] = None
    cancel_requested: bool = False
    tail: deque = field(default_factory=lambda: deque(maxlen=60))

    @property
    def status(self) -> str:
        code = self.proc.poll()
        if code is None:
            return "cancelling" if self.cancel_requested else "running"
        if self.ended is None:
            self.ended = time.time()
        if code == 0:
            return "succeeded"
        return "cancelled" if (self.cancel_requested or code == 130) else "failed"

    def to_dict(self) -> dict[str, Any]:
        status = self.status
        progress = None
        suite_status = None
        manifest = self.run_dir / f"{self.kind}.json"
        if manifest.is_file():
            try:
                m = json.loads(manifest.read_text())
                progress, suite_status = m.get("progress"), m.get("status")
            except (OSError, ValueError):
                pass  # being replaced; next poll reads it
        return {
            "id": self.id,
            "status": status,
            "kind": self.kind,
            "suite_id": self.run_dir.name if manifest.is_file() else None,
            "suite_status": suite_status,
            "progress": progress,
            "command": f"python -m benchmark.{self.kind} " + " ".join(shlex.quote(a) for a in self.argv),
            "started": self.started,
            "ended": self.ended,
            "elapsed_s": (self.ended or time.time()) - self.started,
            "returncode": self.proc.poll(),
            "log_tail": list(self.tail),
        }


class JobManager:
    def __init__(self, runs_root: Path, python: str = sys.executable):
        self.root = Path(runs_root).resolve()
        self.python = python
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _active(self) -> Optional[Job]:
        for j in self.jobs.values():
            if j.status in ("running", "cancelling"):
                return j
        return None

    def start(self, spec: dict[str, Any], benchmarkable: list[str], jev_token: bool) -> dict[str, Any]:
        kind = spec.get("kind", "suite")
        builders = {"suite": build_argv, "shadow": build_shadow_argv, "ablation": build_ablation_argv}
        if kind not in builders:
            raise JobError(400, f"kind must be one of {sorted(builders)}")
        argv = builders[kind](spec, benchmarkable, jev_token)
        with self._lock:
            active = self._active()
            if active:
                raise JobError(409, "a benchmark is already running; cancel it or wait (parallel suites would distort measured latency)")
            if kind == "shadow":
                run_dir = new_run_dir(self.root, "shadow", argv[argv.index("--driver") + 1])
            elif kind == "ablation":
                run_dir = new_run_dir(self.root, "ablation", SPEC_RE.match(argv[1]).group(1))
            else:
                run_dir = new_run_dir(self.root, "suite", argv[argv.index("--param") + 1])
            logs = self.root / ".jobs"
            logs.mkdir(parents=True, exist_ok=True)
            job_id = uuid.uuid4().hex[:10]
            log_path = logs / f"{job_id}.log"
            full = argv + ["--run-dir", str(run_dir)]
            log = open(log_path, "w")
            proc = subprocess.Popen(
                [self.python, "-u", "-m", f"benchmark.{kind}", *full],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            job = Job(job_id, argv, run_dir, log_path, time.time(), proc, kind)
            self.jobs[job_id] = job

        def pump():
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                job.tail.append(line.rstrip("\n"))
            log.close()

        threading.Thread(target=pump, name=f"job-{job_id}", daemon=True).start()
        return job.to_dict()

    def list(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: j.started, reverse=True)]

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise JobError(404, "no such job")
        return job.to_dict()

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise JobError(404, "no such job")
        if job.proc.poll() is None:
            job.cancel_requested = True
            job.proc.terminate()  # SIGTERM: the suite records status "cancelled" and keeps finished points
        return job.to_dict()

    def shutdown(self) -> None:
        for j in self.jobs.values():
            if j.proc.poll() is None:
                j.cancel_requested = True
                j.proc.terminate()
