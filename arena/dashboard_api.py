"""Read-only dashboard API over the ``runs/`` directory (+ static dashboard files).

    python -m arena.dashboard_api                 # http://127.0.0.1:8787
    python -m arena.dashboard_api --runs runs --port 8787

This is an observation layer. It never runs episodes, never changes a
config, and never recomputes a statistic with a different definition:

* per-point aggregates and D-thresholds come from the artifacts written by
  ``benchmark.suite`` (``benchmark.metrics.aggregate`` / ``threshold_summary``);
* distributions (latency, actions, confidence, failure reasons, decision
  accounting) are counted from the logged per-episode results and events;
* replays are exact re-simulations via ``arena.replay`` from config + seed +
  logged actions, and each one is checked against the logged result.

Endpoints (JSON):
    GET /api/meta
    GET /api/runs
    GET /api/suites
    GET /api/suites/<id>
    GET /api/suites/<id>/episodes?controller=<spec>&level=<level>
    GET /api/suites/<id>/compare?level=<level>&seed=<seed>
    GET /api/suites/<id>/replay?episode=<relative episode dir>
    GET /api/ablations
    GET /api/ablations/<id>                                 manifest + summary (+ per-episode rows)
    GET /api/runs/<id>/replay?episode=<relative dir>        verified replay of any episode in a run
    GET /api/runs/<id>/snapshots?episode=<relative dir>     the controller's decision requests
    GET /api/runs/<id>/inspect?episode=<dir>&tick=<k>[&mode=raw]
                     matched snapshot: the rebuilt observation in all modes + local controllers' answers
    GET /api/shadows/<id>                                   manifest + takeover scores + episodes
    GET /api/shadows/<id>/branches?episode=<name>           branch outcomes (no paths)
    GET /api/shadows/<id>/branch?episode=<name>&tick=<k>    one branch with paths + obstacle frames

Benchmark jobs (only when the server binds to localhost, see arena.dashboard_jobs):
    GET  /api/jobs            GET /api/jobs/<id>
    POST /api/jobs            POST /api/jobs/<id>/cancel
Jobs run the unchanged ``python -m benchmark.suite`` / ``benchmark.shadow`` CLIs as subprocesses.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from arena.dashboard_jobs import JobError, JobManager
from arena.difficulty import PRESETS, DifficultyConfig
from arena.dotenv import load_dotenv
from arena.stats import mean, percentile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dashboard" / "dist"
ACTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "STAY")
ANSWER_EVENTS = ("decision", "decision_failed", "decision_superseded", "decision_dropped")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# =============================================================== run listing
def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def effective_status(manifest: dict[str, Any]) -> Optional[str]:
    """'running' only while the writing process exists; otherwise 'interrupted'."""
    status = manifest.get("status")
    if status == "running" and "pid" in manifest and not _pid_alive(manifest["pid"]):
        return "interrupted"
    return status


def classify_run(d: Path) -> Optional[dict[str, Any]]:
    """Describe one run directory, or None if it is not a recognised artifact."""
    stat = d.stat()
    base = {"id": d.name, "mtime": stat.st_mtime}
    if (d / "suite.json").is_file():
        m = _read_json(d / "suite.json")
        return {**base, "kind": "suite", "title": f"suite · {m['param']}", "param": m["param"],
                "levels": m["levels"], "controllers": m["controllers"], "status": effective_status(m),
                "progress": m.get("progress"),
                "episodes_per_point": m.get("episodes_per_point"), "created": m.get("created")}
    if (d / "ablation.json").is_file():
        m = _read_json(d / "ablation.json")
        return {**base, "kind": "ablation", "title": f"ablation · {m['controller']}", "controllers": [m["controller"]],
                "reference": m["reference_controller"], "modes": m["modes"], "timing": m["timing"],
                "experiment_type": m["experiment_type"], "status": effective_status(m),
                "progress": m.get("progress"), "created": m.get("created"),
                "episodes": len(m.get("candidate_seeds", []))}
    if (d / "shadow.json").is_file():
        m = _read_json(d / "shadow.json")
        return {**base, "kind": "shadow", "title": f"shadow · {m['driver']}", "driver": m["driver"],
                "controllers": m["shadows"], "status": effective_status(m), "progress": m.get("progress"),
                "episodes": m.get("episodes"), "interval_s": m.get("interval_s"), "delay_s": m.get("delay_s"),
                "preset_config": m.get("config"), "created": m.get("created"),
                "scored": (d / "scores.json").is_file()}
    if (d / "summary.json").is_file():
        s = _read_json(d / "summary.json")
        if "levels" in s and "config" in s and isinstance(s.get("levels"), list) and "jitter_ms" in s:
            return {**base, "kind": "remote_validation", "title": "remote latency validation"}
        kind = "sweep" if "sweep" in s else "adaptive" if "adaptive" in s else "batch"
        info = {**base, "kind": kind, "title": f"{kind} · {s.get('controller', '?')}",
                "controllers": [s.get("controller")], "episodes": s.get("episodes")}
        if kind == "batch" and "aggregate" in s:
            info["success_rate"] = s["aggregate"].get("success_rate")
        if kind in ("sweep", "adaptive"):
            info["param"] = s[kind]["param"]
        return info
    if (d / "events.jsonl").is_file() and (d / "config.json").is_file():
        c = _read_json(d / "config.json")
        info = {**base, "kind": "episode", "title": f"episode · {c.get('controller', '?')}",
                "controllers": [c.get("controller")], "seed": c.get("seed")}
        if (d / "result.json").is_file():
            r = _read_json(d / "result.json")
            info.update({"success": r.get("success"), "reason": r.get("reason"),
                         "survival_time": r.get("survival_time"), "targets": r.get("targets_collected")})
        return info
    return None


# ================================================================ analysis
def _histogram(values: list[float], lo: float, hi: float, bins: int = 40) -> list[dict[str, float]]:
    width = (hi - lo) / bins if hi > lo else 1.0
    counts = [0] * bins
    for v in values:
        i = int((v - lo) / width) if width > 0 else 0
        counts[min(bins - 1, max(0, i))] += 1
    return [{"x0": lo + i * width, "x1": lo + (i + 1) * width, "count": c} for i, c in enumerate(counts)]


def _latency_block(values: list[float], period_ms: float, hi: float) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": mean(values),
        "over_period": (sum(1 for v in values if v > period_ms) / len(values)) if values else None,
        "histogram": _histogram(values, 0.0, hi) if values else [],
    }


def analyse_suite(d: Path) -> dict[str, Any]:
    manifest = _read_json(d / "suite.json")
    manifest["status"] = effective_status(manifest)
    summary = _read_json(d / "summary.json") if (d / "summary.json").is_file() else {"controllers": {}}
    results = list(_read_jsonl(d / "results.jsonl")) if (d / "results.jsonl").is_file() else []
    cfg = manifest["config"]
    period_ms = 1000.0 / cfg["decision_hz"]
    analysis: dict[str, Any] = {}

    for spec in manifest["controllers"]:
        rs = [r for r in results if r["controller_spec"] == spec]
        lat_by_level: dict[str, list[float]] = {}
        actions: Counter = Counter()
        confidences: list[float] = []
        answers: Counter = Counter()
        for r in rs:
            ev_path = d / r["episode_dir"] / "events.jsonl"
            if not ev_path.is_file():
                continue
            key = f"{r['level']:g}"
            for e in _read_jsonl(ev_path):
                t = e.get("type")
                if t in ANSWER_EVENTS:
                    answers[t] += 1
                    if e.get("latency_ms") is not None:
                        lat_by_level.setdefault(key, []).append(float(e["latency_ms"]))
                if t == "decision":
                    actions[e["action"]] += 1
                    conf = (e.get("meta") or {}).get("confidence")
                    if isinstance(conf, (int, float)):
                        confidences.append(float(conf))

        pooled = [v for vs in lat_by_level.values() for v in vs]
        hi = max(period_ms * 1.5, (percentile(pooled, 99) or 0.0) * 1.1) if pooled else period_ms * 1.5
        failures_total = Counter(r["reason"] for r in rs if not r["success"])
        by_level = {}
        for lv in manifest["levels"]:
            lrs = [r for r in rs if r["level"] == lv]
            if lrs:
                by_level[f"{lv:g}"] = {"episodes": len(lrs),
                                       "reasons": dict(Counter(r["reason"] for r in lrs if not r["success"])),
                                       "successes": sum(1 for r in lrs if r["success"])}
        accounting = {
            "requests": sum(r.get("request_count", 0) for r in rs),
            "applied": sum(r.get("decision_count", 0) for r in rs),
            "failed": sum(r.get("failed_decisions", 0) for r in rs),
            "superseded": sum(r.get("superseded_decisions", 0) for r in rs),
            "dropped_late": sum(r.get("late_dropped", 0) for r in rs),
            "missed_slots": sum(r.get("missed_slots", 0) for r in rs),
            "delayed_slots": sum(r.get("delayed_slots", 0) for r in rs),
        }
        per_level_latency = {}
        for k, vs in lat_by_level.items():
            per_level_latency[k] = _latency_block(vs, period_ms, hi)
        analysis[spec] = {
            "episodes": len(rs),
            "events_available": bool(pooled) or any((d / r["episode_dir"] / "events.jsonl").is_file() for r in rs[:1]),
            "latency": {"pooled": _latency_block(pooled, period_ms, hi), "by_level": per_level_latency},
            "actions": {a: actions.get(a, 0) for a in ACTIONS},
            "confidence": {"n": len(confidences),
                           "histogram": _histogram(confidences, 0.0, 1.0, 20) if confidences else []},
            "answers": dict(answers),
            "accounting": accounting,
            "failures": {"total": dict(failures_total), "episodes_failed": sum(failures_total.values()),
                         "by_level": by_level},
        }

    # Per-point extras for the frontier tooltip / latency metric (from events).
    points_extra: dict[str, dict[str, Any]] = {}
    for spec, a in analysis.items():
        points_extra[spec] = {k: {"p50_latency_ms": v["p50"], "p95_latency_ms": v["p95"]}
                              for k, v in a["latency"]["by_level"].items()}

    return {"id": d.name, "manifest": manifest, "summary": summary, "analysis": analysis,
            "points_extra": points_extra, "decision_period_ms": period_ms,
            "episodes_logged": len(results)}


# ================================================================== replay
def replay_episode(ep_dir: Path, max_frames: int = 2400) -> dict[str, Any]:
    from arena.environment import Environment
    from arena.replay import actions_per_tick

    cfg = _read_json(ep_dir / "config.json")
    result = _read_json(ep_dir / "result.json")
    config = DifficultyConfig.from_dict(cfg["config"])
    total = int(result["ticks"])
    actions = list(actions_per_tick(result["action_changes"], total))
    stride = max(1, math.ceil((total + 1) / max_frames))
    env = Environment(config, cfg["seed"])
    radii: dict[int, float] = {}
    frames = []

    def snap():
        p, t = env.player, env.target
        obs = []
        for o in env.obstacles:
            radii[o.id] = round(o.radius, 2)
            obs.extend((o.id, round(o.x, 1), round(o.y, 1)))
        frames.append([env.tick, round(p.x, 1), round(p.y, 1), round(t.x, 1), round(t.y, 1), env.score, obs])

    for tick in range(total + 1):
        if tick % stride == 0 or tick == total:
            snap()
        if tick < total and not env.done:
            env.step(actions[tick])
    verified = (env.tick == total and env.score == result["targets_collected"]
                and env.outcome.reason == result["reason"])

    timeline = {"decisions": [], "targets": [], "collision": None}
    ev_path = ep_dir / "events.jsonl"
    if ev_path.is_file():
        for e in _read_jsonl(ev_path):
            t = e.get("type")
            if t == "lockstep_decision":
                a = (e.get("answers") or {}).get(cfg.get("controller")) or {}
                if a.get("action"):
                    timeline["decisions"].append({"tick": e["effective_tick"], "request_tick": e["tick"],
                                                  "action": a["action"], "latency_ms": round(a["latency_ms"], 2),
                                                  "confidence": a.get("confidence")})
            elif t == "decision":
                timeline["decisions"].append({"tick": e["applied_tick"], "request_tick": e["request_tick"],
                                              "action": e["action"], "latency_ms": round(e["latency_ms"], 2),
                                              "confidence": (e.get("meta") or {}).get("confidence")})
            elif t == "target_collected":
                timeline["targets"].append({"tick": e["tick"], "score": e["score"]})
            elif t == "collision":
                timeline["collision"] = {"tick": e["tick"], "obstacle_id": e["obstacle_id"], "t": e["t"],
                                         "x": round(env.player.x, 1), "y": round(env.player.y, 1)}
    return {
        "episode": str(ep_dir.name),
        "controller": cfg.get("controller"),
        "seed": cfg["seed"],
        "arena": {"width": config.arena_width, "height": config.arena_height},
        "player_radius": config.player_radius,
        "target_radius": config.target_radius,
        "world_seconds_per_tick": config.world_speed_scale / 60.0,
        "ticks": total,
        "stride": stride,
        "radii": radii,
        "frames": frames,
        "timeline": timeline,
        "result": {k: v for k, v in result.items() if k != "action_changes"},
        "verified": verified,
    }


# ===================================================================== API
class DashboardAPI:
    def __init__(self, runs_root: Path, jobs_enabled: bool = False):
        self.root = Path(runs_root).resolve()
        self.jobs = JobManager(self.root) if jobs_enabled else None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _run_dir(self, run_id: str) -> Path:
        run_id = unquote(run_id)
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            raise ApiError(400, "bad run id")
        d = (self.root / run_id).resolve()
        if d.parent != self.root or not d.is_dir():
            raise ApiError(404, f"run {run_id!r} not found")
        return d

    def _suite_dir(self, run_id: str) -> Path:
        d = self._run_dir(run_id)
        if not (d / "suite.json").is_file():
            raise ApiError(404, f"{run_id!r} is not a benchmark suite")
        return d

    def _cached(self, key: str, stamp: float, compute):
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] == stamp:
                return hit[1]
        value = compute()
        with self._lock:
            self._cache[key] = (stamp, value)
        return value

    # ------------------------------------------------------------ handlers
    def meta(self) -> dict[str, Any]:
        from arena.benchmark import SWEEPABLE
        from controllers import CONTROLLER_NAMES

        jev = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
                   or os.environ.get("JEV_API_KEY"))
        return {
            "runs_root": str(self.root),
            "presets": {k: v.to_dict() for k, v in PRESETS.items()},
            "controllers": [c for c in CONTROLLER_NAMES if c not in ("sleep", "remote")],
            "benchmarkable": [c for c in CONTROLLER_NAMES if c not in ("sleep", "remote", "human")],
            "jev_token": jev,
            "sweepable": list(SWEEPABLE),
            "jobs_enabled": self.jobs is not None,
        }

    def runs(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        out = []
        for d in self.root.iterdir():
            if d.is_dir():
                try:
                    info = classify_run(d)
                except (OSError, ValueError, KeyError):
                    info = None
                if info:
                    out.append(info)
        out.sort(key=lambda r: r["mtime"], reverse=True)
        return out

    def suites(self) -> list[dict[str, Any]]:
        return [r for r in self.runs() if r["kind"] == "suite"]

    def suite(self, run_id: str) -> dict[str, Any]:
        d = self._suite_dir(run_id)
        stamp = max((d / f).stat().st_mtime for f in ("suite.json", "results.jsonl", "summary.json")
                    if (d / f).is_file())
        m = _read_json(d / "suite.json")
        key = f"suite:{d.name}:{effective_status(m)}"  # a crash changes the status without touching files
        return self._cached(key, stamp, lambda: analyse_suite(d))

    def episodes(self, run_id: str, controller: Optional[str], level: Optional[str]) -> list[dict[str, Any]]:
        d = self._suite_dir(run_id)
        out = []
        for r in _read_jsonl(d / "results.jsonl"):
            if controller and r["controller_spec"] != controller:
                continue
            if level is not None and f"{r['level']:g}" != f"{float(level):g}":
                continue
            out.append({k: r.get(k) for k in (
                "controller_spec", "level", "seed", "success", "reason", "survival_time", "targets_collected",
                "collision_time", "p50_latency_ms", "p95_latency_ms", "mean_decision_latency_ms",
                "decision_count", "missed_slots", "episode_dir")})
        return out

    def compare(self, run_id: str, level: str, seed: str) -> dict[str, Any]:
        eps = [e for e in self.episodes(run_id, None, level) if str(e["seed"]) == str(int(float(seed)))]
        return {"level": float(level), "seed": int(float(seed)), "episodes": eps}

    def replay(self, run_id: str, episode: str) -> dict[str, Any]:
        d = self._suite_dir(run_id)
        ep = (d / unquote(episode)).resolve()
        if d not in ep.parents or not (ep / "result.json").is_file():
            raise ApiError(404, "episode not found in this suite")
        stamp = (ep / "result.json").stat().st_mtime
        return self._cached(f"replay:{ep}", stamp, lambda: replay_episode(ep))

    # ----------------------------------------------------------- ablations
    def ablations(self) -> list[dict[str, Any]]:
        return [r for r in self.runs() if r["kind"] == "ablation"]

    def ablation(self, run_id: str) -> dict[str, Any]:
        d = self._run_dir(run_id)
        if not (d / "ablation.json").is_file():
            raise ApiError(404, f"{run_id!r} is not an ablation run")
        m = _read_json(d / "ablation.json")
        m["status"] = effective_status(m)
        summary = None
        if (d / "summary.json").is_file():
            try:
                summary = _read_json(d / "summary.json")
            except ValueError:
                summary = None
        keep = ("role", "mode", "seed", "repeat", "success", "reason", "survival_time", "targets_collected",
                "p50_latency_ms", "mean_decision_latency_ms", "episode_dir", "qualified", "controller_spec",
                "attempt", "valid", "failure_rate")
        rows = [{k: r.get(k) for k in keep} for r in _read_jsonl(d / "results.jsonl")] \
            if (d / "results.jsonl").is_file() else []
        return {"id": d.name, "manifest": m, "summary": summary, "episodes": rows}

    def _episode_in(self, run_id: str, episode: str) -> Path:
        d = self._run_dir(run_id)
        ep = (d / unquote(episode)).resolve()
        if d not in ep.parents or not (ep / "result.json").is_file() or not (ep / "config.json").is_file():
            raise ApiError(404, "episode not found in this run")
        return ep

    def run_replay(self, run_id: str, episode: str) -> dict[str, Any]:
        ep = self._episode_in(run_id, episode)
        return self._cached(f"replay:{ep}", (ep / "result.json").stat().st_mtime, lambda: replay_episode(ep))

    def snapshots(self, run_id: str, episode: str) -> dict[str, Any]:
        from benchmark.snapshot import key_snapshots, requests

        ep = self._episode_in(run_id, episode)
        return {"episode": episode, "requests": requests(ep), "key_ticks": key_snapshots(ep)}

    def inspect(self, run_id: str, episode: str, tick: str, mode: str) -> dict[str, Any]:
        from arena.observation_views import MODES
        from benchmark.snapshot import LOCAL_CONTROLLERS, inspect

        ep = self._episode_in(run_id, episode)
        if mode not in MODES:
            raise ApiError(400, f"mode must be one of {MODES}")
        try:
            k = int(float(tick))
        except ValueError:
            raise ApiError(400, "tick must be a number")
        try:
            # local controllers only: the dashboard never makes paid API calls; the logged answer is shown instead
            return inspect(ep, k, LOCAL_CONTROLLERS, mode)
        except ValueError as e:
            raise ApiError(400, str(e))

    # ------------------------------------------------------------ shadows
    def _shadow_dir(self, run_id: str) -> Path:
        d = self._run_dir(run_id)
        if not (d / "shadow.json").is_file():
            raise ApiError(404, f"{run_id!r} is not a shadow run")
        return d

    def _shadow_episode(self, d: Path, episode: str) -> Path:
        ep = (d / "episodes" / unquote(episode)).resolve()
        if ep.parent != (d / "episodes").resolve() or not ep.is_dir():
            raise ApiError(404, "episode not found in this shadow run")
        return ep

    def shadows(self) -> list[dict[str, Any]]:
        return [r for r in self.runs() if r["kind"] == "shadow"]

    def shadow(self, run_id: str) -> dict[str, Any]:
        d = self._shadow_dir(run_id)
        m = _read_json(d / "shadow.json")
        m["status"] = effective_status(m)
        scores = None
        if (d / "scores.json").is_file():
            try:
                scores = _read_json(d / "scores.json")
            except ValueError:
                scores = None  # being replaced; next poll reads it
        episodes = list(_read_jsonl(d / "results.jsonl")) if (d / "results.jsonl").is_file() else []
        return {"id": d.name, "manifest": m, "scores": scores, "episodes": episodes}

    def shadow_branches(self, run_id: str, episode: str) -> dict[str, Any]:
        ep = self._shadow_episode(self._shadow_dir(run_id), episode)
        path = ep / "branches.jsonl"
        if not path.is_file():
            return {"episode": ep.name, "branches": []}
        keep = ("collided", "end_reason", "ticks", "targets", "approach", "min_clearance", "failed_answers")

        def compute():
            return {"episode": ep.name, "branches": [
                {"tick": b["tick"], "t": b["t"], "driver_action": b.get("driver_action"),
                 "outcomes": {c: {k: o.get(k) for k in keep} for c, o in b["outcomes"].items()}}
                for b in _read_jsonl(path)]}
        return self._cached(f"branches:{ep}", path.stat().st_mtime, compute)

    def shadow_branch(self, run_id: str, episode: str, tick: str) -> dict[str, Any]:
        d = self._shadow_dir(run_id)
        ep = self._shadow_episode(d, episode)
        m = _read_json(d / "shadow.json")
        cfg = DifficultyConfig.from_dict(m["config"])
        try:
            k = int(float(tick))
        except ValueError:
            raise ApiError(400, "tick must be a number")
        path = ep / "branches.jsonl"
        for b in (_read_jsonl(path) if path.is_file() else []):
            if b["tick"] == k:
                return {**b, "arena": {"width": cfg.arena_width, "height": cfg.arena_height},
                        "player_radius": cfg.player_radius, "path_every": 3, "frames_every": 6}
        raise ApiError(404, f"no branch at tick {k}")

    # ------------------------------------------------------------- routing
    def route(self, path: str, query: dict[str, list[str]]) -> Any:
        q = {k: v[0] for k, v in query.items()}
        parts = [p for p in path.split("/") if p][1:]  # drop "api"
        if parts == ["meta"]:
            return self.meta()
        if parts == ["runs"]:
            return self.runs()
        if parts == ["suites"]:
            return self.suites()
        if len(parts) == 2 and parts[0] == "suites":
            return self.suite(parts[1])
        if parts == ["ablations"]:
            return self.ablations()
        if len(parts) == 2 and parts[0] == "ablations":
            return self.ablation(parts[1])
        if len(parts) == 3 and parts[0] == "runs" and parts[2] in ("replay", "snapshots", "inspect"):
            if "episode" not in q:
                raise ApiError(400, "episode is required")
            if parts[2] == "replay":
                return self.run_replay(parts[1], q["episode"])
            if parts[2] == "snapshots":
                return self.snapshots(parts[1], q["episode"])
            if "tick" not in q:
                raise ApiError(400, "tick is required")
            return self.inspect(parts[1], q["episode"], q["tick"], q.get("mode", "raw"))
        if parts == ["shadows"]:
            return self.shadows()
        if len(parts) == 2 and parts[0] == "shadows":
            return self.shadow(parts[1])
        if len(parts) == 3 and parts[0] == "shadows":
            if "episode" not in q:
                raise ApiError(400, "episode is required")
            if parts[2] == "branches":
                return self.shadow_branches(parts[1], q["episode"])
            if parts[2] == "branch":
                if "tick" not in q:
                    raise ApiError(400, "tick is required")
                return self.shadow_branch(parts[1], q["episode"], q["tick"])
        if parts and parts[0] == "jobs":
            jobs = self._jobs()
            if len(parts) == 1:
                return jobs.list()
            if len(parts) == 2:
                return jobs.get(parts[1])
        if len(parts) == 3 and parts[0] == "suites":
            sid, what = parts[1], parts[2]
            if what == "episodes":
                return self.episodes(sid, q.get("controller"), q.get("level"))
            if what == "compare":
                if "level" not in q or "seed" not in q:
                    raise ApiError(400, "level and seed are required")
                return self.compare(sid, q["level"], q["seed"])
            if what == "replay":
                if "episode" not in q:
                    raise ApiError(400, "episode is required")
                return self.replay(sid, q["episode"])
        raise ApiError(404, f"no endpoint {path}")

    def _jobs(self) -> JobManager:
        if self.jobs is None:
            raise ApiError(403, "running benchmarks from the dashboard is disabled on this server")
        return self.jobs

    def route_post(self, path: str, body: Any) -> Any:
        parts = [p for p in path.split("/") if p][1:]
        jobs = self._jobs()
        if parts == ["jobs"]:
            if not isinstance(body, dict):
                raise ApiError(400, "expected a JSON object")
            meta = self.meta()
            return jobs.start(body, meta["benchmarkable"], meta["jev_token"])
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
            return jobs.cancel(parts[1])
        raise ApiError(404, f"no endpoint POST {path}")


def _loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1") or host.startswith("127.")


def make_server(runs_root: Path, host: str = "127.0.0.1", port: int = 8787,
                static_dir: Optional[Path] = DIST, jobs: Optional[bool] = None) -> ThreadingHTTPServer:
    # Starting processes is only offered on a loopback-bound server unless explicitly enabled.
    api = DashboardAPI(runs_root, jobs_enabled=_loopback(host) if jobs is None else jobs)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=60")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            u = urlsplit(self.path)
            if u.path.startswith("/api/"):
                try:
                    data = api.route(u.path, parse_qs(u.query))
                    self._send(200, json.dumps(data, separators=(",", ":")).encode(), "application/json")
                except (ApiError, JobError) as e:
                    self._send(e.status, json.dumps({"error": str(e)}).encode(), "application/json")
                except Exception as e:  # never crash the server on a bad artifact
                    self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")
                return
            self._static(u.path)

        def do_POST(self) -> None:
            u = urlsplit(self.path)
            try:
                if not u.path.startswith("/api/"):
                    raise ApiError(404, "not found")
                # CSRF guard: JSON bodies force a CORS preflight (which we never grant),
                # and a present Origin must be this server.
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
                if ctype != "application/json":
                    raise ApiError(415, "Content-Type must be application/json")
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    raise ApiError(403, "cross-origin request refused")
                length = int(self.headers.get("Content-Length") or 0)
                if length > 64_000:
                    raise ApiError(413, "request too large")
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw or b"{}")
                except ValueError:
                    raise ApiError(400, "invalid JSON")
                data = api.route_post(u.path, body)
                self._send(200, json.dumps(data, separators=(",", ":")).encode(), "application/json")
            except ApiError as e:
                self._send(e.status, json.dumps({"error": str(e)}).encode(), "application/json")
            except JobError as e:
                self._send(e.status, json.dumps({"error": str(e)}).encode(), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

        def _static(self, path: str) -> None:
            if static_dir is None or not (static_dir / "index.html").is_file():
                msg = ("Dashboard UI is not built yet.\n\n  cd dashboard && npm install && npm run build\n\n"
                       "The JSON API is available under /api/.\n")
                self._send(200, msg.encode(), "text/plain; charset=utf-8")
                return
            rel = unquote(path).lstrip("/")
            f = (static_dir / rel).resolve() if rel else static_dir / "index.html"
            if not rel or static_dir.resolve() not in f.parents or not f.is_file():
                f = static_dir / "index.html"  # SPA fallback
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            self._send(200, f.read_bytes(), ctype)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    srv.api = api  # type: ignore[attr-defined]
    return srv


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--enable-jobs", action="store_true", help="allow starting benchmarks even on a non-loopback host")
    ap.add_argument("--disable-jobs", action="store_true", help="read-only: never start benchmarks")
    args = ap.parse_args(argv)
    load_dotenv()
    jobs = False if args.disable_jobs else (True if args.enable_jobs else None)
    srv = make_server(Path(args.runs), args.host, args.port, jobs=jobs)
    host, port = srv.server_address[:2]
    print(f"Decision Arena dashboard: http://{host}:{port}   (runs: {Path(args.runs).resolve()})")
    if not (DIST / "index.html").is_file():
        print("UI not built yet: cd dashboard && npm install && npm run build   (API works meanwhile)")
    print("benchmark jobs: " + ("enabled" if srv.api.jobs else "disabled (read-only)"))  # type: ignore[attr-defined]
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if srv.api.jobs:  # type: ignore[attr-defined]
            srv.api.jobs.shutdown()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
