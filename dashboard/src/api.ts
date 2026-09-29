// Typed client for the read-only dashboard API (arena/dashboard_api.py).

export interface Aggregate {
  level: number;
  episodes: number;
  successes: number;
  success_rate: number;
  success_rate_ci95: [number, number];
  mean_targets: number | null;
  mean_survival_time: number | null;
  mean_latency_ms: number | null;
  p95_latency_ms: number | null;
  mean_missed_slots: number | null;
  failure_reasons: Record<string, number>;
}

export type ThresholdKind = "interpolated" | "above_range" | "below_range" | "not_measured";
export interface Threshold {
  value: number | null;
  kind: ThresholdKind;
  p: number;
}

export interface Histogram {
  x0: number;
  x1: number;
  count: number;
}

export interface LatencyBlock {
  n: number;
  p50: number | null;
  p95: number | null;
  mean: number | null;
  over_period: number | null;
  histogram: Histogram[];
}

export interface ControllerAnalysis {
  episodes: number;
  events_available: boolean;
  latency: { pooled: LatencyBlock; by_level: Record<string, LatencyBlock> };
  actions: Record<string, number>;
  confidence: { n: number; histogram: Histogram[] };
  answers: Record<string, number>;
  accounting: {
    requests: number;
    applied: number;
    failed: number;
    superseded: number;
    dropped_late: number;
    missed_slots: number;
    delayed_slots: number;
  };
  failures: {
    total: Record<string, number>;
    episodes_failed: number;
    by_level: Record<string, { episodes: number; successes: number; reasons: Record<string, number> }>;
  };
}

export interface SuiteManifest {
  kind: "suite";
  status: "running" | "complete" | "cancelled" | "interrupted";
  progress?: SuiteProgress;
  created: string;
  param: string;
  levels: number[];
  controllers: string[];
  episodes_per_point: number;
  base_seed: number;
  seeds: number[];
  config: Record<string, number | null> & { decision_hz: number; max_duration: number; world_speed_scale: number };
  added_latency_ms: number;
  argv: string[];
}

export interface SuiteDetail {
  id: string;
  manifest: SuiteManifest;
  summary: {
    param: string;
    levels: number[];
    controllers: Record<string, { points: Aggregate[]; thresholds: Record<"D90" | "D50" | "D10", Threshold> }>;
  };
  analysis: Record<string, ControllerAnalysis>;
  points_extra: Record<string, Record<string, { p50_latency_ms: number | null; p95_latency_ms: number | null }>>;
  decision_period_ms: number;
  episodes_logged: number;
}

export interface RunInfo {
  id: string;
  kind: "suite" | "batch" | "sweep" | "adaptive" | "episode" | "remote_validation";
  title: string;
  mtime: number;
  param?: string;
  levels?: number[];
  controllers?: string[];
  status?: string;
  progress?: SuiteProgress;
  episodes_per_point?: number;
  episodes?: number;
  success_rate?: number | null;
  success?: boolean;
  reason?: string;
  survival_time?: number;
  targets?: number;
  seed?: number;
}

export interface SuiteProgress {
  episodes_done: number;
  episodes_total: number;
  points_done: number;
  points_total: number;
  current: { controller: string; level: number } | null;
}

export type JobStatus = "running" | "cancelling" | "succeeded" | "failed" | "cancelled";
export interface Job {
  id: string;
  status: JobStatus;
  suite_id: string | null;
  suite_status: string | null;
  progress: SuiteProgress | null;
  command: string;
  started: number;
  ended: number | null;
  elapsed_s: number;
  returncode: number | null;
  log_tail: string[];
}

export interface JobSpec {
  controllers: string[];
  param: string;
  levels: number[];
  episodes: number;
  preset: string;
  seed?: number;
  obstacles?: number;
  decision_hz?: number;
  max_duration?: number;
  max_inflight?: number;
  deadline_ms?: number;
  target_timeout?: number;
}

export interface Meta {
  jobs_enabled: boolean;
  runs_root: string;
  presets: Record<string, Record<string, number | null>>;
  controllers: string[];
  benchmarkable: string[];
  jev_token: boolean;
  sweepable: string[];
}

export interface EpisodeSummary {
  controller_spec: string;
  level: number;
  seed: number;
  success: boolean;
  reason: string;
  survival_time: number;
  targets_collected: number;
  collision_time: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  mean_decision_latency_ms: number | null;
  decision_count: number;
  missed_slots: number;
  episode_dir: string;
}

// frame: [tick, px, py, tx, ty, score, [id, x, y, id, x, y, ...]]
export type Frame = [number, number, number, number, number, number, number[]];

export interface Replay {
  episode: string;
  controller: string;
  seed: number;
  arena: { width: number; height: number };
  player_radius: number;
  target_radius: number;
  world_seconds_per_tick: number;
  ticks: number;
  stride: number;
  radii: Record<string, number>;
  frames: Frame[];
  timeline: {
    decisions: { tick: number; request_tick: number; action: string; latency_ms: number; confidence: number | null }[];
    targets: { tick: number; score: number }[];
    collision: { tick: number; obstacle_id: number; t: number; x: number; y: number } | null;
  };
  result: Record<string, unknown> & { reason: string; success: boolean; survival_time: number; targets_collected: number };
  verified: boolean;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    let msg = `${res.status}`;
    try {
      msg = (await res.json()).error ?? msg;
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data as { error?: string }).error ?? `${res.status}`);
  return data as T;
}

const q = encodeURIComponent;
export const api = {
  meta: () => get<Meta>("/api/meta"),
  runs: () => get<RunInfo[]>("/api/runs"),
  suites: () => get<RunInfo[]>("/api/suites"),
  suite: (id: string) => get<SuiteDetail>(`/api/suites/${q(id)}`),
  episodes: (id: string, controller?: string, level?: number) =>
    get<EpisodeSummary[]>(
      `/api/suites/${q(id)}/episodes?` +
        [controller ? `controller=${q(controller)}` : "", level !== undefined ? `level=${level}` : ""]
          .filter(Boolean)
          .join("&"),
    ),
  compare: (id: string, level: number, seed: number) =>
    get<{ level: number; seed: number; episodes: EpisodeSummary[] }>(
      `/api/suites/${q(id)}/compare?level=${level}&seed=${seed}`,
    ),
  replay: (id: string, episode: string) => get<Replay>(`/api/suites/${q(id)}/replay?episode=${q(episode)}`),
  jobs: () => get<Job[]>("/api/jobs"),
  startJob: (spec: JobSpec) => post<Job>("/api/jobs", spec),
  cancelJob: (id: string) => post<Job>(`/api/jobs/${q(id)}/cancel`, {}),
};
