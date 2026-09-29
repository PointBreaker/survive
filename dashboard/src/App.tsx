import { BarChart3, FlaskConical, Gamepad2, GitCompareArrows, ListTree, Radar } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, type AblationJobSpec, type Job, type JobSpec, type Meta, type RunInfo, type SuiteDetail } from "./api";
import { paramInfo } from "./format";
import { controllerLabel } from "./theme";
import { AblationPage } from "./pages/AblationPage";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { ComparePage, LiveArenaPage, RunsPage } from "./pages/OtherPages";

type Route = "live" | "benchmark" | "ablation" | "compare" | "runs";
const ROUTES: { id: Route; label: string; icon: typeof Radar }[] = [
  { id: "live", label: "Live Arena", icon: Gamepad2 },
  { id: "benchmark", label: "Benchmark", icon: BarChart3 },
  { id: "ablation", label: "Ablation", icon: FlaskConical },
  { id: "compare", label: "Compare", icon: GitCompareArrows },
  { id: "runs", label: "Runs", icon: ListTree },
];

function parseHash(): { route: Route; suite: string | null } {
  const m = window.location.hash.match(/^#\/(\w+)(?:\?suite=([^&]+))?/);
  const route = (ROUTES.find((r) => r.id === m?.[1])?.id ?? "benchmark") as Route;
  return { route, suite: m?.[2] ? decodeURIComponent(m[2]) : null };
}

function suiteLabel(s: RunInfo): string {
  const ctrls = (s.controllers ?? []).map((c) => controllerLabel(String(c))).join(", ");
  const date = new Date(s.mtime * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const st = s.status === "running" && s.progress
    ? ` · running ${Math.round((s.progress.episodes_done / Math.max(1, s.progress.episodes_total)) * 100)}%`
    : s.status && s.status !== "complete" ? ` · ${s.status}` : "";
  return `${paramInfo(s.param ?? "").label} · ${ctrls} · ${date}${st}`;
}

export default function App() {
  const [{ route, suite: hashSuite }, setLoc] = useState(parseHash());
  const [meta, setMeta] = useState<Meta | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [suite, setSuite] = useState<SuiteDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [dismissed, setDismissed] = useState<string | null>(null);
  const [follow, setFollow] = useState<string | null>(null); // job started from this tab

  const suites = runs.filter((r) => r.kind === "suite");
  const defaultSuite = suites.find((s) => s.param === "world_speed_scale") ?? suites[0];
  const selectedId = hashSuite && suites.some((s) => s.id === hashSuite) ? hashSuite : defaultSuite?.id ?? null;

  useEffect(() => {
    const onHash = () => {
      const next = parseHash();
      setLoc((prev) => {
        if (prev.route !== next.route) window.scrollTo(0, 0);
        return next;
      });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((r: Route, s: string | null = selectedId) => {
    window.location.hash = `#/${r}${s ? `?suite=${encodeURIComponent(s)}` : ""}`;
  }, [selectedId]);

  const refreshRuns = useCallback(() => api.runs().then(setRuns).catch((e) => setError(String(e.message ?? e))), []);

  // Jobs: fast polling while one runs, slow otherwise.
  const job = jobs[0] && jobs[0].id !== dismissed ? jobs[0] : null;
  const jobActive = !!job && (job.status === "running" || job.status === "cancelling");
  const refreshJobs = useCallback(() => {
    if (!meta?.jobs_enabled) return Promise.resolve();
    return api.jobs().then(setJobs).catch(() => undefined);
  }, [meta?.jobs_enabled]);
  useEffect(() => {
    refreshJobs();
    const t = setInterval(() => {
      refreshJobs();
      if (jobActive) refreshRuns();
    }, jobActive ? 1500 : 10000);
    return () => clearInterval(t);
  }, [refreshJobs, refreshRuns, jobActive]);

  // Jump to a suite started from this tab as soon as its manifest exists.
  useEffect(() => {
    const j = jobs.find((x) => x.id === follow);
    if (j?.suite_id && runs.some((r) => r.id === j.suite_id)) {
      setFollow(null);
      go(j.kind === "ablation" ? "ablation" : "benchmark", j.suite_id);
    }
  }, [jobs, runs, follow]); // eslint-disable-line react-hooks/exhaustive-deps

  // When a job ends, refresh the run list at once so the suite's final status shows.
  const lastStatus = useRef<string | null>(null);
  useEffect(() => {
    const st = job?.status ?? null;
    if (lastStatus.current && ["running", "cancelling"].includes(lastStatus.current) && st && !["running", "cancelling"].includes(st)) {
      refreshRuns();
    }
    lastStatus.current = st;
  }, [job?.status, refreshRuns]);

  const onRun = async (spec: JobSpec | AblationJobSpec) => {
    const j = await api.startJob(spec);
    setDismissed(null);
    setFollow(j.id);
    setJobs((cur) => [j, ...cur.filter((x) => x.id !== j.id)]);
  };
  const onCancelJob = (id: string) => api.cancelJob(id).then(() => refreshJobs()).catch((e) => setError(String(e.message ?? e)));
  useEffect(() => {
    api.meta().then(setMeta).catch(() => undefined);
    refreshRuns().finally(() => setLoading(false));
    const t = setInterval(refreshRuns, 10000); // pick up new suites started from the terminal
    return () => clearInterval(t);
  }, [refreshRuns]);

  // Load the selected suite; poll while it is still running.
  useEffect(() => {
    if (!selectedId) {
      setSuite(null);
      return;
    }
    let cancelled = false;
    let first = true;
    const load = () => {
      // Dim only when switching suites; live refreshes keep the frame steady.
      if (first) setLoading(true);
      first = false;
      api
        .suite(selectedId)
        .then((s) => !cancelled && (setSuite(s), setError(null)))
        .catch((e) => !cancelled && setError(String(e.message ?? e)))
        .finally(() => !cancelled && setLoading(false));
    };
    load();
    const running = suites.find((s) => s.id === selectedId)?.status === "running";
    const t = running ? setInterval(load, 3000) : undefined;
    return () => {
      cancelled = true;
      if (t) clearInterval(t);
    };
  }, [selectedId, suites.find((s) => s.id === selectedId)?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="app">
      <header className="topnav">
        <div className="brand">
          <div className="brand-mark"><Radar size={15} /></div>
          <div>
            <div className="brand-name">Decision Arena</div>
            <div className="brand-sub">real-time closed-loop decision benchmark</div>
          </div>
        </div>
        <nav className="tabs">
          {ROUTES.map((r) => {
            const Icon = r.icon;
            return (
              <button key={r.id} className={`tab${route === r.id ? " active" : ""}`} onClick={() => go(r.id)}>
                <Icon size={15} /> {r.label}
              </button>
            );
          })}
        </nav>
        <div className="nav-right">
          {error && <span className="error" title={error}>API error</span>}
          {suites.length > 0 && route !== "ablation" && (
            <select className="select" value={selectedId ?? ""} onChange={(e) => go(route === "runs" || route === "live" ? "benchmark" : route, e.target.value)} aria-label="benchmark suite">
              {suites.map((s) => (
                <option key={s.id} value={s.id}>{suiteLabel(s)}</option>
              ))}
            </select>
          )}
        </div>
      </header>
      {route === "benchmark" && (
        <BenchmarkPage
          meta={meta}
          suites={suites}
          suite={suite?.id === selectedId ? suite : null}
          loading={loading}
          onSelectSuite={(id) => go("benchmark", id)}
          job={job}
          onRun={onRun}
          onCancelJob={onCancelJob}
          onDismissJob={() => job && setDismissed(job.id)}
        />
      )}
      {route === "ablation" && (
        <AblationPage meta={meta} runs={runs} selected={hashSuite} onSelect={(id) => go("ablation", id)} onRun={onRun} jobActive={jobActive} />
      )}
      {route === "compare" && <ComparePage suite={suite?.id === selectedId ? suite : null} />}
      {route === "runs" && <RunsPage runs={runs} onOpenSuite={(id) => go(runs.find((r) => r.id === id)?.kind === "ablation" ? "ablation" : "benchmark", id)} />}
      {route === "live" && <LiveArenaPage />}
    </div>
  );
}
