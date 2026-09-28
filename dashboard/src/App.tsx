import { BarChart3, Gamepad2, GitCompareArrows, ListTree, Radar } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, type Meta, type RunInfo, type SuiteDetail } from "./api";
import { paramInfo } from "./format";
import { controllerLabel } from "./theme";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { ComparePage, LiveArenaPage, RunsPage } from "./pages/OtherPages";

type Route = "live" | "benchmark" | "compare" | "runs";
const ROUTES: { id: Route; label: string; icon: typeof Radar }[] = [
  { id: "live", label: "Live Arena", icon: Gamepad2 },
  { id: "benchmark", label: "Benchmark", icon: BarChart3 },
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
  return `${paramInfo(s.param ?? "").label} · ${ctrls} · ${date}${s.status === "running" ? " · running" : ""}`;
}

export default function App() {
  const [{ route, suite: hashSuite }, setLoc] = useState(parseHash());
  const [meta, setMeta] = useState<Meta | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [suite, setSuite] = useState<SuiteDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
    const load = () => {
      setLoading(true);
      api
        .suite(selectedId)
        .then((s) => !cancelled && (setSuite(s), setError(null)))
        .catch((e) => !cancelled && setError(String(e.message ?? e)))
        .finally(() => !cancelled && setLoading(false));
    };
    load();
    const running = suites.find((s) => s.id === selectedId)?.status === "running";
    const t = running ? setInterval(load, 5000) : undefined;
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
          {suites.length > 0 && (
            <select className="select" value={selectedId ?? ""} onChange={(e) => go(route === "runs" || route === "live" ? "benchmark" : route, e.target.value)} aria-label="benchmark suite">
              {suites.map((s) => (
                <option key={s.id} value={s.id}>{suiteLabel(s)}</option>
              ))}
            </select>
          )}
        </div>
      </header>
      {route === "benchmark" && (
        <BenchmarkPage meta={meta} suites={suites} suite={suite?.id === selectedId ? suite : null} loading={loading} onSelectSuite={(id) => go("benchmark", id)} />
      )}
      {route === "compare" && <ComparePage suite={suite?.id === selectedId ? suite : null} />}
      {route === "runs" && <RunsPage runs={runs} onOpenSuite={(id) => go("benchmark", id)} />}
      {route === "live" && <LiveArenaPage />}
    </div>
  );
}
