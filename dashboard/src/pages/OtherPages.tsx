import { Gamepad2, Monitor } from "lucide-react";
import type { RunInfo, SuiteDetail } from "../api";
import { CompareRuns } from "../components/CompareRuns";
import { referenceLevel } from "../components/KeyResults";
import { Card } from "../components/ui";
import { fmtPct, fmtS, paramInfo, reasonLabel } from "../format";
import { controllerLabel, seriesStyles } from "../theme";
import { EmptyFrontier } from "./BenchmarkPage";

export function ComparePage({ suite }: { suite: SuiteDetail | null }) {
  return (
    <div className="page single">
      <main className="main">
        <div className="main-inner">
          <div>
            <h1 className="page-title">Compare</h1>
            <div className="page-sub">Same seed, same initial conditions, one clock. Where do controllers diverge?</div>
          </div>
          {suite ? (
            <CompareRuns suite={suite} styles={seriesStyles(suite.manifest.controllers)} initialLevel={referenceLevel(suite)} />
          ) : (
            <EmptyFrontier />
          )}
        </div>
      </main>
    </div>
  );
}

export function RunsPage({ runs, onOpenSuite }: { runs: RunInfo[]; onOpenSuite: (id: string) => void }) {
  return (
    <div className="page single">
      <main className="main">
        <div className="main-inner">
          <div>
            <h1 className="page-title">Runs</h1>
            <div className="page-sub">Everything under runs/. Benchmark suites open in the Benchmark view.</div>
          </div>
          <Card>
            {runs.length === 0 ? (
              <div className="muted">No runs yet.</div>
            ) : (
              <table className="runs">
                <thead>
                  <tr><th>Run</th><th>Kind</th><th>Controllers</th><th>Details</th><th>Modified</th></tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id} className={r.kind === "suite" ? "clickable" : ""} onClick={() => r.kind === "suite" && onOpenSuite(r.id)}>
                      <td className="mono">{r.id}</td>
                      <td><span className="kind">{r.kind}</span></td>
                      <td>{(r.controllers ?? []).filter(Boolean).map((c) => controllerLabel(String(c))).join(", ") || "–"}</td>
                      <td className="muted">{details(r)}</td>
                      <td className="muted tnum">{new Date(r.mtime * 1000).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>
      </main>
    </div>
  );
}

function details(r: RunInfo): string {
  switch (r.kind) {
    case "suite":
      return `${paramInfo(r.param ?? "").label}: ${(r.levels ?? []).map((l) => paramInfo(r.param ?? "").unit(l)).join(", ")} · ${r.episodes_per_point} seeds/point · ${r.status}`;
    case "batch":
      return `${r.episodes ?? "?"} episodes · success ${fmtPct(r.success_rate ?? null)}`;
    case "sweep":
    case "adaptive":
      return `single-controller ${r.kind} over ${paramInfo(r.param ?? "").short} (aggregates only)`;
    case "episode":
      return `seed ${r.seed} · ${r.reason ? reasonLabel(r.reason) : "unfinished"} · ${fmtS(r.survival_time)} · ${r.targets ?? 0} targets`;
    default:
      return r.title;
  }
}

export function LiveArenaPage() {
  return (
    <div className="page single">
      <main className="main">
        <div className="main-inner" style={{ maxWidth: 820 }}>
          <div>
            <h1 className="page-title">Live Arena</h1>
            <div className="page-sub">Real-time play and inspection run in the native app, not the browser.</div>
          </div>
          <Card title={<span style={{ display: "flex", gap: 8, alignItems: "center" }}><Gamepad2 size={16} /> Desktop console</span>}>
            <div className="muted" style={{ marginBottom: 10 }}>
              Parameters on the left, the arena in the middle, requests / responses / latency on the right. Live episodes
              and their logs land in runs/ and show up in the Runs view.
            </div>
            <div className="cmd">python main.py{"\n"}python main.py --controller jev --max-inflight 3</div>
          </Card>
          <Card title={<span style={{ display: "flex", gap: 8, alignItems: "center" }}><Monitor size={16} /> Replay a logged episode</span>}>
            <div className="cmd">python -m arena.viewer runs/&lt;run&gt; --at collision</div>
          </Card>
        </div>
      </main>
    </div>
  );
}
