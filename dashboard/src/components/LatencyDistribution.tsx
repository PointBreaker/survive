import { useState } from "react";
import type { SuiteDetail } from "../api";
import { fmtMs, fmtPct, paramInfo } from "../format";
import type { SeriesStyle } from "../theme";
import { HistogramChart } from "./charts";
import { Card } from "./ui";

export function LatencyDistribution({ suite, focus }: { suite: SuiteDetail; focus: SeriesStyle }) {
  const [level, setLevel] = useState<string>("all");
  const a = suite.analysis[focus.spec];
  const pinfo = paramInfo(suite.manifest.param);
  const period = suite.decision_period_ms;
  const block = level === "all" ? a?.latency.pooled : a?.latency.by_level[level];
  const levels = Object.keys(a?.latency.by_level ?? {});
  return (
    <Card
      title="Latency Distribution"
      sub={`${focus.label} · runner-measured, snapshot → answer`}
      right={
        levels.length > 1 ? (
          <select className="select" value={level} onChange={(e) => setLevel(e.target.value)} aria-label="level">
            <option value="all">all levels</option>
            {levels.map((l) => (
              <option key={l} value={l}>{pinfo.unit(Number(l))}</option>
            ))}
          </select>
        ) : undefined
      }
    >
      {!block || !block.n ? (
        <div className="muted">No decision events logged for this controller.</div>
      ) : (
        <>
          <div className="stat-inline">
            <div><b>{fmtMs(block.p50)}</b><span>p50</span></div>
            <div><b>{fmtMs(block.p95)}</b><span>p95</span></div>
            <div><b>{fmtMs(period)}</b><span>decision period</span></div>
            <div><b>{fmtPct(block.over_period)}</b><span>over 1 period</span></div>
          </div>
          <HistogramChart
            bins={block.histogram}
            color={focus.color}
            xLabel="latency (ms)"
            fmtX={(v) => `${Math.round(v)}`}
            refLines={[
              { x: period, label: `decision period ${Math.round(period)} ms`, color: "var(--text-2)", dashed: true },
              ...(block.p50 !== null ? [{ x: block.p50, label: `p50 ${fmtMs(block.p50)}`, color: "#e8ecf2" }] : []),
              ...(block.p95 !== null ? [{ x: block.p95, label: `p95 ${fmtMs(block.p95)}`, color: "var(--warning)" }] : []),
            ]}
            tipLabel={(b) => (
              <>
                <div className="tt-title">{Math.round(b.x0)}–{Math.round(b.x1)} ms</div>
                <div className="tt-row"><span className="tt-key" style={{ background: focus.color }} /><span className="tt-name">answers</span><span className="tt-val">{b.count}</span></div>
              </>
            )}
          />
          {block.p95 !== null && block.p95 < 1 && <div className="note">All answers arrive in under 1 ms: this controller computes in-process.</div>}
        </>
      )}
    </Card>
  );
}
