import { useState } from "react";
import type { SuiteDetail } from "../api";
import { fmtPct } from "../format";
import type { SeriesStyle } from "../theme";
import { BarChart, HistogramChart } from "./charts";
import { Card, Seg } from "./ui";

const ACTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "STAY"];

export function DecisionOutcomes({ suite, focus }: { suite: SuiteDetail; focus: SeriesStyle }) {
  const [tab, setTab] = useState<"actions" | "confidence" | "slots">("actions");
  const a = suite.analysis[focus.spec];
  if (!a) return null;
  const totalActions = ACTIONS.reduce((s, k) => s + (a.actions[k] ?? 0), 0);
  const acc = a.accounting;
  const pctOf = (n: number) => (acc.requests ? fmtPct(n / acc.requests, 1) : "–");
  return (
    <Card
      title="Decision Outcomes"
      sub={`${focus.label} · ${acc.requests.toLocaleString()} requests · ${acc.applied.toLocaleString()} applied`}
      right={
        <Seg
          small
          value={tab}
          onChange={setTab}
          options={[
            { value: "actions", label: "Actions" },
            { value: "confidence", label: "Confidence", disabled: a.confidence.n === 0, title: a.confidence.n ? undefined : "this controller reports no confidence" },
            { value: "slots", label: "Slots" },
          ]}
        />
      }
    >
      {tab === "actions" &&
        (totalActions ? (
          <BarChart
            items={ACTIONS.map((k) => ({ key: k, value: (a.actions[k] ?? 0) / totalActions }))}
            color={focus.color}
            fmtY={(v) => `${Math.round(v * 100)}%`}
            tip={(it) => (
              <>
                <div className="tt-title">{it.key}</div>
                <div className="tt-row"><span className="tt-key" style={{ background: focus.color }} /><span className="tt-name">applied</span><span className="tt-val">{fmtPct(it.value, 1)} · {a.actions[it.key]}</span></div>
              </>
            )}
          />
        ) : (
          <div className="muted">No applied decisions logged.</div>
        ))}
      {tab === "confidence" && (
        <HistogramChart
          bins={a.confidence.histogram}
          color={focus.color}
          xLabel="reported confidence of applied answers"
          fmtX={(v) => v.toFixed(1)}
          tipLabel={(b) => (
            <>
              <div className="tt-title">{b.x0.toFixed(2)}–{b.x1.toFixed(2)}</div>
              <div className="tt-row"><span className="tt-key" style={{ background: focus.color }} /><span className="tt-name">answers</span><span className="tt-val">{b.count}</span></div>
            </>
          )}
        />
      )}
      {tab === "slots" && (
        <div className="acct">
          <span>Requests sent</span><span className="v">{acc.requests.toLocaleString()}</span><span className="p">100%</span>
          <span>Applied</span><span className="v">{acc.applied.toLocaleString()}</span><span className="p">{pctOf(acc.applied)}</span>
          <span>Superseded (newer answer already applied)</span><span className="v">{acc.superseded.toLocaleString()}</span><span className="p">{pctOf(acc.superseded)}</span>
          <span>Failed (error / invalid answer)</span><span className="v">{acc.failed.toLocaleString()}</span><span className="p">{pctOf(acc.failed)}</span>
          <span>Dropped past deadline</span><span className="v">{acc.dropped_late.toLocaleString()}</span><span className="p">{pctOf(acc.dropped_late)}</span>
          <span className="muted">Decision slots missed (never served)</span><span className="v">{acc.missed_slots.toLocaleString()}</span><span className="p" />
          <span className="muted">Decision slots delayed (served late)</span><span className="v">{acc.delayed_slots.toLocaleString()}</span><span className="p" />
        </div>
      )}
    </Card>
  );
}
