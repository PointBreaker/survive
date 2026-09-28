import { CheckCircle2, Clock, OctagonX } from "lucide-react";
import { useState } from "react";
import type { SuiteDetail } from "../api";
import { paramInfo } from "../format";
import { OUTCOME_COLOR, type SeriesStyle } from "../theme";
import { StackedColumns } from "./charts";
import { Card, Seg } from "./ui";

// The environment defines exactly these episode endings (arena/environment.py).
const OUTCOMES = [
  { key: "survived", label: "Survived", icon: CheckCircle2 },
  { key: "collision", label: "Collision", icon: OctagonX },
  { key: "target_timeout", label: "Target timeout", icon: Clock },
];

export function FailureAnalysis({ suite, focus }: { suite: SuiteDetail; focus: SeriesStyle }) {
  const [tab, setTab] = useState<"reason" | "level">("reason");
  const a = suite.analysis[focus.spec];
  const pinfo = paramInfo(suite.manifest.param);
  if (!a) return null;
  const total = a.episodes;
  const counts: Record<string, number> = {
    survived: total - a.failures.episodes_failed,
    collision: a.failures.total.collision ?? 0,
    target_timeout: a.failures.total.target_timeout ?? 0,
  };
  const other = Object.entries(a.failures.total).filter(([k]) => !(k in counts));
  const columns = suite.manifest.levels
    .filter((lv) => a.failures.by_level[`${lv}`] || a.failures.by_level[pinfo.unit(lv)] || a.failures.by_level[String(lv)])
    .map((lv) => {
      const b = a.failures.by_level[String(lv)] ?? a.failures.by_level[`${lv}`];
      return { key: String(lv), label: pinfo.unit(lv), total: b?.episodes ?? 0, parts: { survived: b?.successes ?? 0, ...(b?.reasons ?? {}) } };
    });
  return (
    <Card
      title="Failure Analysis"
      sub={`${focus.label} · ${total} episodes · ${a.failures.episodes_failed} failed`}
      right={<Seg small value={tab} onChange={setTab} options={[{ value: "reason", label: "By reason" }, { value: "level", label: `By ${pinfo.short}` }]} />}
    >
      {tab === "reason" ? (
        <>
          <div className="outcome-bar" role="img" aria-label="episode outcomes">
            {OUTCOMES.map((o) => (counts[o.key] ? <div key={o.key} style={{ flex: counts[o.key], background: OUTCOME_COLOR[o.key] }} /> : null))}
          </div>
          <div className="outcome-list">
            {OUTCOMES.map((o) => {
              const Icon = o.icon;
              return (
                <div className="outcome-item" key={o.key}>
                  <Icon size={15} color={OUTCOME_COLOR[o.key]} />
                  <span>{o.label}</span>
                  <span className="pct">{total ? Math.round((counts[o.key] / total) * 100) : 0}%</span>
                  <span className="n">{counts[o.key]}</span>
                </div>
              );
            })}
            {other.map(([k, n]) => (
              <div className="outcome-item" key={k}><span /><span>{k}</span><span className="pct">{Math.round((n / total) * 100)}%</span><span className="n">{n}</span></div>
            ))}
          </div>
          <div className="note">
            Collision and target timeout are the only ways an episode can fail. Failed or late decisions never end an
            episode (the previous action continues); they are counted under Decision Outcomes.
          </div>
        </>
      ) : (
        <StackedColumns
          columns={columns}
          xLabel={pinfo.label}
          segments={OUTCOMES.map((o) => ({ key: o.key, label: o.label, color: OUTCOME_COLOR[o.key] }))}
        />
      )}
    </Card>
  );
}
