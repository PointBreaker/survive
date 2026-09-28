import type { SuiteDetail } from "../api";
import { fmtMs, fmtPct, fmtThreshold, paramInfo } from "../format";
import type { SeriesStyle } from "../theme";
import { referenceLevel } from "./KeyResults";

/** Purely computed statements about this benchmark. No judgements. */
export function Facts({ suite, focus }: { suite: SuiteDetail; focus: SeriesStyle }) {
  const { manifest, summary, analysis, decision_period_ms } = suite;
  const pinfo = paramInfo(manifest.param);
  const a = analysis[focus.spec];
  const ref = referenceLevel(suite);
  const lat = a?.latency.by_level[String(ref)] ?? a?.latency.pooled;
  const d50 = fmtThreshold(summary.controllers[focus.spec]?.thresholds.D50, manifest.param);
  const seeds = manifest.seeds;
  return (
    <div className="facts" aria-label="Observed in this benchmark">
      <div className="fact">
        <div className="fact-k">{focus.label} · empirical D50 {pinfo.short}</div>
        <div className="fact-v">{d50.text}</div>
      </div>
      <div className="fact">
        <div className="fact-k">{focus.label} · median decision latency @ {pinfo.unit(ref)}</div>
        <div className="fact-v">{fmtMs(lat?.p50)}<small>p95 {fmtMs(lat?.p95)}</small></div>
      </div>
      <div className="fact">
        <div className="fact-k">Decision period at {manifest.config.decision_hz} Hz</div>
        <div className="fact-v">{fmtMs(decision_period_ms)}<small>max {manifest.config.max_inflight ?? 1} in flight</small></div>
      </div>
      <div className="fact">
        <div className="fact-k">{focus.label} · answers slower than one decision period @ {pinfo.unit(ref)}</div>
        <div className="fact-v">{lat?.n ? fmtPct(lat.over_period) : "–"}<small>{lat?.n ? `of ${lat.n.toLocaleString()} answers` : "no decision events"}</small></div>
      </div>
      <div className="fact">
        <div className="fact-k">Paired seeds per point</div>
        <div className="fact-v">{manifest.episodes_per_point}<small>seeds {seeds[0]}–{seeds[seeds.length - 1]}, identical across controllers</small></div>
      </div>
    </div>
  );
}
