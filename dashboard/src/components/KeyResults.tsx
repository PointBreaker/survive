import type { SuiteDetail } from "../api";
import { fmtMs, fmtS, fmtThreshold, paramInfo } from "../format";
import type { SeriesStyle } from "../theme";
import { Card } from "./ui";

/** Level used for "at the reference condition" stats: the base config value if tested. */
export function referenceLevel(suite: SuiteDetail): number {
  const { manifest } = suite;
  const base = manifest.param === "latency_ms" ? manifest.added_latency_ms : (manifest.config[manifest.param] as number | null);
  if (base !== null && base !== undefined && manifest.levels.includes(base)) return base;
  return manifest.levels[Math.floor(manifest.levels.length / 2)];
}

export function KeyResults({ suite, styles }: { suite: SuiteDetail; styles: SeriesStyle[] }) {
  const { manifest, summary, analysis } = suite;
  const pinfo = paramInfo(manifest.param);
  const ref = referenceLevel(suite);
  return (
    <Card
      title="Key Results"
      sub={`Empirical D50: largest ${pinfo.short} with ≥ 50% observed success`}
    >
      <div className="kr-grid">
        {styles.map((s) => {
          const c = summary.controllers[s.spec];
          const d50 = fmtThreshold(c?.thresholds.D50, manifest.param);
          const lat = analysis[s.spec]?.latency.by_level[String(ref)] ?? analysis[s.spec]?.latency.pooled;
          const refPoint = c?.points.find((p) => p.level === ref);
          return (
            <div className="kr" key={s.spec} style={{ ["--c" as string]: s.color }} title={d50.note}>
              <div className="kr-name">
                <span className="dot" style={{ background: s.color }} />
                {s.label}
              </div>
              <div className={`kr-value${d50.text === "Not measured" ? " muted" : ""}`}>{d50.text}</div>
              <div className="kr-label">D50 {pinfo.short}{d50.note ? ` · ${d50.note}` : ""}</div>
              <div className="kr-stats">
                <span>p50 latency @ {pinfo.unit(ref)}</span><span>{fmtMs(lat?.p50)}</span>
                <span>p95 latency @ {pinfo.unit(ref)}</span><span>{fmtMs(lat?.p95)}</span>
                <span>avg survival @ {pinfo.unit(ref)}</span><span>{fmtS(refPoint?.mean_survival_time)}</span>
                <span>targets / ep @ {pinfo.unit(ref)}</span><span>{refPoint?.mean_targets?.toFixed(1) ?? "–"}</span>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
