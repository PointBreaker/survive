import {
  Area,
  CartesianGrid,
  ComposedChart,
  Customized,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Aggregate, SuiteDetail } from "../api";
import { fmtMs, fmtPct, fmtS, paramInfo } from "../format";
import type { SeriesStyle } from "../theme";
import { ShapeMark } from "./ui";

export type Metric = "success" | "survival" | "targets" | "latency";

export const METRICS: { value: Metric; label: string }[] = [
  { value: "success", label: "Success Rate" },
  { value: "survival", label: "Avg Survival" },
  { value: "targets", label: "Targets Reached" },
  { value: "latency", label: "Latency" },
];

function metricValue(p: Aggregate, metric: Metric, p50?: number | null): number | null {
  switch (metric) {
    case "success":
      return p.success_rate;
    case "survival":
      return p.mean_survival_time;
    case "targets":
      return p.mean_targets;
    case "latency":
      return p50 ?? p.mean_latency_ms;
  }
}

function fmtMetric(v: number | null | undefined, metric: Metric): string {
  if (v === null || v === undefined) return "–";
  return metric === "success" ? fmtPct(v) : metric === "survival" ? fmtS(v) : metric === "latency" ? fmtMs(v) : v.toFixed(1);
}

export function FrontierChart({ suite, metric, styles }: { suite: SuiteDetail; metric: Metric; styles: SeriesStyle[] }) {
  const { manifest, summary, points_extra } = suite;
  const pinfo = paramInfo(manifest.param);
  const lookup = new Map<string, Aggregate>();
  for (const s of styles) for (const p of summary.controllers[s.spec]?.points ?? []) lookup.set(`${s.spec}@${p.level}`, p);

  const rows = manifest.levels.map((lv) => {
    const row: Record<string, unknown> = { level: lv, label: pinfo.unit(lv) };
    for (const s of styles) {
      const p = lookup.get(`${s.spec}@${lv}`);
      if (!p) continue;
      const p50 = points_extra[s.spec]?.[String(lv)]?.p50_latency_ms ?? null;
      row[s.spec] = metricValue(p, metric, p50);
      if (metric === "success") row[`${s.spec}__band`] = p.success_rate_ci95;
    }
    return row;
  });

  const yMax =
    metric === "success"
      ? 1
      : metric === "survival"
        ? manifest.config.max_duration
        : Math.max(1, ...rows.flatMap((r) => styles.map((s) => (typeof r[s.spec] === "number" ? (r[s.spec] as number) : 0)))) * 1.1;

  const yFmt = (v: number) =>
    metric === "success" ? `${Math.round(v * 100)}%` : metric === "survival" ? `${Math.round(v)} s` : metric === "latency" ? `${Math.round(v)} ms` : `${Math.round(v)}`;

  const lastIndex = rows.length - 1;

  return (
    <div style={{ width: "100%", height: 410 }}>
      <ResponsiveContainer>
        <ComposedChart data={rows} margin={{ top: 10, right: 150, bottom: 22, left: 4 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: "var(--text-3)", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border-strong)" }}
            label={{ value: `${pinfo.label}${manifest.param === "world_speed_scale" ? " (world time ÷ real time)" : ""}`, position: "insideBottom", offset: -14, fill: "var(--text-3)", fontSize: 11 }}
            padding={{ left: 18, right: 18 }}
          />
          <YAxis
            domain={[0, yMax]}
            tickFormatter={yFmt}
            tick={{ fill: "var(--text-3)", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
            ticks={metric === "success" ? [0, 0.25, 0.5, 0.75, 1] : undefined}
          />
          {metric === "success" && (
            <ReferenceLine
              y={0.5}
              stroke="var(--text-3)"
              strokeDasharray="4 4"
              label={{ value: "50% · D50 threshold", position: "insideTopLeft", fill: "var(--text-3)", fontSize: 10.5 }}
            />
          )}
          {metric === "success" &&
            styles.map((s) => (
              <Area
                key={`${s.spec}-band`}
                dataKey={`${s.spec}__band`}
                stroke="none"
                fill={s.color}
                fillOpacity={0.05}
                isAnimationActive={false}
                activeDot={false}
                connectNulls
              />
            ))}
          {styles.map((s) => (
            <Line
              key={s.spec}
              dataKey={s.spec}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              isAnimationActive={false}
              connectNulls
              dot={(p: { cx?: number; cy?: number; index?: number }) =>
                p.cx === undefined || p.cy === undefined ? <g key={`${s.spec}-${p.index}`} /> : (
                  <ShapeMark key={`${s.spec}-${p.index}`} shape={s.shape} x={p.cx} y={p.cy} r={4.2} fill={s.color} stroke="var(--surface-1)" />
                )
              }
              activeDot={(p: { cx?: number; cy?: number }) =>
                p.cx === undefined || p.cy === undefined ? <g /> : (
                  <ShapeMark shape={s.shape} x={p.cx} y={p.cy} r={5.6} fill={s.color} stroke="var(--surface-1)" />
                )
              }
            />
          ))}
          <Tooltip
            cursor={{ stroke: "var(--border-strong)", strokeWidth: 1 }}
            content={({ active, label }) =>
              active ? <FrontierTip suite={suite} styles={styles} label={String(label)} metric={metric} lookup={lookup} /> : null
            }
          />
          <Customized component={(p: unknown) => <EndLabels chart={p} styles={styles} rows={rows} lastIndex={lastIndex} metric={metric} />} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Direct labels at each line's last point, nudged apart so they never overlap. */
function EndLabels({ chart, styles, rows, lastIndex, metric }: {
  chart: unknown; styles: SeriesStyle[]; rows: Record<string, unknown>[]; lastIndex: number; metric: Metric;
}) {
  const c = chart as {
    xAxisMap?: Record<string, { scale: ((v: string) => number) & { bandwidth?: () => number } }>;
    yAxisMap?: Record<string, { scale: (v: number) => number }>;
    offset?: { left: number; top: number; width: number; height: number };
  };
  const xa = c.xAxisMap && Object.values(c.xAxisMap)[0];
  const ya = c.yAxisMap && Object.values(c.yAxisMap)[0];
  if (!xa || !ya || lastIndex < 0) return null;
  const items = styles
    .map((s) => {
      // last level that has a value for this series
      for (let i = lastIndex; i >= 0; i--) {
        const v = rows[i][s.spec];
        if (typeof v === "number") {
          const x = xa.scale(rows[i].label as string) + (xa.scale.bandwidth ? xa.scale.bandwidth() / 2 : 0);
          return { s, x, y: ya.scale(v), v };
        }
      }
      return null;
    })
    .filter((i): i is { s: SeriesStyle; x: number; y: number; v: number } => i !== null)
    .sort((a, b) => a.y - b.y);
  const gap = 15;
  const top = c.offset?.top ?? 0;
  const bottom = top + (c.offset?.height ?? 300);
  const lx = (c.offset ? c.offset.left + c.offset.width : Math.max(...items.map((i) => i.x))) + 14;
  for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < gap) items[i].y = items[i - 1].y + gap;
  // keep the stack inside the plot: shift up if the last label overflows
  const overflow = items.length ? items[items.length - 1].y - bottom : 0;
  if (overflow > 0) for (const it of items) it.y -= overflow;
  for (let i = 0; i < items.length; i++) items[i].y = Math.max(items[i].y, top + i * gap);
  return (
    <g>
      {items.map(({ s, x, y, v }) => (
        <g key={s.spec}>
          <line x1={x + 6} x2={lx - 4} y1={ya.scale(v)} y2={y} stroke="var(--border-strong)" />
          <ShapeMark shape={s.shape} x={lx + 3} y={y} r={3.4} fill={s.color} />
          <text x={lx + 12} y={y + 3.8} fill="var(--text-2)" fontSize="11.5" fontWeight={500}>
            {s.label} <tspan fill="var(--text-3)">{fmtMetric(v, metric)}</tspan>
          </text>
        </g>
      ))}
    </g>
  );
}

function FrontierTip({ suite, styles, label, metric, lookup }: {
  suite: SuiteDetail; styles: SeriesStyle[]; label: string; metric: Metric; lookup: Map<string, Aggregate>;
}) {
  const pinfo = paramInfo(suite.manifest.param);
  const lv = suite.manifest.levels.find((l) => pinfo.unit(l) === label);
  if (lv === undefined) return null;
  return (
    <div className="tt">
      <div className="tt-title">{pinfo.label} {label}</div>
      {styles.map((s) => {
        const p = lookup.get(`${s.spec}@${lv}`);
        if (!p) return null;
        const p50 = suite.points_extra[s.spec]?.[String(lv)]?.p50_latency_ms ?? null;
        const [lo, hi] = p.success_rate_ci95;
        const main = metricValue(p, metric, p50);
        return (
          <div className="tt-row" key={s.spec}>
            <span className="tt-key" style={{ background: s.color }} />
            <span className="tt-name">{s.label}</span>
            <span className="tt-val">{fmtMetric(main, metric)}</span>
            <span className="tt-detail">
              success {fmtPct(p.success_rate)} (95% CI {fmtPct(lo)}–{fmtPct(hi)}) · {p.episodes} ep ·
              survival {fmtS(p.mean_survival_time)} · targets {p.mean_targets?.toFixed(1) ?? "–"} · p50 {fmtMs(p50)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
