// Small hand-rolled SVG charts: exact numeric axes, recessive grid, thin marks,
// 2px surface gaps between adjacent fills, and a hover tooltip on every mark.
import { useEffect, useState, type ReactNode, type RefObject } from "react";
import type { Histogram } from "../api";
import { FloatTip, useHover } from "./ui";

const AXIS = "var(--text-3)";
const GRID = "var(--grid)";

/** Draw in real pixels: the SVG viewBox follows the container width. */
export function useWidth(ref: RefObject<HTMLDivElement>, fallback = 480): number {
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(200, el.clientWidth)));
    ro.observe(el);
    setW(Math.max(200, el.clientWidth));
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

function niceStep(range: number, target = 5): number {
  const raw = range / target;
  const p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

export interface RefLine {
  x: number;
  label: string;
  color: string;
  dashed?: boolean;
}

export function HistogramChart(props: {
  bins: Histogram[];
  color: string;
  height?: number;
  xLabel: string;
  refLines?: RefLine[];
  fmtX: (v: number) => string;
  tipLabel: (b: Histogram) => ReactNode;
}) {
  const { bins, color } = props;
  const H = props.height ?? 190;
  const hov = useHover<Histogram>();
  const W = useWidth(hov.ref);
  const m = { l: 40, r: 12, t: 22, b: 34 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  if (!bins.length) return null;
  const x0 = bins[0].x0;
  const x1 = bins[bins.length - 1].x1;
  const ymax = niceMax(Math.max(...bins.map((b) => b.count)));
  const X = (v: number) => m.l + ((v - x0) / (x1 - x0)) * iw;
  const Y = (v: number) => m.t + ih - (v / ymax) * ih;
  const ticks = [0, ymax / 2, ymax];
  const xstep = niceStep(x1 - x0, Math.max(3, Math.floor(iw / 80)));
  const xticks: number[] = [];
  for (let v = Math.ceil(x0 / xstep) * xstep; v <= x1 + 1e-9; v += xstep) xticks.push(v);
  return (
    <div className="chart-wrap" ref={hov.ref} onPointerLeave={hov.clear}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="histogram">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} stroke={GRID} />
            <text x={m.l - 6} y={Y(t) + 3.5} fill={AXIS} fontSize="10.5" textAnchor="end" className="tnum">{Math.round(t)}</text>
          </g>
        ))}
        {bins.map((b, i) => {
          const bx = X(b.x0) + 1;
          const bw = Math.max(1, X(b.x1) - X(b.x0) - 2);
          const active = hov.hover?.data === b;
          return (
            <g key={i} onPointerMove={(e) => hov.onMove(e, b)}>
              <rect x={X(b.x0)} y={m.t} width={X(b.x1) - X(b.x0)} height={ih} fill="transparent" />
              {b.count > 0 && (
                <rect x={bx} y={Y(b.count)} width={bw} height={Y(0) - Y(b.count)} rx={1.5} fill={color} opacity={active ? 1 : 0.85} />
              )}
            </g>
          );
        })}
        {(props.refLines ?? []).map((r, i) =>
          r.x >= x0 && r.x <= x1 ? (
            <g key={r.label}>
              <line x1={X(r.x)} x2={X(r.x)} y1={m.t - 4} y2={m.t + ih} stroke={r.color} strokeWidth={1.4} strokeDasharray={r.dashed ? "4 3" : undefined} />
              <text
                x={X(r.x) > W - m.r - 110 ? X(r.x) - 4 : X(r.x) + 4}
                textAnchor={X(r.x) > W - m.r - 110 ? "end" : "start"}
                y={m.t + 6 + i * 12}
                fill={r.color}
                fontSize="10.5"
                fontWeight={600}
              >
                {r.label}
              </text>
            </g>
          ) : null,
        )}
        <line x1={m.l} x2={W - m.r} y1={Y(0)} y2={Y(0)} stroke="var(--border-strong)" />
        {xticks.map((t) => (
          <text key={t} x={X(t)} y={H - m.b + 15} fill={AXIS} fontSize="10.5" textAnchor="middle" className="tnum">{props.fmtX(t)}</text>
        ))}
        <text x={m.l + iw / 2} y={H - 4} fill={AXIS} fontSize="11" textAnchor="middle">{props.xLabel}</text>
      </svg>
      {hov.hover && (
        <FloatTip x={hov.hover.x} y={hov.hover.y} width={hov.ref.current?.clientWidth ?? 400}>
          {props.tipLabel(hov.hover.data)}
        </FloatTip>
      )}
    </div>
  );
}

export function BarChart(props: {
  items: { key: string; value: number; label?: string }[];
  color: string;
  height?: number;
  fmtY: (v: number) => string;
  tip: (item: { key: string; value: number }) => ReactNode;
}) {
  const H = props.height ?? 190;
  const hov = useHover<{ key: string; value: number }>();
  const W = useWidth(hov.ref);
  const m = { l: 40, r: 10, t: 12, b: 26 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const ymax = niceMax(Math.max(0.0001, ...props.items.map((i) => i.value)));
  const step = iw / props.items.length;
  const bw = Math.min(34, step - 8);
  const Y = (v: number) => m.t + ih - (v / ymax) * ih;
  return (
    <div className="chart-wrap" ref={hov.ref} onPointerLeave={hov.clear}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="bar chart">
        {[0, ymax / 2, ymax].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} stroke={GRID} />
            <text x={m.l - 6} y={Y(t) + 3.5} fill={AXIS} fontSize="10.5" textAnchor="end" className="tnum">{props.fmtY(t)}</text>
          </g>
        ))}
        {props.items.map((it, i) => {
          const cx = m.l + step * i + step / 2;
          const active = hov.hover?.data.key === it.key;
          return (
            <g key={it.key} onPointerMove={(e) => hov.onMove(e, it)}>
              <rect x={cx - step / 2} y={m.t} width={step} height={ih} fill="transparent" />
              {it.value > 0 && (
                <path
                  d={roundedTop(cx - bw / 2, Y(it.value), bw, Y(0) - Y(it.value), 4)}
                  fill={props.color}
                  opacity={active ? 1 : 0.85}
                />
              )}
              <text x={cx} y={H - 8} fill={active ? "var(--text-1)" : AXIS} fontSize="10.5" textAnchor="middle">{it.label ?? it.key}</text>
            </g>
          );
        })}
        <line x1={m.l} x2={W - m.r} y1={Y(0)} y2={Y(0)} stroke="var(--border-strong)" />
      </svg>
      {hov.hover && (
        <FloatTip x={hov.hover.x} y={hov.hover.y} width={hov.ref.current?.clientWidth ?? 400}>
          {props.tip(hov.hover.data)}
        </FloatTip>
      )}
    </div>
  );
}

function roundedTop(x: number, y: number, w: number, h: number, r: number): string {
  const rr = Math.min(r, w / 2, h);
  return `M${x} ${y + h}V${y + rr}Q${x} ${y} ${x + rr} ${y}H${x + w - rr}Q${x + w} ${y} ${x + w} ${y + rr}V${y + h}Z`;
}

export interface StackSeg {
  key: string;
  label: string;
  color: string;
}

/** 100% stacked columns, one per category; 2px surface gap between segments. */
export function StackedColumns(props: {
  columns: { key: string; label: string; total: number; parts: Record<string, number> }[];
  segments: StackSeg[];
  height?: number;
  xLabel: string;
}) {
  const H = props.height ?? 190;
  const hov = useHover<(typeof props.columns)[number]>();
  const W = useWidth(hov.ref);
  const m = { l: 38, r: 10, t: 10, b: 34 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const step = iw / Math.max(1, props.columns.length);
  const bw = Math.min(40, step - 10);
  const Y = (f: number) => m.t + ih - f * ih;
  return (
    <div className="chart-wrap" ref={hov.ref} onPointerLeave={hov.clear}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="stacked columns">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} stroke={GRID} />
            <text x={m.l - 6} y={Y(t) + 3.5} fill={AXIS} fontSize="10.5" textAnchor="end">{Math.round(t * 100)}%</text>
          </g>
        ))}
        {props.columns.map((c, i) => {
          const cx = m.l + step * i + step / 2;
          let acc = 0;
          return (
            <g key={c.key} onPointerMove={(e) => hov.onMove(e, c)}>
              <rect x={cx - step / 2} y={m.t} width={step} height={ih} fill="transparent" />
              {props.segments.map((s) => {
                const n = c.parts[s.key] ?? 0;
                if (!n || !c.total) return null;
                const f0 = acc / c.total;
                acc += n;
                const f1 = acc / c.total;
                const y = Y(f1);
                const h = Math.max(0, Y(f0) - Y(f1) - 2);
                return <rect key={s.key} x={cx - bw / 2} y={y} width={bw} height={h} rx={2} fill={s.color} opacity={0.88} />;
              })}
              <text x={cx} y={H - m.b + 15} fill={AXIS} fontSize="10.5" textAnchor="middle" className="tnum">{c.label}</text>
            </g>
          );
        })}
        <text x={m.l + iw / 2} y={H - 4} fill={AXIS} fontSize="11" textAnchor="middle">{props.xLabel}</text>
      </svg>
      {hov.hover && (
        <FloatTip x={hov.hover.x} y={hov.hover.y} width={hov.ref.current?.clientWidth ?? 400}>
          <div className="tt-title">{hov.hover.data.label} · {hov.hover.data.total} episodes</div>
          {props.segments.map((s) => {
            const n = hov.hover!.data.parts[s.key] ?? 0;
            return (
              <div className="tt-row" key={s.key}>
                <span className="tt-key" style={{ background: s.color }} />
                <span className="tt-name">{s.label}</span>
                <span className="tt-val">{hov.hover!.data.total ? Math.round((n / hov.hover!.data.total) * 100) : 0}% · {n}</span>
              </div>
            );
          })}
        </FloatTip>
      )}
    </div>
  );
}
