import { useRef, useState, type ReactNode } from "react";
import type { SeriesStyle, Shape } from "../theme";

export function Seg<T extends string | number>(props: {
  options: { value: T; label: string; disabled?: boolean; title?: string }[];
  value: T;
  onChange: (v: T) => void;
  small?: boolean;
}) {
  return (
    <div className={`seg${props.small ? " small" : ""}`} role="tablist">
      {props.options.map((o) => (
        <button
          key={String(o.value)}
          className={o.value === props.value ? "on" : ""}
          disabled={o.disabled}
          title={o.title}
          onClick={() => props.onChange(o.value)}
          role="tab"
          aria-selected={o.value === props.value}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Card(props: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${props.className ?? ""}`}>
      {(props.title || props.right) && (
        <div className="card-head">
          <div>
            {props.title && <h3 className="card-title">{props.title}</h3>}
            {props.sub && <div className="card-sub">{props.sub}</div>}
          </div>
          {props.right}
        </div>
      )}
      {props.children}
    </section>
  );
}

/** SVG marker shape (secondary encoding next to colour). */
export function ShapeMark({ shape, x, y, r, fill, stroke }: { shape: Shape; x: number; y: number; r: number; fill: string; stroke?: string }) {
  const sw = stroke ? 2 : 0;
  switch (shape) {
    case "square":
      return <rect x={x - r * 0.85} y={y - r * 0.85} width={r * 1.7} height={r * 1.7} rx={1.5} fill={fill} stroke={stroke} strokeWidth={sw} />;
    case "diamond":
      return <path d={`M${x} ${y - r * 1.15}L${x + r * 1.15} ${y}L${x} ${y + r * 1.15}L${x - r * 1.15} ${y}Z`} fill={fill} stroke={stroke} strokeWidth={sw} />;
    case "triangle":
      return <path d={`M${x} ${y - r * 1.15}L${x + r * 1.1} ${y + r * 0.8}L${x - r * 1.1} ${y + r * 0.8}Z`} fill={fill} stroke={stroke} strokeWidth={sw} />;
    case "triangleDown":
      return <path d={`M${x} ${y + r * 1.15}L${x + r * 1.1} ${y - r * 0.8}L${x - r * 1.1} ${y - r * 0.8}Z`} fill={fill} stroke={stroke} strokeWidth={sw} />;
    case "cross":
      return <path d={`M${x - r} ${y - r}L${x + r} ${y + r}M${x + r} ${y - r}L${x - r} ${y + r}`} stroke={fill} strokeWidth={2.2} />;
    default:
      return <circle cx={x} cy={y} r={r} fill={fill} stroke={stroke} strokeWidth={sw} />;
  }
}

export function Legend({ styles }: { styles: SeriesStyle[] }) {
  return (
    <div className="legend">
      {styles.map((s) => (
        <span className="legend-item" key={s.spec}>
          <svg width="26" height="12" aria-hidden>
            <line x1="1" x2="25" y1="6" y2="6" stroke={s.color} strokeWidth="2" />
            <ShapeMark shape={s.shape} x={13} y={6} r={3.6} fill={s.color} stroke="var(--surface-1)" />
          </svg>
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** Hover tooltip positioned inside a relative container. */
export function useHover<T>() {
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ x: number; y: number; data: T } | null>(null);
  const onMove = (e: React.PointerEvent, data: T) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    setHover({ x: e.clientX - box.left, y: e.clientY - box.top, data });
  };
  return { ref, hover, onMove, clear: () => setHover(null) };
}

export function FloatTip({ x, y, width, children }: { x: number; y: number; width: number; children: ReactNode }) {
  const left = Math.min(Math.max(8, x + 14), Math.max(8, width - 230));
  return (
    <div className="tt" style={{ position: "absolute", left, top: Math.max(4, y - 10), pointerEvents: "none", minWidth: 180, zIndex: 5 }}>
      {children}
    </div>
  );
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <h3>{title}</h3>
      {children}
    </div>
  );
}
