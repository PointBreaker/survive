import type { Threshold } from "./api";

export const PARAMS: Record<string, { label: string; short: string; unit: (v: number) => string }> = {
  world_speed_scale: { label: "World Speed", short: "world speed", unit: (v) => `${fmtNum(v)}×` },
  obstacle_count: { label: "Obstacle Count", short: "obstacles", unit: (v) => `${Math.round(v)}` },
  latency_ms: { label: "Added Latency", short: "added latency", unit: (v) => `${fmtNum(v)} ms` },
  obstacle_speed_max: { label: "Obstacle Speed", short: "obstacle speed", unit: (v) => `${fmtNum(v)}` },
  spawn_rate: { label: "Spawn Rate", short: "spawn rate", unit: (v) => `${fmtNum(v)}/s` },
};

export function paramInfo(p: string) {
  return PARAMS[p] ?? { label: p, short: p, unit: (v: number) => fmtNum(v) };
}

export function fmtNum(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "–";
  const s = v.toFixed(digits);
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
}

export function fmtPct(v: number | null | undefined, digits = 0): string {
  return v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`;
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined) return "–";
  if (v < 1) return `${v.toFixed(2)} ms`;
  if (v < 10) return `${v.toFixed(1)} ms`;
  return `${Math.round(v)} ms`;
}

export function fmtS(v: number | null | undefined): string {
  return v === null || v === undefined ? "–" : `${v.toFixed(1)} s`;
}

/** D-threshold as measured: never presented as more precise than it is. */
export function fmtThreshold(t: Threshold | undefined, param: string): { text: string; note: string } {
  const unit = paramInfo(param).unit;
  if (!t || t.kind === "not_measured" || t.value === null) return { text: "Not measured", note: "" };
  if (t.kind === "above_range") return { text: `≥ ${unit(t.value)}`, note: "never fell below 50% in tested range" };
  if (t.kind === "below_range")
    return t.value > 0
      ? { text: `< ${unit(t.value)}`, note: "already below 50% at the lowest tested level" }
      : { text: "Below range", note: `below 50% even at ${unit(t.value)}` };
  return { text: unit(t.value), note: "interpolated between measured levels" };
}

export function reasonLabel(r: string): string {
  return { collision: "Collision", target_timeout: "Target timeout", max_duration: "Survived", target_goal: "Goal reached" }[r] ?? r;
}
