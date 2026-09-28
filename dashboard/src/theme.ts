// Series identity. Order + hexes validated with the dataviz palette checker
// on the dark surface (#0f131a): blue, amber, green, coral, violet, magenta.
// Legend order follows this list so amber never sits next to coral.
// Green/coral land in the CVD 6-8 band -> lines also carry distinct marker
// shapes and direct end labels (secondary encoding).

export const SERIES_ORDER = ["jev", "greedy", "simple_avoid", "random"];
const FIXED: Record<string, string> = {
  jev: "#3987e5",
  greedy: "#c98500",
  simple_avoid: "#199e70",
  random: "#e66767",
};
const EXTRA = ["#9085e9", "#d55181"];
const OVERFLOW = "#7d8594";
export const SHAPES = ["circle", "square", "diamond", "triangle", "triangleDown", "cross"] as const;
export type Shape = (typeof SHAPES)[number];

// Status palette (reserved; always shown with icon + label).
export const STATUS = { good: "#0ca30c", warning: "#fab219", serious: "#ec835a", critical: "#d03b3b" };
export const OUTCOME_COLOR: Record<string, string> = {
  survived: STATUS.good,
  collision: STATUS.critical,
  target_timeout: STATUS.warning,
};

export interface SeriesStyle {
  spec: string;
  color: string;
  shape: Shape;
  label: string;
}

/** Stable styles: identity decides colour, never rank; extras in manifest order. */
export function seriesStyles(specs: string[]): SeriesStyle[] {
  const known = SERIES_ORDER.filter((s) => specs.includes(s));
  const extras = specs.filter((s) => !SERIES_ORDER.includes(s));
  return [...known, ...extras].map((spec, i) => {
    const color = FIXED[spec] ?? EXTRA[extras.indexOf(spec)] ?? OVERFLOW;
    return { spec, color, shape: SHAPES[i % SHAPES.length], label: controllerLabel(spec) };
  });
}

const NAMES: Record<string, string> = {
  jev: "Jev",
  simple_avoid: "SimpleAvoid",
  greedy: "Greedy",
  random: "Random",
  human: "Human",
};

export function controllerLabel(spec: string): string {
  const m = spec.match(/^([a-z_]+)(?:\+(\d+(?:\.\d+)?)ms)?$/);
  if (!m) return spec;
  const base = NAMES[m[1]] ?? m[1];
  return m[2] ? `${base} +${m[2]} ms` : base;
}
