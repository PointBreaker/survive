import { Check, ChevronDown, ChevronRight, Copy, FlaskConical, Terminal } from "lucide-react";
import { useMemo, useState } from "react";
import type { Meta } from "../api";
import { controllerLabel, seriesStyles } from "../theme";
import { paramInfo } from "../format";

const LEVEL_CHOICES: Record<string, number[]> = {
  world_speed_scale: [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8],
  obstacle_count: [5, 10, 15, 20, 30, 40, 60, 80, 120, 160],
  latency_ms: [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000],
};
const DEFAULT_LEVELS: Record<string, number[]> = {
  world_speed_scale: [0.25, 0.5, 1, 2, 4, 8],
  obstacle_count: [5, 10, 20, 40, 80],
  latency_ms: [0, 100, 200, 300, 500],
};

/**
 * Experiment configuration -> the exact CLI command that runs it.
 * Phase 1 does not start benchmarks from the browser; it never pretends to.
 */
export function Sidebar({ meta }: { meta: Meta | null }) {
  const [preset, setPreset] = useState("medium");
  const [controllers, setControllers] = useState<string[]>(["jev", "simple_avoid", "greedy", "random"]);
  const [param, setParam] = useState("world_speed_scale");
  const [levels, setLevels] = useState<number[]>(DEFAULT_LEVELS.world_speed_scale);
  const [obstacles, setObstacles] = useState<string>("");
  const [hz, setHz] = useState("10");
  const [duration, setDuration] = useState("60");
  const [seed, setSeed] = useState("0");
  const [episodes, setEpisodes] = useState("20");
  const [adv, setAdv] = useState(false);
  const [inflight, setInflight] = useState("1");
  const [deadline, setDeadline] = useState("");
  const [timeout, setTimeoutS] = useState("");
  const [matched, setMatched] = useState("");
  const [copied, setCopied] = useState(false);

  const presets = meta?.presets ?? {};
  const available = meta?.benchmarkable ?? ["jev", "simple_avoid", "greedy", "random"];
  const styles = seriesStyles([...available, "human"]);

  const command = useMemo(() => {
    const specs = [...controllers];
    if (matched && Number(matched) > 0) specs.push(`simple_avoid+${Number(matched)}ms`);
    const parts = [
      "python -m benchmark.suite",
      `--controllers ${specs.join(",")}`,
      `--param ${param}`,
      `--levels ${[...levels].sort((a, b) => a - b).join(",")}`,
      `--episodes ${episodes || 20}`,
      `--preset ${preset}`,
    ];
    if (obstacles && param !== "obstacle_count") parts.push(`--obstacles ${obstacles}`);
    if (hz !== "10") parts.push(`--decision-hz ${hz}`);
    if (duration !== "60") parts.push(`--max-duration ${duration}`);
    if (seed !== "0") parts.push(`--seed ${seed}`);
    if (inflight !== "1") parts.push(`--max-inflight ${inflight}`);
    if (deadline) parts.push(`--deadline-ms ${deadline}`);
    if (timeout) parts.push(`--target-timeout ${timeout}`);
    return parts.join(" \\\n  ");
  }, [controllers, matched, param, levels, episodes, preset, obstacles, hz, duration, seed, inflight, deadline, timeout]);

  const pointCount = levels.length * (controllers.length + (matched && Number(matched) > 0 ? 1 : 0));
  const toggle = (c: string) => setControllers((cs) => (cs.includes(c) ? cs.filter((x) => x !== c) : [...cs, c]));
  const toggleLevel = (v: number) => setLevels((ls) => (ls.includes(v) ? ls.filter((x) => x !== v) : [...ls, v]));
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command.replace(/ \\\n {2}/g, " "));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable: the command stays selectable */
    }
  };

  return (
    <aside className="sidebar">
      <h2><FlaskConical size={17} /> Benchmark</h2>
      <p className="lede">Configure a measured frontier: controllers × one swept pressure × paired seeds.</p>

      <div className="side-section">
        <div className="side-label">Preset</div>
        {Object.entries(presets).map(([name, cfg]) => (
          <button key={name} className={`preset${preset === name ? " active" : ""}`} onClick={() => setPreset(name)}>
            <span className="radio" />
            <span>
              <div className="p-name">{name[0].toUpperCase() + name.slice(1)}</div>
              <div className="p-sub">{cfg.obstacle_count} obstacles · speed {cfg.obstacle_speed_min}–{cfg.obstacle_speed_max}</div>
            </span>
          </button>
        ))}
      </div>

      <div className="side-section">
        <div className="side-label">Controllers</div>
        {styles.map((s) => {
          const disabled = s.spec === "human";
          return (
            <label key={s.spec} className={`check-row${disabled ? " disabled" : ""}`} title={disabled ? "Human play is interactive (python main.py); it cannot run in a batch benchmark" : undefined}>
              <input type="checkbox" disabled={disabled} checked={!disabled && controllers.includes(s.spec)} onChange={() => toggle(s.spec)} />
              <span className="dot" style={{ background: disabled ? "var(--text-3)" : s.color }} />
              {controllerLabel(s.spec)}
              {s.spec === "jev" && meta && !meta.jev_token && <span className="badge-warn" style={{ marginLeft: "auto" }}>no token</span>}
            </label>
          );
        })}
      </div>

      <div className="side-section">
        <div className="side-label">Swept pressure</div>
        <select
          className="select"
          style={{ width: "100%" }}
          value={param}
          onChange={(e) => {
            setParam(e.target.value);
            setLevels(DEFAULT_LEVELS[e.target.value] ?? []);
          }}
        >
          {Object.keys(LEVEL_CHOICES).map((p) => (
            <option key={p} value={p}>{paramInfo(p).label}</option>
          ))}
        </select>
        <div className="levels">
          {(LEVEL_CHOICES[param] ?? []).map((v) => (
            <button key={v} className={`level-chip${levels.includes(v) ? " on" : ""}`} onClick={() => toggleLevel(v)}>
              {paramInfo(param).unit(v)}
            </button>
          ))}
        </div>
      </div>

      <div className="side-section">
        <div className="side-label">Parameters</div>
        {param !== "obstacle_count" && (
          <div className="param-row">
            <label>Obstacles</label>
            <input className="field" placeholder={String(presets[preset]?.obstacle_count ?? "")} value={obstacles} onChange={(e) => setObstacles(e.target.value.replace(/\D/g, ""))} />
          </div>
        )}
        <div className="param-row"><label>Decision rate (Hz)</label><input className="field" value={hz} onChange={(e) => setHz(e.target.value)} /></div>
        <div className="param-row"><label>Episode length (world s)</label><input className="field" value={duration} onChange={(e) => setDuration(e.target.value)} /></div>
        <div className="param-row"><label>Base seed</label><input className="field" value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} /></div>
        <div className="param-row"><label>Episodes per point</label><input className="field" value={episodes} onChange={(e) => setEpisodes(e.target.value.replace(/\D/g, ""))} /></div>
      </div>

      <div className="side-section">
        <button className="adv-toggle" onClick={() => setAdv(!adv)}>
          {adv ? <ChevronDown size={14} /> : <ChevronRight size={14} />} Advanced
        </button>
        {adv && (
          <>
            <div className="param-row"><label>Max requests in flight</label><input className="field" value={inflight} onChange={(e) => setInflight(e.target.value.replace(/\D/g, ""))} /></div>
            <div className="param-row"><label>Decision deadline (ms)</label><input className="field" placeholder="none" value={deadline} onChange={(e) => setDeadline(e.target.value.replace(/[^\d.]/g, ""))} /></div>
            <div className="param-row"><label>Target timeout (world s)</label><input className="field" placeholder="15" value={timeout} onChange={(e) => setTimeoutS(e.target.value.replace(/[^\d.]/g, ""))} /></div>
            <div className="param-row" title="Adds SimpleAvoid with this much simulated latency: a latency-matched baseline">
              <label>Latency-matched SimpleAvoid (ms)</label>
              <input className="field" placeholder="off" value={matched} onChange={(e) => setMatched(e.target.value.replace(/\D/g, ""))} />
            </div>
          </>
        )}
      </div>

      <div className="side-section">
        <button className="btn primary block" onClick={copy} disabled={!controllers.length || !levels.length}>
          {copied ? <Check size={15} /> : <Copy size={15} />} {copied ? "Copied" : "Copy benchmark command"}
        </button>
        <div className="cmd" aria-label="benchmark command"><Terminal size={11} style={{ verticalAlign: -1, marginRight: 5 }} />{command}</div>
        <div className="hint">
          {pointCount} points × {episodes || 20} episodes. Run it in your terminal; the suite appears in the run selector
          as soon as it starts and fills in live.
        </div>
      </div>
    </aside>
  );
}
