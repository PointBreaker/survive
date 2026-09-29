import { Check, ChevronDown, ChevronRight, Copy, FlaskConical, Loader2, Play, Terminal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { Job, JobSpec, Meta } from "../api";
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
 * Experiment configuration. "Run benchmark" asks the local dashboard server to
 * run exactly the command shown below (python -m benchmark.suite ...).
 */
export function Sidebar({ meta, activeJob, onRun }: {
  meta: Meta | null;
  activeJob: Job | null;
  onRun: (spec: JobSpec) => Promise<void>;
}) {
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
  const [confirming, setConfirming] = useState(false);
  const [starting, setStarting] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  // Without a token Jev can't run: start with it unchecked (still selectable, marked "no token").
  useEffect(() => {
    if (meta && !meta.jev_token) setControllers((cs) => cs.filter((c) => c !== "jev"));
  }, [meta]);

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

  const spec: JobSpec = useMemo(() => {
    const ctrls = [...controllers];
    if (matched && Number(matched) > 0) ctrls.push(`simple_avoid+${Number(matched)}ms`);
    const num = (v: string) => (v.trim() === "" ? undefined : Number(v));
    return {
      controllers: ctrls,
      param,
      levels: [...levels].sort((a, b) => a - b),
      episodes: Number(episodes || 20),
      preset,
      seed: num(seed),
      obstacles: param === "obstacle_count" ? undefined : num(obstacles),
      decision_hz: num(hz),
      max_duration: num(duration),
      max_inflight: num(inflight),
      deadline_ms: num(deadline),
      target_timeout: num(timeout),
    };
  }, [controllers, matched, param, levels, episodes, preset, seed, obstacles, hz, duration, inflight, deadline, timeout]);

  // Upper bound on Jev API calls: every episode lasting its full length, one
  // request per decision slot (episodes that end early make fewer).
  const jevRequestBound = useMemo(() => {
    if (!controllers.includes("jev")) return 0;
    const hzN = Number(hz) || 10;
    const dur = Number(duration) || 60;
    const eps = Number(episodes) || 20;
    const baseSpeed = 1;
    return Math.round(levels.reduce((sum, lv) => sum + eps * (dur / (param === "world_speed_scale" ? lv : baseSpeed)) * hzN, 0));
  }, [controllers, hz, duration, episodes, levels, param]);

  const run = async () => {
    if (jevRequestBound && !confirming) {
      setConfirming(true);
      return;
    }
    setConfirming(false);
    setStarting(true);
    setRunError(null);
    try {
      await onRun(spec);
    } catch (e) {
      setRunError(String((e as Error).message ?? e));
    } finally {
      setStarting(false);
    }
  };
  const busy = !!activeJob && (activeJob.status === "running" || activeJob.status === "cancelling");
  const canRun = !!meta?.jobs_enabled && controllers.length + (matched ? 1 : 0) > 0 && levels.length > 0 && !busy && !starting;

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
        {meta?.jobs_enabled ? (
          <>
            {confirming ? (
              <div className="confirm">
                <div className="confirm-title">Jev makes real API calls</div>
                <div className="confirm-body">
                  Up to <b className="tnum">{jevRequestBound.toLocaleString()}</b> requests (every episode at full length,
                  one per decision slot). Episodes that end early use fewer.
                </div>
                <div className="confirm-actions">
                  <button className="btn" onClick={() => setConfirming(false)}>Back</button>
                  <button className="btn primary" onClick={run}><Play size={14} /> Run anyway</button>
                </div>
              </div>
            ) : (
              <button className="btn primary block" onClick={run} disabled={!canRun} title={busy ? "a benchmark is already running" : undefined}>
                {starting ? <Loader2 size={15} className="spin" /> : <Play size={15} />}
                {busy ? "Benchmark running…" : starting ? "Starting…" : "Run benchmark"}
              </button>
            )}
            {runError && <div className="error" style={{ marginTop: 8, fontSize: 12 }}>{runError}</div>}
            <button className="btn-link" onClick={copy} disabled={!controllers.length || !levels.length}>
              {copied ? <Check size={12} /> : <Copy size={12} />} {copied ? "copied" : "copy as command"}
            </button>
          </>
        ) : (
          <button className="btn primary block" onClick={copy} disabled={!controllers.length || !levels.length}>
            {copied ? <Check size={15} /> : <Copy size={15} />} {copied ? "Copied" : "Copy benchmark command"}
          </button>
        )}
        <div className="cmd" aria-label="benchmark command"><Terminal size={11} style={{ verticalAlign: -1, marginRight: 5 }} />{command}</div>
        <div className="hint">
          {pointCount} points × {episodes || 20} episodes = {(pointCount * Number(episodes || 20)).toLocaleString()} episodes.
          {meta?.jobs_enabled
            ? " Runs locally through the same CLI; one benchmark at a time so measured latency isn't distorted."
            : " This server is read-only: run the command in your terminal; the suite appears here and fills in live."}
        </div>
      </div>
    </aside>
  );
}
