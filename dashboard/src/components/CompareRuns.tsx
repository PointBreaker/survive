import { CheckCircle2, Clock, OctagonX, Pause, Play, ShieldAlert, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Frame, type Replay, type SuiteDetail } from "../api";
import { fmtMs, paramInfo } from "../format";
import { OUTCOME_COLOR, type SeriesStyle } from "../theme";
import { Card, Seg } from "./ui";

const SPEEDS = [0.25, 0.5, 1, 2, 4];

/** Same-seed comparison: identical initial conditions, one shared scrubber. */
export function CompareRuns({ suite, styles, initialLevel }: { suite: SuiteDetail; styles: SeriesStyle[]; initialLevel: number }) {
  const { manifest } = suite;
  const pinfo = paramInfo(manifest.param);
  const [level, setLevel] = useState(initialLevel);
  const [seed, setSeed] = useState(manifest.seeds[0]);
  const [shown, setShown] = useState<string[]>(styles.slice(0, 2).map((s) => s.spec));
  const [replays, setReplays] = useState<Record<string, Replay | null>>({});
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => setLevel(initialLevel), [initialLevel, suite.id]);
  useEffect(() => setShown(styles.slice(0, 2).map((s) => s.spec)), [suite.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load the replays for (level, seed) of every shown controller.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    setPlaying(false);
    setTick(0);
    api
      .compare(suite.id, level, seed)
      .then(async (cmp) => {
        const out: Record<string, Replay | null> = {};
        await Promise.all(
          shown.map(async (spec) => {
            const ep = cmp.episodes.find((e) => e.controller_spec === spec);
            out[spec] = ep ? await api.replay(suite.id, ep.episode_dir) : null;
          }),
        );
        if (!cancelled) setReplays(out);
      })
      .catch((e) => !cancelled && setError(String(e.message ?? e)));
    return () => {
      cancelled = true;
    };
  }, [suite.id, level, seed, shown]);

  const maxTick = Math.max(1, ...Object.values(replays).map((r) => r?.ticks ?? 0));
  const tickRef = useRef(tick);
  tickRef.current = tick;

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const next = tickRef.current + dt * 60 * speed;
      if (next >= maxTick) {
        setTick(maxTick);
        setPlaying(false);
        return;
      }
      setTick(next);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, maxTick]);

  const anyReplay = Object.values(replays).find((r) => r) ?? null;
  const worldT = anyReplay ? tick * anyReplay.world_seconds_per_tick : 0;
  const worldEnd = anyReplay ? maxTick * anyReplay.world_seconds_per_tick : 0;

  const toggleShown = (spec: string) =>
    setShown((cur) => (cur.includes(spec) ? (cur.length > 1 ? cur.filter((s) => s !== spec) : cur) : [...cur, spec].slice(-4)));

  return (
    <Card
      title="Compare Runs · Same Seed"
      sub="Exact re-simulations from the logged actions: identical initial conditions, one shared clock."
      right={
        <div className="compare-controls">
          <select className="select" value={seed} onChange={(e) => setSeed(Number(e.target.value))} aria-label="seed">
            {manifest.seeds.map((s) => (
              <option key={s} value={s}>seed {s}</option>
            ))}
          </select>
          <select className="select" value={level} onChange={(e) => setLevel(Number(e.target.value))} aria-label="level">
            {manifest.levels.map((l) => (
              <option key={l} value={l}>{pinfo.label} {pinfo.unit(l)}</option>
            ))}
          </select>
          <Seg
            small
            value={"__"}
            onChange={toggleShown}
            options={styles.map((s) => ({ value: s.spec, label: (shown.includes(s.spec) ? "● " : "○ ") + s.label }))}
          />
        </div>
      }
    >
      {error && <div className="error">Could not load replays: {error}</div>}
      <div className="compare-grid">
        {shown.map((spec) => {
          const st = styles.find((s) => s.spec === spec)!;
          const r = replays[spec];
          return <ArenaPanel key={spec} style={st} replay={r} tick={tick} />;
        })}
      </div>
      <div className="scrub">
        <button className="icon-btn" onClick={() => (tick >= maxTick ? (setTick(0), setPlaying(true)) : setPlaying(!playing))} aria-label={playing ? "pause" : "play"}>
          {playing ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <span className="mono tnum" style={{ minWidth: 118 }}>t = {worldT.toFixed(1)} / {worldEnd.toFixed(1)} s</span>
        <input type="range" min={0} max={maxTick} step={1} value={Math.round(tick)} onChange={(e) => { setPlaying(false); setTick(Number(e.target.value)); }} aria-label="time" />
        <Seg small value={speed} onChange={setSpeed} options={SPEEDS.map((s) => ({ value: s, label: `${s}×` }))} />
      </div>
    </Card>
  );
}

function outcomePill(r: Replay) {
  const res = r.result;
  if (res.success) return { text: `Survived ${res.survival_time.toFixed(1)} s`, color: OUTCOME_COLOR.survived, Icon: CheckCircle2 };
  if (res.reason === "collision") return { text: `Collision @ ${res.survival_time.toFixed(1)} s`, color: OUTCOME_COLOR.collision, Icon: OctagonX };
  return { text: `Target timeout @ ${res.survival_time.toFixed(1)} s`, color: OUTCOME_COLOR.target_timeout, Icon: Clock };
}

function frameAt(r: Replay, tick: number): { frame: Frame; index: number } {
  const i = Math.min(r.frames.length - 1, Math.max(0, Math.floor(tick / r.stride)));
  return { frame: r.frames[i], index: i };
}

function ArenaPanel({ style, replay, tick }: { style: SeriesStyle; replay: Replay | null | undefined; tick: number }) {
  if (replay === undefined) return <div className="arena-panel"><div className="muted">loading…</div></div>;
  if (replay === null) return <div className="arena-panel"><div className="muted">{style.label}: no episode for this seed and level.</div></div>;
  const pill = outcomePill(replay);
  const { frame } = frameAt(replay, tick);
  const p50 = replay.result.p50_latency_ms as number | null;
  const Icon = pill.Icon;
  return (
    <div className="arena-panel">
      <div className="arena-head">
        <span className="name"><span className="dot" style={{ background: style.color }} />{style.label}</span>
        <span className="pill" style={{ color: pill.color, borderColor: pill.color }}><Icon size={12} />{pill.text}</span>
        <span className="arena-stats">
          <span>score <b>{frame[5]}</b></span>
          <span>p50 <b>{fmtMs(p50)}</b></span>
        </span>
      </div>
      <ArenaCanvas replay={replay} tick={tick} color={style.color} />
      <div className="note" style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8 }}>
        {replay.verified ? <ShieldCheck size={13} color="var(--good)" /> : <ShieldAlert size={13} color="var(--warning)" />}
        {replay.verified ? "replay re-simulated and verified against the logged result" : "replay does not match the logged result"}
      </div>
    </div>
  );
}

export function ArenaCanvas({ replay, tick, color }: { replay: Replay; tick: number; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(600);
  const aspect = replay.arena.height / replay.arena.width;

  useEffect(() => {
    const el = ref.current?.parentElement;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth - 24));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const trail = useMemo(() => replay.frames.map((f) => [f[1], f[2]] as const), [replay]);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(100, w);
    const ch = cw * aspect;
    c.width = cw * dpr;
    c.height = ch * dpr;
    c.style.height = `${ch}px`;
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const s = cw / replay.arena.width;
    // surface + grid
    g.fillStyle = "#0d1118";
    g.fillRect(0, 0, cw, ch);
    g.strokeStyle = "#161c26";
    g.lineWidth = 1;
    for (let x = 0; x <= replay.arena.width; x += 100) { g.beginPath(); g.moveTo(x * s + 0.5, 0); g.lineTo(x * s + 0.5, ch); g.stroke(); }
    for (let y = 0; y <= replay.arena.height; y += 100) { g.beginPath(); g.moveTo(0, y * s + 0.5); g.lineTo(cw, y * s + 0.5); g.stroke(); }

    const { frame, index } = frameAt(replay, tick);
    // obstacle short trails (previous ~8 frames)
    const back = Math.max(0, index - 8);
    const prev = replay.frames[back][6];
    const prevPos = new Map<number, [number, number]>();
    for (let i = 0; i < prev.length; i += 3) prevPos.set(prev[i], [prev[i + 1], prev[i + 2]]);
    const obs = frame[6];
    for (let i = 0; i < obs.length; i += 3) {
      const id = obs[i], x = obs[i + 1], y = obs[i + 2];
      const r = replay.radii[String(id)] ?? 15;
      const pp = prevPos.get(id);
      if (pp) {
        const grad = g.createLinearGradient(pp[0] * s, pp[1] * s, x * s, y * s);
        grad.addColorStop(0, "rgba(230,103,103,0)");
        grad.addColorStop(1, "rgba(230,103,103,0.35)");
        g.strokeStyle = grad;
        g.lineWidth = Math.max(1.5, r * s * 0.9);
        g.lineCap = "round";
        g.beginPath(); g.moveTo(pp[0] * s, pp[1] * s); g.lineTo(x * s, y * s); g.stroke();
      }
      g.fillStyle = "#e0564f";
      g.beginPath(); g.arc(x * s, y * s, r * s, 0, Math.PI * 2); g.fill();
    }
    // target
    g.strokeStyle = "#4ade80";
    g.lineWidth = Math.max(2, replay.target_radius * s * 0.45);
    g.beginPath(); g.arc(frame[3] * s, frame[4] * s, replay.target_radius * s * 0.75, 0, Math.PI * 2); g.stroke();
    // player trajectory
    g.strokeStyle = color;
    g.globalAlpha = 0.55;
    g.lineWidth = 1.5;
    g.beginPath();
    for (let i = 0; i <= index; i++) { const [x, y] = trail[i]; if (i === 0) g.moveTo(x * s, y * s); else g.lineTo(x * s, y * s); }
    g.stroke();
    g.globalAlpha = 1;
    // player
    g.fillStyle = color;
    g.beginPath(); g.arc(frame[1] * s, frame[2] * s, replay.player_radius * s, 0, Math.PI * 2); g.fill();
    g.strokeStyle = "rgba(255,255,255,0.85)";
    g.lineWidth = 1.2;
    g.stroke();
    // collision marker
    const col = replay.timeline.collision;
    if (col && tick >= col.tick) {
      const x = col.x * s, y = col.y * s, k = 9;
      g.strokeStyle = "#ff8a80";
      g.lineWidth = 2.4;
      g.beginPath(); g.moveTo(x - k, y - k); g.lineTo(x + k, y + k); g.moveTo(x + k, y - k); g.lineTo(x - k, y + k); g.stroke();
      g.beginPath(); g.arc(x, y, 16, 0, Math.PI * 2); g.stroke();
    }
    // clock
    g.fillStyle = "rgba(170,178,192,0.9)";
    g.font = "500 11px Inter, system-ui, sans-serif";
    g.textAlign = "right";
    const ended = tick > replay.ticks;
    g.fillText(`t = ${(Math.min(tick, replay.ticks) * replay.world_seconds_per_tick).toFixed(1)} s${ended ? " · ended" : ""}`, cw - 10, ch - 10);
  }, [replay, tick, w, aspect, color, trail]);

  return <canvas ref={ref} className="arena" />;
}
