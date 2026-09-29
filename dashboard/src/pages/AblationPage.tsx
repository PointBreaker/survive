import { CheckCircle2, Clock, Loader2, OctagonX, Play, ShieldAlert, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type AblationDetail,
  type AblationJobSpec,
  type AblationMode,
  type Inspection,
  type Meta,
  type Replay,
  type RequestInfo,
  type RunInfo,
} from "../api";
import { useWidth } from "../components/charts";
import { ArenaCanvas } from "../components/CompareRuns";
import { Card, Seg } from "../components/ui";
import { fmtMs, fmtNum, fmtPct, fmtS, reasonLabel } from "../format";
import { controllerLabel, OUTCOME_COLOR, seriesStyles } from "../theme";

const MODE_LABEL: Record<string, string> = { raw: "RAW", relative: "RELATIVE", physics: "PHYSICS" };
const MODE_NOTE: Record<string, string> = {
  raw: "canonical state only",
  relative: "+ ego-relative coordinates",
  physics: "+ linear closest-approach projection",
};
const STEP_NOTE: Record<string, string> = {
  "raw>relative": "sensitivity to coordinate representation",
  "relative>physics": "sensitivity to explicit physical projection",
  "raw>physics": "sensitivity to representation + projection",
  ">reference": "remaining gap to the reference controller on its qualified seeds",
};

function stepNote(from: string, to: string): string {
  return STEP_NOTE[to === "reference" ? ">reference" : `${from}>${to}`] ?? "";
}

export function AblationPage({ meta, runs, selected, onSelect, onRun, jobActive }: {
  meta: Meta | null;
  runs: RunInfo[];
  selected: string | null;
  onSelect: (id: string) => void;
  onRun: (spec: AblationJobSpec) => Promise<void>;
  jobActive: boolean;
}) {
  const ablations = runs.filter((r) => r.kind === "ablation");
  const id = selected && ablations.some((a) => a.id === selected) ? selected : ablations[0]?.id ?? null;
  const status = ablations.find((a) => a.id === id)?.status;
  const [data, setData] = useState<AblationDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pick, setPick] = useState<{ mode: string; seed: number; role: "candidate" | "reference" } | null>(null);

  useEffect(() => {
    if (!id) return setData(null);
    let cancelled = false;
    const load = () => api.ablation(id).then((d) => !cancelled && (setData(d), setError(null))).catch((e) => !cancelled && setError(String(e.message ?? e)));
    load();
    const t = status === "running" ? setInterval(load, 2500) : undefined;
    return () => {
      cancelled = true;
      if (t) clearInterval(t);
    };
  }, [id, status]);
  useEffect(() => setPick(null), [id]);

  return (
    <div className="page single">
      <main className="main">
        <div className="main-inner">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 16, flexWrap: "wrap" }}>
            <div>
              <h1 className="page-title">Observation Ablation</h1>
              <div className="page-sub">
                Same controller, three observation modes, only on seeds a reference controller completed under identical
                conditions and matched latency. Closed-loop outcomes only.
              </div>
            </div>
            {ablations.length > 0 && (
              <select className="select" value={id ?? ""} onChange={(e) => onSelect(e.target.value)} aria-label="ablation run">
                {ablations.map((a) => (
                  <option key={a.id} value={a.id}>
                    {controllerLabel(String(a.controllers?.[0]))} · {(a.modes ?? []).join("/")} · {a.timing} · {new Date(a.mtime * 1000).toLocaleString()}
                    {a.status && a.status !== "complete" ? ` · ${a.status}` : ""}
                  </option>
                ))}
              </select>
            )}
          </div>
          {error && <div className="error">{error}</div>}
          {meta?.jobs_enabled && <Launcher meta={meta} onRun={onRun} busy={jobActive} />}
          {!id && <Card><div className="muted">No ablation runs yet. Start one above or run <span className="mono">python -m benchmark.ablation</span>.</div></Card>}
          {data && <AblationBody data={data} pick={pick} setPick={setPick} />}
        </div>
      </main>
    </div>
  );
}

function AblationBody({ data, pick, setPick }: {
  data: AblationDetail;
  pick: { mode: string; seed: number; role: "candidate" | "reference" } | null;
  setPick: (p: { mode: string; seed: number; role: "candidate" | "reference" } | null) => void;
}) {
  const m = data.manifest;
  const s = data.summary;
  const cand = seriesStyles([m.controller.split("+")[0]])[0].color;
  const ref = seriesStyles([m.reference_controller])[0].color;
  const modes = m.modes.filter((x) => s?.modes[x]);
  const running = m.status === "running";
  return (
    <>
      {running && m.progress && (
        <div className="job-strip running" role="status">
          <div className="job-state" style={{ color: "var(--accent)" }}><Loader2 size={16} className="spin" /> Running</div>
          <div className="job-mid">
            <div className="job-bar"><div style={{ width: `${Math.round((m.progress.done / Math.max(1, m.progress.total)) * 100)}%` }} /></div>
            <div className="job-meta">
              <span>{m.progress.stage} · <b>{MODE_LABEL[m.progress.mode ?? ""] ?? "–"}</b></span>
              <span>reference <b>{m.progress.done}</b> / {m.progress.total}</span>
              {m.progress.cand_total !== undefined && <span>candidate <b>{m.progress.cand_done}</b> / {m.progress.cand_total}</span>}
            </div>
          </div>
        </div>
      )}
      <div className="facts">
        <Fact k="Candidate" v={controllerLabel(m.controller)} />
        <Fact k="Reference (feasibility baseline)" v={`${controllerLabel(m.reference_controller)}${m.reference_repeats > 1 ? ` · ${m.reference_repeats}× each` : ""}`} />
        <Fact k="Timing" v={m.timing === "lockstep" ? `lockstep · ${fmtNum((m.interval_s ?? 0.1) * 1000)} ms/decision` : `real time · ${fmtNum(m.config.world_speed_scale)}× world speed`} />
        <Fact k="Control loop" v={m.timing === "lockstep" ? "world waits for every answer" : `${fmtNum(m.config.decision_hz)} Hz · ${m.config.max_inflight} in flight`} />
        <Fact k="Reference latency" v={m.timing === "lockstep" ? "none (lockstep)" : `${fmtMs(m.latency.reference_ms)}`} small={m.timing === "lockstep" ? "" : m.latency.how} />
        <Fact k="Seeds" v={`${m.candidate_seeds.length}`} small={`${m.config.obstacle_count} obstacles · ${fmtS(m.config.max_duration)}`} />
      </div>
      {!s ? (
        <Card><div className="muted">Waiting for the first completed mode…</div></Card>
      ) : (
        <>
          <Card
            title="Success Rate by Observation Mode"
            sub={`${controllerLabel(m.controller)} on reference-qualified seeds · 95% Wilson CI · the reference bar is 100% on those seeds by construction; its qualification rate is shown below it`}
          >
            <ModeBars modes={modes} s={s.modes} cand={cand} refColor={ref} refLabel={controllerLabel(m.reference_controller)} />
          </Card>
          <div className="grid3">
            <Card title="Avg Survival" sub="world seconds, qualified seeds">
              <SmallBars ctx={s.modes} modes={modes} color={cand} value={(v) => v.candidate?.mean_survival_time ?? null} fmt={fmtS} max={m.config.max_duration} />
            </Card>
            <Card title="Targets / Episode" sub="qualified seeds">
              <SmallBars ctx={s.modes} modes={modes} color={cand} value={(v) => v.candidate?.mean_targets ?? null} fmt={(x) => (x === null ? "–" : x.toFixed(1))} />
            </Card>
            <Card title="Latency p50" sub="measured per mode, not adjusted; dashed = reference latency">
              <SmallBars
                ctx={s.modes}
                modes={modes}
                color={cand}
                value={(v) => v.candidate_latency.p50_ms}
                fmt={fmtMs}
                refLine={m.timing === "lockstep" ? null : m.latency.reference_ms}
                flag={(v) => v.latency_match.within_tolerance === false}
              />
            </Card>
          </div>
          <Card title="Capability Decomposition" sub="success-rate change on the seeds both sides evaluated (paired); counts of seeds that flipped">
            <div className="decomp">
              {s.decomposition.map((st) => (
                <div className="decomp-row" key={`${st.from}-${st.to}`}>
                  <div className="decomp-step">{MODE_LABEL[st.from] ?? st.from} → {st.to === "reference" ? "Reference" : MODE_LABEL[st.to] ?? st.to}</div>
                  <div className="decomp-val tnum">{st.delta_pp === null ? "–" : `${st.delta_pp > 0 ? "+" : ""}${st.delta_pp.toFixed(1)} pp`}</div>
                  <div className="decomp-detail">
                    {st.from_success} → {st.to_success} of {st.seeds} seeds · {st.gained} gained · {st.lost} lost
                    <span className="muted"> · {stepNote(st.from, st.to)}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
          <SeedTable data={data} modes={modes} pick={pick} setPick={setPick} />
          {pick && <EpisodeInspector data={data} pick={pick} color={pick.role === "candidate" ? cand : ref} />}
        </>
      )}
    </>
  );
}

function Fact({ k, v, small }: { k: string; v: string; small?: string }) {
  return (
    <div className="fact">
      <div className="fact-k">{k}</div>
      <div className="fact-v">{v}{small ? <small>{small}</small> : null}</div>
    </div>
  );
}

function ModeBars({ modes, s, cand, refColor, refLabel }: {
  modes: string[]; s: Record<string, AblationMode>; cand: string; refColor: string; refLabel: string;
}) {
  const last = s[modes[modes.length - 1]];
  const items = [
    ...modes.map((md) => ({ key: md, label: MODE_LABEL[md], note: MODE_NOTE[md], agg: s[md].candidate, color: cand, sub: `${s[md].candidate?.successes ?? 0} / ${s[md].candidate?.episodes ?? 0}` })),
    { key: "reference", label: `Reference`, note: `${refLabel} @ same latency`, agg: last?.reference_qualified ?? null, color: refColor,
      sub: last ? `qualified ${last.qualified} / ${last.candidate_seeds} (${fmtPct(last.reference_all?.success_rate)})` : "" },
  ];
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref, 760), H = 270, m = { l: 44, r: 12, t: 22, b: 58 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const step = iw / items.length, bw = Math.min(70, step * 0.46);
  const Y = (v: number) => m.t + ih - v * ih;
  return (
    <div className="chart-wrap" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="success rate by observation mode">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" />
            <text x={m.l - 8} y={Y(t) + 3.5} fontSize="10.5" fill="var(--text-3)" textAnchor="end">{Math.round(t * 100)}%</text>
          </g>
        ))}
        {items.map((it, i) => {
          const cx = m.l + step * i + step / 2;
          const v = it.agg?.success_rate ?? null;
          const [lo, hi] = it.agg?.success_rate_ci95 ?? [0, 0];
          return (
            <g key={it.key}>
              {i === items.length - 1 && <line x1={cx - step / 2} x2={cx - step / 2} y1={m.t} y2={m.t + ih} stroke="var(--border-strong)" strokeDasharray="3 4" />}
              {v !== null && v > 0 && <rect x={cx - bw / 2} y={Y(v)} width={bw} height={Y(0) - Y(v)} rx={3} fill={it.color} opacity={0.88} />}
              {v !== null && (
                <>
                  <line x1={cx} x2={cx} y1={Y(hi)} y2={Y(lo)} stroke="var(--text-2)" />
                  <line x1={cx - 6} x2={cx + 6} y1={Y(hi)} y2={Y(hi)} stroke="var(--text-2)" />
                  <line x1={cx - 6} x2={cx + 6} y1={Y(lo)} y2={Y(lo)} stroke="var(--text-2)" />
                  <text x={cx + 12} y={Y(hi) - 7} fontSize="12.5" fontWeight={650} fill="var(--text-1)" textAnchor="start">{fmtPct(v)}</text>
                </>
              )}
              {v === null && <text x={cx} y={Y(0) - 8} fontSize="11" fill="var(--text-3)" textAnchor="middle">not run</text>}
              <text x={cx} y={H - 38} fontSize="12" fontWeight={600} fill="var(--text-1)" textAnchor="middle">{it.label}</text>
              <text x={cx} y={H - 23} fontSize="10.5" fill="var(--text-3)" textAnchor="middle">{it.note}</text>
              <text x={cx} y={H - 9} fontSize="10.5" fill="var(--text-3)" textAnchor="middle">{it.sub}</text>
            </g>
          );
        })}
        <line x1={m.l} x2={W - m.r} y1={Y(0)} y2={Y(0)} stroke="var(--border-strong)" />
      </svg>
    </div>
  );
}

function SmallBars({ ctx, modes, color, value, fmt, max, refLine, flag }: {
  ctx: Record<string, AblationMode>; modes: string[]; color: string; value: (v: AblationMode) => number | null; fmt: (v: number | null) => string;
  max?: number; refLine?: number | null; flag?: (v: AblationMode) => boolean;
}) {
  const vals = modes.map((md) => (ctx[md] ? value(ctx[md]) : null));
  const top = max ?? Math.max(1e-6, refLine ?? 0, ...vals.map((v) => v ?? 0)) * 1.15;
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref, 300), H = 150, m = { l: 8, r: 8, t: 18, b: 24 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const step = iw / modes.length, bw = Math.min(46, step * 0.5);
  const Y = (v: number) => m.t + ih - (v / top) * ih;
  return (
    <div ref={ref}>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img">
      {modes.map((md, i) => {
        const v = vals[i];
        const cx = m.l + step * i + step / 2;
        const bad = ctx[md] && flag?.(ctx[md]);
        return (
          <g key={md}>
            {v !== null && v > 0 && <rect x={cx - bw / 2} y={Y(v)} width={bw} height={Y(0) - Y(v)} rx={3} fill={color} opacity={0.85} />}
            <text x={cx} y={v === null ? Y(0) - 6 : Y(v) - 5} fontSize="11.5" fontWeight={600} fill={bad ? "var(--warning)" : "var(--text-1)"} textAnchor="middle">
              {fmt(v)}{bad ? " ⚠" : ""}
            </text>
            <text x={cx} y={H - 7} fontSize="10.5" fill="var(--text-3)" textAnchor="middle">{MODE_LABEL[md]}</text>
          </g>
        );
      })}
      {refLine !== undefined && refLine !== null && refLine > 0 && (
        <line x1={m.l} x2={W - m.r} y1={Y(refLine)} y2={Y(refLine)} stroke="var(--text-2)" strokeDasharray="4 4" />
      )}
      <line x1={m.l} x2={W - m.r} y1={Y(0)} y2={Y(0)} stroke="var(--border-strong)" />
    </svg>
    </div>
  );
}

function outcome(reason: string | null | undefined) {
  if (!reason) return { text: "–", color: "var(--text-3)", Icon: null as null | typeof CheckCircle2 };
  if (reason === "max_duration" || reason === "target_goal") return { text: reasonLabel(reason), color: OUTCOME_COLOR.survived, Icon: CheckCircle2 };
  if (reason === "collision") return { text: "Collision", color: OUTCOME_COLOR.collision, Icon: OctagonX };
  return { text: "Timeout", color: OUTCOME_COLOR.target_timeout, Icon: Clock };
}

function SeedTable({ data, modes, pick, setPick }: {
  data: AblationDetail; modes: string[];
  pick: { mode: string; seed: number; role: string } | null;
  setPick: (p: { mode: string; seed: number; role: "candidate" | "reference" }) => void;
}) {
  const rows = data.episodes;
  const seeds = data.manifest.candidate_seeds;
  const get = (role: string, mode: string, seed: number) => rows.find((r) => r.role === role && r.mode === mode && r.seed === seed && r.repeat === 0);
  return (
    <Card title="Paired Seeds" sub="each row is one world (same seed, config and latency for every cell); click a cell to replay and inspect decisions">
      <div className="table-scroll">
        <table className="runs seeds">
          <thead>
            <tr>
              <th>Seed</th>
              <th>Reference</th>
              {modes.map((md) => <th key={md}>{MODE_LABEL[md]}</th>)}
            </tr>
          </thead>
          <tbody>
            {seeds.map((seed) => {
              const r = get("reference", modes[0], seed);
              if (!r) return null;
              return (
                <tr key={seed}>
                  <td className="mono">{seed}</td>
                  <Cell row={r} active={pick?.seed === seed && pick.role === "reference"} onClick={() => setPick({ mode: modes[0], seed, role: "reference" })} />
                  {modes.map((md) => {
                    const c = get("candidate", md, seed);
                    const q = data.summary?.modes[md]?.qualified_seeds.includes(seed);
                    return c ? (
                      <Cell key={md} row={c} dim={!q} active={pick?.seed === seed && pick.mode === md && pick.role === "candidate"}
                            onClick={() => setPick({ mode: md, seed, role: "candidate" })} />
                    ) : (
                      <td key={md} className="muted">{q ? "pending" : "not qualified"}</td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Cell({ row, onClick, active, dim }: { row: { reason: string; survival_time: number; targets_collected: number }; onClick: () => void; active: boolean; dim?: boolean }) {
  const o = outcome(row.reason);
  const Icon = o.Icon;
  return (
    <td className={`clickable${active ? " on" : ""}`} onClick={onClick} style={{ opacity: dim ? 0.55 : 1 }}>
      <span className="pill" style={{ color: o.color, borderColor: o.color }}>{Icon && <Icon size={12} />}{o.text}</span>
      <span className="muted tnum" style={{ marginLeft: 8 }}>{fmtS(row.survival_time)} · {row.targets_collected} tg</span>
    </td>
  );
}

function EpisodeInspector({ data, pick, color }: { data: AblationDetail; pick: { mode: string; seed: number; role: "candidate" | "reference" }; color: string }) {
  const row = data.episodes.find((r) => r.role === pick.role && r.mode === pick.mode && r.seed === pick.seed && r.repeat === 0);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [reqs, setReqs] = useState<{ requests: RequestInfo[]; key_ticks: number[] } | null>(null);
  const [tick, setTick] = useState(0);
  const [insp, setInsp] = useState<Inspection | null>(null);
  const [inspErr, setInspErr] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState(pick.mode);

  useEffect(() => {
    if (!row) return;
    setReplay(null);
    setInsp(null);
    setViewMode(pick.mode);
    api.runReplay(data.id, row.episode_dir).then((r) => {
      setReplay(r);
      setTick(r.timeline.collision ? Math.max(0, r.timeline.collision.tick - 30) : r.ticks);
    });
    api.snapshots(data.id, row.episode_dir).then(setReqs);
  }, [data.id, row?.episode_dir]); // eslint-disable-line react-hooks/exhaustive-deps

  const nearest = useMemo(() => {
    if (!reqs?.requests.length) return null;
    return reqs.requests.reduce((b, r) => (Math.abs(r.tick - tick) < Math.abs(b.tick - tick) ? r : b), reqs.requests[0]);
  }, [reqs, tick]);

  const inspect = (t: number) => {
    if (!row) return;
    setTick(t);
    setInspErr(null);
    api.inspect(data.id, row.episode_dir, t, pick.mode).then(setInsp).catch((e) => setInspErr(String(e.message ?? e)));
  };

  if (!row) return null;
  const label = pick.role === "candidate" ? `${controllerLabel(data.manifest.controller)} · ${MODE_LABEL[pick.mode]}` : `Reference ${controllerLabel(data.manifest.reference_controller)}`;
  const tickS = replay?.world_seconds_per_tick ?? 1 / 60;
  return (
    <Card title={`Episode · seed ${pick.seed} · ${label}`} sub={`${reasonLabel(row.reason)} at ${fmtS(row.survival_time)} · ${row.targets_collected} targets`}>
      <div className="inspector">
        <div className="arena-panel">
          {replay ? <ArenaCanvas replay={replay} tick={tick} color={color} /> : <div className="muted">loading replay…</div>}
          {replay && (
            <>
              <input type="range" min={0} max={replay.ticks} value={tick} onChange={(e) => setTick(Number(e.target.value))} style={{ width: "100%" }} aria-label="tick" />
              <div className="note" style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <span className="tnum">t = {(tick * tickS).toFixed(2)} s</span>
                {replay.verified ? <><ShieldCheck size={13} color="var(--good)" /> verified re-simulation</> : <><ShieldAlert size={13} color="var(--warning)" /> replay mismatch</>}
                {nearest && (
                  <button className="btn" onClick={() => inspect(nearest.tick)}>Inspect decision @ {(nearest.tick * tickS).toFixed(2)} s</button>
                )}
              </div>
            </>
          )}
          {reqs && reqs.key_ticks.length > 0 && (
            <div className="snap-list">
              <span className="muted">last decisions before the end:</span>
              {reqs.key_ticks.map((t) => (
                <button key={t} className={`chip${insp?.tick === t ? " on" : ""}`} onClick={() => inspect(t)}>{(t * tickS).toFixed(2)} s</button>
              ))}
            </div>
          )}
        </div>
        <div className="inspect-panel">
          {inspErr && <div className="error">{inspErr}</div>}
          {!insp ? (
            <div className="muted">Pick a decision to see the exact snapshot this controller received, in every observation mode, and what other controllers would answer to it. Inspection only: several actions can be workable, none is ground truth.</div>
          ) : (
            <>
              <div className="tt-title">Snapshot t = {insp.t.toFixed(2)} s (tick {insp.tick})</div>
              <table className="runs answers">
                <tbody>
                  <tr>
                    <td>{controllerLabel(insp.controller)} <span className="muted">(logged)</span></td>
                    <td className="mono">{insp.logged?.action ?? insp.logged?.status ?? "–"}</td>
                    <td className="muted tnum">
                      {insp.logged?.confidence != null ? `confidence ${insp.logged.confidence.toFixed(2)} · ` : ""}
                      {insp.logged?.latency_ms != null ? fmtMs(insp.logged.latency_ms) : ""}
                    </td>
                  </tr>
                  {Object.entries(insp.answers).map(([n, a]) => (
                    <tr key={n}>
                      <td>{controllerLabel(n)}</td>
                      <td className="mono">{a.action ?? a.error ?? "–"}</td>
                      <td className="muted">{a.primed ? "primed with previous snapshot" : "cold start"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "12px 0 6px" }}>
                <span className="muted">Observation</span>
                <Seg small value={viewMode} onChange={setViewMode} options={Object.keys(insp.views).map((k) => ({ value: k, label: MODE_LABEL[k] ?? k }))} />
              </div>
              <ObsView obs={insp.views[viewMode]} />
            </>
          )}
        </div>
      </div>
    </Card>
  );
}

function ObsView({ obs }: { obs: Record<string, unknown> }) {
  const obstacles = (obs.obstacles as Record<string, number>[]) ?? [];
  const cols = obstacles.length ? Object.keys(obstacles[0]) : [];
  const num = (v: unknown) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) < 10 ? 2 : 1)) : JSON.stringify(v));
  const flat = (o: unknown) => Object.entries((o as Record<string, unknown>) ?? {}).map(([k, v]) => `${k} ${typeof v === "object" && v ? JSON.stringify(v, (_k, x) => (typeof x === "number" ? Math.round(x * 10) / 10 : x)) : num(v)}`).join(" · ");
  return (
    <div className="obs-view">
      <div className="mono small"><b>player</b> {flat(obs.player)}</div>
      <div className="mono small"><b>target</b> {flat(obs.target)}</div>
      <div className="mono small"><b>control</b> {flat(obs.control)}</div>
      {"prediction_horizon_s" in obs && <div className="mono small"><b>prediction_horizon_s</b> {String(obs.prediction_horizon_s)}</div>}
      <div className="table-scroll">
        <table className="runs obs-table">
          <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {obstacles.map((o, i) => (
              <tr key={i}>{cols.map((c) => <td key={c} className="mono tnum">{num(o[c])}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Launcher({ meta, onRun, busy }: { meta: Meta; onRun: (s: AblationJobSpec) => Promise<void>; busy: boolean }) {
  const cands = meta.benchmarkable.filter((c) => c !== "jev" || meta.jev_token);
  const [controller, setController] = useState(cands.includes("jev") ? "jev" : cands[0] ?? "greedy");
  const [modes, setModes] = useState<string[]>(["raw", "relative", "physics"]);
  const [preset, setPreset] = useState("easy");
  const [episodes, setEpisodes] = useState(30);
  const [timing, setTiming] = useState<"realtime" | "lockstep">("realtime");
  const [speed, setSpeed] = useState(0.5);
  const [lat, setLat] = useState<string>("auto");
  const [err, setErr] = useState<string | null>(null);
  const toggle = (md: string) => setModes((cur) => (cur.includes(md) ? cur.filter((x) => x !== md) : [...cur, md]));
  const submit = async () => {
    setErr(null);
    try {
      await onRun({
        kind: "ablation", controller, reference: "simple_avoid", modes, episodes, preset, timing,
        ...(timing === "realtime" ? { world_speed: speed, match_latency: lat === "auto" ? "auto" : Number(lat) } : { interval: 0.1 }),
      });
    } catch (e) {
      setErr(String((e as Error).message ?? e));
    }
  };
  const calls = controller === "jev" ? episodes * modes.length : 0;
  return (
    <Card title="Run ablation" sub="python -m benchmark.ablation, started as a subprocess; one job at a time">
      <div className="launcher">
        <label>Controller
          <select className="select" value={controller} onChange={(e) => setController(e.target.value)}>
            {cands.map((c) => <option key={c} value={c}>{controllerLabel(c)}</option>)}
          </select>
        </label>
        <label>Modes
          <span className="checks">
            {["raw", "relative", "physics"].map((md) => (
              <label key={md}><input type="checkbox" checked={modes.includes(md)} onChange={() => toggle(md)} /> {MODE_LABEL[md]}</label>
            ))}
          </span>
        </label>
        <label>Preset
          <select className="select" value={preset} onChange={(e) => setPreset(e.target.value)}>
            {Object.keys(meta.presets).map((p) => <option key={p}>{p}</option>)}
          </select>
        </label>
        <label>Seeds <input className="input" type="number" min={1} max={500} value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} /></label>
        <label>Timing <Seg small value={timing} onChange={setTiming} options={[{ value: "realtime", label: "Real time" }, { value: "lockstep", label: "Lockstep" }]} /></label>
        {timing === "realtime" && (
          <>
            <label>World speed <input className="input" type="number" step={0.05} min={0.05} max={64} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} /></label>
            <label>Reference latency (ms or auto) <input className="input" value={lat} onChange={(e) => setLat(e.target.value)} /></label>
          </>
        )}
        <button className="btn primary" disabled={busy || modes.length === 0} onClick={submit}><Play size={14} /> Run</button>
      </div>
      {calls > 0 && <div className="note">Jev: up to {calls} episodes of real API calls (only on qualified seeds).</div>}
      {err && <div className="error">{err}</div>}
    </Card>
  );
}
