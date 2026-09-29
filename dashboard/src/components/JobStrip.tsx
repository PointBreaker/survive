import { CheckCircle2, CircleSlash, Eye, FileText, Loader2, OctagonX, Square, X } from "lucide-react";
import { useState } from "react";
import type { Job } from "../api";
import { controllerLabel } from "../theme";

function fmtDur(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "–";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} min ${Math.round(s % 60)} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

const STATE = {
  running: { label: "Running", Icon: Loader2, color: "var(--accent)", spin: true },
  cancelling: { label: "Cancelling", Icon: Loader2, color: "var(--warning)", spin: true },
  succeeded: { label: "Finished", Icon: CheckCircle2, color: "var(--good)", spin: false },
  failed: { label: "Failed", Icon: OctagonX, color: "var(--critical)", spin: false },
  cancelled: { label: "Cancelled", Icon: CircleSlash, color: "var(--text-2)", spin: false },
} as const;

export function JobStrip({ job, viewing, onView, onCancel, onDismiss }: {
  job: Job;
  viewing: boolean;
  onView: () => void;
  onCancel: () => void;
  onDismiss: () => void;
}) {
  const [showLog, setShowLog] = useState(job.status === "failed");
  const st = STATE[job.status];
  const p = job.kind && job.kind !== "suite" ? null : job.progress;  // other job kinds report on their own page
  const frac = p && p.episodes_total ? p.episodes_done / p.episodes_total : 0;
  const active = job.status === "running" || job.status === "cancelling";
  // Rough ETA from average time per finished episode (episodes differ in length).
  const eta = active && p && p.episodes_done > 0 ? (job.elapsed_s / p.episodes_done) * (p.episodes_total - p.episodes_done) : null;
  const Icon = st.Icon;
  return (
    <div className={`job-strip ${job.status}`} role="status" aria-live="polite">
      <div className="job-state" style={{ color: st.color }}>
        <Icon size={16} className={st.spin ? "spin" : undefined} /> {st.label}
      </div>
      <div className="job-mid">
        <div className="job-bar" aria-label="progress"><div style={{ width: `${Math.round(frac * 100)}%` }} /></div>
        <div className="job-meta">
          {p ? (
            <>
              <span><b>{p.episodes_done.toLocaleString()}</b> / {p.episodes_total.toLocaleString()} episodes</span>
              <span><b>{p.points_done}</b> / {p.points_total} points</span>
              {p.current && active && <span>now: <b>{controllerLabel(p.current.controller)}</b> @ {p.current.level}</span>}
            </>
          ) : (
            <span>{job.kind && job.kind !== "suite" ? `${job.kind} run` : "starting…"}</span>
          )}
          <span>elapsed <b>{fmtDur(job.elapsed_s)}</b></span>
          {eta !== null && <span>≈ {fmtDur(eta)} left</span>}
          {job.status === "cancelled" && p && <span>{p.points_done} completed points kept</span>}
          {job.status === "failed" && <span>exit code {job.returncode}</span>}
        </div>
      </div>
      <div className="job-actions">
        {job.suite_id && !viewing && (
          <button className="btn" onClick={onView}><Eye size={14} /> {active ? "View live" : "View results"}</button>
        )}
        <button className="btn" onClick={() => setShowLog(!showLog)} aria-expanded={showLog}><FileText size={14} /> Log</button>
        {active ? (
          <button className="btn" onClick={onCancel} disabled={job.status === "cancelling"}><Square size={13} /> Cancel</button>
        ) : (
          <button className="icon-btn" onClick={onDismiss} aria-label="dismiss"><X size={14} /></button>
        )}
      </div>
      {showLog && (
        <pre className="job-log">{job.command}{"\n\n"}{job.log_tail.join("\n") || "(no output yet)"}</pre>
      )}
    </div>
  );
}
