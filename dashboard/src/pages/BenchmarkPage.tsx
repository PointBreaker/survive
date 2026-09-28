import { Activity } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { Meta, RunInfo, SuiteDetail } from "../api";
import { CompareRuns } from "../components/CompareRuns";
import { DecisionOutcomes } from "../components/DecisionOutcomes";
import { FailureAnalysis } from "../components/FailureAnalysis";
import { Facts } from "../components/Facts";
import { FrontierChart, METRICS, type Metric } from "../components/FrontierChart";
import { KeyResults, referenceLevel } from "../components/KeyResults";
import { LatencyDistribution } from "../components/LatencyDistribution";
import { Sidebar } from "../components/Sidebar";
import { Card, Empty, Legend, Seg } from "../components/ui";
import { paramInfo } from "../format";
import { seriesStyles } from "../theme";

const X_AXES = ["world_speed_scale", "obstacle_count", "latency_ms"];

export function EmptyFrontier() {
  return (
    <Card>
      <Empty icon={<Activity size={28} />} title="No frontier benchmark yet">
        <div>A frontier needs a benchmark suite: several controllers, one swept pressure, paired seeds.</div>
        <div className="cmd">
          python -m benchmark.suite --controllers jev,simple_avoid,greedy,random \{"\n"}  --param world_speed_scale --levels 0.25,0.5,1,2,4,8 --episodes 20
        </div>
        <div className="hint">Jev needs OPENROUTER_API_KEY (or TYPESAFE_API_KEY) in .env. The suite shows up here while it runs.</div>
      </Empty>
    </Card>
  );
}

export function BenchmarkPage(props: {
  meta: Meta | null;
  suites: RunInfo[];
  suite: SuiteDetail | null;
  loading: boolean;
  onSelectSuite: (id: string) => void;
}) {
  const { suite, suites } = props;
  const [metric, setMetric] = useState<Metric>("success");
  const styles = useMemo(() => (suite ? seriesStyles(suite.manifest.controllers) : []), [suite]);
  const [focusSpec, setFocusSpec] = useState<string | null>(null);
  useEffect(() => {
    if (!suite) return;
    const specs = suite.manifest.controllers;
    if (!focusSpec || !specs.includes(focusSpec)) {
      // Jev if present; otherwise the controller with the highest measured latency
      // (the latency panels are most informative for it).
      const slowest = [...specs].sort(
        (a, b) => (suite.analysis[b]?.latency.pooled.p50 ?? 0) - (suite.analysis[a]?.latency.pooled.p50 ?? 0),
      )[0];
      setFocusSpec(specs.includes("jev") ? "jev" : slowest ?? styles[0]?.spec ?? null);
    }
  }, [suite, styles, focusSpec]);
  const focus = styles.find((s) => s.spec === focusSpec) ?? styles[0];

  const latestFor = (param: string) => suites.find((s) => s.param === param);
  const xAxis = suite?.manifest.param ?? "world_speed_scale";

  return (
    <div className="page">
      <Sidebar meta={props.meta} />
      <main className="main">
        <div className="main-inner">
          <div className="hero-head">
            <div>
              <h1 className="page-title">Capability Frontier</h1>
              <div className="page-sub">How much real-time pressure can each controller survive?</div>
            </div>
            <div className="controls">
              <Seg value={metric} onChange={setMetric} options={METRICS} />
              <Seg
                value={xAxis}
                onChange={(p) => {
                  const s = latestFor(p);
                  if (s) props.onSelectSuite(s.id);
                }}
                options={X_AXES.map((p) => ({
                  value: p,
                  label: `vs ${paramInfo(p).label}`,
                  disabled: !latestFor(p),
                  title: latestFor(p) ? undefined : `No suite sweeping ${paramInfo(p).short} yet: python -m benchmark.suite --param ${p} …`,
                }))}
              />
            </div>
          </div>

          {!suite ? (
            props.loading ? <Card><div className="muted">Loading…</div></Card> : <EmptyFrontier />
          ) : (
            <div className={props.loading ? "loading-dim" : ""} style={{ display: "grid", gap: 20 }}>
              <div className="grid-top">
                <Card
                  title={`${METRICS.find((m) => m.value === metric)!.label} vs. ${paramInfo(suite.manifest.param).label}`}
                  sub={`Observed over ${suite.manifest.episodes_per_point} paired seeds per point${metric === "success" ? " · band = 95% Wilson interval" : ""}${suite.manifest.status === "running" ? " · suite still running, updating live" : ""}`}
                >
                  <Legend styles={styles} />
                  <FrontierChart suite={suite} metric={metric} styles={styles} />
                </Card>
                <KeyResults suite={suite} styles={styles} />
              </div>

              {focus && <Facts suite={suite} focus={focus} />}

              {focus && (
                <>
                  <div className="focus-row">
                    <span className="label">Detail for</span>
                    <Seg value={focus.spec} onChange={setFocusSpec} options={styles.map((s) => ({ value: s.spec, label: s.label }))} />
                  </div>
                  <div className="grid-three">
                    <FailureAnalysis suite={suite} focus={focus} />
                    <LatencyDistribution suite={suite} focus={focus} />
                    <DecisionOutcomes suite={suite} focus={focus} />
                  </div>
                </>
              )}

              <CompareRuns suite={suite} styles={styles} initialLevel={referenceLevel(suite)} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
