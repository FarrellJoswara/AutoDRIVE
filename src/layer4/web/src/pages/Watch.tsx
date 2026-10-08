import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchMaps,
  getOfficialRunLiveTelemetry,
  getOfficialRunTelemetry,
  getOfficialRuns,
  type EvaluationResult,
  type FleetTelemetry,
  type MapCatalogEntry,
  type OfficialRunList,
  type OfficialRunTelemetry,
  type PpoDiagnostics,
  type RunSummary,
} from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";
import { ingestOfficialRunTelemetry, setOfficialRunSelection } from "../store";

const ACTIVE = new Set(["starting", "running", "stopping"]);
const STALE_AFTER_MS = 3000;
const REFRESH_RUNS_MS = 2500;
// Run telemetry is sampled at up to 15 Hz. Poll close to that rate so a fast
// car does not appear to jump through walls between Watch frames.
const REFRESH_TELEMETRY_MS = 100;
const TELEMETRY_CONCURRENCY = 8;

type ViewMode = "overview" | "focus";
interface RunFrame {
  data: OfficialRunTelemetry | null;
  receivedAt: number;
  sourceAt: number | null;
  error: string | null;
}

function formatNumber(value: number | null | undefined, digits = 2, suffix = ""): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)}${suffix}`;
}

function mapIdForWatch(snapshotMapId: string | null | undefined, runMapId: string | null | undefined, maps: MapCatalogEntry[]): string {
  // Older official-run telemetry serializes a missing map as "none". Resolve
  // that through the active map catalog so these runs still get their underlay.
  if (snapshotMapId && snapshotMapId !== "none") return snapshotMapId;
  if (runMapId && runMapId !== "none") return runMapId;
  return maps.find((entry) => entry.active)?.id ?? "none";
}

function latestFleet(snapshot: OfficialRunTelemetry | null, run: RunSummary | null): FleetTelemetry | null {
  return snapshot?.fleet ?? (run?.latest_telemetry && "cars" in run.latest_telemetry
    ? run.latest_telemetry
    : run?.latest_fleet ?? null);
}

function latestPpo(snapshot: OfficialRunTelemetry | null): Partial<PpoDiagnostics> | null {
  const updates = snapshot?.ppo_diagnostics?.updates ?? [];
  return updates[updates.length - 1]?.values ?? snapshot?.metrics?.ppo ?? null;
}

function sourceTimestamp(data: OfficialRunTelemetry | null): number | null {
  const value = data?.fleet?.ts ?? data?.metrics?.ts ?? data?.updated_at ?? "";
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : null;
}

function ageLabel(frame: RunFrame | undefined, fleet: FleetTelemetry | null): { stale: boolean; text: string } {
  if (!fleet) return { stale: true, text: "Waiting for telemetry" };
  const sourceAt = frame?.sourceAt ?? (Number.isFinite(Date.parse(fleet.ts)) ? Date.parse(fleet.ts) : null);
  if (sourceAt == null) return { stale: true, text: "Telemetry age unknown" };
  const age = Math.max(0, Date.now() - sourceAt);
  return age > STALE_AFTER_MS
    ? { stale: true, text: `Stale · ${Math.floor(age / 1000)}s` }
    : { stale: false, text: `Live · ${Math.floor(age / 1000)}s ago` };
}

async function mapPool<T>(items: T[], concurrency: number, worker: (item: T) => Promise<void>): Promise<void> {
  let next = 0;
  const count = Math.min(items.length, Math.max(1, concurrency));
  await Promise.all(Array.from({ length: count }, async () => {
    while (next < items.length) {
      const item = items[next++];
      await worker(item);
    }
  }));
}

function OverviewTile({
  run,
  frame,
  maps,
  selected,
  onSelect,
}: {
  run: RunSummary;
  frame: RunFrame | undefined;
  maps: MapCatalogEntry[];
  selected: boolean;
  onSelect: () => void;
}) {
  const cardRef = useRef<HTMLElement>(null);
  const [canvasVisible, setCanvasVisible] = useState(false);
  const runIsActive = ACTIVE.has(run.state);
  const snapshot = frame?.data ?? null;
  const fleet = latestFleet(snapshot, run);
  const metrics = snapshot?.metrics ?? run.latest_metrics ?? null;
  const age = ageLabel(frame, fleet);
  const cars = fleet?.cars ?? [];
  const speeds = cars.map((car) => car.speed).filter((speed): speed is number => speed != null && Number.isFinite(speed));
  const movingSpeed = speeds.length ? speeds.reduce((sum, speed) => sum + speed, 0) / speeds.length : null;
  const lapCount = cars.reduce((max, car) => Math.max(max, car.lap_count ?? 0), 0);
  const bestLap = cars.reduce<number | null>((best, car) => {
    const value = car.best_lap_time_s;
    return value != null && Number.isFinite(value) && (best == null || value < best) ? value : best;
  }, null);
  const step = Math.max(metrics?.step ?? 0, fleet?.step ?? 0);
  const mapId = mapIdForWatch(snapshot?.map_id, run.map_id, maps);
  const map = maps.find((entry) => entry.id === mapId);
  const canvasAge = age.stale ? STALE_AFTER_MS + 1 : 0;

  useEffect(() => {
    if (!runIsActive || typeof IntersectionObserver === "undefined") {
      setCanvasVisible(runIsActive);
      return;
    }
    const node = cardRef.current;
    if (!node) return;
    const observer = new IntersectionObserver(([entry]) => {
      setCanvasVisible(entry.isIntersecting);
    }, { rootMargin: "240px 0px" });
    observer.observe(node);
    return () => observer.disconnect();
  }, [run.run_id, runIsActive]);

  return (
    <article
      ref={cardRef}
      className="panel"
      style={{
        minWidth: 0, overflow: "hidden",
        padding: "0.8rem", border: selected ? "1px solid var(--accent, #37d99a)" : undefined,
        background: "var(--panel-bg, rgba(10, 20, 17, 0.84))", color: "inherit",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.5rem" }}>
        <div style={{ minWidth: 0 }}>
          <button type="button" onClick={onSelect} aria-pressed={selected} style={{ maxWidth: "100%", padding: 0, border: 0, background: "transparent", color: "inherit", textAlign: "left", cursor: "pointer" }}>
            <strong style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{run.display_name || run.run_id}</strong>
          </button>
          <small className="meta" title={run.run_id}>{run.run_id}</small>
        </div>
        <span className="badge" data-state={run.state}>{run.state}</span>
      </div>
      {runIsActive && <div className="watch-overview-canvas" style={{ height: 132, minHeight: 0, marginTop: "0.6rem", overflow: "hidden", border: "1px solid var(--line, #263b34)" }}>
        {canvasVisible ? <>
        <FleetCanvas
          showMap
          showFleet
          showFrontier={false}
          showCurrentProgress={false}
          showLidar={false}
          selectedEnvId={-1}
          mapId={mapId}
          mapYamlUrl={mapId === "none" ? null : map?.yaml_url ?? null}
          fleetPanel={fleet}
          telemetryOverride={fleet}
          telemetryAgeMs={canvasAge}
          runState={run.state}
          phaseOverride={snapshot?.training_phase ?? null}
          trainingStep={step}
          rolloutSize={Math.max(1, fleet?.rollout_size ?? metrics?.rollout_size ?? 1)}
        />
        </> : <div className="meta" style={{ display: "grid", height: "100%", placeItems: "center" }}>Map preview loads as this card comes into view</div>}
      </div>}
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", marginTop: "0.6rem", fontSize: "0.82rem" }}>
        <span>Step <strong>{step.toLocaleString()}</strong></span>
        <span className="meta" title={age.text} style={{ color: age.stale ? "var(--warning, #e6b94f)" : undefined }}>{age.text}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "0.5rem", marginTop: "0.7rem" }}>
        <div><small className="meta">Cars</small><strong style={{ display: "block" }}>{cars.length || "—"}</strong></div>
        <div><small className="meta">{cars.every((car) => car.speed_source === "lidar_estimate") ? "Avg LiDAR speed" : "Avg wheel speed"}</small><strong style={{ display: "block" }}>{formatNumber(movingSpeed, 2, " m/s")}</strong></div>
        <div><small className="meta">Laps</small><strong style={{ display: "block" }}>{cars.length ? lapCount : "—"}</strong></div>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", marginTop: "0.55rem", fontSize: "0.78rem" }}>
        <span className="meta">Best lap {formatNumber(bestLap, 3, " s")}</span>
        <span className="meta">10-lap {formatNumber(fleet?.best_10_lap_time_s, 3, " s")}</span>
      </div>
      {frame?.error && <small className="msg err" style={{ display: "block", marginTop: "0.5rem" }}>{frame.error}</small>}
    </article>
  );
}

export function WatchPage() {
  const [runList, setRunList] = useState<OfficialRunList>({ runs: [], max_concurrent_runs: 2 });
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [frames, setFrames] = useState<Record<string, RunFrame>>({});
  const [maps, setMaps] = useState<MapCatalogEntry[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("overview");
  const [showMap, setShowMap] = useState(true);
  const [showFleet, setShowFleet] = useState(true);
  const [showFrontier, setShowFrontier] = useState(true);
  const [showProgress, setShowProgress] = useState(true);
  const [showLidar, setShowLidar] = useState(true);
  const [selectedEnvId, setSelectedEnvId] = useState<number | null>(null);
  const selectedRunRef = useRef<string | null>(null);

  const selectedRun = runList.runs.find((run) => run.run_id === selectedRunId) ?? null;
  const selectedFrame = selectedRunId ? frames[selectedRunId] : undefined;
  const snapshot = selectedFrame?.data ?? null;
  const fleet = latestFleet(snapshot, selectedRun);
  const metrics = snapshot?.metrics ?? selectedRun?.latest_metrics ?? null;
  const phase = snapshot?.training_phase ?? null;
  const cars = fleet?.cars ?? [];
  const mapId = mapIdForWatch(snapshot?.map_id, selectedRun?.map_id, maps);
  const selectedMap = maps.find((map) => map.id === mapId) ?? null;
  const age = ageLabel(selectedFrame, fleet);
  const active = selectedRun != null && ACTIVE.has(selectedRun.state);
  const ppo = latestPpo(snapshot);
  const evaluation = snapshot?.evaluation ?? null;
  const evalResult: EvaluationResult | null = evaluation?.latest_result ?? null;
  const currentStep = Math.max(metrics?.step ?? 0, fleet?.step ?? 0);
  const rolloutSize = Math.max(1, fleet?.rollout_size ?? metrics?.rollout_size ?? 1);
  const nextUpdate = phase?.phase === "rollout" ? phase.step + rolloutSize : Math.ceil((currentStep + 1) / rolloutSize) * rolloutSize;
  const visibleRuns = useMemo(() => showHistory ? runList.runs : runList.runs.filter((run) => ACTIVE.has(run.state)), [runList.runs, showHistory]);
  const runIdsToPoll = useMemo(() => {
    const ids = new Set(runList.runs.filter((run) => ACTIVE.has(run.state)).map((run) => run.run_id));
    if (selectedRunId) ids.add(selectedRunId);
    return [...ids];
  }, [runList.runs, selectedRunId]);
  const runIdsKey = runIdsToPoll.join("|");

  useEffect(() => {
    let cancelled = false;
    void Promise.all([getOfficialRuns(), fetchMaps().catch(() => [] as MapCatalogEntry[])])
      .then(([runs, mapEntries]) => {
        if (cancelled) return;
        setRunList(runs);
        setMaps(mapEntries);
        setSelectedRunId((current) => current && runs.runs.some((run) => run.run_id === current)
          ? current
          : runs.runs.find((run) => ACTIVE.has(run.state))?.run_id ?? runs.runs[0]?.run_id ?? null);
        setLoadError(null);
      })
      .catch((error: unknown) => { if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error)); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const refreshRuns = async () => {
      try {
        const list = await getOfficialRuns();
        if (cancelled) return;
        setRunList(list);
        setSelectedRunId((current) => current && list.runs.some((run) => run.run_id === current)
          ? current
          : list.runs.find((run) => ACTIVE.has(run.state))?.run_id ?? list.runs[0]?.run_id ?? null);
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error));
      }
    };
    void refreshRuns();
    const timer = window.setInterval(() => void refreshRuns(), REFRESH_RUNS_MS);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    selectedRunRef.current = selectedRunId;
    setOfficialRunSelection(selectedRun);
    if (selectedRunId && frames[selectedRunId]?.data) ingestOfficialRunTelemetry(frames[selectedRunId].data!);
  }, [selectedRunId, selectedRun?.state]);

  useEffect(() => {
    if (!runIdsToPoll.length) return;
    let cancelled = false;
    let inFlight = false;
    const ids = runIdsToPoll;
    const refreshTelemetry = async () => {
      if (inFlight || document.visibilityState !== "visible") return;
      inFlight = true;
      await mapPool(ids, TELEMETRY_CONCURRENCY, async (runId) => {
        try {
          const data = viewMode === "focus" && selectedRunRef.current === runId
            ? await getOfficialRunTelemetry(runId)
            : await getOfficialRunLiveTelemetry(runId);
          if (cancelled || data.run_id !== runId) return;
          const frame: RunFrame = { data, receivedAt: Date.now(), sourceAt: sourceTimestamp(data), error: null };
          setFrames((current) => ({ ...current, [runId]: frame }));
          if (selectedRunRef.current === runId) ingestOfficialRunTelemetry(data);
        } catch (error) {
          if (!cancelled) setFrames((current) => ({
            ...current,
            [runId]: { ...current[runId], data: current[runId]?.data ?? null, receivedAt: current[runId]?.receivedAt ?? 0,
              sourceAt: current[runId]?.sourceAt ?? null, error: error instanceof Error ? error.message : String(error) },
          }));
        }
      });
      inFlight = false;
    };
    void refreshTelemetry();
    const timer = window.setInterval(() => void refreshTelemetry(), REFRESH_TELEMETRY_MS);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [runIdsKey, viewMode, selectedRunId]);

  useEffect(() => {
    if (selectedEnvId == null || selectedEnvId === -1) return;
    if (!cars.some((car) => car.env_id === selectedEnvId)) setSelectedEnvId(-1);
  }, [fleet?.run_id, cars.map((car) => car.env_id).join(",")]);

  const focusRun = (runId: string) => {
    setSelectedRunId(runId);
    setViewMode("focus");
  };

  return (
    <section className="panel fleet-panel official-watch-panel">
      <style>{`.watch-overview-canvas .fleet-canvas-wrap { min-height: 0; height: 132px; }`}</style>
      <div className="watch-heading">
        <h2>Watch <small className="watch-runtime-badge">Official simulator</small></h2>
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "flex-end", gap: "0.75rem" }}>
          <div className="watch-view-toggle" role="group" aria-label="Watch layout">
            <button type="button" aria-pressed={viewMode === "overview"} onClick={() => setViewMode("overview")}>All active</button>
            <button type="button" aria-pressed={viewMode === "focus"} onClick={() => setViewMode("focus")}>Focus run</button>
          </div>
          {viewMode === "focus" && <label className="field watch-run-select">Run
            <select value={selectedRunId ?? ""} onChange={(event) => setSelectedRunId(event.target.value || null)}>
              <option value="">Select an official run</option>
              {runList.runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.display_name || run.run_id} · {run.state}</option>)}
            </select>
          </label>}
        </div>
      </div>
      {loadError && <p className="msg err" role="alert">{loadError}</p>}

      {viewMode === "overview" && <>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.75rem", margin: "0.5rem 0 1rem" }}>
          <p className="meta" style={{ margin: 0 }}>
            {showHistory ? `${visibleRuns.length} runs · active and history` : `${visibleRuns.length} active runs`}
            {visibleRuns.length > 0 ? ` · telemetry refreshes every ${REFRESH_TELEMETRY_MS / 1000}s` : ""}
          </p>
          <label><input type="checkbox" checked={showHistory} onChange={(event) => setShowHistory(event.target.checked)} /> Include history</label>
        </div>
        {visibleRuns.length === 0
          ? <p className="meta watch-empty">No active official runs. Start one from Train, or include history to inspect completed runs.</p>
          : <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(min(100%, 270px), 1fr))", gap: "0.75rem", alignItems: "stretch", maxHeight: "72vh", overflowY: "auto", paddingRight: 4 }}>
            {visibleRuns.map((run) => <OverviewTile key={run.run_id} run={run} frame={frames[run.run_id]} maps={maps} selected={run.run_id === selectedRunId} onSelect={() => focusRun(run.run_id)} />)}
          </div>}
      </>}

      {viewMode === "focus" && !selectedRun && <p className="meta watch-empty">No official run selected. Start a run from Train, then select it here.</p>}
      {viewMode === "focus" && selectedRun && <>
        <div className="watch-run-status">
          <span className="badge" data-state={selectedRun.state}>{selectedRun.state}</span>
          <strong>{selectedRun.display_name || selectedRun.run_id}</strong>
          <span>Run ID <code>{selectedRun.run_id}</code></span>
          <span>Step {currentStep.toLocaleString()}</span>
          <span>{age.text}</span>
        </div>
        <div className="fleet-toolbar">
          <label><input type="checkbox" checked={showMap} onChange={(e) => setShowMap(e.target.checked)} /> Map</label>
          <label><input type="checkbox" checked={showFleet} onChange={(e) => setShowFleet(e.target.checked)} /> Cars</label>
          <label><input type="checkbox" checked={showFrontier} onChange={(e) => setShowFrontier(e.target.checked)} /> Frontier</label>
          <label><input type="checkbox" checked={showProgress} onChange={(e) => setShowProgress(e.target.checked)} /> Progress</label>
          <label><input type="checkbox" checked={showLidar} onChange={(e) => setShowLidar(e.target.checked)} /> LiDAR</label>
          {cars.length > 0 && <label>Focus <select value={selectedEnvId ?? cars[0].env_id} onChange={(e) => setSelectedEnvId(Number(e.target.value))}><option value={-1}>All cars</option>{cars.map((car) => <option key={car.env_id} value={car.env_id}>Env {car.env_id}</option>)}</select></label>}
        </div>
        <div className="fleet-layout">
          <div className="fleet-canvas-stack">
            <FleetCanvas showMap={showMap} showFleet={showFleet} showFrontier={showFrontier} showCurrentProgress={showProgress} showLidar={showLidar} selectedEnvId={selectedEnvId ?? -1} mapId={mapId} mapYamlUrl={mapId === "none" ? null : selectedMap?.yaml_url ?? null} fleetPanel={fleet} trainingStep={currentStep} rolloutSize={rolloutSize} />
          </div>
          <aside className="fleet-side">
            <h3>Run telemetry</h3>
            {!fleet ? <p className="meta">No car frame has arrived for this run yet. The run may still be starting.</p> : <>
              <dl className="fleet-stats">
                <div><dt>Episode</dt><dd>{fleet.episode}</dd></div>
                <div><dt>Cars</dt><dd>{cars.length}</dd></div>
                <div><dt>Best 10-lap time</dt><dd>{formatNumber(fleet.best_10_lap_time_s, 3, " s")}</dd></div>
                {selectedRun.log_path && <div><dt>Log</dt><dd title={selectedRun.log_path}>{selectedRun.log_path}</dd></div>}
              </dl>
              <div className="official-car-list">{cars.map((car) => <button type="button" key={car.env_id} onClick={() => setSelectedEnvId(car.env_id)} data-selected={selectedEnvId === car.env_id}>
                <span>Env {car.env_id}</span><strong>{formatNumber(car.speed, 2)} m/s</strong><small>{car.speed_source === "lidar_estimate" ? "LiDAR speed estimate" : "Wheel speed estimate"} · Lap {car.lap_count ?? 0} · collisions {car.collision_count ?? (car.collision ? 1 : 0)}</small>
              </button>)}</div>
            </>}
          </aside>
        </div>

        <section className="official-watch-data">
          <article className="training-progress-panel"><h3>Training metrics</h3>
            {metrics ? <div className="official-metric-grid">
              <div><span>Step</span><strong>{metrics.step.toLocaleString()}</strong></div>
              <div><span>Rollout reward</span><strong>{formatNumber(metrics.reward, 3)}</strong></div>
              <div><span>Episode</span><strong>{metrics.episode.toLocaleString()}</strong></div>
              <div><span>Loss</span><strong>{formatNumber(metrics.loss, 5)}</strong></div>
              <div><span>PPO phase</span><strong>{phase?.phase ?? "—"}</strong></div>
              <div><span>Next update</span><strong>{active ? nextUpdate.toLocaleString() : "—"}</strong></div>
            </div> : <p className="meta">Waiting for this run’s training metrics.</p>}
          </article>
          <article className="training-progress-panel"><h3>Evaluation</h3>
            {evaluation ? <>
              <p><span className="badge" data-state={evaluation.state}>{evaluation.state}</span></p>
              {evalResult ? <div className="official-metric-grid">
                <div><span>10-lap time</span><strong>{formatNumber(evalResult.best_10_lap_time_s, 3, " s")}</strong></div>
                <div><span>Laps observed</span><strong>{evalResult.laps_observed}</strong></div>
                <div><span>Frontier pace</span><strong>{formatNumber(evalResult.frontier_speed_mps, 3, " m/s")}</strong></div>
                <div><span>Collisions</span><strong>{evalResult.collisions}</strong></div>
                <div><span>Failed episodes</span><strong>{evalResult.failed_episodes}</strong></div>
                <div><span>Evaluation attempts</span><strong>{evalResult.evaluation_runs ?? "—"}</strong></div>
              </div> : <p className="meta">No completed evaluation result for this run yet.</p>}
              {snapshot?.evaluation_history?.length ? <p className="meta">{snapshot.evaluation_history.length} saved evaluations for this run.</p> : null}
            </> : <p className="meta">Waiting for evaluation data from this run.</p>}
          </article>
          <article className="training-progress-panel"><h3>PPO diagnostics</h3>
            {ppo ? <div className="official-metric-grid">
              {([ ["Learning rate", ppo.learning_rate], ["Explained variance", ppo.explained_variance], ["Approx. KL", ppo.approx_kl], ["Clip fraction", ppo.clip_fraction], ["Entropy loss", ppo.entropy_loss], ["Policy loss", ppo.policy_gradient_loss], ["Value loss", ppo.value_loss], ["Action std", ppo.std], ["FPS", ppo.fps] ] as const).map(([label, value]) => <div key={label}><span>{label}</span><strong>{formatNumber(value, 5)}</strong></div>)}
            </div> : <p className="meta">PPO diagnostics appear after the first policy update.</p>}
          </article>
        </section>
        <p className="meta official-watch-footnote">This view is scoped to run <code>{selectedRun.run_id}</code>. {age.stale ? "The most recent frame is stale; metrics remain visible with their last received values." : "Telemetry is current."}</p>
      </>}
    </section>
  );
}
