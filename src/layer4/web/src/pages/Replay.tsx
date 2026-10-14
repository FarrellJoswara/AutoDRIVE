import { useEffect, useState } from "react";
import { fetchReplayModels, getOfficialModelMetadata, getReplayStatus, resetReplay, startReplay, stopReplay, getBestOfficialEvaluation, getOfficialEvaluationHistory,
  getOfficialRunCameraUrl, type ModelCheckpoint, type OfficialModelMetadata, type ReplayStatus, type OfficialChampion, type OfficialEvaluationRecord } from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";

function modelIdFromPath(path: string | null | undefined, models: ModelCheckpoint[]): string | null {
  if (!path) return null;
  let normalized = path.replace(/\\/g, "/");
  normalized = normalized.replace(/^.*\/logs\/rl\//i, "");
  normalized = normalized.replace(/^\/(?:runs\/rl|app\/logs\/rl)\//i, "").replace(/^\/+/, "");
  return models.some((model) => model.id === normalized) ? normalized : null;
}

export function ReplayPage() {
  const [models, setModels] = useState<ModelCheckpoint[]>([]);
  const [modelId, setModelId] = useState("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [bestLoading, setBestLoading] = useState(true);
  const [status, setStatus] = useState<ReplayStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [best, setBest] = useState<OfficialChampion | null>(null);
  const [bestError, setBestError] = useState<string | null>(null);
  const [history, setHistory] = useState<OfficialEvaluationRecord[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedMetadata, setSelectedMetadata] = useState<OfficialModelMetadata | null>(null);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [metadataRefreshVersion, setMetadataRefreshVersion] = useState(0);
  const [cameraRefresh, setCameraRefresh] = useState(0);
  const [cameraReady, setCameraReady] = useState(false);
  async function refreshModels() {
    setModelsLoading(true);
    try {
      const available = await fetchReplayModels();
      setModels(available);
    } finally {
      setModelsLoading(false);
    }
  }
  async function refreshBest() {
    try {
      const result = await getBestOfficialEvaluation();
      setBest(result.best);
      setBestError(null);
    } catch (reason) {
      setBestError(`Best-model information could not be refreshed: ${String(reason)}`);
    } finally {
      setBestLoading(false);
    }
  }
  async function refreshHistory() {
    try {
      setHistory(await getOfficialEvaluationHistory());
      setHistoryError(null);
    } catch (reason) {
      setHistoryError(`Evaluation history could not be refreshed: ${String(reason)}`);
    }
  }
  async function refreshAll() {
    await Promise.all([refreshModels(), refreshBest(), refreshHistory()]);
    setMetadataRefreshVersion((value) => value + 1);
  }
  useEffect(() => {
    void refreshModels().catch((reason) => setError(String(reason)));
    void refreshBest();
    void refreshHistory();
    let cancelled = false;
    let timer: number;
    async function poll() {
      try {
        const next = await getReplayStatus();
        if (!cancelled) {
          setStatus(next);
          setPollError(null);
        }
      } catch (reason) {
        if (!cancelled) setPollError(String(reason));
      }
      if (!cancelled) timer = window.setTimeout(poll, 500);
    }
    void poll();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, []);
  useEffect(() => {
    const timer = window.setInterval(() => {
      void refreshBest();
      void refreshHistory();
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    if (modelsLoading || bestLoading) return;
    setModelId((current) => {
      if (models.some((model) => model.id === current)) return current;
      return modelIdFromPath(best?.model_path, models) ?? models[0]?.id ?? "";
    });
  }, [models, modelsLoading, best, bestLoading]);
  useEffect(() => {
    if (!modelId) {
      setSelectedMetadata(null);
      setMetadataError(null);
      return;
    }
    let cancelled = false;
    setSelectedMetadata(null);
    setMetadataError(null);
    void getOfficialModelMetadata(modelId).then((metadata) => {
      if (!cancelled && metadata.id === modelId) setSelectedMetadata(metadata);
    }).catch((reason) => {
      if (!cancelled) setMetadataError(`Selected checkpoint identity could not be verified: ${String(reason)}`);
    });
    return () => { cancelled = true; };
  }, [modelId, metadataRefreshVersion]);
  useEffect(() => {
    const timer = window.setInterval(() => setCameraRefresh((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => { setCameraReady(false); }, [status?.run_id]);
  async function perform(action: () => Promise<ReplayStatus>) {
    setBusy(true); setError(null);
    try { setStatus(await action()); } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  }
  async function startSelected(validation_mode?: "bc_diagnostic" | "training_probe") {
    if (!modelId || !selectedMetadata) return;
    setBusy(true);
    setError(null);
    try {
      const currentMetadata = await getOfficialModelMetadata(modelId);
      if (currentMetadata.sha256 !== selectedMetadata.sha256) {
        setSelectedMetadata(currentMetadata);
        setError("This checkpoint changed since it was reviewed. Check the updated SHA-256 before starting.");
        return;
      }
      setStatus(await startReplay({ model_id: modelId, map_id: "none", seed: 0, device: "cpu", validation_mode }));
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(false);
    }
  }
  const active = ["queued", "starting", "running", "stopping"].includes(status?.state ?? "idle");
  const fleet = status?.last_fleet ?? null;
  const sourceTime = fleet ? Date.parse(fleet.ts) : NaN;
  const age = Number.isFinite(sourceTime) ? Math.max(0, Date.now() - sourceTime) : Infinity;
  const result = status?.evaluation;
  const validationMode = status?.validation_mode ?? result?.validation_mode ?? null;
  const diagnostic = validationMode === "bc_diagnostic" || validationMode === "training_probe";
  const diagnosticLabel = validationMode === "training_probe" ? "Training probe" : "BC diagnostic";
  const diagnosticRun = diagnostic && result ? result.runs[0] : undefined;
  const scoredLapTarget = diagnosticRun?.race_laps_required ?? result?.race_laps_required ?? (validationMode === "training_probe" ? 3 : 10);
  const car = fleet?.cars.find((item) => item.env_id === 0);
  const selectedModel = models.find((model) => model.id === modelId);
  const selectedIsBest = Boolean(selectedMetadata && best?.model_sha256 && selectedMetadata.sha256 === best.model_sha256);
  const resultMatchesSelection = Boolean(result?.model_sha256 && selectedMetadata?.sha256 === result.model_sha256);
  const progress = car?.frontier_progress_m ?? diagnosticRun?.frontier_progress_m;
  const progressFraction = car?.frontier_progress_fraction ?? diagnosticRun?.frontier_progress_fraction;
  const projectionValid = car?.frontier_projection_valid ?? car?.route_projection_valid ?? diagnosticRun?.frontier_projection_valid;
  const seconds = (value?: number | null) => Number.isFinite(value) ? `${value!.toFixed(2)} s` : "—";
  const meters = (value?: number | null) => Number.isFinite(value) ? `${value!.toFixed(2)} m` : "—";
  return <section className="panel fleet-panel">
    <h2>Official evaluation / replay</h2>
    <p className="lede">Run a competitive evaluation with one warm-up lap and ten scored laps across three fresh simulator processes, or select a noncompetitive diagnostic for a single checkpoint.</p>
    <p className="meta">Diagnostics run without teacher control or PPO updates. BC diagnostics require an authenticated behavior-cloning checkpoint and use one warm-up plus ten scored laps. Training probes require authenticated PPO lineage with an approved BC ancestor and use one warm-up plus three scored laps. Both are noncompetitive and never promote a champion.</p>
    <section className="panel" aria-label="Best official model">
      <h3 style={{ marginTop: 0 }}>Best official model across runs</h3>
      {best ? <p>{best.model_path} · median adjusted <strong>10-lap total {best.median_adjusted_race_time_s.toFixed(2)} s</strong> over {best.attempts} valid races. SHA-256 <code>{best.model_sha256}</code>.</p>
        : <p className="meta">No checkpoint has completed three valid official races yet.</p>}
      {bestError && <p className="error" role="alert">{bestError}</p>}
    </section>
    <label className="field">Checkpoint
      <select value={modelId} disabled={active || busy} onChange={(event) => setModelId(event.target.value)}>
        {(modelsLoading || bestLoading) && <option value="">Loading official checkpoints and verified best model…</option>}
        {!modelsLoading && !models.length && <option value="">No compatible official checkpoints found</option>}
        {models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
      </select>
    </label>
    <div className="actions">
      <button className="primary" disabled={busy || active || modelsLoading || bestLoading || !modelId || !selectedMetadata} onClick={() => void startSelected()}>Start competitive evaluation</button>
      <button disabled={busy || active || modelsLoading || bestLoading || !modelId || !selectedMetadata} aria-label="Start noncompetitive BC diagnostic" onClick={() => void startSelected("bc_diagnostic")}>Start BC diagnostic · noncompetitive</button>
      <button disabled={busy || active || modelsLoading || bestLoading || !modelId || !selectedMetadata} aria-label="Start noncompetitive PPO training probe" onClick={() => void startSelected("training_probe")}>Start PPO training probe · noncompetitive</button>
      <button disabled={busy || !active} onClick={() => void perform(stopReplay)}>Stop evaluation</button>
      <button disabled={busy || active || !status?.run_id} onClick={() => void perform(resetReplay)}>Run another evaluation</button>
      <button disabled={busy || modelsLoading} onClick={() => void refreshAll().catch((reason) => setError(String(reason)))}>Refresh checkpoints and evaluation history</button>
    </div>
    <p role="status">{status?.state ?? "idle"} · {status?.validation_mode ? `${status.validation_mode} · noncompetitive` : "competitive"} · {status?.attempts_completed ?? 0}/{status?.attempts_required ?? 3} attempts · {status?.run_id ?? "No simulator attempt active"} · {fleet ? (age > 1500 ? "Telemetry stale" : "Live telemetry") : "Waiting for telemetry"}</p>
    {selectedModel && <div className="meta" aria-label="Selected checkpoint identity">
      <p>Selected checkpoint: <strong>{selectedModel.label}</strong>{status?.model ? ` · evaluated ${status.model}` : ""}</p>
      {selectedMetadata ? <p>SHA-256 <code>{selectedMetadata.sha256}</code>{selectedIsBest ? " · matches current best evaluated model" : ""} · {selectedMetadata.observation_profile} · {selectedMetadata.simulator_backend} · throttle {selectedMetadata.throttle_mode}/{selectedMetadata.negative_throttle_mode} · steering {selectedMetadata.steering_mode}</p>
        : !metadataError ? <p>Checking the selected checkpoint’s file hash and saved settings…</p> : null}
      {metadataError && <p className="error" role="alert">{metadataError}</p>}
    </div>}
    {status?.simulator_backend === "windows_external" && status.simulator_connections?.length ? <section className="panel" aria-label="Windows simulator connection" style={{ marginTop: 12 }}>
      <h3 style={{ marginTop: 0 }}>Connect the official Windows simulator</h3>
      <p className="meta">This checkpoint was trained with the graphics-enabled official Windows simulator. Start it on the assigned bridge port so replay uses the same camera and sensor workflow.</p>
      <p>Environment 0: <code>{status.simulator_connections[0].host}:{status.simulator_connections[0].port}</code></p>
      <p className="meta">Run <code>scripts/start_official_windows_simulators.ps1 -SimulatorExePath &lt;AutoDRIVE Simulator.exe&gt; -Ports {status.simulator_connections.map((connection) => connection.port).join(",")} -RunId {status.run_id} -HubUrl {window.location.origin}</code> in PowerShell.</p>
    </section> : null}
    {(error || status?.error || pollError) && <p className="error" role="alert">{error || status?.error || pollError}</p>}
    {diagnostic && <section className="panel" aria-label={`${diagnosticLabel} telemetry`}>
      <h3>{diagnosticLabel} · noncompetitive</h3>
      <p>Route: official IROS 2026 centerline · projection {projectionValid == null ? "waiting" : projectionValid ? "valid" : "invalid"} · frontier progress {meters(progress)} · scored target {Number.isFinite(progressFraction) ? `${Math.round(progressFraction! * 100)}%` : "—"} · {fleet ? (age > 1500 ? "telemetry stale" : "telemetry current") : "waiting for telemetry"} · camera {car?.camera_fresh === true ? "fresh" : car?.camera_fresh === false ? "stale" : "waiting"}{car?.camera_age_s != null ? ` (${seconds(car.camera_age_s)})` : ""}</p>
      <div className="training-progress-panel" style={{ width: "100%", aspectRatio: "16 / 9", background: "#080d0b", overflow: "hidden", display: "grid", placeItems: "center" }}>
        {status?.run_id && cameraReady
          ? <img src={getOfficialRunCameraUrl(status.run_id, 0, cameraRefresh)} alt={`Official simulator camera during ${diagnosticLabel.toLowerCase()}`} style={{ width: "100%", height: "100%", objectFit: "contain" }} />
          : <span className="meta">{active ? "Waiting for a fresh diagnostic camera frame…" : "No diagnostic camera frame is available."}</span>}
        {status?.run_id && <img src={getOfficialRunCameraUrl(status.run_id, 0, cameraRefresh)} alt="" aria-hidden="true" onLoad={() => setCameraReady(true)} onError={() => setCameraReady(false)} style={{ display: "none" }} />}
      </div>
      {diagnosticRun && <div className="official-metric-grid">
        <div><span>Readiness</span><strong>{result?.readiness_passed ? "Passed" : "Not passed"}</strong></div>
        <div><span>Warm-up laps</span><strong>{diagnosticRun.warmup_laps ?? 0}/1 · {seconds(diagnosticRun.warmup_lap_times_s?.[0])}</strong></div>
        <div><span>Scored laps</span><strong>{diagnosticRun.race_laps_completed ?? 0}/{scoredLapTarget}</strong></div>
        <div><span>Race time</span><strong>{seconds(diagnosticRun.race_time_s)}</strong></div>
        <div><span>Wall time</span><strong>{seconds(diagnosticRun.wall_duration_s)}</strong></div>
        <div><span>Collisions</span><strong>{diagnosticRun.collision_count ?? 0}</strong></div>
        <div><span>Camera samples</span><strong>{diagnosticRun.camera_sample_count ?? "—"}</strong></div>
        <div><span>Fresh / stale / missing / unknown</span><strong>{diagnosticRun.camera_fresh_sample_count ?? "—"} / {diagnosticRun.camera_stale_sample_count ?? "—"} / {diagnosticRun.camera_missing_sample_count ?? "—"} / {diagnosticRun.camera_freshness_unknown_count ?? "—"}</strong></div>
        <div><span>Camera age mean / max</span><strong>{seconds(diagnosticRun.camera_age_s?.mean)} / {seconds(diagnosticRun.camera_age_s?.max)}</strong></div>
        <div><span>Frontier offset</span><strong>{meters(diagnosticRun.frontier_lateral_offset_m)}</strong></div>
      </div>}
    </section>}
    {result && <div className="panel" aria-label={diagnostic ? `Noncompetitive ${diagnosticLabel} result` : "Official attempt result"}>
      <h3>{diagnostic ? `${diagnosticLabel} result · noncompetitive` : "Official attempt result"}</h3>
      <p className="meta">Protocol: {result.protocol ?? (diagnostic ? validationMode : "official_10_lap_race")} · {diagnostic ? "noncompetitive; not a champion score" : "competitive; three fresh attempts, one warm-up plus ten scored laps each"} · evaluated model SHA-256 <code>{result.model_sha256 ?? "not reported"}</code>{result.model_sha256 && selectedMetadata ? (resultMatchesSelection ? " · matches selection" : " · differs from current selection") : ""}</p>
      {diagnostic
        ? <p>{result.readiness_passed ? "Driving gate passed: " : "Driving gate not passed: "}one noncompetitive attempt; no competitive score or champion promotion. Attempts: {result.completed_attempts}/{result.attempts_required ?? 1}.</p>
        : <p>{result.valid ? `Median adjusted 10-lap total: ${result.median_adjusted_race_time_s?.toFixed(2)} s.` : "This batch did not produce three valid attempts and cannot promote the best model."} Valid attempts: {result.completed_attempts}/{result.attempts_required ?? 3}.</p>}
      {result.runs.map((run, index) => <div key={index}>
        <p>Attempt {index + 1}: {run.stop_reason ?? run.score_status ?? "finished"} · warm-up {run.warmup_laps ?? 0}/1, scored {run.race_laps_completed ?? 0}/{run.race_laps_required ?? scoredLapTarget}, collisions {run.collision_count ?? 0}{run.race_collisions != null ? ` (race ${run.race_collisions}${run.warmup_collisions != null ? `, warm-up ${run.warmup_collisions}` : ""})` : ""}</p>
        <p className="meta">Race time {seconds(run.race_time_s)} · adjusted total {seconds(run.adjusted_race_time_s)} · wall time {seconds(run.wall_duration_s)}{run.race_lap_times_s?.length ? ` · scored laps: ${run.race_lap_times_s.map((lap) => seconds(lap)).join(", ")}` : ""}</p>
        {!diagnostic && <p className="meta">Camera samples fresh / stale / missing / unknown: {run.camera_fresh_sample_count ?? "—"} / {run.camera_stale_sample_count ?? "—"} / {run.camera_missing_sample_count ?? "—"} / {run.camera_freshness_unknown_count ?? "—"} of {run.camera_sample_count ?? "—"}{run.camera_age_s ? ` · camera age mean / max ${seconds(run.camera_age_s.mean)} / ${seconds(run.camera_age_s.max)}` : ""}; route projection {run.route_projection_valid == null ? "not reported" : run.route_projection_valid ? "valid" : "invalid"}{run.route_monitor ? ` · ${run.route_monitor.projection_valid_sample_count ?? 0}/${run.route_monitor.projection_sample_count ?? 0} valid route samples${run.route_monitor_only ? " (observational only; no scoring/stop effect)" : ""}` : ""}</p>}
        {!diagnostic && run.official_scene_sha256 && run.route_sha256 && <p className="meta">Official scene SHA-256 <code>{run.official_scene_sha256}</code> · route SHA-256 <code>{run.route_sha256}</code></p>}
        {!diagnostic && run.simulator_identity && <p className="meta">Simulator PID {run.simulator_identity.pid ?? "—"} · port {run.simulator_identity.port ?? "—"} · process manifest SHA-256 <code>{run.simulator_identity.manifest_sha256 ?? "not reported"}</code></p>}
        {diagnostic && run.race_lap_times_s?.length ? <p className="meta">Scored lap times: {run.race_lap_times_s.map((lap, lapIndex) => `${lapIndex + 1}: ${seconds(lap)}`).join(" · ")}</p> : null}
        {diagnostic && run.policy_inference_latency_s && <p className="meta">Policy inference latency: mean {seconds(run.policy_inference_latency_s.mean)} · max {seconds(run.policy_inference_latency_s.max)}</p>}
      </div>)}
    </div>}
    <section className="panel" aria-label="Official evaluation history">
      <h3 style={{ marginTop: 0 }}>Evaluation history across all training runs</h3>
      {historyError && <p className="error" role="alert">{historyError}</p>}
      {history.length ? <ol>{history.slice(0, 20).map((item) => <li key={item.evaluation_id}>
        {item.source} · {item.model_path} · {item.result?.valid
          ? `median ${item.result.median_adjusted_race_time_s?.toFixed(2) ?? "—"} s`
          : `${item.result?.completed_attempts ?? 0}/${item.result?.attempts_required ?? 3} valid attempts (${item.state})`}
        {item.parent_run_id ? ` · training ${item.parent_run_id}` : ""}
      </li>)}</ol> : <p className="meta">Completed official evaluations will appear here.</p>}
    </section>
    <FleetCanvas fleetSource="replay" showMap={false} showFleet showFrontier={diagnostic} showCurrentProgress={diagnostic} showLidar
      selectedEnvId={0} mapId="none" mapYamlUrl={null} fleetPanel={fleet}
      telemetryOverride={fleet} telemetryAgeMs={age} runState={status?.state} />
    {status?.log_path && <p className="meta">Log: {status.log_path}</p>}
  </section>;
}
