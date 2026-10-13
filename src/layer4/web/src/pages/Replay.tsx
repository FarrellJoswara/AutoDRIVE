import { useEffect, useState } from "react";
import { fetchReplayModels, getReplayStatus, resetReplay, startReplay, stopReplay, getBestOfficialEvaluation, getOfficialEvaluationHistory,
  getOfficialRunCameraUrl, type ModelCheckpoint, type ReplayStatus, type OfficialChampion, type OfficialEvaluationRecord } from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";

export function ReplayPage() {
  const [models, setModels] = useState<ModelCheckpoint[]>([]);
  const [modelId, setModelId] = useState("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [status, setStatus] = useState<ReplayStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [best, setBest] = useState<OfficialChampion | null>(null);
  const [history, setHistory] = useState<OfficialEvaluationRecord[]>([]);
  const [cameraRefresh, setCameraRefresh] = useState(0);
  const [cameraReady, setCameraReady] = useState(false);
  async function refreshModels() {
    setModelsLoading(true);
    try {
      const available = await fetchReplayModels();
      setModels(available);
      setModelId((current) => available.some((model) => model.id === current) ? current : available[0]?.id ?? "");
    } finally {
      setModelsLoading(false);
    }
  }
  useEffect(() => {
    void refreshModels().catch((reason) => setError(String(reason)));
    void getBestOfficialEvaluation().then((result) => setBest(result.best)).catch(() => undefined);
    void getOfficialEvaluationHistory().then(setHistory).catch(() => undefined);
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
      void getBestOfficialEvaluation().then((result) => setBest(result.best)).catch(() => undefined);
      void getOfficialEvaluationHistory().then(setHistory).catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);
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
      {best ? <p>{best.model_path} · median adjusted time <strong>{best.median_adjusted_race_time_s.toFixed(2)} s</strong> over {best.attempts} valid races.</p>
        : <p className="meta">No checkpoint has completed three valid official races yet.</p>}
    </section>
    <label className="field">Checkpoint
      <select value={modelId} disabled={active || busy} onChange={(event) => setModelId(event.target.value)}>
        {modelsLoading && <option value="">Loading official checkpoints…</option>}
        {!modelsLoading && !models.length && <option value="">No compatible official checkpoints found</option>}
        {models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
      </select>
    </label>
    <div className="actions">
      <button className="primary" disabled={busy || active || modelsLoading || !modelId} onClick={() => void perform(() => startReplay({model_id: modelId, map_id: "none", seed: 0, device: "cpu"}))}>Start competitive evaluation</button>
      <button disabled={busy || active || modelsLoading || !modelId} aria-label="Start noncompetitive BC diagnostic" onClick={() => void perform(() => startReplay({model_id: modelId, map_id: "none", seed: 0, device: "cpu", validation_mode: "bc_diagnostic"}))}>Start BC diagnostic · noncompetitive</button>
      <button disabled={busy || active || modelsLoading || !modelId} aria-label="Start noncompetitive PPO training probe" onClick={() => void perform(() => startReplay({model_id: modelId, map_id: "none", seed: 0, device: "cpu", validation_mode: "training_probe"}))}>Start PPO training probe · noncompetitive</button>
      <button disabled={busy || !active} onClick={() => void perform(stopReplay)}>Stop evaluation</button>
      <button disabled={busy || active || !status?.run_id} onClick={() => void perform(resetReplay)}>Run another evaluation</button>
      <button disabled={busy || modelsLoading} onClick={() => void refreshModels().catch((reason) => setError(String(reason)))}>Refresh checkpoints</button>
    </div>
    <p role="status">{status?.state ?? "idle"} · {status?.attempts_completed ?? 0}/{status?.attempts_required ?? 3} attempts · {status?.run_id ?? "No simulator attempt active"} · {fleet ? (age > 1500 ? "Telemetry stale" : "Live telemetry") : "Waiting for telemetry"}</p>
    {selectedModel && <p className="meta">Selected checkpoint: <strong>{selectedModel.label}</strong>{status?.model ? ` · evaluated ${status.model}` : ""}</p>}
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
      {diagnostic
        ? <p>{result.readiness_passed ? "Driving gate passed: " : "Driving gate not passed: "}one noncompetitive attempt; no competitive score or champion promotion. Attempts: {result.completed_attempts}/{result.attempts_required ?? 1}.</p>
        : <p>{result.valid ? `Median adjusted time: ${result.median_adjusted_race_time_s?.toFixed(2)} s.` : "This batch did not produce three valid attempts and cannot promote the best model."} Valid attempts: {result.completed_attempts}/{result.attempts_required ?? 3}.</p>}
      {result.runs.map((run, index) => <div key={index}>
        <p>Attempt {index + 1}: {run.stop_reason ?? run.score_status ?? "finished"}{diagnostic ? ` · warm-up ${run.warmup_laps ?? 0}, scored ${run.race_laps_completed ?? 0}/${run.race_laps_required ?? scoredLapTarget}, collisions ${run.collision_count ?? 0}` : ""}</p>
        {diagnostic && run.race_lap_times_s?.length ? <p className="meta">Scored lap times: {run.race_lap_times_s.map((lap, lapIndex) => `${lapIndex + 1}: ${seconds(lap)}`).join(" · ")}</p> : null}
        {diagnostic && run.policy_inference_latency_s && <p className="meta">Policy inference latency: mean {seconds(run.policy_inference_latency_s.mean)} · max {seconds(run.policy_inference_latency_s.max)}</p>}
      </div>)}
    </div>}
    <section className="panel" aria-label="Official evaluation history">
      <h3 style={{ marginTop: 0 }}>Evaluation history across all training runs</h3>
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
