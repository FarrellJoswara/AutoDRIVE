import { useEffect, useState } from "react";
import { fetchReplayModels, getReplayStatus, resetReplay, startReplay, stopReplay, getBestOfficialEvaluation, getOfficialEvaluationHistory,
  type ModelCheckpoint, type ReplayStatus, type OfficialChampion, type OfficialEvaluationRecord } from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";

export function ReplayPage() {
  const [models, setModels] = useState<ModelCheckpoint[]>([]);
  const [modelId, setModelId] = useState("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [status, setStatus] = useState<ReplayStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [best, setBest] = useState<OfficialChampion | null>(null);
  const [history, setHistory] = useState<OfficialEvaluationRecord[]>([]);
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
      try { const next = await getReplayStatus(); if (!cancelled) setStatus(next); }
      catch (reason) { if (!cancelled) setError(String(reason)); }
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
  return <section className="panel fleet-panel">
    <h2>Official evaluation / replay</h2>
    <p className="lede">Evaluate a saved policy on the official simulator: one warm-up lap followed by ten scored laps, repeated in three fresh simulator processes. Automatic training evaluations use the same protocol.</p>
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
      <button className="primary" disabled={busy || active || modelsLoading || !modelId} onClick={() => void perform(() => startReplay({model_id: modelId, map_id: "none", seed: 0, device: "cpu"}))}>Start evaluation</button>
      <button disabled={busy || !active} onClick={() => void perform(stopReplay)}>Stop evaluation</button>
      <button disabled={busy || active || !status?.run_id} onClick={() => void perform(resetReplay)}>Run another evaluation</button>
      <button disabled={busy || modelsLoading} onClick={() => void refreshModels().catch((reason) => setError(String(reason)))}>Refresh checkpoints</button>
    </div>
    <p role="status">{status?.state ?? "idle"} · {status?.attempts_completed ?? 0}/{status?.attempts_required ?? 3} attempts · {status?.run_id ?? "No simulator attempt active"} · {fleet ? (age > 1500 ? "Telemetry stale" : "Live telemetry") : "Waiting for telemetry"}</p>
    {status?.simulator_backend === "windows_external" && status.simulator_connections?.length ? <section className="panel" aria-label="Windows simulator connection" style={{ marginTop: 12 }}>
      <h3 style={{ marginTop: 0 }}>Connect the official Windows simulator</h3>
      <p className="meta">This checkpoint was trained with the graphics-enabled official Windows simulator. Start it on the assigned bridge port so replay uses the same camera and sensor workflow.</p>
      <p>Environment 0: <code>{status.simulator_connections[0].host}:{status.simulator_connections[0].port}</code></p>
      <p className="meta">Run <code>scripts/start_official_windows_simulators.ps1 -SimulatorExePath &lt;AutoDRIVE Simulator.exe&gt; -Ports {status.simulator_connections.map((connection) => connection.port).join(",")} -RunId {status.run_id} -HubUrl {window.location.origin}</code> in PowerShell.</p>
    </section> : null}
    {(error || status?.error) && <p className="error" role="alert">{error || status?.error}</p>}
    {result && <div className="panel">
      <h3>Official attempt result</h3>
      <p>{result.valid ? `Median adjusted time: ${result.median_adjusted_race_time_s?.toFixed(2)} s.` : "This batch did not produce three valid attempts and cannot promote the best model."} Valid attempts: {result.completed_attempts}/{result.attempts_required ?? 3}.</p>
      {result.runs.map((run, index) => <p key={index}>Attempt {index + 1}: {run.stop_reason ?? run.score_status ?? "finished"}</p>)}
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
    <FleetCanvas fleetSource="replay" showMap={false} showFleet showFrontier={false} showCurrentProgress={false} showLidar
      selectedEnvId={0} mapId="none" mapYamlUrl={null} fleetPanel={fleet}
      telemetryOverride={fleet} telemetryAgeMs={age} runState={status?.state} />
    {status?.log_path && <p className="meta">Log: {status.log_path}</p>}
  </section>;
}
