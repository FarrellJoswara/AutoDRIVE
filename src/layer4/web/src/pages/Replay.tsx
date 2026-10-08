import { useEffect, useState } from "react";
import { fetchReplayModels, getReplayStatus, resetReplay, startReplay, stopReplay,
  type ModelCheckpoint, type ReplayStatus } from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";

export function ReplayPage() {
  const [models, setModels] = useState<ModelCheckpoint[]>([]);
  const [modelId, setModelId] = useState("");
  const [status, setStatus] = useState<ReplayStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function refreshModels() {
    const available = await fetchReplayModels();
    setModels(available);
    setModelId((current) => available.some((model) => model.id === current) ? current : available[0]?.id ?? "");
  }
  useEffect(() => {
    void refreshModels().catch((reason) => setError(String(reason)));
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
  async function perform(action: () => Promise<ReplayStatus>) {
    setBusy(true); setError(null);
    try { setStatus(await action()); } catch (reason) { setError(String(reason)); }
    finally { setBusy(false); }
  }
  const active = ["starting", "running", "stopping"].includes(status?.state ?? "idle");
  const fleet = status?.last_fleet ?? null;
  const sourceTime = fleet ? Date.parse(fleet.ts) : NaN;
  const age = Number.isFinite(sourceTime) ? Math.max(0, Date.now() - sourceTime) : Infinity;
  const result = status?.evaluation;
  return <section className="panel fleet-panel">
    <h2>Official evaluation / replay</h2>
    <p className="lede">Run a saved policy with deterministic actions in a fresh official simulator. Simulator timing can still vary. Its observation profile and controls come from the training record. Restart creates a new race attempt.</p>
    <label className="field">Checkpoint
      <select value={modelId} disabled={active || busy} onChange={(event) => setModelId(event.target.value)}>
        {!models.length && <option value="">No compatible official checkpoints found</option>}
        {models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
      </select>
    </label>
    <div className="actions">
      <button className="primary" disabled={busy || active || !modelId} onClick={() => void perform(() => startReplay({model_id: modelId, map_id: "none", seed: 0, device: "cpu"}))}>Start evaluation</button>
      <button disabled={busy || !active} onClick={() => void perform(stopReplay)}>Stop evaluation</button>
      <button disabled={busy || !status?.run_id} onClick={() => void perform(resetReplay)}>Restart attempt</button>
      <button disabled={busy} onClick={() => void refreshModels().catch((reason) => setError(String(reason)))}>Refresh checkpoints</button>
    </div>
    <p role="status">{status?.state ?? "idle"} · {status?.run_id ?? "No attempt selected"} · {fleet ? (age > 1500 ? "Telemetry stale" : "Live telemetry") : "Waiting for telemetry"}</p>
    {(error || status?.error) && <p className="error" role="alert">{error || status?.error}</p>}
    {result && <div className="panel">
      <h3>Official attempt result</h3>
      <p>Completed valid attempts: {result.completed_attempts}. Adjusted race time: {result.best_adjusted_race_time_s == null ? "No valid completed score" : `${result.best_adjusted_race_time_s.toFixed(2)} s`}.</p>
      {result.runs.map((run, index) => <p key={index}>Attempt {index + 1}: {run.stop_reason ?? run.score_status ?? "finished"}</p>)}
    </div>}
    <FleetCanvas fleetSource="replay" showMap={false} showFleet showFrontier={false} showCurrentProgress={false} showLidar
      selectedEnvId={0} mapId="none" mapYamlUrl={null} fleetPanel={fleet}
      telemetryOverride={fleet} telemetryAgeMs={age} runState={status?.state} />
    {status?.log_path && <p className="meta">Log: {status.log_path}</p>}
  </section>;
}
