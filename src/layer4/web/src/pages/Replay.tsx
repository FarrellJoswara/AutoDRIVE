import { useEffect, useMemo, useState } from "react";
import {
  fetchMaps,
  fetchReplayModels,
  getSettings,
  resetReplay,
  startReplay,
  stopReplay,
  type MapCatalogEntry,
  type ModelCheckpoint,
} from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";
import { useHubStore } from "../store";

export function ReplayPage() {
  const { replayStatus, replayFleet, replayFleetAgeMs } = useHubStore();
  const [models, setModels] = useState<ModelCheckpoint[]>([]);
  const [maps, setMaps] = useState<MapCatalogEntry[]>([]);
  const [modelId, setModelId] = useState("");
  const [mapId, setMapId] = useState("none");
  const [seed, setSeed] = useState(0);
  const [device, setDevice] = useState("auto");
  const [showMap, setShowMap] = useState(true);
  const [showFrontier, setShowFrontier] = useState(true);
  const [showCurrentProgress, setShowCurrentProgress] = useState(true);
  const [showLidar, setShowLidar] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      fetchReplayModels().catch(() => [] as ModelCheckpoint[]),
      fetchMaps().catch(() => [] as MapCatalogEntry[]),
      getSettings().catch(() => null),
    ]).then(([availableModels, availableMaps, settings]) => {
      if (cancelled) return;
      setModels(availableModels);
      setModelId((current) => current || availableModels[0]?.id || "");
      const replayMaps = availableMaps.filter((map) => map.id === "none" || map.mesh_status === "ready");
      setMaps(replayMaps);
      if (settings?.map_id && replayMaps.some((map) => map.id === settings.map_id)) {
        setMapId(settings.map_id);
      }
    });
    return () => { cancelled = true; };
  }, []);

  const running = ["starting", "running", "stopping"].includes(replayStatus?.state ?? "idle");
  const selectedMap = useMemo(() => maps.find((map) => map.id === mapId) ?? null, [maps, mapId]);
  const car = replayFleet?.cars.find((entry) => entry.env_id === 0) ?? null;
  const stale = replayFleetAgeMs > 1000;

  async function perform(action: () => Promise<unknown>, success: string) {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await action();
      setMessage(success);
    } catch (ex) {
      setError(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel fleet-panel">
      <div className="run-strip">
        <div className="run-strip-main">
          <h2>Replay</h2>
          <span className="badge" data-state={replayStatus?.state ?? "idle"}>{replayStatus?.state ?? "idle"}</span>
          <span className="meta">{replayStatus?.run_id ?? "Choose a checkpoint and track to begin"}</span>
        </div>
        <div className="actions" style={{ marginTop: 0 }}>
          <button
            type="button"
            className="primary"
            disabled={busy || running || !modelId}
            onClick={() => void perform(
              () => startReplay({ model_id: modelId, map_id: mapId, seed, device }),
              "Replay started"
            )}
          >Start</button>
          <button
            type="button"
            disabled={busy || !running}
            onClick={() => void perform(stopReplay, "Replay stopped")}
          >Stop</button>
          <button
            type="button"
            disabled={busy || !running}
            onClick={() => void perform(resetReplay, "Replay reset with the same checkpoint, map, and seed")}
          >Reset</button>
        </div>
      </div>
      <p className="lede">Evaluate a checkpoint in its own simulator session. Training can keep running independently.</p>

      <div className="replay-config">
        <label className="field">
          Checkpoint
          <select value={modelId} disabled={running} onChange={(event) => setModelId(event.target.value)}>
            {models.length === 0 && <option value="">No checkpoints found in logs/rl</option>}
            {models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
          </select>
        </label>
        <label className="field">
          Track
          <select value={mapId} disabled={running} onChange={(event) => setMapId(event.target.value)}>
            <option value="none">Builtin</option>
            {maps.map((map) => <option key={map.id} value={map.id}>{map.label}</option>)}
          </select>
        </label>
        <label className="field">
          Seed
          <input type="number" value={seed} disabled={running} onChange={(event) => setSeed(Number(event.target.value) || 0)} />
        </label>
        <label className="field">
          Device
          <select value={device} disabled={running} onChange={(event) => setDevice(event.target.value)}>
            <option value="auto">Auto</option>
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA</option>
          </select>
        </label>
      </div>

      {(message || error || replayStatus?.error) && (
        <p className={`msg ${error || replayStatus?.error ? "err" : "ok"}`}>
          {error || replayStatus?.error || message}
        </p>
      )}

      <div className="watch-metrics replay-metrics">
        <div className="watch-metrics-row">
          <span><strong>step</strong> {replayFleet?.step ?? "—"}</span>
          <span><strong>episode</strong> {replayFleet?.episode ?? "—"}</span>
          <span><strong>map</strong> {replayStatus?.map_id ?? mapId}</span>
          <span><strong>sample</strong> {stale ? "waiting" : car ? "live" : "—"}</span>
        </div>
      </div>

      <div className="fleet-toolbar">
        <label className="check-inline"><input type="checkbox" checked={showMap} onChange={(e) => setShowMap(e.target.checked)} />Map</label>
        <label className="check-inline"><input type="checkbox" checked={showFrontier} onChange={(e) => setShowFrontier(e.target.checked)} />Frontier</label>
        <label className="check-inline"><input type="checkbox" checked={showCurrentProgress} onChange={(e) => setShowCurrentProgress(e.target.checked)} />Current position</label>
        <label className="check-inline"><input type="checkbox" checked={showLidar} onChange={(e) => setShowLidar(e.target.checked)} />LiDAR</label>
      </div>

      <div className="fleet-layout">
        <div className="fleet-canvas-stack">
          <FleetCanvas
            showMap={showMap}
            showFleet
            showFrontier={showFrontier}
            showCurrentProgress={showCurrentProgress}
            showLidar={showLidar}
            fleetSource="replay"
            selectedEnvId={0}
            mapId={mapId}
            mapYamlUrl={selectedMap?.yaml_url ?? null}
            fleetPanel={replayFleet}
          />
          <div className="canvas-legend" aria-hidden>
            <span className="leg-car">▸ replay car</span>
            <span className="leg-collision">red · collision</span>
            <span className="leg-lap">┄ finish gate</span>
          </div>
        </div>
        <aside className="fleet-side">
          <h3>Run results</h3>
          {!car ? <p className="meta">Start replay to see the model on track.</p> : (
            <dl className="fleet-stats replay-stats">
              <div><dt>Completed laps</dt><dd>{car.lap_count ?? 0}</dd></div>
              <div><dt>Current lap</dt><dd>{car.lap_elapsed_s != null ? `${car.lap_elapsed_s.toFixed(2)} s` : "—"}</dd></div>
              <div><dt>Last lap</dt><dd>{car.last_lap_time_s != null ? `${car.last_lap_time_s.toFixed(2)} s` : "—"}</dd></div>
              <div><dt>Best lap</dt><dd>{car.best_lap_time_s != null ? `${car.best_lap_time_s.toFixed(2)} s` : "—"}</dd></div>
              <div><dt>Speed</dt><dd>{car.speed != null ? `${car.speed.toFixed(2)} m/s` : "—"}</dd></div>
              <div><dt>Episode return</dt><dd>{car.episode_return != null ? car.episode_return.toFixed(2) : "—"}</dd></div>
              {car.frontier_progress_m != null && <div><dt>Frontier</dt><dd>{car.frontier_progress_m.toFixed(2)} m</dd></div>}
            </dl>
          )}
          {replayStatus?.log_path && <p className="meta replay-log">Log: {replayStatus.log_path}</p>}
        </aside>
      </div>
    </section>
  );
}
