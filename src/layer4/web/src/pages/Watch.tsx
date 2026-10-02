import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchMaps,
  getSettings,
  type MapCatalogEntry,
} from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";
import { useHubStore } from "../store";

function Sparkline({
  steps,
  rewards,
  len,
}: {
  steps: Float32Array;
  rewards: Float32Array;
  len: number;
}) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = ref.current;
    if (!svg || len < 2) return;
    const w = 800;
    const h = 56;
    const n = Math.min(len, steps.length);
    let minR = Infinity;
    let maxR = -Infinity;
    for (let i = 0; i < n; i++) {
      minR = Math.min(minR, rewards[i]);
      maxR = Math.max(maxR, rewards[i]);
    }
    if (!Number.isFinite(minR) || !Number.isFinite(maxR) || minR === maxR) {
      maxR = minR + 1;
    }
    const pts: string[] = [];
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * (w - 4) + 2;
      const y = h - 4 - ((rewards[i] - minR) / (maxR - minR)) * (h - 8);
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    const poly = svg.querySelector("polyline");
    if (poly) poly.setAttribute("points", pts.join(" "));
  }, [steps, rewards, len]);

  return (
    <svg ref={ref} className="spark spark-compact" viewBox="0 0 800 56" preserveAspectRatio="none">
      <polyline fill="none" stroke="#3ecf8e" strokeWidth="2" points="" />
    </svg>
  );
}

/**
 * Observe surface — underlay from Train-selected map_id; no map picker / Activate.
 */
export function WatchPage() {
  const { fleet, fleetAgeMs, metrics, steps, rewards, metricsLen } = useHubStore();
  const stale = fleetAgeMs > 500;
  const cars = useMemo(
    () => (fleet?.cars ?? []).filter((c) => !c.reset),
    [fleet]
  );
  const [showMap, setShowMap] = useState(true);
  const [showFleet, setShowFleet] = useState(true);
  const [showLidar, setShowLidar] = useState(true);
  const [showDetails, setShowDetails] = useState(false);
  const [mapId, setMapId] = useState("none");
  const [maps, setMaps] = useState<MapCatalogEntry[]>([]);
  const [mapNote, setMapNote] = useState<string | null>(null);
  const [mapLoadErr, setMapLoadErr] = useState<string | null>(null);
  const [selectedEnvId, setSelectedEnvId] = useState(0);
  const mapsRef = useRef<MapCatalogEntry[]>([]);
  mapsRef.current = maps;

  const applyMapId = (mid: string, list: MapCatalogEntry[]) => {
    const id = mid.trim() || "none";
    setMapId(id);
    if (id === "none") {
      setMapNote(
        "Builtin track — no occupancy underlay (grid only). Set a mesh-ready map on Train."
      );
      setMapLoadErr(null);
      return;
    }
    const entry = list.find((m) => m.id === id);
    if (!entry) {
      setMapNote(`Train map "${id}" not in catalog.`);
      setMapLoadErr(`Unknown map ${id}`);
    } else if (!entry.yaml_url) {
      setMapNote(`Map "${id}" has no occupancy yaml — grid only.`);
      setMapLoadErr(null);
    } else {
      setMapNote(null);
      setMapLoadErr(null);
    }
  };

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      getSettings().catch(() => null),
      fetchMaps().catch(() => [] as MapCatalogEntry[]),
    ]).then(([settings, list]) => {
      if (cancelled) return;
      setMaps(list);
      applyMapId(settings?.map_id ?? "none", list);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Follow Train map_id: event (immediate) + short poll + focus refresh
  useEffect(() => {
    const syncFromSettings = () => {
      void getSettings()
        .then((s) => {
          applyMapId(s.map_id ?? "none", mapsRef.current);
        })
        .catch(() => undefined);
    };

    const onMapEvent = (ev: Event) => {
      applyMapId(String((ev as CustomEvent).detail ?? "none"), mapsRef.current);
    };

    window.addEventListener("aicar-map-id", onMapEvent);
    window.addEventListener("focus", syncFromSettings);
    const timer = window.setInterval(syncFromSettings, 1000);
    return () => {
      window.removeEventListener("aicar-map-id", onMapEvent);
      window.removeEventListener("focus", syncFromSettings);
      window.clearInterval(timer);
    };
  }, []);

  const selectedMap = useMemo(
    () => maps.find((m) => m.id === mapId) ?? null,
    [maps, mapId]
  );

  const selected = useMemo(() => {
    if (!cars.length) return null;
    return cars.find((c) => c.env_id === selectedEnvId) ?? cars[0];
  }, [cars, selectedEnvId]);

  const multi = cars.length > 1;

  return (
    <section className="panel fleet-panel">
      <h2>Watch</h2>
      <p className="lede">
        Fleet + live metrics from hub telemetry. Map underlay follows the Train
        selection (locked on Start) — no map controls here.
      </p>

      <div className="watch-metrics">
        <div className="watch-metrics-row">
          <span>
            <strong>step</strong> {metrics?.step ?? "—"}
          </span>
          <span>
            <strong>reward</strong>{" "}
            {metrics?.reward != null ? metrics.reward.toFixed(3) : "—"}
          </span>
          <span>
            <strong>episode</strong> {metrics?.episode ?? "—"}
          </span>
          <span className="meta">
            underlay <strong>{mapId}</strong>
          </span>
          <button
            type="button"
            className="btn"
            onClick={() => setShowDetails((v) => !v)}
          >
            {showDetails ? "Hide details" : "Details"}
          </button>
        </div>
        <Sparkline steps={steps} rewards={rewards} len={metricsLen} />
        {showDetails && (
          <table className="live-table">
            <tbody>
              <tr>
                <td>loss</td>
                <td>{metrics?.loss != null ? metrics.loss.toFixed(6) : "—"}</td>
              </tr>
              <tr>
                <td>run_id</td>
                <td>{metrics?.run_id ?? "—"}</td>
              </tr>
              <tr>
                <td>checkpoint</td>
                <td>{metrics?.checkpoint ?? "—"}</td>
              </tr>
              <tr>
                <td>ts</td>
                <td>{metrics?.ts ?? "—"}</td>
              </tr>
            </tbody>
          </table>
        )}
      </div>

      <div className="fleet-toolbar">
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showMap}
            onChange={(e) => setShowMap(e.target.checked)}
          />
          Map
        </label>
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showFleet}
            onChange={(e) => setShowFleet(e.target.checked)}
          />
          Fleet
        </label>
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showLidar}
            onChange={(e) => setShowLidar(e.target.checked)}
          />
          LiDAR
        </label>

        {multi && (
          <label className="field-inline">
            Focus car
            <select
              value={selected?.env_id ?? 0}
              onChange={(e) => setSelectedEnvId(Number(e.target.value))}
            >
              {cars.map((c) => (
                <option key={c.env_id} value={c.env_id}>
                  env {c.env_id}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

  {mapNote && <p className="meta fleet-note">{mapNote}</p>}
  {mapLoadErr && (
    <p className="meta fleet-note" data-bad="1">
      {mapLoadErr}
    </p>
  )}
  {mapId !== "none" &&
    maps.some((m) => m.active && m.id !== mapId) && (
      <p className="meta fleet-note">
        Underlay is <strong>{mapId}</strong>; sims still run{" "}
        <strong>{maps.find((m) => m.active)?.id}</strong> until Start locks
        the Train map. Cars may sit off-track until then.
      </p>
    )}

      <div className="fleet-layout">
        <div className="fleet-canvas-stack">
          <FleetCanvas
            showMap={showMap}
            showFleet={showFleet}
            showLidar={showLidar}
            selectedEnvId={selected?.env_id ?? selectedEnvId}
            mapId={mapId}
            mapYamlUrl={selectedMap?.yaml_url ?? null}
            fleetPanel={fleet}
          />
          <div className="canvas-legend" aria-hidden>
            <span className="leg-car">▸ car</span>
            <span className="leg-x">✕ collision</span>
            <span className="leg-lap">┄ finish gate</span>
          </div>
        </div>

        <aside className="fleet-side">
          <h3>Focused</h3>
          {!selected ? (
            <p className="meta">
              No fleet sample yet. Start a train job with HUB_URL set.
            </p>
          ) : (
            <dl className="fleet-stats">
              <div>
                <dt>Env</dt>
                <dd>{selected.env_id}</dd>
              </div>
              <div>
                <dt>Episode #</dt>
                <dd>{fleet?.episode ?? "—"}</dd>
              </div>
              <div>
                <dt>Steps</dt>
                <dd>{fleet?.step ?? "—"}</dd>
              </div>
              <div>
                <dt>Return</dt>
                <dd>
                  {selected.episode_return != null
                    ? selected.episode_return.toFixed(2)
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Contacts this episode</dt>
                <dd data-bad={selected.collision ? "1" : "0"}>
                  {selected.collision_count ?? (selected.collision ? "1+" : 0)}
                </dd>
              </div>
              <div>
                <dt>Speed</dt>
                <dd>
                  {selected.speed != null
                    ? `${selected.speed.toFixed(2)} m/s`
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Laps this episode</dt>
                <dd>
                  {selected.lap_supported
                    ? selected.lap_count ?? 0
                    : "unavailable"}
                </dd>
              </div>
              {selected.lap_supported && (
                <>
                  <div>
                    <dt>Current lap</dt>
                    <dd>
                      {selected.lap_elapsed_s != null
                        ? `${selected.lap_elapsed_s.toFixed(1)} s`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Last lap</dt>
                    <dd>
                      {selected.last_lap_time_s != null
                        ? `${selected.last_lap_time_s.toFixed(2)} s`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Best this episode</dt>
                    <dd>
                      {selected.best_lap_time_s != null
                        ? `${selected.best_lap_time_s.toFixed(2)} s`
                        : "—"}
                    </dd>
                  </div>
                </>
              )}
              {selected.frontier_progress_m != null && (
                <>
                  <div>
                    <dt>Frontier progress</dt>
                    <dd>{selected.frontier_progress_m.toFixed(2)} m</dd>
                  </div>
                  <div>
                    <dt>Frontier speed</dt>
                    <dd>{(selected.frontier_speed_mps ?? 0).toFixed(2)} m/s</dd>
                  </div>
                  <div>
                    <dt>Since last push</dt>
                    <dd>{(selected.time_since_frontier_push_s ?? 0).toFixed(1)} s</dd>
                  </div>
                </>
              )}
              <div>
                <dt>Pose (x,z)</dt>
                <dd>
                  {selected.pose
                    ? `${selected.pose[0].toFixed(2)}, ${selected.pose[1].toFixed(2)}`
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Yaw</dt>
                <dd>
                  {selected.yaw != null ? `${selected.yaw.toFixed(3)} rad` : "—"}
                </dd>
              </div>
              <div>
                <dt>Sample</dt>
                <dd data-stale={stale ? "1" : "0"}>
                  {stale ? "stale" : "fresh"}
                </dd>
              </div>
              <div>
                <dt>LiDAR beams</dt>
                <dd>{selected.lidar?.length ?? 0}</dd>
              </div>
            </dl>
          )}
        </aside>
      </div>
    </section>
  );
}
