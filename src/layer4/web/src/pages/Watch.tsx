import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchMaps,
  getSettings,
  type MapCatalogEntry,
} from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";
import { useHubStore } from "../store";

/**
 * Observe surface — underlay from Train-selected map_id; no map picker / Activate.
 */
export function WatchPage() {
  const { fleet, fleetAgeMs, metrics } = useHubStore();
  const stale = fleetAgeMs > 500;
  const cars = useMemo(() => fleet?.cars ?? [], [fleet]);
  const [showMap, setShowMap] = useState(true);
  const [showFleet, setShowFleet] = useState(true);
  const [showFrontier, setShowFrontier] = useState(true);
  const [showCurrentProgress, setShowCurrentProgress] = useState(true);
  const [showLidar, setShowLidar] = useState(true);
  const [showDetails, setShowDetails] = useState(false);
  const [mapId, setMapId] = useState("none");
  const [maps, setMaps] = useState<MapCatalogEntry[]>([]);
  const [mapNote, setMapNote] = useState<string | null>(null);
  const [mapLoadErr, setMapLoadErr] = useState<string | null>(null);
  const [selectedEnvId, setSelectedEnvId] = useState<number | null>(null);
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

  const selected = useMemo(
    () => selectedEnvId == null ? null : cars.find((c) => c.env_id === selectedEnvId) ?? null,
    [cars, selectedEnvId]
  );

  const lapSupported = cars.filter((c) => c.lap_supported);

  return (
    <section className="panel fleet-panel">
      <h2>Watch</h2>

      <div className="watch-metrics">
        <div className="watch-metrics-row">
          <span>
            <strong>step</strong> {metrics?.step ?? "—"}
          </span>
          <span>
            <strong>episode</strong> {metrics?.episode ?? "—"}
          </span>
          <span><strong>cars</strong> {cars.length}</span>
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
            checked={showFrontier}
            onChange={(e) => setShowFrontier(e.target.checked)}
          />
          Frontier
        </label>
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showCurrentProgress}
            onChange={(e) => setShowCurrentProgress(e.target.checked)}
          />
          Current position
        </label>
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showLidar}
            disabled={selectedEnvId == null}
            onChange={(e) => setShowLidar(e.target.checked)}
          />
          LiDAR {selectedEnvId == null ? "(select a car)" : ""}
        </label>

        {cars.length > 0 && (
          <label className="field-inline">
            Focus
            <select
              value={selectedEnvId ?? "all"}
              onChange={(e) => setSelectedEnvId(e.target.value === "all" ? null : Number(e.target.value))}
            >
              <option value="all">All cars</option>
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
            showFrontier={showFrontier}
            showCurrentProgress={showCurrentProgress}
            showLidar={showLidar}
            selectedEnvId={selected?.env_id ?? -1}
            mapId={mapId}
            mapYamlUrl={selectedMap?.yaml_url ?? null}
            fleetPanel={fleet}
          />
          <div className="canvas-legend" aria-hidden>
            <span className="leg-car">▸ car</span>
            <span className="leg-collision">red car · collision event</span>
            <span className="leg-current">━ current route position</span>
            <span className="leg-frontier">┄ best progress</span>
            <span className="leg-lap">┄ finish gate</span>
          </div>
        </div>

        <aside className="fleet-side">
          <h3>{selected ? `Env ${selected.env_id}` : "Fleet overview"}</h3>
          {!selected ? (
            cars.length === 0 ? <p className="meta">No fleet sample yet. Start a train job with HUB_URL set.</p> : <>
              {lapSupported.length === 0 ? <p className="meta">Lap tracking is unavailable on this map.</p> :
                <div className="lap-list">{lapSupported.map((car) => (
                  <details className="lap-disclosure" key={car.env_id}>
                    <summary>
                      <span>Env {car.env_id}</span>
                      <strong>{car.lap_count ?? 0} laps</strong>
                    </summary>
                    <dl className="lap-times">
                      <div><dt>Current lap</dt><dd>{car.lap_elapsed_s != null ? `${car.lap_elapsed_s.toFixed(2)} s` : "—"}</dd></div>
                      <div><dt>Last lap</dt><dd>{car.last_lap_time_s != null ? `${car.last_lap_time_s.toFixed(2)} s` : "—"}</dd></div>
                      <div><dt>Best lap</dt><dd>{car.best_lap_time_s != null ? `${car.best_lap_time_s.toFixed(2)} s` : "—"}</dd></div>
                    </dl>
                  </details>
                ))}</div>}
            </>
          ) : (
            <dl className="fleet-stats">
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
                <dt>Speed</dt>
                <dd>
                  {selected.speed != null
                    ? `${selected.speed.toFixed(2)} m/s`
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Forward speed (signed)</dt>
                <dd>
                  {selected.v_long != null
                    ? `${selected.v_long.toFixed(2)} m/s`
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Throttle command</dt>
                <dd>
                  {selected.throttle_command != null
                    ? selected.throttle_command.toFixed(2)
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Steering command</dt>
                <dd>
                  {selected.steering_command != null
                    ? selected.steering_command.toFixed(2)
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
              {selected.current_progress_m != null && (
                <div>
                  <dt>Current route position</dt>
                  <dd>
                    {selected.current_progress_m.toFixed(2)} m
                    {selected.signed_route_delta_m != null &&
                      ` (${selected.current_route_speed_mps != null && selected.current_route_speed_mps >= 0 ? "+" : ""}${selected.current_route_speed_mps?.toFixed(2) ?? "—"} m/s route)`}
                  </dd>
                </div>
              )}
              {selected.reward_components && (
                <>
                  <div><dt>Reward · progress</dt><dd>{(selected.reward_components.route_progress ?? 0).toFixed(3)}</dd></div>
                  <div><dt>Reward · reverse</dt><dd>{(selected.reward_components.backward_motion ?? 0).toFixed(3)}</dd></div>
                  <div><dt>Reward · collision</dt><dd>{(selected.reward_components.collision ?? 0).toFixed(1)}</dd></div>
                  <div><dt>Reward · total</dt><dd>{(selected.reward_components.total ?? 0).toFixed(3)}</dd></div>
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
