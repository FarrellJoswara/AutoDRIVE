import { useMemo, useState } from "react";
import { FleetCanvas } from "../fleet/FleetCanvas";
import { listMaps, type MapId } from "../fleet/mapLoader";
import { useHubStore } from "../store";

export function FleetPage() {
  const { fleet, fleetAgeMs } = useHubStore();
  const stale = fleetAgeMs > 500;
  const cars = fleet?.cars ?? [];
  const [showMap, setShowMap] = useState(true);
  const [showFleet, setShowFleet] = useState(true);
  const [showLidar, setShowLidar] = useState(true);
  const [mapId, setMapId] = useState<MapId>("porto");
  const [selectedEnvId, setSelectedEnvId] = useState(0);

  const selected = useMemo(() => {
    if (!cars.length) return null;
    return cars.find((c) => c.env_id === selectedEnvId) ?? cars[0];
  }, [cars, selectedEnvId]);

  const multi = cars.length > 1;

  return (
    <section className="panel fleet-panel">
      <h2>Fleet</h2>
      <p className="lede">
        Bird&apos;s-eye from hub telemetry only — map → cars → LiDAR → collision X.
        No second stepper.
      </p>

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

        <label className="field-inline">
          Track
          <select
            value={mapId}
            onChange={(e) => setMapId(e.target.value as MapId)}
          >
            {listMaps().map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>

        {multi && (
          <label className="field-inline">
            LiDAR car
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

        <div className="fleet-legend" aria-hidden>
          <span className="leg-car">▸ car</span>
          <span className="leg-x">✕ collision</span>
        </div>
      </div>

      <div className="fleet-layout">
        <FleetCanvas
          showMap={showMap}
          showFleet={showFleet}
          showLidar={showLidar}
          selectedEnvId={selected?.env_id ?? selectedEnvId}
          mapId={mapId}
          fleetPanel={fleet}
        />

        <aside className="fleet-side">
          <h3>Selected</h3>
          {!selected ? (
            <p className="meta">No fleet sample yet. Start a train job with HUB_URL set.</p>
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
                <dt>Collision</dt>
                <dd data-bad={selected.collision ? "1" : "0"}>
                  {selected.collision ? "yes" : "no"}
                </dd>
              </div>
              <div>
                <dt>Speed</dt>
                <dd>
                  {selected.speed != null ? `${selected.speed.toFixed(2)} m/s` : "—"}
                </dd>
              </div>
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
                <dd data-stale={stale ? "1" : "0"}>{stale ? "stale" : "fresh"}</dd>
              </div>
              <div>
                <dt>LiDAR beams</dt>
                <dd>{selected.lidar?.length ?? 0}</dd>
              </div>
            </dl>
          )}
          <p className="meta fleet-note">
            LiDAR angles are defaults (360° / beam 0 = forward) — see{" "}
            <code>lidarCalibration.ts</code> TODO(F7).
          </p>
        </aside>
      </div>
    </section>
  );
}
