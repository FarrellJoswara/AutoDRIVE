import { useEffect, useMemo, useRef, useState } from "react";
import {
  activateMap,
  centerlineDownloadUrl,
  fetchActiveMap,
  fetchMaps,
  generateMapCenterline,
  generateMapMesh,
  generateMapMeshPreview,
  uploadMapZip,
  type MapCatalogEntry,
} from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";
import { useHubStore } from "../store";

const LS_MAP_KEY = "aicar.fleet.mapId";

export function FleetPage() {
  const { fleet, fleetAgeMs } = useHubStore();
  const stale = fleetAgeMs > 500;
  const cars = fleet?.cars ?? [];
  const [showMap, setShowMap] = useState(true);
  const [showFleet, setShowFleet] = useState(true);
  const [showLidar, setShowLidar] = useState(true);
  const [maps, setMaps] = useState<MapCatalogEntry[]>([]);
  const [mapsError, setMapsError] = useState<string | null>(null);
  const [meshBusy, setMeshBusy] = useState(false);
  const [activateBusy, setActivateBusy] = useState(false);
  const [opsBusy, setOpsBusy] = useState(false);
  const [meshMsg, setMeshMsg] = useState<string | null>(null);
  const [mismatchWarnings, setMismatchWarnings] = useState<string[]>([]);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [mapId, setMapId] = useState<string>(() => {
    try {
      return localStorage.getItem(LS_MAP_KEY) || "porto";
    } catch {
      return "porto";
    }
  });
  const [selectedEnvId, setSelectedEnvId] = useState(0);

  const refreshMaps = (list: MapCatalogEntry[]) => {
    setMaps(list);
    setMapId((prev) => {
      if (list.some((m) => m.id === prev)) return prev;
      const preferred = list.find((m) => m.id === "porto") ?? list[0];
      return preferred?.id ?? "none";
    });
  };

  useEffect(() => {
    let cancelled = false;
    void fetchMaps()
      .then((list) => {
        if (cancelled) return;
        refreshMaps(list);
        setMapsError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setMapsError(err instanceof Error ? err.message : String(err));
        setMaps([
          {
            id: "none",
            label: "Grid only",
            yaml_url: null,
            image_url: null,
            mesh_status: "none",
            overlay_only: true,
            active: true,
            source: null,
          },
        ]);
        setMapId("none");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(LS_MAP_KEY, mapId);
    } catch {
      /* ignore */
    }
  }, [mapId]);

  useEffect(() => {
    let cancelled = false;
    void fetchActiveMap(mapId)
      .then((res) => {
        if (cancelled) return;
        setMismatchWarnings(res.warnings ?? []);
        if (res.maps?.length) setMaps(res.maps);
      })
      .catch(() => {
        if (!cancelled) setMismatchWarnings([]);
      });
    return () => {
      cancelled = true;
    };
  }, [mapId]);

  const selectedMap = useMemo(
    () => maps.find((m) => m.id === mapId) ?? null,
    [maps, mapId]
  );

  const activeMap = useMemo(() => maps.find((m) => m.active) ?? null, [maps]);

  const selectionMismatch = useMemo(() => {
    const activeId = activeMap?.id === "none" ? null : activeMap?.id ?? null;
    const sel = mapId === "none" ? null : mapId;
    return sel !== activeId;
  }, [activeMap, mapId]);

  useEffect(() => {
    if (selectedMap?.mesh_preview_url) {
      setPreviewUrl(`${selectedMap.mesh_preview_url}?t=${Date.now()}`);
    } else {
      setPreviewUrl(null);
    }
  }, [selectedMap?.id, selectedMap?.mesh_preview_url]);

  const selected = useMemo(() => {
    if (!cars.length) return null;
    return cars.find((c) => c.env_id === selectedEnvId) ?? cars[0];
  }, [cars, selectedEnvId]);

  const multi = cars.length > 1;

  const onGenerateMesh = () => {
    if (!selectedMap || selectedMap.id === "none" || meshBusy) return;
    setMeshBusy(true);
    setMeshMsg(null);
    void generateMapMesh(selectedMap.id)
      .then((res) => {
        setMaps(res.maps);
        const r = res.result;
        setMeshMsg(
          `Mesh ready — contours=${String(r.n_contours)} visual=${String(r.n_verts_visual)}v collider=${String(r.n_verts_collider)}v`
        );
      })
      .catch((err: unknown) => {
        setMeshMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setMeshBusy(false));
  };

  const onActivate = () => {
    if (!selectedMap || activateBusy) return;
    setActivateBusy(true);
    setMeshMsg(null);
    void activateMap(selectedMap.id)
      .then((res) => {
        setMaps(res.maps);
        const id = res.active?.id ?? "builtin";
        const n = res.restarted?.length ?? 0;
        let msg = `Activated ${id}`;
        if (n > 0) msg += ` — restarted ${n} sim(s)`;
        else if (res.restart_error) msg += ` — restart failed: ${res.restart_error}`;
        else msg += " — no running compose sims to restart";
        if (res.note) msg += ` (${res.note})`;
        setMeshMsg(msg);
        setMismatchWarnings(res.note ? [res.note] : []);
      })
      .catch((err: unknown) => {
        setMeshMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setActivateBusy(false));
  };

  const onUpload = (file: File | null) => {
    if (!file || opsBusy) return;
    setOpsBusy(true);
    setMeshMsg(null);
    void uploadMapZip(file, { overwrite: false })
      .then((res) => {
        setMaps(res.maps);
        setMapId(res.id);
        setMeshMsg(`Uploaded map ${res.id} (${res.files.length} files) — Generate mesh then Activate`);
      })
      .catch((err: unknown) => {
        setMeshMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        setOpsBusy(false);
        if (fileRef.current) fileRef.current.value = "";
      });
  };

  const onCenterline = () => {
    if (!selectedMap || selectedMap.id === "none" || opsBusy) return;
    setOpsBusy(true);
    setMeshMsg(null);
    void generateMapCenterline(selectedMap.id)
      .then((res) => {
        setMaps(res.maps);
        const n = res.result.n_points;
        setMeshMsg(`Centerline ready — ${String(n)} points`);
      })
      .catch((err: unknown) => {
        setMeshMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setOpsBusy(false));
  };

  const onMeshPreview = () => {
    if (!selectedMap || selectedMap.id === "none" || opsBusy) return;
    setOpsBusy(true);
    setMeshMsg(null);
    void generateMapMeshPreview(selectedMap.id)
      .then((res) => {
        setMaps(res.maps);
        setPreviewUrl(`${res.result.url}?t=${Date.now()}`);
        setMeshMsg(
          `Mesh preview — ${res.result.n_verts}v / ${res.result.n_faces}f`
        );
      })
      .catch((err: unknown) => {
        setMeshMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setOpsBusy(false));
  };

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
            onChange={(e) => setMapId(e.target.value)}
            disabled={!maps.length}
          >
            {maps.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
                {m.active ? " ●" : ""}
                {m.id !== "none" ? ` [${m.mesh_status}]` : ""}
              </option>
            ))}
          </select>
        </label>

        {selectedMap?.thumbnail_url && (
          <img
            className="fleet-thumb"
            src={selectedMap.thumbnail_url}
            alt=""
            width={40}
            height={40}
          />
        )}

        <button
          type="button"
          className="btn"
          disabled={
            meshBusy || !selectedMap || selectedMap.id === "none" || !selectedMap.yaml_url
          }
          onClick={onGenerateMesh}
        >
          {meshBusy ? "Generating…" : "Generate mesh"}
        </button>

        <button
          type="button"
          className="btn"
          disabled={
            activateBusy ||
            !selectedMap ||
            (selectedMap.id !== "none" && selectedMap.mesh_status !== "ready")
          }
          onClick={onActivate}
          title="Write active map + restart compose sims (hub owns the logic)"
        >
          {activateBusy ? "Activating…" : "Activate"}
        </button>

        <button
          type="button"
          className="btn"
          disabled={opsBusy || !selectedMap || selectedMap.id === "none"}
          onClick={onCenterline}
        >
          Centerline
        </button>

        {selectedMap &&
          selectedMap.id !== "none" &&
          selectedMap.centerline_status === "ready" && (
            <a
              className="btn"
              href={centerlineDownloadUrl(selectedMap.id)}
              download={`${selectedMap.id}_centerline.csv`}
            >
              Download CSV
            </a>
          )}

        <button
          type="button"
          className="btn"
          disabled={
            opsBusy ||
            !selectedMap ||
            selectedMap.id === "none" ||
            selectedMap.mesh_status !== "ready"
          }
          onClick={onMeshPreview}
        >
          Mesh preview
        </button>

        <label className="btn fleet-upload">
          {opsBusy ? "Working…" : "Upload zip"}
          <input
            ref={fileRef}
            type="file"
            accept=".zip,application/zip"
            hidden
            disabled={opsBusy}
            onChange={(e) => onUpload(e.target.files?.[0] ?? null)}
          />
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

      {selectionMismatch && (
        <p className="meta fleet-note fleet-warn" data-bad="1">
          Selection ({mapId}) ≠ active physics ({activeMap?.id ?? "builtin"}).
          Overlay only until you Activate
          {activeMap?.id && activeMap.id !== "none"
            ? ` — sims should be on ${activeMap.id} after last Activate+restart`
            : ""}
          .
        </p>
      )}
      {mismatchWarnings.map((w) => (
        <p key={w} className="meta fleet-note fleet-warn" data-bad="1">
          {w}
        </p>
      ))}

      {selectedMap?.overlay_only && selectedMap.id !== "none" && (
        <p className="meta fleet-note">
          {selectedMap.mesh_status !== "ready"
            ? "Generate mesh first, then Activate to switch sim physics."
            : selectedMap.active
              ? "Active map is set, but TrackLoader player is not marked installed (touch simulator/.aicar_trackloader after rebuild)."
              : `Mesh ready — Activate to make this the physics track (active: ${activeMap?.id ?? "builtin"}).`}
        </p>
      )}
      {!selectedMap?.overlay_only && selectedMap && selectedMap.id !== "none" && (
        <p className="meta fleet-note">
          Physics map active — TrackLoader should load this mesh on sim boot.
        </p>
      )}
      {meshMsg && <p className="meta fleet-note">{meshMsg}</p>}
      {mapsError && (
        <p className="meta fleet-note" data-bad="1">
          Map catalog failed: {mapsError}
        </p>
      )}

      {previewUrl && (
        <div className="fleet-mesh-preview">
          <img src={previewUrl} alt={`Mesh preview ${mapId}`} />
        </div>
      )}

      <div className="fleet-layout">
        <FleetCanvas
          showMap={showMap}
          showFleet={showFleet}
          showLidar={showLidar}
          selectedEnvId={selected?.env_id ?? selectedEnvId}
          mapId={mapId}
          mapYamlUrl={selectedMap?.yaml_url ?? null}
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
              <div>
                <dt>Mesh</dt>
                <dd>{selectedMap?.mesh_status ?? "—"}</dd>
              </div>
              <div>
                <dt>Physics</dt>
                <dd>{activeMap?.id ?? "builtin"}</dd>
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
