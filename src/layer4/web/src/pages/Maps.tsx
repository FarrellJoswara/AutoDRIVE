import { useEffect, useMemo, useRef, useState } from "react";
import {
  centerlineDownloadUrl,
  fetchMaps,
  generateMapCenterline,
  generateMapMesh,
  generateMapMeshPreview,
  generateMapThumbnail,
  uploadMapZip,
  type MapCatalogEntry,
} from "../api";

/**
 * Map authoring workshop — does not Start/Stop train or Activate physics.
 * Train dropdown only lists builtin + mesh-ready maps; prep them here.
 */
export function MapsPage() {
  const [maps, setMaps] = useState<MapCatalogEntry[]>([]);
  const [mapsError, setMapsError] = useState<string | null>(null);
  const [mapId, setMapId] = useState<string>("porto");
  const [meshBusy, setMeshBusy] = useState(false);
  const [opsBusy, setOpsBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [thumbUrl, setThumbUrl] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

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
        refreshMaps(list.filter((m) => m.id !== "none"));
        setMapsError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setMapsError(err instanceof Error ? err.message : String(err));
        setMaps([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedMap = useMemo(
    () => maps.find((m) => m.id === mapId) ?? null,
    [maps, mapId]
  );

  useEffect(() => {
    if (!selectedMap) {
      setPreviewUrl(null);
      setThumbUrl(null);
      return;
    }
    const bust = Date.now();
    setPreviewUrl(
      selectedMap.mesh_preview_url
        ? `${selectedMap.mesh_preview_url}?t=${bust}`
        : null
    );
    setThumbUrl(
      selectedMap.thumbnail_url
        ? `${selectedMap.thumbnail_url}?t=${bust}`
        : null
    );
    // Ensure browser-safe occupancy thumb when catalog has none yet.
    if (!selectedMap.thumbnail_url && selectedMap.yaml_url) {
      void generateMapThumbnail(selectedMap.id)
        .then((res) => {
          refreshMaps(res.maps.filter((m) => m.id !== "none"));
        })
        .catch(() => undefined);
    }
  }, [selectedMap?.id, selectedMap?.mesh_preview_url, selectedMap?.thumbnail_url, selectedMap?.yaml_url]);

  const onGenerateMesh = () => {
    if (!selectedMap || meshBusy) return;
    setMeshBusy(true);
    setMsg(null);
    void generateMapMesh(selectedMap.id)
      .then((res) => {
        refreshMaps(res.maps.filter((m) => m.id !== "none"));
        const r = res.result;
        setMsg(
          `Mesh ready — contours=${String(r.n_contours)} visual=${String(r.n_verts_visual)}v collider=${String(r.n_verts_collider)}v. Available in Train map dropdown.`
        );
      })
      .catch((err: unknown) => {
        setMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setMeshBusy(false));
  };

  const onUpload = (file: File | null) => {
    if (!file || opsBusy) return;
    setOpsBusy(true);
    setMsg(null);
    void uploadMapZip(file, { overwrite: false })
      .then((res) => {
        refreshMaps(res.maps.filter((m) => m.id !== "none"));
        setMapId(res.id);
        setMsg(`Uploaded ${res.id} — Generate mesh to make it selectable on Train.`);
      })
      .catch((err: unknown) => {
        setMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        setOpsBusy(false);
        if (fileRef.current) fileRef.current.value = "";
      });
  };

  const onCenterline = () => {
    if (!selectedMap || opsBusy) return;
    setOpsBusy(true);
    setMsg(null);
    void generateMapCenterline(selectedMap.id)
      .then((res) => {
        refreshMaps(res.maps.filter((m) => m.id !== "none"));
        setMsg(`Centerline ready — ${String(res.result.n_points)} points`);
      })
      .catch((err: unknown) => {
        setMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setOpsBusy(false));
  };

  const onMeshPreview = () => {
    if (!selectedMap || opsBusy) return;
    setOpsBusy(true);
    setMsg(null);
    void generateMapMeshPreview(selectedMap.id)
      .then((res) => {
        refreshMaps(res.maps.filter((m) => m.id !== "none"));
        setPreviewUrl(`${res.result.url}?t=${Date.now()}`);
        setMsg(
          `Mesh preview ready — ${res.result.n_verts}v / ${res.result.n_faces}f (shown on Train)`
        );
      })
      .catch((err: unknown) => {
        setMsg(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setOpsBusy(false));
  };

  // Maps tab hero = occupancy; compact mesh preview panel lives on Train.
  const heroSrc = thumbUrl ?? previewUrl;

  return (
    <section className="panel maps-panel">
      <h2>Maps</h2>
      <p className="lede">
        Prepare tracks here (upload, mesh, centerline). This tab does not start
        training or switch sim physics — pick a mesh-ready map on{" "}
        <strong>Train</strong> and lock it in with Start.
      </p>

      <div className="fleet-toolbar">
        <label className="field-inline">
          Map
          <select
            value={mapId}
            onChange={(e) => setMapId(e.target.value)}
            disabled={!maps.length}
          >
            {maps.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} [{m.mesh_status}]
                {m.active ? " ● active" : ""}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          className="btn"
          disabled={meshBusy || !selectedMap || !selectedMap.yaml_url}
          onClick={onGenerateMesh}
        >
          {meshBusy ? "Generating…" : "Generate mesh"}
        </button>

        <button
          type="button"
          className="btn"
          disabled={opsBusy || !selectedMap}
          onClick={onCenterline}
        >
          Centerline
        </button>

        {selectedMap && selectedMap.centerline_status === "ready" && (
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
      </div>

      <div className="maps-preview">
        {heroSrc ? (
          <img
            src={heroSrc}
            alt={`Map preview ${mapId}`}
          />
        ) : (
          <p className="meta maps-preview-empty">
            {selectedMap
              ? "Generating occupancy preview…"
              : "Select a map to preview."}
          </p>
        )}
        {heroSrc && (
          <span className="maps-preview-label">
            {thumbUrl ? "occupancy" : "mesh preview"}
          </span>
        )}
      </div>

      {selectedMap && (
        <div className="meta" style={{ marginTop: "0.75rem" }}>
          <div>
            <strong>id</strong> {selectedMap.id}
          </div>
          <div>
            <strong>mesh</strong> {selectedMap.mesh_status}
          </div>
          <div>
            <strong>centerline</strong> {selectedMap.centerline_status ?? "—"}
          </div>
          <div>
            <strong>source</strong> {selectedMap.source ?? "—"}
          </div>
          {selectedMap.mesh_status === "ready" ? (
            <p className="msg ok">Ready for Train map dropdown.</p>
          ) : (
            <p className="lede" style={{ marginTop: "0.5rem" }}>
              Generate mesh before this map appears as a Train option (besides
              builtin).
            </p>
          )}
        </div>
      )}

      {msg && <p className="meta fleet-note">{msg}</p>}
      {mapsError && (
        <p className="meta fleet-note" data-bad="1">
          Map catalog failed: {mapsError}
        </p>
      )}
      {!maps.length && !mapsError && (
        <p className="lede">No maps found under simulator/maps/.</p>
      )}
    </section>
  );
}
