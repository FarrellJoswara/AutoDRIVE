import { useEffect, useRef } from "react";
import type { FleetTelemetry } from "../api";
import {
  DrawFrameOpts,
  ViewState,
  drawDynamic,
  drawStaticMap,
  initialViewState,
  resolveBounds,
} from "./draw";
import type { LoadedMap, MapId } from "./mapLoader";
import { loadMap } from "./mapLoader";
import {
  getFleetHot,
  getFleetHotAgeMs,
  getReplayFleetHot,
  getReplayFleetHotAgeMs,
  getTrainingPhaseAgeMs,
  getTrainingPhaseHot,
  getTrainingStateHot,
} from "../store";

export interface FleetCanvasProps {
  showMap: boolean;
  showFleet: boolean;
  showFrontier: boolean;
  showCurrentProgress: boolean;
  showLidar: boolean;
  fleetSource?: "train" | "replay";
  selectedEnvId: number;
  mapId: MapId;
  /** Hub catalog yaml URL; null when mapId is none or unknown. */
  mapYamlUrl: string | null;
  /** Side-panel sync — last React-visible fleet (throttled) */
  fleetPanel: FleetTelemetry | null;
}

/**
 * Two canvases: static map underlayer + dynamic cars/LiDAR.
 * rAF reads getFleetHot() — WS does not drive React for every sample.
 */
export function FleetCanvas(props: FleetCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const staticRef = useRef<HTMLCanvasElement>(null);
  const dynRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<LoadedMap | null>(null);
  const viewRef = useRef<ViewState>(initialViewState());
  const lastFleetSeq = useRef(0);
  const mapDirty = useRef(true);
  const propsRef = useRef(props);
  propsRef.current = props;

  // Sync toggle / selection into view ref without React→canvas render storm
  useEffect(() => {
    const v = viewRef.current;
    v.showMap = props.showMap;
    v.showFleet = props.showFleet;
    v.showFrontier = props.showFrontier;
    v.showCurrentProgress = props.showCurrentProgress;
    v.showLidar = props.showLidar;
    v.selectedEnvId = props.selectedEnvId;
    mapDirty.current = true;
  }, [props.showMap, props.showFleet, props.showFrontier, props.showCurrentProgress, props.showLidar, props.selectedEnvId]);

  // Load occupancy map
  useEffect(() => {
    let cancelled = false;
    mapRef.current = null;
    mapDirty.current = true;
    // Reset grow-only pose fit so a new underlay re-frames around current cars.
    const v = viewRef.current;
    v.fitMinX = Infinity;
    v.fitMaxX = -Infinity;
    v.fitMinZ = Infinity;
    v.fitMaxZ = -Infinity;
    if (props.mapId === "none" || !props.mapYamlUrl) {
      return;
    }
    void loadMap(props.mapId, props.mapYamlUrl)
      .then((m) => {
        if (cancelled) return;
        mapRef.current = m;
        mapDirty.current = true;
      })
      .catch((err) => {
        console.warn("fleet map load failed", err);
        if (!cancelled) {
          mapRef.current = null;
          mapDirty.current = true;
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.mapYamlUrl]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const sc = staticRef.current;
    const dc = dynRef.current;
    if (!wrap || !sc || !dc) return;

    let raf = 0;
    let alive = true;
    let lastDrawnKey = "";
    let lastBoundsKey = "";

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const cssW = Math.max(1, Math.floor(rect.width));
      const cssH = Math.max(1, Math.floor(rect.height));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      for (const c of [sc, dc]) {
        c.width = Math.floor(cssW * dpr);
        c.height = Math.floor(cssH * dpr);
        c.style.width = `${cssW}px`;
        c.style.height = `${cssH}px`;
      }
      mapDirty.current = true;
      lastDrawnKey = "";
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const tick = () => {
      if (!alive) return;
      raf = requestAnimationFrame(tick);

      const p = propsRef.current;
      const fleet = p.fleetSource === "replay" ? getReplayFleetHot() : getFleetHot();
      const age = p.fleetSource === "replay" ? getReplayFleetHotAgeMs() : getFleetHotAgeMs();
      const stale = age > 500;
      const phase = p.fleetSource === "train" ? getTrainingPhaseHot() : null;
      const trainState = getTrainingStateHot();
      const trainActive = trainState === "starting" || trainState === "running" || trainState === "stopping";
      const ppoUpdating =
        trainActive &&
        phase?.phase === "ppo_update" &&
        getTrainingPhaseAgeMs() < 60_000 &&
        phase.step >= (fleet?.step ?? 0);
      const staleLabel = stale
        ? ppoUpdating
          ? "PPO updating — simulator paused"
          : p.fleetSource === "train" && !trainActive
            ? null
            : `No recent ${p.fleetSource === "replay" ? "replay " : ""}fleet telemetry (${Math.max(1, Math.floor(age / 1000))}s)`
        : null;
      const view = viewRef.current;
      view.showMap = p.showMap;
      view.showFleet = p.showFleet;
      view.showFrontier = p.showFrontier;
      view.showCurrentProgress = p.showCurrentProgress;
      view.showLidar = p.showLidar;
      view.selectedEnvId = p.selectedEnvId;

      const cssW = sc.clientWidth;
      const cssH = sc.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const map = mapRef.current;
      const bounds = resolveBounds(map, view, fleet);
      const boundsKey =
        bounds.minX.toFixed(2) +
        ":" +
        bounds.maxX.toFixed(2) +
        ":" +
        bounds.minZ.toFixed(2) +
        ":" +
        bounds.maxZ.toFixed(2);

      const seq =
        (fleet?.ts ?? "") +
        ":" +
        (fleet?.step ?? "") +
        ":" +
        view.showMap +
        view.showFleet +
        view.showFrontier +
        view.showCurrentProgress +
        view.showLidar +
        view.selectedEnvId +
        (map?.id ?? "none") +
        boundsKey +
        staleLabel;

      // Redraw static underlay when map OR fitted bounds change.
      if (mapDirty.current || boundsKey !== lastBoundsKey) {
        lastBoundsKey = boundsKey;
        const sctx = sc.getContext("2d");
        if (sctx) {
          drawStaticMap(sctx, map, view.showMap, cssW, cssH, dpr, bounds);
        }
        mapDirty.current = false;
        lastDrawnKey = "";
      }

      if (seq === lastDrawnKey) return;
      lastDrawnKey = seq;

      const dctx = dc.getContext("2d");
      if (!dctx) return;
      const opts: DrawFrameOpts = {
        map,
        fleet,
        staleLabel,
        view,
        cssW,
        cssH,
        dpr,
      };
      drawDynamic(dctx, opts, bounds);
      lastFleetSeq.current += 1;
    };

    raf = requestAnimationFrame(tick);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <div className="fleet-canvas-wrap" ref={wrapRef}>
      <canvas className="fleet-canvas fleet-canvas-static" ref={staticRef} />
      <canvas className="fleet-canvas fleet-canvas-dyn" ref={dynRef} />
    </div>
  );
}
