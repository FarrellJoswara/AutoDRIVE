import { useEffect, useRef } from "react";
import type { EvaluatorCarTelemetry, EvaluatorLiveTelemetry, FleetCar, FleetTelemetry } from "../api";
import {
  DrawFrameOpts,
  ViewState,
  drawEvaluatorCarIncremental,
  drawFleetCarsIncremental,
  drawStaticMap,
  drawTelemetryOverlay,
  initialViewState,
  resolveBounds,
  worldToScreenTransform,
} from "./draw";
import type { LoadedMap, MapId } from "./mapLoader";
import { loadMap } from "./mapLoader";
import {
  getFleetHot,
  getFleetHotAgeMs,
  getEvaluatorLiveHot,
  getReplayFleetHot,
  getReplayFleetHotAgeMs,
  getTrainingPhaseHot,
  getTrainingStateHot,
  subscribeFleetFrames,
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
  evaluatorCar?: EvaluatorCarTelemetry | null;
  evaluatorActive?: boolean;
  evaluatorSnapshotTimesteps?: number | null;
  trainingStep?: number;
  rolloutSize?: number;
}

/**
 * Draws the latest received fleet and evaluator poses directly. No synthetic
 * between-sample motion is added; incoming telemetry schedules each update.
 */
export function FleetCanvas(props: FleetCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const staticRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const fleetCarsRef = useRef<HTMLCanvasElement>(null);
  const evaluatorRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<LoadedMap | null>(null);
  const viewRef = useRef<ViewState>(initialViewState());
  const mapDirty = useRef(true);
  const propsRef = useRef(props);
  const scheduleFrameRef = useRef<() => void>(() => {});
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
    scheduleFrameRef.current();
  }, [props.showMap, props.showFleet, props.showFrontier, props.showCurrentProgress, props.showLidar, props.selectedEnvId]);

  useEffect(() => {
    scheduleFrameRef.current();
  }, [props.fleetSource, props.evaluatorActive, props.evaluatorCar, props.evaluatorSnapshotTimesteps, props.trainingStep, props.rolloutSize]);

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
        scheduleFrameRef.current();
      })
      .catch((err) => {
        console.warn("fleet map load failed", err);
        if (!cancelled) {
          mapRef.current = null;
          mapDirty.current = true;
          scheduleFrameRef.current();
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.mapId, props.mapYamlUrl]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const sc = staticRef.current;
    const oc = overlayRef.current;
    const fc = fleetCarsRef.current;
    const ec = evaluatorRef.current;
    if (!wrap || !sc || !oc || !fc || !ec) return;

    let raf = 0;
    let idleTimer = 0;
    let alive = true;
    let cssW = 1;
    let cssH = 1;
    let dpr = 1;
    let lastOverlayKey = "";
    let lastEvaluatorBoundsKey = "";
    let previousEvaluatorCar: EvaluatorCarTelemetry | null = null;
    let lastBoundsKey = "";
    let previousFleetCars: FleetCar[] = [];
    let lastFleetBoundsKey = "";
    let lastFleetCarsKey = "";

    const scheduleFrame = () => {
      if (!alive) return;
      if (idleTimer) {
        window.clearTimeout(idleTimer);
        idleTimer = 0;
      }
      if (!raf) raf = requestAnimationFrame(tick);
    };

    const scheduleIdleFrame = () => {
      if (!alive || idleTimer) return;
      // Check staleness and low-frequency status changes without a permanent
      // 60 Hz polling loop while the canvas is otherwise still.
      idleTimer = window.setTimeout(() => {
        idleTimer = 0;
        scheduleFrame();
      }, 500);
    };

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      cssW = Math.max(1, Math.floor(rect.width));
      cssH = Math.max(1, Math.floor(rect.height));
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      for (const c of [sc, oc, fc]) {
        c.width = Math.floor(cssW * dpr);
        c.height = Math.floor(cssH * dpr);
        c.style.width = `${cssW}px`;
        c.style.height = `${cssH}px`;
      }
      ec.width = Math.floor(120 * dpr);
      ec.height = Math.floor(64 * dpr);
      ec.style.width = "120px";
      ec.style.height = "64px";
      mapDirty.current = true;
      lastOverlayKey = "";
      lastFleetCarsKey = "";
      lastEvaluatorBoundsKey = "";
      previousEvaluatorCar = null;
      previousFleetCars = [];
      lastFleetBoundsKey = "";
      lastFleetCarsKey = "";
      scheduleFrame();
    };

    const tick = () => {
      if (!alive) return;
      raf = 0;

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
        phase.step >= (fleet?.step ?? 0);
      const currentStep = Math.max(p.trainingStep ?? 0, fleet?.step ?? 0);
      const rolloutActive = trainActive && phase?.phase === "rollout" && (p.rolloutSize ?? 0) > 0;
      const nextUpdateStep = rolloutActive ? phase.step + (p.rolloutSize ?? 0) : null;
      const ppoProgress = ppoUpdating
        ? phase?.runtime === "official"
          ? { label: "PPO updating · last action held", progress: 1, detail: "Pose refreshes next rollout" }
          : { label: "PPO updating · simulator paused", progress: 1, detail: "Update in progress" }
          : rolloutActive && !stale && nextUpdateStep != null
          ? {
            label: `${Math.max(0, nextUpdateStep - currentStep).toLocaleString()} steps until PPO update`,
            progress: Math.max(0, Math.min(1, (currentStep - phase.step) / (p.rolloutSize ?? 1))),
            detail: `Step ${currentStep.toLocaleString()} / ${nextUpdateStep.toLocaleString()}`,
          }
          : null;
      const staleLabel = stale && !ppoUpdating
        ? p.fleetSource === "train" && !trainActive
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

      const map = mapRef.current;
      let evaluatorCar: EvaluatorCarTelemetry | null = null;
      if (p.fleetSource !== "replay" && p.evaluatorActive) {
        const live = getEvaluatorLiveHot();
        const matchingLive: EvaluatorLiveTelemetry | null =
          live && live.snapshot_timesteps === p.evaluatorSnapshotTimesteps ? live : null;
        evaluatorCar = matchingLive ?? p.evaluatorCar ?? null;
      }
      const bounds = resolveBounds(map, view, fleet, evaluatorCar?.pose);
      const boundsKey =
        bounds.minX.toFixed(2) +
        ":" +
        bounds.maxX.toFixed(2) +
        ":" +
        bounds.minZ.toFixed(2) +
        ":" +
        bounds.maxZ.toFixed(2);

      const currentFleetCars = view.showFleet
        ? fleet?.cars ?? []
        : [];

      const overlayKey =
        (fleet?.ts ?? "") +
        ":" +
        (fleet?.step ?? "") +
        ":evaluator:" +
        Boolean(evaluatorCar) +
        (ppoProgress ? `${ppoProgress.label}:${ppoProgress.progress.toFixed(3)}:${ppoProgress.detail}` : "no-ppo-progress") +
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
        lastOverlayKey = "";
        lastFleetCarsKey = "";
      }

      const opts: DrawFrameOpts = {
        map,
        fleet,
        evaluatorCar,
        staleLabel,
        ppoProgress,
        view,
        cssW,
        cssH,
        dpr,
      };
      if (overlayKey !== lastOverlayKey) {
        lastOverlayKey = overlayKey;
        const octx = oc.getContext("2d");
        if (octx) drawTelemetryOverlay(octx, opts, bounds);
      }

      const fleetCarsKey = `${fleet?.ts ?? ""}:${fleet?.step ?? ""}:${view.showFleet}:${view.selectedEnvId}:${boundsKey}`;
      const fleetPoseKey = `${view.showFleet}:${view.selectedEnvId}:${boundsKey}:${fleetCarsKey}`;
      const fleetBoundsChanged = boundsKey !== lastFleetBoundsKey;
      if (fleetPoseKey !== lastFleetCarsKey || fleetBoundsChanged) {
        lastFleetCarsKey = fleetPoseKey;
        lastFleetBoundsKey = boundsKey;
        const fctx = fc.getContext("2d");
        if (fctx) {
          drawFleetCarsIncremental(
            fctx,
            currentFleetCars,
            previousFleetCars,
            view,
            bounds,
            cssW,
            cssH,
            dpr,
            fleetBoundsChanged
          );
        }
        previousFleetCars = currentFleetCars;
      }

      const evaluatorBoundsChanged = boundsKey !== lastEvaluatorBoundsKey;
      if (evaluatorBoundsChanged) {
        lastEvaluatorBoundsKey = boundsKey;
        previousEvaluatorCar = null;
      }
      const visibleEvaluatorCar = view.showFleet ? evaluatorCar : null;
      const evaluatorKey = visibleEvaluatorCar
        ? `${visibleEvaluatorCar.pose[0]}:${visibleEvaluatorCar.pose[1]}:${visibleEvaluatorCar.yaw}:${visibleEvaluatorCar.speed}`
        : "none";
      const previousEvaluatorKey = previousEvaluatorCar
        ? `${previousEvaluatorCar.pose[0]}:${previousEvaluatorCar.pose[1]}:${previousEvaluatorCar.yaw}:${previousEvaluatorCar.speed}`
        : "none";
      if (evaluatorKey !== previousEvaluatorKey || evaluatorBoundsChanged) {
        const ectx = ec.getContext("2d");
        if (ectx) {
          if (visibleEvaluatorCar) {
            const { scale, ox, oy } = worldToScreenTransform(bounds, cssW, cssH);
            const x = ox + visibleEvaluatorCar.pose[0] * scale - 16;
            const y = oy - visibleEvaluatorCar.pose[1] * scale - 30;
            ec.style.display = "block";
            ec.style.transform = `translate3d(${x}px, ${y}px, 0)`;
          } else {
            ec.style.display = "none";
          }
          drawEvaluatorCarIncremental(
            ectx,
            visibleEvaluatorCar,
            bounds,
            cssW,
            cssH,
            dpr
          );
        }
        previousEvaluatorCar = visibleEvaluatorCar;
      }

      scheduleIdleFrame();
    };

    scheduleFrameRef.current = scheduleFrame;
    const ro = new ResizeObserver(resize);
    resize();
    ro.observe(wrap);
    window.addEventListener("resize", resize);
    const unsubscribeFleetFrames = subscribeFleetFrames(scheduleFrame);
    scheduleFrame();
    return () => {
      alive = false;
      if (idleTimer) window.clearTimeout(idleTimer);
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("resize", resize);
      unsubscribeFleetFrames();
      scheduleFrameRef.current = () => {};
    };
  }, []);

  return (
    <div className="fleet-canvas-wrap" ref={wrapRef}>
      <canvas className="fleet-canvas fleet-canvas-static" ref={staticRef} />
      <canvas className="fleet-canvas fleet-canvas-overlay" ref={overlayRef} />
      <canvas className="fleet-canvas fleet-canvas-cars" ref={fleetCarsRef} />
      <canvas className="fleet-canvas fleet-canvas-evaluator" ref={evaluatorRef} />
    </div>
  );
}
