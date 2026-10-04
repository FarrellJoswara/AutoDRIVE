import { useEffect, useRef } from "react";
import type { EvaluatorCarTelemetry, EvaluatorLiveTelemetry, FleetTelemetry } from "../api";
import {
  DrawFrameOpts,
  ViewState,
  drawEvaluatorCarIncremental,
  drawFleetCars,
  drawStaticMap,
  drawTelemetryOverlay,
  initialViewState,
  resolveBounds,
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

interface EvaluatorTransition {
  from: EvaluatorCarTelemetry;
  to: EvaluatorCarTelemetry;
  startedAt: number;
  durationMs: number;
}

function interpolateEvaluator(
  transition: EvaluatorTransition,
  now: number,
): EvaluatorCarTelemetry {
  const t = transition.durationMs <= 0
    ? 1
    : Math.max(0, Math.min(1, (now - transition.startedAt) / transition.durationMs));
  let angleDelta = (transition.to.yaw - transition.from.yaw) % (Math.PI * 2);
  if (angleDelta > Math.PI) angleDelta -= Math.PI * 2;
  if (angleDelta < -Math.PI) angleDelta += Math.PI * 2;
  return {
    pose: [
      transition.from.pose[0] + (transition.to.pose[0] - transition.from.pose[0]) * t,
      transition.from.pose[1] + (transition.to.pose[1] - transition.from.pose[1]) * t,
    ],
    yaw: transition.from.yaw + angleDelta * t,
    speed: transition.from.speed + (transition.to.speed - transition.from.speed) * t,
  };
}

/**
 * Two canvases: static map underlayer + dynamic cars/LiDAR.
 * rAF reads getFleetHot() — WS does not drive React for every sample.
 */
export function FleetCanvas(props: FleetCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const staticRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const fleetCarsRef = useRef<HTMLCanvasElement>(null);
  const evaluatorRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<LoadedMap | null>(null);
  const viewRef = useRef<ViewState>(initialViewState());
  const lastFleetSeq = useRef(0);
  const mapDirty = useRef(true);
  const propsRef = useRef(props);
  const evaluatorTransitionRef = useRef<EvaluatorTransition | null>(null);
  const evaluatorTargetRef = useRef<{ key: string; car: EvaluatorCarTelemetry; receivedAt: number } | null>(null);
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
    const oc = overlayRef.current;
    const fc = fleetCarsRef.current;
    const ec = evaluatorRef.current;
    if (!wrap || !sc || !oc || !fc || !ec) return;

    let raf = 0;
    let alive = true;
    let lastOverlayKey = "";
    let lastFleetCarsKey = "";
    let lastEvaluatorBoundsKey = "";
    let previousEvaluatorCar: EvaluatorCarTelemetry | null = null;
    let lastBoundsKey = "";

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const cssW = Math.max(1, Math.floor(rect.width));
      const cssH = Math.max(1, Math.floor(rect.height));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      for (const c of [sc, oc, fc, ec]) {
        c.width = Math.floor(cssW * dpr);
        c.height = Math.floor(cssH * dpr);
        c.style.width = `${cssW}px`;
        c.style.height = `${cssH}px`;
      }
      mapDirty.current = true;
      lastOverlayKey = "";
      lastFleetCarsKey = "";
      lastEvaluatorBoundsKey = "";
      previousEvaluatorCar = null;
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
        phase.step >= (fleet?.step ?? 0);
      const currentStep = Math.max(p.trainingStep ?? 0, fleet?.step ?? 0);
      const rolloutActive = trainActive && phase?.phase === "rollout" && (p.rolloutSize ?? 0) > 0;
      const nextUpdateStep = rolloutActive ? phase.step + (p.rolloutSize ?? 0) : null;
      const ppoProgress = ppoUpdating
        ? { label: "PPO updating · simulator paused", progress: 1, detail: "Update in progress" }
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

      const cssW = sc.clientWidth;
      const cssH = sc.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const map = mapRef.current;
      let evaluatorCar: EvaluatorCarTelemetry | null = null;
      if (p.fleetSource !== "replay" && p.evaluatorActive) {
        const live = getEvaluatorLiveHot();
        const matchingLive: EvaluatorLiveTelemetry | null =
          live && live.snapshot_timesteps === p.evaluatorSnapshotTimesteps ? live : null;
        const target = matchingLive ?? p.evaluatorCar ?? null;
        if (target) {
          const now = performance.now();
          const key = `${matchingLive ? "ws" : "poll"}:${target.pose[0]}:${target.pose[1]}:${target.yaw}:${target.speed}`;
          const previousTarget = evaluatorTargetRef.current;
          if (!previousTarget || previousTarget.key !== key) {
            const current = evaluatorTransitionRef.current
              ? interpolateEvaluator(evaluatorTransitionRef.current, now)
              : previousTarget?.car ?? target;
            const durationMs = previousTarget
              ? Math.max(40, Math.min(500, now - previousTarget.receivedAt))
              : 0;
            evaluatorTransitionRef.current = { from: current, to: target, startedAt: now, durationMs };
            evaluatorTargetRef.current = { key, car: target, receivedAt: now };
          }
          evaluatorCar = evaluatorTransitionRef.current
            ? interpolateEvaluator(evaluatorTransitionRef.current, now)
            : target;
        }
      } else {
        evaluatorTransitionRef.current = null;
        evaluatorTargetRef.current = null;
      }
      // The fitted view follows sampled evaluator targets rather than its
      // interpolated pose, so animation frames cannot continuously reframe it.
      const bounds = resolveBounds(map, view, fleet, evaluatorTargetRef.current?.car.pose);
      const boundsKey =
        bounds.minX.toFixed(2) +
        ":" +
        bounds.maxX.toFixed(2) +
        ":" +
        bounds.minZ.toFixed(2) +
        ":" +
        bounds.maxZ.toFixed(2);

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
      if (fleetCarsKey !== lastFleetCarsKey) {
        lastFleetCarsKey = fleetCarsKey;
        const fctx = fc.getContext("2d");
        if (fctx) drawFleetCars(fctx, opts, bounds);
        lastFleetSeq.current += 1;
      }

      const evaluatorBoundsChanged = boundsKey !== lastEvaluatorBoundsKey;
      if (evaluatorBoundsChanged) {
        lastEvaluatorBoundsKey = boundsKey;
        previousEvaluatorCar = null;
      }
      const visibleEvaluatorCar = view.showFleet ? evaluatorCar : null;
      const evaluatorKey = visibleEvaluatorCar
        ? `${visibleEvaluatorCar.pose[0].toFixed(4)}:${visibleEvaluatorCar.pose[1].toFixed(4)}:${visibleEvaluatorCar.yaw.toFixed(4)}:${visibleEvaluatorCar.speed.toFixed(3)}`
        : "none";
      const previousEvaluatorKey = previousEvaluatorCar
        ? `${previousEvaluatorCar.pose[0].toFixed(4)}:${previousEvaluatorCar.pose[1].toFixed(4)}:${previousEvaluatorCar.yaw.toFixed(4)}:${previousEvaluatorCar.speed.toFixed(3)}`
        : "none";
      if (evaluatorKey !== previousEvaluatorKey || evaluatorBoundsChanged) {
        const ectx = ec.getContext("2d");
        if (ectx) {
          drawEvaluatorCarIncremental(
            ectx,
            visibleEvaluatorCar,
            previousEvaluatorCar,
            bounds,
            cssW,
            cssH,
            dpr,
            evaluatorBoundsChanged
          );
        }
        previousEvaluatorCar = evaluatorCar;
      }
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
      <canvas className="fleet-canvas fleet-canvas-overlay" ref={overlayRef} />
      <canvas className="fleet-canvas fleet-canvas-cars" ref={fleetCarsRef} />
      <canvas className="fleet-canvas fleet-canvas-evaluator" ref={evaluatorRef} />
    </div>
  );
}
