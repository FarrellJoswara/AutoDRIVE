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

interface EvaluatorTransition {
  from: EvaluatorCarTelemetry;
  to: EvaluatorCarTelemetry;
  startedAt: number;
  durationMs: number;
}

interface FleetCarTransition {
  from: FleetCar;
  to: FleetCar;
  startedAt: number;
  durationMs: number;
}

function interpolateFleetCar(transition: FleetCarTransition, now: number): FleetCar {
  const t = transition.durationMs <= 0
    ? 1
    : Math.max(0, Math.min(1, (now - transition.startedAt) / transition.durationMs));
  const { from, to } = transition;
  let yaw = to.yaw;
  if (from.yaw != null && to.yaw != null) {
    let angleDelta = (to.yaw - from.yaw) % (Math.PI * 2);
    if (angleDelta > Math.PI) angleDelta -= Math.PI * 2;
    if (angleDelta < -Math.PI) angleDelta += Math.PI * 2;
    yaw = from.yaw + angleDelta * t;
  }
  const pose = from.pose && to.pose
    ? [
      from.pose[0] + (to.pose[0] - from.pose[0]) * t,
      from.pose[1] + (to.pose[1] - from.pose[1]) * t,
    ] as [number, number]
    : to.pose;
  const speed = from.speed != null && to.speed != null
    ? from.speed + (to.speed - from.speed) * t
    : to.speed;
  return { ...to, pose, yaw, speed };
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
 * Static map and telemetry layers redraw on changes; rAF is active only while
 * fleet/evaluator interpolation is in progress.
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
    let lastFleetSampleKey = "";
    let lastFleetSampleAt = 0;
    let fleetCarTransitions = new Map<number, FleetCarTransition>();
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
      const now = performance.now();

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

      const map = mapRef.current;
      const fleetSampleKey = `${fleet?.run_id ?? ""}:${fleet?.step ?? ""}:${fleet?.ts ?? ""}`;
      if (fleetSampleKey !== lastFleetSampleKey) {
        const sampleIntervalMs = lastFleetSampleAt > 0
          ? Math.max(40, Math.min(500, now - lastFleetSampleAt))
          : 0;
        const previousById = new Map(previousFleetCars.map((car) => [car.env_id, car]));
        const nextTransitions = new Map<number, FleetCarTransition>();
        for (const target of fleet?.cars ?? []) {
          if (target.reset || !target.pose) continue;
          const from = previousById.get(target.env_id) ?? target;
          nextTransitions.set(target.env_id, {
            from,
            to: target,
            startedAt: now,
            durationMs: previousById.has(target.env_id) ? sampleIntervalMs : 0,
          });
        }
        fleetCarTransitions = nextTransitions;
        lastFleetSampleKey = fleetSampleKey;
        lastFleetSampleAt = now;
      }

      let evaluatorCar: EvaluatorCarTelemetry | null = null;
      if (p.fleetSource !== "replay" && p.evaluatorActive) {
        const live = getEvaluatorLiveHot();
        const matchingLive: EvaluatorLiveTelemetry | null =
          live && live.snapshot_timesteps === p.evaluatorSnapshotTimesteps ? live : null;
        const target = matchingLive ?? p.evaluatorCar ?? null;
        if (target) {
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

      const fleetAnimating = view.showFleet && [...fleetCarTransitions.values()].some(
        (transition) => transition.durationMs > 0 && now < transition.startedAt + transition.durationMs,
      );
      const currentFleetCars = view.showFleet
        ? [...fleetCarTransitions.values()].map((transition) => interpolateFleetCar(transition, now))
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
      const fleetPoseKey = `${view.showFleet}:${view.selectedEnvId}:${boundsKey}:` + (fleetAnimating
        ? currentFleetCars.map((car) => `${car.env_id}:${car.pose?.[0].toFixed(4)}:${car.pose?.[1].toFixed(4)}:${car.yaw?.toFixed(4)}:${car.speed?.toFixed(3)}:${car.collision}`).join("|")
        : fleetCarsKey);
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
        ? `${visibleEvaluatorCar.pose[0].toFixed(4)}:${visibleEvaluatorCar.pose[1].toFixed(4)}:${visibleEvaluatorCar.yaw.toFixed(4)}:${visibleEvaluatorCar.speed.toFixed(3)}`
        : "none";
      const previousEvaluatorKey = previousEvaluatorCar
        ? `${previousEvaluatorCar.pose[0].toFixed(4)}:${previousEvaluatorCar.pose[1].toFixed(4)}:${previousEvaluatorCar.yaw.toFixed(4)}:${previousEvaluatorCar.speed.toFixed(3)}`
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

      const evaluatorAnimating = Boolean(
        visibleEvaluatorCar && evaluatorTransitionRef.current &&
        evaluatorTransitionRef.current.durationMs > 0 &&
        now < evaluatorTransitionRef.current.startedAt + evaluatorTransitionRef.current.durationMs,
      );
      if (fleetAnimating || evaluatorAnimating) {
        raf = requestAnimationFrame(tick);
      } else {
        scheduleIdleFrame();
      }
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
