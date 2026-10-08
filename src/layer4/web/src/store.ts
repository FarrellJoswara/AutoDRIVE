import { useSyncExternalStore } from "react";
import {
  ConnState,
  EvaluatorLiveTelemetry,
  FleetTelemetry,
  MetricsTelemetry,
  ReplayStatus,
  TrainingPhaseTelemetry,
  TrainStatus,
  OfficialRunTelemetry,
  RunSummary,
  getReplayStatus,
  wsUrl,
} from "./api";

const METRICS_CAP = 2000;
const FLEET_UI_HZ = 4;
const metricsHistoryHot: MetricsTelemetry[] = new Array(METRICS_CAP);
let metricsHistoryLength = 0;
let metricsHistoryNext = 0;

type Listener = () => void;

interface StoreState {
  conn: ConnState;
  status: TrainStatus | null;
  replayStatus: ReplayStatus | null;
  trainingPhase: TrainingPhaseTelemetry | null;
  metrics: MetricsTelemetry | null;
  steps: Float32Array;
  rewards: Float32Array;
  metricsLen: number;
  fleet: FleetTelemetry | null;
  fleetAgeMs: number;
  replayFleet: FleetTelemetry | null;
  replayFleetAgeMs: number;
  officialRunId: string | null;
}

let state: StoreState = {
  conn: "offline",
  status: null,
  replayStatus: null,
  trainingPhase: null,
  metrics: null,
  steps: new Float32Array(METRICS_CAP),
  rewards: new Float32Array(METRICS_CAP),
  metricsLen: 0,
  fleet: null,
  fleetAgeMs: Infinity,
  replayFleet: null,
  replayFleetAgeMs: Infinity,
  officialRunId: null,
};

/** Last-value fleet for rAF canvas — updated every WS sample without React. */
let fleetHot: FleetTelemetry | null = null;
let fleetHotReceivedAt = 0;
let lastFleetUiEmit = 0;
let replayFleetHot: FleetTelemetry | null = null;
let replayFleetReceivedAt = 0;
let lastReplayUiEmit = 0;

let trainingPhaseHot: TrainingPhaseTelemetry | null = null;
let evaluatorLiveHot: EvaluatorLiveTelemetry | null = null;
let officialMetricKey: string | null = null;

/** Latest evaluator pose for the animation-frame canvas path. */
export function getEvaluatorLiveHot(): EvaluatorLiveTelemetry | null {
  return evaluatorLiveHot;
}

const listeners = new Set<Listener>();
const fleetFrameListeners = new Set<Listener>();
let ws: WebSocket | null = null;
let backoff = 250;
let reconnectTimer: number | null = null;
let started = false;

function emit() {
  listeners.forEach((l) => l());
}

/** Wake canvas animation only when new telemetry or training state arrives. */
export function subscribeFleetFrames(listener: Listener): () => void {
  fleetFrameListeners.add(listener);
  return () => fleetFrameListeners.delete(listener);
}

function emitFleetFrame() {
  fleetFrameListeners.forEach((listener) => listener());
}

function setState(partial: Partial<StoreState>) {
  state = { ...state, ...partial };
  emit();
}

function pushMetric(m: MetricsTelemetry) {
  metricsHistoryHot[metricsHistoryNext] = m;
  metricsHistoryNext = (metricsHistoryNext + 1) % METRICS_CAP;
  metricsHistoryLength = Math.min(metricsHistoryLength + 1, METRICS_CAP);
  const i = state.metricsLen % METRICS_CAP;
  state.steps[i] = m.step;
  state.rewards[i] = m.reward;
  const nextLen = Math.min(state.metricsLen + 1, METRICS_CAP);
  state = {
    ...state,
    metrics: m,
    metricsLen: state.metricsLen >= METRICS_CAP ? METRICS_CAP : nextLen,
  };
  emit();
}

/** Return recent metrics in time order for charts mounted after training starts. */
export function getMetricsHistoryHot(runId?: string): MetricsTelemetry[] {
  const history: MetricsTelemetry[] = [];
  const start = (metricsHistoryNext - metricsHistoryLength + METRICS_CAP) % METRICS_CAP;
  for (let offset = 0; offset < metricsHistoryLength; offset += 1) {
    const sample = metricsHistoryHot[(start + offset) % METRICS_CAP];
    if (!runId || sample.run_id === runId) history.push(sample);
  }
  return history;
}

export function setOfficialRunSelection(run: RunSummary | null) {
  const runId = run?.run_id ?? null;
  if (state.officialRunId === runId) {
    if (run) setState({ status: runToLegacyStatus(run) });
    return;
  }
  fleetHot = null;
  fleetHotReceivedAt = 0;
  trainingPhaseHot = null;
  evaluatorLiveHot = null;
  officialMetricKey = null;
  state = {
    ...state,
    officialRunId: runId,
    fleet: null,
    fleetAgeMs: Infinity,
    metrics: null,
    metricsLen: 0,
    trainingPhase: null,
    status: run ? runToLegacyStatus(run) : null,
  };
  emit();
  emitFleetFrame();
}

function runToLegacyStatus(run: RunSummary): TrainStatus {
  const status = state.status;
  return {
    state: run.state === "failed" ? "error"
      : run.state === "completed" || run.state === "stopped" ? "exited"
      : run.state,
    pid: run.pid ?? null,
    started_at: run.started_at ?? null,
    argv: [],
    exit_code: run.exit_code ?? null,
    stop_reason: run.stop_reason ?? null,
    log_path: run.log_path ?? null,
    hub_url: status?.hub_url ?? "",
    error: run.error ?? null,
    cleanup_error: run.cleanup_error ?? null,
    stopped_containers: [],
  };
}

export function ingestOfficialRunTelemetry(snapshot: OfficialRunTelemetry) {
  if (!snapshot.run_id || snapshot.run_id !== state.officialRunId) return;
  if (snapshot.fleet) {
    const sourceTs = Date.parse(snapshot.fleet.ts);
    const ageMs = Number.isFinite(sourceTs) ? Math.max(0, Date.now() - sourceTs) : 0;
    ingestFleet(snapshot.fleet, ageMs);
  }
  if (snapshot.metrics) {
    const key = `${snapshot.metrics.run_id}:${snapshot.metrics.step}:${snapshot.metrics.ts}`;
    if (key !== officialMetricKey) {
      officialMetricKey = key;
      pushMetric(snapshot.metrics);
    }
  }
  if (snapshot.training_phase) ingestTrainingPhase({ ...snapshot.training_phase, runtime: snapshot.training_phase.runtime ?? "official" });
  const selected = state.status;
  if (selected) setState({ status: { ...selected, state: selected.state } });
}

function ingestFleet(p: FleetTelemetry, sourceAgeMs = 0) {
  fleetHot = p;
  fleetHotReceivedAt = performance.now() - Math.max(0, sourceAgeMs);
  emitFleetFrame();
  const now = performance.now();
  // Throttle React side-panel updates (~4 Hz); canvas reads getFleetHot()
  if (now - lastFleetUiEmit >= 1000 / FLEET_UI_HZ) {
    lastFleetUiEmit = now;
    setState({ fleet: p, fleetAgeMs: Math.max(0, sourceAgeMs) });
  } else {
    // Keep age fresh without full panel churn when possible
    state = { ...state, fleetAgeMs: Math.max(0, sourceAgeMs) };
  }
}

function ingestReplayFleet(p: FleetTelemetry) {
  replayFleetHot = p;
  replayFleetReceivedAt = performance.now();
  emitFleetFrame();
  const now = performance.now();
  if (now - lastReplayUiEmit >= 1000 / FLEET_UI_HZ) {
    lastReplayUiEmit = now;
    setState({ replayFleet: p, replayFleetAgeMs: 0 });
  } else {
    state = { ...state, replayFleetAgeMs: 0 };
  }
}

export function getFleetHot(): FleetTelemetry | null {
  return fleetHot;
}

export function getFleetHotAgeMs(): number {
  if (!fleetHot) return Infinity;
  return performance.now() - fleetHotReceivedAt;
}

export function getReplayFleetHot(): FleetTelemetry | null {
  return replayFleetHot;
}

export function getReplayFleetHotAgeMs(): number {
  if (!replayFleetHot) return Infinity;
  return performance.now() - replayFleetReceivedAt;
}

export function getTrainingPhaseHot(): TrainingPhaseTelemetry | null {
  return trainingPhaseHot;
}

export function getTrainingStateHot(): TrainStatus["state"] | null {
  const statusState = state.status?.state ?? null;
  if (statusState && statusState !== "idle") return statusState;
  if (trainingPhaseHot?.runtime === "official") {
    if (trainingPhaseHot.phase === "stopped") return statusState ?? "idle";
    if (getTrainingPhaseAgeMs() < 180_000) return "running";
  }
  return statusState;
}

export function getTrainingPhaseAgeMs(): number {
  if (!trainingPhaseHot) return Infinity;
  const sentAt = Date.parse(trainingPhaseHot.ts);
  return Number.isFinite(sentAt) ? Math.max(0, Date.now() - sentAt) : Infinity;
}

function ingestTrainingPhase(phase: TrainingPhaseTelemetry | null) {
  trainingPhaseHot = phase;
  setState({ trainingPhase: phase });
  emitFleetFrame();
}

async function resyncStatus() {
  try {
    const replay = await getReplayStatus();
    setState({ replayStatus: replay });
    if (replay.last_fleet) {
      ingestReplayFleet(replay.last_fleet);
      setState({ replayFleet: replay.last_fleet, replayFleetAgeMs: 0 });
    }
  } catch {
    /* hub may be down during reconnect */
  }
}

// WebSocket frames provide the live path. Poll the persisted latest frame as a
// recovery path too, so Watch survives a missed/reconnecting socket or hub
// restart and does not depend on opening the page at exactly the right time.
function scheduleReconnect() {
  setState({ conn: "reconnecting" });
  if (reconnectTimer != null) window.clearTimeout(reconnectTimer);
  const jitter = Math.random() * 100;
  reconnectTimer = window.setTimeout(() => {
    connect();
  }, backoff + jitter);
  backoff = Math.min(5000, backoff * 1.8);
}

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  try {
    ws = new WebSocket(wsUrl());
  } catch {
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    backoff = 250;
    setState({ conn: "live" });
    void resyncStatus();
  };

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data as string) as {
        type: string;
        payload: unknown;
      };
      if (msg.type === "status") {
        const status = msg.payload as TrainStatus;
        if (!state.officialRunId) setState({ status });
        emitFleetFrame();
        if (status.state === "starting" && !state.officialRunId) ingestTrainingPhase(null);
      } else if (msg.type === "train_phase") {
        ingestTrainingPhase(msg.payload as TrainingPhaseTelemetry);
      } else if (msg.type === "evaluator_live") {
        evaluatorLiveHot = msg.payload as EvaluatorLiveTelemetry;
        emitFleetFrame();
      } else if (msg.type === "replay_status") {
        setState({ replayStatus: msg.payload as ReplayStatus });
      } else if (msg.type === "replay_telemetry") {
        ingestReplayFleet(msg.payload as FleetTelemetry);
      } else if (msg.type === "telemetry") {
        const p = msg.payload as MetricsTelemetry & FleetTelemetry;
        if (p.kind === "fleet") {
          if (!state.officialRunId || p.run_id === state.officialRunId) ingestFleet(p as FleetTelemetry);
        } else if (p.kind === "phase") {
          if (!state.officialRunId || p.run_id === state.officialRunId) ingestTrainingPhase(p as unknown as TrainingPhaseTelemetry);
        } else {
          if (!state.officialRunId || p.run_id === state.officialRunId) pushMetric(p as MetricsTelemetry);
        }
      }
    } catch {
      /* ignore bad frames */
    }
  };

  ws.onclose = () => {
    ws = null;
    scheduleReconnect();
  };

  ws.onerror = () => {
    try {
      ws?.close();
    } catch {
      /* ignore */
    }
  };
}

export function startStore() {
  if (started) return;
  started = true;
  void resyncStatus();
  connect();
  window.setInterval(() => {
    if (fleetHot) {
      const age = getFleetHotAgeMs();
      // Only emit when crossing stale threshold or for panel age display
      if (Math.abs(age - state.fleetAgeMs) > 200) {
        setState({ fleetAgeMs: age });
      }
    }
    if (replayFleetHot) {
      const age = getReplayFleetHotAgeMs();
      if (Math.abs(age - state.replayFleetAgeMs) > 200) {
        setState({ replayFleetAgeMs: age });
      }
    }
  }, 500);
}

export function getSnapshot(): StoreState {
  return state;
}

export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useHubStore(): StoreState {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
