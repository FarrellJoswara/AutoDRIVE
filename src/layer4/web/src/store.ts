import { useSyncExternalStore } from "react";
import {
  ConnState,
  FleetTelemetry,
  MetricsTelemetry,
  ReplayStatus,
  TrainingPhaseTelemetry,
  TrainStatus,
  getTrainStatus,
  getReplayStatus,
  wsUrl,
} from "./api";

const METRICS_CAP = 2000;
const FLEET_UI_HZ = 4;

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
};

/** Last-value fleet for rAF canvas — updated every WS sample without React. */
let fleetHot: FleetTelemetry | null = null;
let fleetHotReceivedAt = 0;
let lastFleetUiEmit = 0;
let replayFleetHot: FleetTelemetry | null = null;
let replayFleetReceivedAt = 0;
let lastReplayUiEmit = 0;

let trainingPhaseHot: TrainingPhaseTelemetry | null = null;

const listeners = new Set<Listener>();
let ws: WebSocket | null = null;
let backoff = 250;
let reconnectTimer: number | null = null;
let started = false;

function emit() {
  listeners.forEach((l) => l());
}

function setState(partial: Partial<StoreState>) {
  state = { ...state, ...partial };
  emit();
}

function pushMetric(m: MetricsTelemetry) {
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

function ingestFleet(p: FleetTelemetry) {
  fleetHot = p;
  fleetHotReceivedAt = performance.now();
  const now = performance.now();
  // Throttle React side-panel updates (~4 Hz); canvas reads getFleetHot()
  if (now - lastFleetUiEmit >= 1000 / FLEET_UI_HZ) {
    lastFleetUiEmit = now;
    setState({ fleet: p, fleetAgeMs: 0 });
  } else {
    // Keep age fresh without full panel churn when possible
    state = { ...state, fleetAgeMs: 0 };
  }
}

function ingestReplayFleet(p: FleetTelemetry) {
  replayFleetHot = p;
  replayFleetReceivedAt = performance.now();
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
  return state.status?.state ?? null;
}

export function getTrainingPhaseAgeMs(): number {
  if (!trainingPhaseHot) return Infinity;
  const sentAt = Date.parse(trainingPhaseHot.ts);
  return Number.isFinite(sentAt) ? Math.max(0, Date.now() - sentAt) : Infinity;
}

function ingestTrainingPhase(phase: TrainingPhaseTelemetry | null) {
  trainingPhaseHot = phase;
  setState({ trainingPhase: phase });
}

async function resyncStatus() {
  try {
    const status = await getTrainStatus();
    setState({ status });
    ingestTrainingPhase(status.last_train_phase ?? null);
    if (status.last_telemetry) {
      pushMetric(status.last_telemetry);
    }
    if (status.last_fleet) {
      ingestFleet(status.last_fleet);
      setState({ fleet: status.last_fleet, fleetAgeMs: 0 });
    }
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
        setState({ status });
        if (status.state === "starting") ingestTrainingPhase(null);
      } else if (msg.type === "train_phase") {
        ingestTrainingPhase(msg.payload as TrainingPhaseTelemetry);
      } else if (msg.type === "replay_status") {
        setState({ replayStatus: msg.payload as ReplayStatus });
      } else if (msg.type === "replay_telemetry") {
        ingestReplayFleet(msg.payload as FleetTelemetry);
      } else if (msg.type === "telemetry") {
        const p = msg.payload as MetricsTelemetry & FleetTelemetry;
        if (p.kind === "fleet") {
          ingestFleet(p as FleetTelemetry);
        } else if (p.kind === "phase") {
          ingestTrainingPhase(p as unknown as TrainingPhaseTelemetry);
        } else {
          pushMetric(p as MetricsTelemetry);
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
