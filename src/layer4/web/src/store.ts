import { useSyncExternalStore } from "react";
import {
  ConnState,
  FleetTelemetry,
  MetricsTelemetry,
  TrainStatus,
  getTrainStatus,
  wsUrl,
} from "./api";

const METRICS_CAP = 2000;
const FLEET_UI_HZ = 4;

type Listener = () => void;

interface StoreState {
  conn: ConnState;
  status: TrainStatus | null;
  metrics: MetricsTelemetry | null;
  steps: Float32Array;
  rewards: Float32Array;
  metricsLen: number;
  fleet: FleetTelemetry | null;
  fleetAgeMs: number;
}

let state: StoreState = {
  conn: "offline",
  status: null,
  metrics: null,
  steps: new Float32Array(METRICS_CAP),
  rewards: new Float32Array(METRICS_CAP),
  metricsLen: 0,
  fleet: null,
  fleetAgeMs: Infinity,
};

/** Last-value fleet for rAF canvas — updated every WS sample without React. */
let fleetHot: FleetTelemetry | null = null;
let fleetHotReceivedAt = 0;
let lastFleetUiEmit = 0;

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

export function getFleetHot(): FleetTelemetry | null {
  return fleetHot;
}

export function getFleetHotAgeMs(): number {
  if (!fleetHot) return Infinity;
  return performance.now() - fleetHotReceivedAt;
}

async function resyncStatus() {
  try {
    const status = await getTrainStatus();
    setState({ status });
    if (status.last_telemetry) {
      pushMetric(status.last_telemetry);
    }
    if (status.last_fleet) {
      ingestFleet(status.last_fleet);
      setState({ fleet: status.last_fleet, fleetAgeMs: 0 });
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
        setState({ status: msg.payload as TrainStatus });
      } else if (msg.type === "telemetry") {
        const p = msg.payload as MetricsTelemetry & FleetTelemetry;
        if (p.kind === "fleet") {
          ingestFleet(p as FleetTelemetry);
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
