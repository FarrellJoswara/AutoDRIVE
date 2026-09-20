export type TrainState =
  | "idle"
  | "starting"
  | "running"
  | "stopping"
  | "exited"
  | "error";

export interface TrainStatus {
  state: TrainState;
  pid: number | null;
  started_at: string | null;
  argv: string[];
  exit_code: number | null;
  log_path: string | null;
  hub_url: string;
  error: string | null;
  cleanup_error: string | null;
  stopped_containers: string[];
  last_telemetry?: MetricsTelemetry | null;
  last_fleet?: FleetTelemetry | null;
}

export interface Settings {
  n_envs: number;
  base_port: number;
  timesteps: number;
  out: string | null;
  run_name: string | null;
  seed: number;
  device: "auto" | "cpu" | "cuda";
  resume: string | null;
  headless: boolean;
  auto_launch: boolean;
  connect_timeout: number;
  frame_skip: number;
  max_episode_steps: number;
  stagnation_speed_threshold: number;
  stagnation_steps: number;
  forward_scale: number;
  collision_penalty: number;
  slip_penalty: number;
  steer_jerk_penalty: number;
  telemetry_every_n: number;
  fleet_hz: number;
  lidar_display_beams: number;
  telemetry_lidar_max_envs: number;
  docker_mode: boolean;
  stop_sims_on_train_exit: boolean;
  stop_stack_on_train_exit: boolean;
}

export interface MetricsTelemetry {
  kind?: string;
  step: number;
  reward: number;
  episode: number;
  loss: number | null;
  checkpoint: string | null;
  run_id: string;
  ts: string;
}

export interface FleetCar {
  env_id: number;
  pose: [number, number] | null;
  yaw: number | null;
  collision: boolean;
  speed: number | null;
  episode_return: number | null;
  lidar?: number[];
}

export interface FleetTelemetry {
  kind: "fleet";
  step: number;
  episode: number;
  run_id: string;
  ts: string;
  cars: FleetCar[];
  /** Metres — denorm for normalised lidar [0,1]. Defaults match Layer 1. */
  lidar_range_min?: number;
  lidar_range_max?: number;
}

export type ConnState = "live" | "reconnecting" | "offline";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function getSettings(): Promise<Settings> {
  return jsonFetch("/settings");
}

export function putSettings(s: Settings): Promise<{ ok: boolean; settings: Settings }> {
  return jsonFetch("/settings", { method: "PUT", body: JSON.stringify(s) });
}

export function getTrainStatus(): Promise<TrainStatus> {
  return jsonFetch("/train/status");
}

export function startTrain(s: Settings): Promise<TrainStatus> {
  return jsonFetch("/train/start", { method: "POST", body: JSON.stringify(s) });
}

export function stopTrain(): Promise<TrainStatus> {
  return jsonFetch("/train/stop", { method: "POST" });
}

export function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}
