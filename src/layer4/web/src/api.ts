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
  stop_reason?: string | null;
  log_path: string | null;
  hub_url: string;
  error: string | null;
  cleanup_error: string | null;
  stopped_containers: string[];
  last_telemetry?: MetricsTelemetry | null;
  last_fleet?: FleetTelemetry | null;
  last_train_phase?: TrainingPhaseTelemetry | null;
}

export interface TrainingPhaseTelemetry {
  kind: "phase";
  phase: "rollout" | "ppo_update" | "stopped";
  step: number;
  run_id: string;
  ts: string;
}

export interface Settings {
  n_envs: number;
  timesteps: number;
  max_duration_seconds: number;
  stop_after_laps: number;
  plateau_min_timesteps: number;
  plateau_window_timesteps: number;
  plateau_patience: number;
  plateau_min_improvement_pct: number;
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
  frontier_stagnation_seconds: number;
  terminate_on_collision: boolean;
  forward_scale: number;
  backward_speed_penalty_scale: number;
  route_progress_scale: number;
  collision_penalty: number;
  slip_penalty: number;
  steer_jerk_penalty: number;
  lap_time_reward_scale: number;
  telemetry_every_n: number;
  fleet_hz: number;
  lidar_display_beams: number;
  telemetry_lidar_max_envs: number;
  docker_mode: boolean;
  stop_sims_on_train_exit: boolean;
  stop_stack_on_train_exit: boolean;
  /** Train-selected map id; locked in on Start. "none" = builtin. */
  map_id: string;
  laps_per_episode: number;
}

export interface ReplayStatus {
  state: TrainState;
  pid: number | null;
  started_at: string | null;
  argv: string[];
  exit_code: number | null;
  error: string | null;
  log_path: string | null;
  model: string | null;
  map_id: string | null;
  seed: number | null;
  run_id: string | null;
  last_fleet?: FleetTelemetry | null;
}

export interface ModelCheckpoint {
  id: string;
  label: string;
  modified: number;
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
  /** True only for the sampled collision event, not for the whole episode. */
  collision: boolean;
  /** Contacts counted since this environment's current episode began. */
  collision_count?: number;
  speed: number | null;
  /** Signed velocity along the vehicle's forward axis; negative means reversing. */
  v_long?: number | null;
  /** Applied continuous action sent to the simulator, each in [-1, 1]. */
  throttle_command?: number | null;
  steering_command?: number | null;
  episode_return: number | null;
  /** Backend-computed across-track frontier segment in world (x,z) coordinates. */
  frontier_line?: [[number, number], [number, number]] | null;
  frontier_progress_m?: number | null;
  /** Full-width cross-section at the car's current projected route position. */
  current_progress_line?: [[number, number], [number, number]] | null;
  current_progress_m?: number | null;
  signed_route_delta_m?: number | null;
  current_route_speed_mps?: number | null;
  route_projection_valid?: boolean;
  reward_components?: {
    route_progress?: number;
    backward_motion?: number;
    collision?: number;
    lap_bonus?: number;
    slip?: number;
    steering_change?: number;
    raw_forward_velocity?: number;
    total?: number;
  } | null;
  time_since_frontier_push_s?: number | null;
  frontier_speed_mps?: number | null;
  lap_supported?: boolean;
  lap_count?: number;
  last_lap_time_s?: number | null;
  best_lap_time_s?: number | null;
  lap_elapsed_s?: number | null;
  lidar?: number[];
  /** True on done/respawn frames — clear LiDAR; pose may be null. */
  reset?: boolean;
}

export interface FleetTelemetry {
  kind: "fleet";
  step: number;
  episode: number;
  run_id: string;
  ts: string;
  cars: FleetCar[];
  /** Metres â€” denorm for normalised lidar [0,1]. Defaults match Layer 1. */
  lidar_range_min?: number;
  lidar_range_max?: number;
  /** Backend-computed virtual finish gate in map world (x,z) coordinates. */
  lap_gate?: [[number, number], [number, number]] | null;
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

/** Hub-discovered occupancy maps under simulator/maps/. */
export interface MapCatalogEntry {
  id: string;
  label: string;
  yaml_url: string | null;
  image_url: string | null;
  mesh_status: string;
  overlay_only: boolean;
  active?: boolean;
  source: string | null;
  thumbnail_url?: string | null;
  centerline_url?: string | null;
  centerline_status?: string;
  mesh_preview_url?: string | null;
}

export function getReplayStatus(): Promise<ReplayStatus> {
  return jsonFetch("/api/replay/status");
}

export function fetchReplayModels(): Promise<ModelCheckpoint[]> {
  return jsonFetch("/api/replay/models");
}

export function startReplay(settings: { model_id: string; map_id: string; seed: number; device: string }): Promise<ReplayStatus> {
  return jsonFetch("/api/replay/start", { method: "POST", body: JSON.stringify(settings) });
}

export function stopReplay(): Promise<ReplayStatus> {
  return jsonFetch("/api/replay/stop", { method: "POST" });
}

export function resetReplay(): Promise<ReplayStatus> {
  return jsonFetch("/api/replay/reset", { method: "POST" });
}

export interface MapLapGate {
  supported: boolean;
  reason?: string | null;
  route_length_m?: number;
  progress_m?: number;
  default_progress_m?: number;
  customized?: boolean;
  centerline_built_at?: string | null;
  gate_line?: [[number, number], [number, number]] | null;
  preview?: {
    width: number;
    height: number;
    gate_line: [[number, number], [number, number]];
  } | null;
}

export function fetchMaps(): Promise<MapCatalogEntry[]> {
  return jsonFetch("/api/maps");
}

export function fetchMapLapGate(mapId: string): Promise<MapLapGate> {
  return jsonFetch(`/api/maps/${encodeURIComponent(mapId)}/lap-gate`);
}

export function saveMapLapGate(
  mapId: string,
  progress_m: number | null
): Promise<MapLapGate> {
  return jsonFetch(`/api/maps/${encodeURIComponent(mapId)}/lap-gate`, {
    method: "PUT",
    body: JSON.stringify({ progress_m }),
  });
}

export function generateMapMesh(mapId: string): Promise<{
  ok: boolean;
  result: Record<string, unknown>;
  maps: MapCatalogEntry[];
}> {
  return jsonFetch(`/api/maps/${encodeURIComponent(mapId)}/generate-mesh`, {
    method: "POST",
  });
}

export function activateMap(
  mapId: string,
  opts?: { force?: boolean; restart?: boolean }
): Promise<{
  ok: boolean;
  active: { id: string | null; activated_at: string | null; path: string };
  restarted: string[];
  restart_error: string | null;
  trackloader: boolean;
  note: string | null;
  maps: MapCatalogEntry[];
}> {
  const q = opts?.force ? "?force=1" : "";
  return jsonFetch(`/api/maps/${encodeURIComponent(mapId)}/activate${q}`, {
    method: "POST",
    body: JSON.stringify({
      force: Boolean(opts?.force),
      restart: opts?.restart !== false,
    }),
  });
}

export function fetchActiveMap(selected?: string): Promise<{
  active: { id: string | null; activated_at: string | null; path: string | null };
  trackloader: boolean;
  selected_id?: string | null;
  selection_mismatch?: boolean;
  warnings?: string[];
  maps: MapCatalogEntry[];
}> {
  const q =
    selected != null && selected !== ""
      ? `?selected=${encodeURIComponent(selected)}`
      : "";
  return jsonFetch(`/api/maps/active${q}`);
}

export async function uploadMapZip(
  file: Blob,
  opts?: { mapId?: string; overwrite?: boolean; label?: string }
): Promise<{
  ok: boolean;
  id: string;
  path: string;
  files: string[];
  maps: MapCatalogEntry[];
}> {
  const q = new URLSearchParams();
  if (opts?.mapId) q.set("map_id", opts.mapId);
  if (opts?.overwrite) q.set("overwrite", "1");
  if (opts?.label) q.set("label", opts.label);
  const qs = q.toString();
  const res = await fetch(`/api/maps/upload${qs ? `?${qs}` : ""}`, {
    method: "POST",
    headers: { "Content-Type": "application/zip" },
    body: file,
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
  return res.json();
}

export function generateMapCenterline(mapId: string): Promise<{
  ok: boolean;
  result: Record<string, unknown>;
  maps: MapCatalogEntry[];
}> {
  return jsonFetch(`/api/maps/${encodeURIComponent(mapId)}/generate-centerline`, {
    method: "POST",
  });
}

export function generateMapMeshPreview(mapId: string): Promise<{
  ok: boolean;
  result: { url: string; n_verts: number; n_faces: number; size_px: number };
  maps: MapCatalogEntry[];
}> {
  return jsonFetch(
    `/api/maps/${encodeURIComponent(mapId)}/generate-mesh-preview`,
    { method: "POST" }
  );
}

export function generateMapThumbnail(mapId: string): Promise<{
  ok: boolean;
  path: string;
  url: string;
  maps: MapCatalogEntry[];
}> {
  return jsonFetch(
    `/api/maps/${encodeURIComponent(mapId)}/generate-thumbnail`,
    { method: "POST" }
  );
}

export function centerlineDownloadUrl(mapId: string): string {
  return `/api/maps/${encodeURIComponent(mapId)}/centerline.csv`;
}

export async function shutdownHub(opts?: {
  stack?: boolean;
}): Promise<{
  ok: boolean;
  shutting_down: boolean;
  stack: boolean;
  stopped_containers: string[];
  train: TrainStatus | null;
  stack_error?: string;
}> {
  const q = opts?.stack ? "?stack=1" : "";
  return jsonFetch(`/api/shutdown${q}`, { method: "POST" });
}
