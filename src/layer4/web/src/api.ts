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
  /** Age of last_fleet's original frame; serving it again does not make it fresh. */
  last_fleet_age_ms?: number | null;
  last_train_phase?: TrainingPhaseTelemetry | null;
  last_evaluation?: EvaluationStatus | null;
}

export interface EvaluationResult {
  timesteps: number;
  simulated_seconds: number;
  frontier_distance_m: number;
  frontier_speed_mps: number;
  reward_per_simulated_second: number;
  total_reward?: number;
  collisions: number;
  failed_episodes: number;
  laps_observed: number;
  lap_times_s: number[];
  best_10_lap_time_s: number | null;
  selection_score?: number | null;
  selection_score_median?: number;
  selection_score_mean?: number;
  selection_score_best?: number;
  evaluation_runs?: number;
  successful_attempts?: number;
  collision_rate?: number;
  attempts?: EvaluationAttemptResult[];
  selection_metric?: "frontier_speed" | "reward_per_simulated_second" | "total_reward" | "ten_lap_time";
  improved?: boolean;
  stale_evaluations?: number;
  exploration_std?: number[];
  exploration_adaptation?: string;
  stop_reason?: string | null;
}

export interface EvaluationAttemptResult {
  simulated_seconds: number;
  frontier_distance_m: number;
  frontier_speed_mps: number;
  reward_per_simulated_second: number;
  total_reward: number;
  collisions: number;
  failed_episodes: number;
  laps_observed: number;
  lap_times_s: number[];
  best_10_lap_time_s: number | null;
  stop_reason?: string;
}

export function getEvaluationHistory(): Promise<EvaluationResult[]> {
  return jsonFetch("/train/evaluations");
}

export interface EvaluationStatus {
  state: "waiting" | "running" | "complete";
  snapshot_timesteps?: number | null;
  selection_metric: "frontier_speed" | "reward_per_simulated_second" | "total_reward" | "ten_lap_time";
  evaluation_runs_per_snapshot?: number;
  best_selection_score?: number | null;
  stale_evaluations: number;
  plateau_patience: number;
  latest_result?: EvaluationResult | null;
  live_result?: EvaluationLiveResult | null;
  evaluator_car?: EvaluatorCarTelemetry | null;
  stop_reason?: string | null;
  updated_utc: string;
}

export interface EvaluationLiveResult {
  attempt_index?: number;
  attempt_count?: number;
  frontier_distance_m: number;
  frontier_speed_mps: number;
  simulated_seconds: number;
  laps_observed: number;
  lap_times_s: number[];
  best_10_lap_time_s: number | null;
  reward_per_simulated_second: number;
  total_reward?: number;
  collisions: number;
}

export interface EvaluatorCarTelemetry {
  pose: [number, number];
  yaw: number;
  speed: number;
}

export interface EvaluatorLiveTelemetry extends EvaluatorCarTelemetry {
  run_id: string;
  snapshot_timesteps: number;
}

export interface TrainingPhaseTelemetry {
  kind: "phase";
  phase: "rollout" | "ppo_update" | "stopped";
  step: number;
  run_id: string;
  runtime?: "custom" | "official";
  rollout_size?: number;
  ts: string;
}

export interface Settings {
  n_envs: number;
  timesteps: number;
  max_duration_seconds: number;
  plateau_min_timesteps: number;
  plateau_patience: number;
  plateau_min_improvement_pct: number;
  evaluation_every_timesteps: number;
  evaluation_runs_per_snapshot: number;
  /** PPO rollout transitions collected per environment before each update. */
  ppo_n_steps: number;
  evaluation_metric: "frontier_speed" | "reward_per_simulated_second" | "total_reward" | "ten_lap_time";
  ppo_learning_rate: number;
  ppo_n_epochs: number;
  ppo_gamma: number;
  ppo_gae_lambda: number;
  exploration_std_min: number;
  exploration_std_max: number;
  exploration_improvement_scale: number;
  exploration_plateau_scale: number;
  out: string | null;
  run_name: string | null;
  seed: number;
  device: "auto" | "cpu" | "cuda";
  resume: string | null;
  headless: boolean;
  auto_launch: boolean;
  simulator_mode: "legacy" | "fixed_camera_on" | "fixed_camera_off";
  action_interval_s: number | null;
  observation_profile: "simulator" | "official_sensors" | "official_sensors_history";
  throttle_mode: "bidirectional" | "forward_only";
  policy_architecture: "lidar_cnn" | "lidar_cnn_pooled" | "temporal_lidar_cnn";
  steering_action_scale: number;
  straight_throttle_gain: number;
  straight_throttle_steering_threshold: number;
  connect_timeout: number;
  frame_skip: number;
  max_episode_steps: number;
  laps_per_episode: number;
  stagnation_speed_threshold: number;
  stagnation_steps: number;
  frontier_stagnation_seconds: number;
  terminate_on_collision: boolean;
  forward_scale: number;
  backward_speed_penalty_scale: number;
  route_progress_scale: number;
  frontier_pace_target_mps: number;
  frontier_pace_bonus_strength: number;
  frontier_pace_source: "episode_average" | "current_push";
  time_penalty_per_second: number;
  collision_penalty_magnitude: number;
  collision_reward_percent: number;
  episode_failure_penalty_magnitude: number;
  episode_failure_reward_percent: number;
  slip_penalty: number;
  steer_jerk_penalty: number;
  telemetry_every_n: number;
  fleet_hz: number;
  lidar_display_beams: number;
  telemetry_lidar_max_envs: number;
  docker_mode: boolean;
  stop_sims_on_train_exit: boolean;
  stop_stack_on_train_exit: boolean;
  /** Train-selected map id; locked in on Start. "none" = builtin. */
  map_id: string;
}

export interface ReplayStatus {
  evaluation?: { completed_attempts: number; best_adjusted_race_time_s: number | null; runs: {stop_reason?: string; score_status?: string}[] } | null;
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
  runtime?: "custom" | "official";
  rollout_size?: number;
  step: number;
  reward: number;
  episode: number;
  loss: number | null;
  /** Step count at which the latest completed PPO update was logged. */
  ppo_step?: number | null;
  /** TensorBoard train/* scalars from the latest completed PPO update. */
  ppo?: Partial<PpoDiagnostics>;
  completed_laps?: number;
  clean_episode_wins?: number;
  best_lap_time_s?: number | null;
  checkpoint: string | null;
  run_id: string;
  ts: string;
}

/** Settings accepted by the official AutoDRIVE trainer. */
export interface OfficialTrainSettings {
  n_envs: number;
  total_timesteps: number;
  stop_on_plateau: boolean;
  plateau_min_timesteps: number;
  plateau_window_episodes: number;
  plateau_patience: number;
  plateau_min_improvement_pct: number;
  max_duration_hours: number;
  seed: number;
  device: "cpu" | "cuda";
  resume: string | null;
  checkpoint_every: number;
  n_steps: number;
  learning_rate: number;
  n_epochs: number;
  gamma: number;
  gae_lambda: number;
  timeout_s: number;
  training_timeout_s: number;
  race_laps: number;
  warmup_laps: number;
  time_cost_per_simulated_second: number;
  lap_completion_reward: number;
  collision_penalty_base: number;
  failed_episode_penalty: number;
  observation_profile: "official_sensors" | "official_sensors_history" | "official_sensors_camera";
  steering_action_scale: number;
  straight_throttle_gain: number;
  straight_throttle_steering_threshold: number;
  throttle_mode: string;
  negative_throttle_mode: string;
  steering_mode: string;
}

export type OfficialRunState = "starting" | "running" | "stopping" | "exited" | "error" | "completed" | "stopped" | "failed";

export interface RunSummary {
  run_id: string;
  kind?: "training" | "evaluation";
  state: OfficialRunState;
  started_at?: string | null;
  pid?: number | null;
  exit_code?: number | null;
  stop_reason?: string | null;
  log_path?: string | null;
  error?: string | null;
  cleanup_error?: string | null;
  config: OfficialTrainSettings;
  map_id?: string | null;
  containers?: string[] | Record<string, unknown>;
  display_name?: string | null;
  created_at?: string | null;
  finished_at?: string | null;
  output_dir?: string | null;
  network_name?: string | null;
  api_container_name?: string | null;
  sim_container_name?: string | null;
  latest_fleet?: FleetTelemetry | null;
  latest_metrics?: MetricsTelemetry | null;
  latest_telemetry?: FleetTelemetry | MetricsTelemetry | null;
}

export interface OfficialRunTelemetry {
  run_id: string;
  updated_at?: string | null;
  fleet?: FleetTelemetry | null;
  metrics?: MetricsTelemetry | null;
  training_phase?: TrainingPhaseTelemetry | null;
  evaluation?: EvaluationStatus | null;
  evaluation_history?: EvaluationResult[];
  ppo_diagnostics?: PpoDiagnosticsHistory | null;
  map_id?: string | null;
}

interface OfficialRunTelemetryWire {
  run_id: string;
  state?: string;
  fleet?: FleetTelemetry | null;
  last_fleet?: FleetTelemetry | null;
  last_metrics?: MetricsTelemetry | null;
  last_train_phase?: TrainingPhaseTelemetry | null;
  last_evaluator_live?: EvaluatorLiveTelemetry | null;
  last_evaluation?: EvaluationStatus | null;
  map_id?: string | null;
  updated_at?: string | null;
}

export interface OfficialRunList {
  runs: RunSummary[];
  max_concurrent_runs: number;
}

const runDetailCache = new Map<string, {
  expiresAt: number;
  evaluations: Promise<EvaluationResult[]>;
  diagnostics: Promise<PpoDiagnosticsHistory | null>;
}>();

export function getOfficialSettings(): Promise<{ settings: OfficialTrainSettings }> {
  return jsonFetch("/train/official-settings");
}

export function getOfficialRuns(): Promise<OfficialRunList> {
  return jsonFetch<OfficialRunList>("/train/runs").then((result) => ({
    ...result,
    runs: result.runs.map(normalizeRunSummary),
  }));
}

export function createOfficialRun(config: OfficialTrainSettings, display_name?: string): Promise<RunSummary> {
  return jsonFetch<RunSummary>("/train/runs", {
    method: "POST",
    body: JSON.stringify({ config, ...(display_name?.trim() ? { display_name: display_name.trim() } : {}) }),
  }).then(normalizeRunSummary);
}

export function getOfficialRun(runId: string): Promise<RunSummary> {
  return jsonFetch<RunSummary>(`/train/runs/${encodeURIComponent(runId)}`).then(normalizeRunSummary);
}

export function stopOfficialRun(runId: string): Promise<RunSummary> {
  return jsonFetch<RunSummary>(`/train/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }).then(normalizeRunSummary);
}

export function getOfficialRunTelemetry(runId: string): Promise<OfficialRunTelemetry> {
  const encoded = encodeURIComponent(runId);
  const now = Date.now();
  let details = runDetailCache.get(runId);
  if (!details || details.expiresAt <= now) {
    details = {
      expiresAt: now + 5000,
      evaluations: jsonFetch<EvaluationResult[]>(`/train/runs/${encoded}/evaluations`).catch(() => []),
      diagnostics: jsonFetch<PpoDiagnosticsHistory>(`/train/runs/${encoded}/ppo-diagnostics`).catch(() => null),
    };
    runDetailCache.set(runId, details);
  }
  return Promise.all([
    jsonFetch<OfficialRunTelemetryWire>(`/train/runs/${encoded}/telemetry`),
    details.evaluations,
    details.diagnostics,
  ]).then(([wire, evaluationHistory, ppoDiagnostics]) => ({
    run_id: wire.run_id,
    updated_at: wire.updated_at ?? wire.last_fleet?.ts ?? wire.last_metrics?.ts ?? null,
    fleet: wire.last_fleet ?? wire.fleet ?? null,
    metrics: wire.last_metrics ?? null,
    training_phase: wire.last_train_phase ?? null,
    evaluation: wire.last_evaluation ?? null,
    evaluation_history: evaluationHistory,
    ppo_diagnostics: ppoDiagnostics,
    map_id: wire.map_id ?? null,
  }));
}

/** Lightweight live frame for multi-run overviews; avoids per-run history scans. */
export function getOfficialRunLiveTelemetry(runId: string): Promise<OfficialRunTelemetry> {
  const encoded = encodeURIComponent(runId);
  return jsonFetch<OfficialRunTelemetryWire>(`/train/runs/${encoded}/telemetry`).then((wire) => ({
    run_id: wire.run_id,
    updated_at: wire.updated_at ?? wire.last_fleet?.ts ?? wire.last_metrics?.ts ?? null,
    fleet: wire.last_fleet ?? wire.fleet ?? null,
    metrics: wire.last_metrics ?? null,
    training_phase: wire.last_train_phase ?? null,
    evaluation: wire.last_evaluation ?? null,
    map_id: wire.map_id ?? null,
  }));
}

function normalizeRunSummary(raw: RunSummary): RunSummary {
  const state = raw.state as string;
  return {
    ...raw,
    state: state === "stopped" ? "exited" : state === "failed" || state === "unknown" ? "error" : raw.state,
    pid: raw.pid ?? null,
    started_at: raw.started_at ?? null,
    exit_code: raw.exit_code ?? null,
    stop_reason: raw.stop_reason ?? null,
    log_path: raw.log_path ?? null,
    error: raw.error ?? null,
    cleanup_error: raw.cleanup_error ?? null,
    containers: raw.containers ?? {
      api: raw.api_container_name,
      simulator: raw.sim_container_name,
      network: raw.network_name,
    },
  };
}

export interface PpoDiagnostics {
  fps: number;
  approx_kl: number;
  clip_fraction: number;
  entropy_loss: number;
  explained_variance: number;
  learning_rate: number;
  loss: number;
  policy_gradient_loss: number;
  std: number;
  value_loss: number;
}

export interface PpoDiagnosticsUpdate {
  step: number;
  values: Partial<PpoDiagnostics>;
}

export interface PpoDiagnosticsHistory {
  run_id: string;
  updates: PpoDiagnosticsUpdate[];
}

export function getPpoDiagnostics(): Promise<PpoDiagnosticsHistory> {
  return jsonFetch("/train/ppo-diagnostics");
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
  /** Source used for the displayed speed estimate. */
  speed_source?: "lidar_estimate" | "wheel_encoder" | "simulator";
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
    reverse_direction_gate?: number;
    backward_motion?: number;
    time_cost?: number;
    collision?: number;
    episode_failure?: number;
    slip?: number;
    steering_change?: number;
    total?: number;
  } | null;
  time_since_frontier_push_s?: number | null;
  frontier_speed_mps?: number | null;
  lap_supported?: boolean;
  lap_count?: number;
  /** Lap splits for this car's current episode. Cleared on respawn. */
  lap_times_s?: number[];
  last_lap_time_s?: number | null;
  best_lap_time_s?: number | null;
  lap_elapsed_s?: number | null;
  /** Best rolling ten-lap total for this fixed-model replay session. */
  best_10_lap_time_s?: number | null;
  lidar?: number[];
  /** True on done/respawn frames — clear LiDAR; pose may be null. */
  reset?: boolean;
}

export interface FleetTelemetry {
  kind: "fleet";
  runtime?: "custom" | "official";
  rollout_size?: number;
  step: number;
  episode: number;
  run_id: string;
  ts: string;
  cars: FleetCar[];
  /** Fastest completed rolling ten-lap total across all cars in this train run. */
  best_10_lap_time_s?: number | null;
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
