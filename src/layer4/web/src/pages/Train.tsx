import { FormEvent, forwardRef, useEffect, useImperativeHandle, useState } from "react";
import {
  Settings,
  fetchMaps,
  getSettings,
  putSettings,
  startTrain,
  stopTrain,
  type MapCatalogEntry,
} from "../api";
import { useHubStore } from "../store";

const empty: Settings = {
  n_envs: 1,
  timesteps: 0,
  max_duration_seconds: 0,
  stop_after_laps: 0,
  plateau_min_timesteps: 100000,
  plateau_window_timesteps: 25000,
  plateau_patience: 5,
  plateau_min_improvement_pct: 1,
  plateau_min_successful_laps: 10,
  expert_pretrain_steps: 0,
  curriculum_single_lap_successes: 10,
  out: null,
  run_name: null,
  seed: 0,
  device: "auto",
  resume: null,
  headless: true,
  auto_launch: true,
  connect_timeout: 90,
  frame_skip: 4,
  max_episode_steps: 0,
  stagnation_speed_threshold: 0.15,
  stagnation_steps: 50,
  frontier_stagnation_seconds: 5,
  terminate_on_collision: true,
  forward_scale: 0,
  backward_speed_penalty_scale: 1,
  route_progress_scale: 10,
  time_penalty_per_second: 1,
  collision_penalty: -100,
  episode_failure_penalty: -100,
  slip_penalty: 0.2,
  steer_jerk_penalty: 0.05,
  lap_time_reward_scale: 1000,
  telemetry_every_n: 200,
  fleet_hz: 15,
  lidar_display_beams: 120,
  telemetry_lidar_max_envs: 4,
  docker_mode: false,
  stop_sims_on_train_exit: true,
  stop_stack_on_train_exit: false,
  map_id: "none",
  laps_per_episode: 10,
};

type NumKey = {
  [K in keyof Settings]: Settings[K] extends number ? K : never;
}[keyof Settings];

function runnableMaps(list: MapCatalogEntry[]): MapCatalogEntry[] {
  const builtin: MapCatalogEntry = {
    id: "none",
    label: "Builtin (no custom mesh)",
    yaml_url: null,
    image_url: null,
    mesh_status: "none",
    overlay_only: true,
    active: false,
    source: null,
  };
  const ready = list.filter(
    (m) => m.id !== "none" && m.mesh_status === "ready"
  );
  return [builtin, ...ready];
}

export interface TrainPageHandle {
  start(): Promise<void>;
  stop(): Promise<void>;
}

interface TrainPageProps {
  onBusyChange?: (busy: boolean) => void;
  onReadyChange?: (ready: boolean) => void;
}

export const TrainPage = forwardRef<TrainPageHandle, TrainPageProps>(function TrainPage(
  { onBusyChange, onReadyChange },
  ref
) {
  const { status } = useHubStore();
  const [form, setForm] = useState<Settings>(empty);
  const [mapChoices, setMapChoices] = useState<MapCatalogEntry[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const state = status?.state ?? "idle";
  const running =
    state === "running" || state === "starting" || state === "stopping";
  const selectedTrainMap =
    mapChoices.find((m) => m.id === form.map_id) ?? null;

  useEffect(() => {
    let cancelled = false;
    Promise.all([getSettings(), fetchMaps().catch(() => [] as MapCatalogEntry[])])
      .then(([s, maps]) => {
        if (cancelled) return;
        const choices = runnableMaps(maps);
        setMapChoices(choices);
        const merged = { ...empty, ...s };
        if (!choices.some((m) => m.id === merged.map_id)) {
          merged.map_id = "none";
        }
        if (merged.map_id === "none") merged.laps_per_episode = 0;
        setForm(merged);
      })
      .catch((e: Error) => {
        if (!cancelled) setErr(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!loading) onReadyChange?.(true);
  }, [loading, onReadyChange]);

  function num(key: NumKey, v: string) {
    const n = Number(v);
    setForm((f) => ({ ...f, [key]: Number.isFinite(n) ? n : f[key] }));
  }

  function str(key: "out" | "run_name" | "resume", v: string) {
    setForm((f) => ({ ...f, [key]: v.trim() === "" ? null : v }));
  }

  async function onSave(e: FormEvent) {
    e.preventDefault();
    setMsg(null);
    setErr(null);
    try {
      const res = await putSettings(form);
      setForm({ ...empty, ...res.settings });
      setMsg("Settings saved");
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    }
  }

  function applyDockerPreset() {
    setForm((f) => ({
      ...f,
      docker_mode: true,
      auto_launch: false,
      headless: true,
      stop_sims_on_train_exit: true,
    }));
  }

  async function onStart() {
    setErr(null);
    setMsg(null);
    setBusy(true);
    onBusyChange?.(true);
    try {
      const res = await putSettings(form);
      const saved = { ...empty, ...res.settings };
      setForm(saved);
      await startTrain(saved);
      setMsg(`Started — map locked: ${saved.map_id || "none"}`);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  }

  async function onStop() {
    setErr(null);
    setBusy(true);
    onBusyChange?.(true);
    try {
      await stopTrain();
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  }

  useImperativeHandle(ref, () => ({ start: onStart, stop: onStop }), [form, onBusyChange, busy]);

  if (loading) {
    return (
      <section className="panel">
        <h2>Train</h2>
        <p className="lede">Loading…</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="run-strip">
        <div className="run-strip-main">
          <h2>Train</h2>
          <span className="badge" data-state={state}>
            {state}
          </span>
          <span className="meta">
            map <strong>{form.map_id || "none"}</strong>
            {running ? " (locked for this run)" : ""}
          </span>
        </div>
      </div>

      <p className="lede">
        Start locks in the map below (activates physics if needed), then runs{" "}
        <code>python -m src.layer3.train</code>. Prep meshes on the Maps tab.
      </p>

      <div className="meta" style={{ marginBottom: "1rem" }}>
        <div>
          <strong>pid</strong> {status?.pid ?? "—"}
        </div>
        <div>
          <strong>log</strong> {status?.log_path ?? "—"}
        </div>
        {status?.error && (
          <div>
            <strong>error</strong> {status.error}
          </div>
        )}
        {status?.argv && status.argv.length > 0 && (
          <div style={{ wordBreak: "break-all" }}>
            <strong>argv</strong> {status.argv.join(" ")}
          </div>
        )}
      </div>

      <form onSubmit={onSave}>
        <div className="grid">
          <div className="section-title">Map</div>
          <div className="train-map-row">
            <label className="field">
              map (builtin + mesh-ready only)
              <select
                value={form.map_id}
                disabled={running}
                onChange={(e) => {
                  const map_id = e.target.value;
                  setForm((f) => {
                    const next = {
                      ...f,
                      map_id,
                      laps_per_episode: map_id === "none"
                        ? 0
                        : (f.laps_per_episode || 10),
                    };
                    // Persist immediately so Watch underlay follows without waiting for Start.
                    void putSettings(next)
                      .then((res) => {
                        setForm((cur) => ({ ...cur, ...res.settings, map_id }));
                        window.dispatchEvent(
                          new CustomEvent("aicar-map-id", { detail: map_id })
                        );
                      })
                      .catch((ex) =>
                        setErr(ex instanceof Error ? ex.message : String(ex))
                      );
                    return next;
                  });
                }}
              >
                {mapChoices.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                    {m.id !== "none" ? " [ready]" : ""}
                  </option>
                ))}
              </select>
            </label>
            {(selectedTrainMap?.thumbnail_url ||
              selectedTrainMap?.mesh_preview_url) && (
              <div className="train-map-preview">
                {selectedTrainMap.thumbnail_url && (
                  <img
                    className="fleet-thumb"
                    src={selectedTrainMap.thumbnail_url}
                    alt=""
                    width={40}
                    height={40}
                  />
                )}
                {selectedTrainMap.mesh_preview_url && (
                  <div className="fleet-mesh-preview">
                    <img
                      src={selectedTrainMap.mesh_preview_url}
                      alt={`Mesh preview ${selectedTrainMap.id}`}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
          {form.map_id === "none" && (
            <p className="meta run-stop-help">
              Frontier reward needs a validated centerline map. The builtin map has
              no positive driving reward under this objective.
            </p>
          )}

          <div className="section-title">Train / job</div>
          <label className="field">
            Environments (simulators scale automatically)
            <input
              type="number"
              min={1}
              max={16}
              value={form.n_envs}
              onChange={(e) => num("n_envs", e.target.value)}
            />
          </label>
          <label className="field">
            Maximum training timesteps (0 = no limit)
            <input
              type="number"
              min={0}
              value={form.timesteps}
              onChange={(e) => num("timesteps", e.target.value)}
            />
          </label>
          <p className="meta run-stop-help">
            With no timestep limit, training stops after the measured improvement rate
            plateaus. Route maps use frontier meters per environment step; builtin maps
            use reward per environment step.
          </p>
          <label className="field">
            run_name (→ out stamp)
            <input
              type="text"
              value={form.run_name ?? ""}
              onChange={(e) => str("run_name", e.target.value)}
              placeholder="ppo"
            />
          </label>
          <label className="field">
            out (optional path)
            <input
              type="text"
              value={form.out ?? ""}
              onChange={(e) => str("out", e.target.value)}
              placeholder="logs/rl/…"
            />
          </label>
          <label className="field">
            seed
            <input
              type="number"
              value={form.seed}
              onChange={(e) => num("seed", e.target.value)}
            />
          </label>
          <label className="field">
            device
            <select
              value={form.device}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  device: e.target.value as Settings["device"],
                }))
              }
            >
              <option value="auto">auto</option>
              <option value="cpu">cpu</option>
              <option value="cuda">cuda</option>
            </select>
          </label>
          <label className="field">
            resume (.zip path)
            <input
              type="text"
              value={form.resume ?? ""}
              onChange={(e) => str("resume", e.target.value)}
            />
          </label>

          <div className="section-title">Env kwargs</div>
          <label className="field check">
            <input
              type="checkbox"
              checked={form.headless}
              onChange={(e) =>
                setForm((f) => ({ ...f, headless: e.target.checked }))
              }
            />
            headless
          </label>
          <label className="field check">
            <input
              type="checkbox"
              checked={form.auto_launch}
              disabled={form.docker_mode}
              onChange={(e) =>
                setForm((f) => ({ ...f, auto_launch: e.target.checked }))
              }
            />
            auto_launch {form.docker_mode ? "(forced off by docker_mode)" : ""}
          </label>
          <label className="field">
            connect_timeout
            <input
              type="number"
              step="any"
              value={form.connect_timeout}
              onChange={(e) => num("connect_timeout", e.target.value)}
            />
          </label>
          <label className="field">
            frame_skip
            <input
              type="number"
              min={1}
              value={form.frame_skip}
              onChange={(e) => num("frame_skip", e.target.value)}
            />
          </label>
          <label className="field">
            Per-car episode step cap (0 disables)
            <input
              type="number"
              min={0}
              value={form.max_episode_steps}
              onChange={(e) => num("max_episode_steps", e.target.value)}
            />
          </label>
          <label className="field">
            Idle speed threshold (builtin map)
            <input
              type="number"
              step="any"
              value={form.stagnation_speed_threshold}
              onChange={(e) => num("stagnation_speed_threshold", e.target.value)}
            />
          </label>
          <label className="field">
            Idle episode timeout in steps (builtin map)
            <input
              type="number"
              value={form.stagnation_steps}
              onChange={(e) => num("stagnation_steps", e.target.value)}
            />
          </label>
          <label className="field">
            Laps to complete a car’s life (0 disables; success after target)
            <input
              type="number"
              min={0}
              step={1}
              disabled={form.map_id === "none"}
              value={form.laps_per_episode}
              onChange={(e) => num("laps_per_episode", e.target.value)}
            />
          </label>
          <div className="section-title">Run stopping (0 disables optional limits)</div>
          <label className="field">
            Maximum training duration (seconds)
            <input
              type="number"
              min={0}
              step="any"
              value={form.max_duration_seconds}
              onChange={(e) => num("max_duration_seconds", e.target.value)}
            />
          </label>
          <label className="field">
            Stop the entire training run at this cumulative lap count (optional)
            <input
              type="number"
              min={0}
              step={1}
              value={form.stop_after_laps}
              onChange={(e) => num("stop_after_laps", e.target.value)}
            />
          </label>
          <label className="field">
            Minimum training steps before plateau can stop
            <input
              type="number"
              min={0}
              value={form.plateau_min_timesteps}
              onChange={(e) => num("plateau_min_timesteps", e.target.value)}
            />
          </label>
          <label className="field">
            Steps per improvement measurement window
            <input
              type="number"
              min={1}
              value={form.plateau_window_timesteps}
              onChange={(e) => num("plateau_window_timesteps", e.target.value)}
            />
          </label>
          <label className="field">
            Consecutive windows without improvement before stopping
            <input
              type="number"
              min={1}
              value={form.plateau_patience}
              onChange={(e) => num("plateau_patience", e.target.value)}
            />
          </label>
          <label className="field">
            Minimum improvement to reset patience (%)
            <input
              type="number"
              min={0}
              step="any"
              value={form.plateau_min_improvement_pct}
              onChange={(e) => num("plateau_min_improvement_pct", e.target.value)}
            />
          </label>
          <label className="field">
            Successful laps required before plateau stopping can end the run
            <input
              type="number"
              min={0}
              step={1}
              value={form.plateau_min_successful_laps}
              onChange={(e) => num("plateau_min_successful_laps", e.target.value)}
            />
          </label>
          <p className="meta run-stop-help">
            Plateau stopping compares the best measured progress rate against each new
            window and waits for this many completed laps on closed centerline maps.
            Collision and frontier stagnation reset only the affected car’s episode.
          </p>
          <div className="section-title">Learning warm-up</div>
          <label className="field">
            Optional centerline-teacher warm-up steps per car (0 disables)
            <input
              type="number"
              min={0}
              step={1000}
              value={form.expert_pretrain_steps}
              disabled={form.map_id === "none"}
              onChange={(e) => num("expert_pretrain_steps", e.target.value)}
            />
          </label>
          <label className="field">
            Clean one-lap episodes before training the full lap target
            <input
              type="number"
              min={0}
              step={1}
              value={form.curriculum_single_lap_successes}
              disabled={form.map_id === "none" || form.laps_per_episode <= 1}
              onChange={(e) => num("curriculum_single_lap_successes", e.target.value)}
            />
          </label>
          <p className="meta run-stop-help">
            Teacher warm-up is disabled by default. When enabled, the map-aware
            driver supplies labels only; the learned policy still receives LiDAR
            and vehicle-state inputs. Closed tracks begin with one-lap episodes and
            advance to the configured target after repeated clean finishes.
          </p>
          <label className="field">
            End episode after no frontier progress (centerline maps; 0 disables)
            <input
              type="number"
              min={0}
              step="any"
              value={form.frontier_stagnation_seconds}
              onChange={(e) => num("frontier_stagnation_seconds", e.target.value)}
            />
          </label>
          <label className="field check">
            <input
              type="checkbox"
              checked={form.terminate_on_collision}
              onChange={(e) =>
                setForm((f) => ({ ...f, terminate_on_collision: e.target.checked }))
              }
            />
            End that car’s episode on collision
          </label>
          <label className="field">
            Reward per meter of new frontier progress
            <input
              type="number"
              min={0}
              step="any"
              value={form.route_progress_scale}
              onChange={(e) => num("route_progress_scale", e.target.value)}
            />
          </label>
          <label className="field">
            Time cost per simulated second
            <input
              type="number"
              min={0}
              step="any"
              value={form.time_penalty_per_second}
              onChange={(e) => num("time_penalty_per_second", e.target.value)}
            />
          </label>
          <label className="field">
            Secondary body-relative reverse penalty per meter
            <input
              type="number"
              min={0}
              step="any"
              value={form.backward_speed_penalty_scale}
              onChange={(e) => num("backward_speed_penalty_scale", e.target.value)}
            />
          </label>
          <label className="field">
            Clean target-lap bonus (scale × average frontier speed across all target laps)
            <input
              type="number"
              min={0}
              step="any"
              value={form.lap_time_reward_scale}
              onChange={(e) => num("lap_time_reward_scale", e.target.value)}
            />
          </label>
          <label className="field">
            Collision penalty (charged once when a collision is detected)
            <input
              type="number"
              step="any"
              value={form.collision_penalty}
              onChange={(e) => num("collision_penalty", e.target.value)}
            />
          </label>
          <label className="field">
            Failed episode penalty (stall or timeout)
            <input
              type="number"
              step="any"
              value={form.episode_failure_penalty}
              onChange={(e) => num("episode_failure_penalty", e.target.value)}
            />
          </label>
          <p className="meta run-stop-help">
            Only new frontier advances earn progress reward; retracing ground and raw
            speed earn nothing. Simulated time has a running cost, and reversing has
            an additional body-relative cost. A crash receives the collision penalty;
            stalls and timeouts receive the failed-episode penalty. Clean completion
            still earns the configured target-lap pace bonus.
          </p>
          <label className="field">
            slip_penalty
            <input
              type="number"
              step="any"
              value={form.slip_penalty}
              onChange={(e) => num("slip_penalty", e.target.value)}
            />
          </label>
          <label className="field">
            steer_jerk_penalty
            <input
              type="number"
              step="any"
              value={form.steer_jerk_penalty}
              onChange={(e) => num("steer_jerk_penalty", e.target.value)}
            />
          </label>

          <div className="section-title">Hub / telemetry</div>
          <label className="field">
            telemetry_every_n
            <input
              type="number"
              min={1}
              value={form.telemetry_every_n}
              onChange={(e) => num("telemetry_every_n", e.target.value)}
            />
          </label>
          <label className="field">
            fleet_hz
            <input
              type="number"
              step="any"
              value={form.fleet_hz}
              onChange={(e) => num("fleet_hz", e.target.value)}
            />
          </label>
          <label className="field">
            lidar_display_beams
            <input
              type="number"
              min={1}
              value={form.lidar_display_beams}
              onChange={(e) => num("lidar_display_beams", e.target.value)}
            />
          </label>
          <label className="field">
            telemetry_lidar_max_envs
            <input
              type="number"
              min={0}
              value={form.telemetry_lidar_max_envs}
              onChange={(e) => num("telemetry_lidar_max_envs", e.target.value)}
            />
          </label>
          <label className="field check">
            <input
              type="checkbox"
              checked={form.docker_mode}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  docker_mode: e.target.checked,
                  auto_launch: e.target.checked ? false : f.auto_launch,
                }))
              }
            />
            docker_mode (force --no-auto-launch)
          </label>
          <label className="field check">
            <input
              type="checkbox"
              checked={form.stop_sims_on_train_exit}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  stop_sims_on_train_exit: e.target.checked,
                }))
              }
            />
            stop_sims_on_train_exit
          </label>
          <label className="field check">
            <input
              type="checkbox"
              checked={form.stop_stack_on_train_exit}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  stop_stack_on_train_exit: e.target.checked,
                }))
              }
            />
            stop_stack_on_train_exit (also stops Mission Control)
          </label>
        </div>

        <div className="actions">
          <button type="submit" className="primary">
            Save
          </button>
          <button type="button" onClick={applyDockerPreset}>
            Docker preset
          </button>
        </div>
      </form>
      {msg && <p className="msg ok">{msg}</p>}
      {err && <p className="msg err">{err}</p>}
    </section>
  );
});
