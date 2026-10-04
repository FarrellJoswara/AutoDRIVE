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

type NumKey = {
  [K in keyof Settings]-?:
    Settings[K] extends number ? K : never;
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
  // All defaults come from the backend settings model.
  const [form, setForm] = useState<Settings>({} as Settings);
  const [mapChoices, setMapChoices] = useState<MapCatalogEntry[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [busy, setBusy] = useState(false);

  const state = status?.state ?? "idle";
  const running =
    state === "running" || state === "starting" || state === "stopping";
  const selectedTrainMap =
    mapChoices.find((m) => m.id === form.map_id) ?? null;
  // Older hubs don't send ppo_n_steps yet; this fallback matches Layer 3's
  // current PPO default until the backend is safely restarted.
  const ppoStepsPerEnv = form.ppo_n_steps ?? 1024;
  const rolloutSize = Math.max(1, form.n_envs * ppoStepsPerEnv);
  const firstEvaluationStep = Math.ceil(form.evaluation_every_timesteps / rolloutSize) * rolloutSize;
  const secondEvaluationStep = Math.ceil((form.evaluation_every_timesteps * 2) / rolloutSize) * rolloutSize;

  useEffect(() => {
    let cancelled = false;
    Promise.all([getSettings(), fetchMaps().catch(() => [] as MapCatalogEntry[])])
      .then(([s, maps]) => {
        if (cancelled) return;
        const choices = runnableMaps(maps);
        setMapChoices(choices);
        const merged = { ...s };
        if (!choices.some((m) => m.id === merged.map_id)) {
          merged.map_id = "none";
        }
        setForm(merged);
        setSettingsLoaded(true);
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
    if (!loading) onReadyChange?.(settingsLoaded);
  }, [loading, settingsLoaded, onReadyChange]);

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
      setForm(res.settings);
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
      const saved = res.settings;
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
      const result = await stopTrain();
      if (result.stopped_containers.length > 0) {
        setMsg(`Stopped simulators: ${result.stopped_containers.join(", ")}`);
      } else if (!running) {
        setMsg("No running training job or simulator containers to stop.");
      }
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  }

  useImperativeHandle(ref, () => ({ start: onStart, stop: onStop }), [form, onBusyChange, busy, running]);

  if (loading || !settingsLoaded) {
    return (
      <section className="panel">
        <h2>Train</h2>
        <p className={err ? "error" : "lede"}>{err ?? "Loading settings…"}</p>
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
            Successful laps per car episode (0 disables)
            <input
              type="number"
              min={0}
              value={form.laps_per_episode}
              onChange={(e) => num("laps_per_episode", e.target.value)}
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
            Minimum training steps before plateau can stop
            <input
              type="number"
              min={0}
              value={form.plateau_min_timesteps}
              onChange={(e) => num("plateau_min_timesteps", e.target.value)}
            />
          </label>
          <label className="field">
            Minimum training steps between evaluations
            <input
              type="number"
              min={1}
              value={form.evaluation_every_timesteps}
              onChange={(e) => num("evaluation_every_timesteps", e.target.value)}
            />
            <small className="meta">
              With {form.n_envs} environments × {ppoStepsPerEnv.toLocaleString()} PPO steps per rollout, snapshots align to rollout boundaries. The first two nominal snapshot points are {firstEvaluationStep.toLocaleString()} and {secondEvaluationStep.toLocaleString()} steps; a still-running evaluation can delay the next one.
            </small>
          </label>
          <label className="field">
            Evaluation score for checkpoint selection and plateau stopping
            <select
              value={form.evaluation_metric}
              onChange={(e) => setForm((f) => ({
                ...f,
                evaluation_metric: e.target.value as Settings["evaluation_metric"],
              }))}
            >
              <option value="frontier_speed">Frontier pace</option>
              <option value="reward_per_simulated_second">Reward per simulated second</option>
              <option value="total_reward">Total reward per attempt</option>
            </select>
          </label>
          <label className="field">
            Consecutive evaluations without improvement before stopping
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
          <p className="meta run-stop-help">
            A separate simulator evaluates a fixed policy snapshot in parallel until
            it crashes, stalls, or completes 10 laps. Total attempt reward includes
            frontier progress, time and reverse costs, and any terminal failure cost.
            The selected score chooses the best checkpoint and drives exploration
            and plateau stopping. Evaluations do not overlap.
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
            Fixed collision cost (positive magnitude)
            <input
              type="number"
              min={0}
              step="any"
              value={form.collision_penalty_magnitude}
              onChange={(e) => num("collision_penalty_magnitude", e.target.value)}
            />
          </label>
          <label className="field">
            Positive frontier reward clawed back on collision (%)
            <input
              type="number"
              min={0}
              max={100}
              step="any"
              value={form.collision_reward_percent}
              onChange={(e) => num("collision_reward_percent", e.target.value)}
            />
          </label>
          <label className="field">
            Failed episode cost (positive magnitude; stall or timeout)
            <input
              type="number"
              min={0}
              step="any"
              value={form.episode_failure_penalty_magnitude}
              onChange={(e) => num("episode_failure_penalty_magnitude", e.target.value)}
            />
          </label>
          <label className="field">
            Positive frontier reward clawed back on stall/timeout (%)
            <input
              type="number"
              min={0}
              max={100}
              step="any"
              value={form.episode_failure_reward_percent}
              onChange={(e) => num("episode_failure_reward_percent", e.target.value)}
            />
          </label>
          <p className="meta run-stop-help">
            Only new frontier advances earn progress reward; retracing ground and raw
            speed earn nothing. Simulated time has a running cost, and reversing has
            an additional body-relative cost. Crashes and other failed episodes claw
            back the configured share of positive frontier reward earned during that
            car’s life, then pay their fixed failure cost. Ten-lap completion ends an
            episode successfully without a failure deduction.
          </p>
          <label className="field">
            Minimum action standard deviation
            <input type="number" min={0.01} step="any" value={form.exploration_std_min}
              onChange={(e) => num("exploration_std_min", e.target.value)} />
          </label>
          <label className="field">
            Maximum action standard deviation
            <input type="number" min={0.01} step="any" value={form.exploration_std_max}
              onChange={(e) => num("exploration_std_max", e.target.value)} />
          </label>
          <label className="field">
            Exploration scale after an evaluation improvement
            <input type="number" min={0.01} max={1} step="any" value={form.exploration_improvement_scale}
              onChange={(e) => num("exploration_improvement_scale", e.target.value)} />
          </label>
          <label className="field">
            Exploration scale after repeated plateau evaluations
            <input type="number" min={1} step="any" value={form.exploration_plateau_scale}
              onChange={(e) => num("exploration_plateau_scale", e.target.value)} />
          </label>
          <p className="meta run-stop-help">
            Evaluation improvement gently reduces policy randomness. Repeated flat
            evaluations raise it modestly, always within the configured bounds.
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
