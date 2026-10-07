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

function RewardTip({ text }: { text: string }) {
  return (
    <span className="reward-tip" role="img" tabIndex={0} aria-label={text} data-tooltip={text}>
      ?
    </span>
  );
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
  const rewardBase = Math.max(0, Number(form.route_progress_scale) || 0);
  const paceTarget = Math.max(0.1, Number(form.frontier_pace_target_mps) || 6);
  const rewardAtPace = (pace: number) =>
    rewardBase * (1 + Math.min(1, Math.max(0, pace) / paceTarget) ** 2);
  const rewardNumber = (value: number) =>
    Number(value.toFixed(2)).toLocaleString(undefined, { maximumFractionDigits: 2 });

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
            PPO learning rate
            <input
              type="number"
              min={0.000001}
              max={0.01}
              step="any"
              value={form.ppo_learning_rate}
              onChange={(e) => num("ppo_learning_rate", e.target.value)}
            />
            <small className="meta">Controls how far each PPO update moves the policy. Smaller values make resumed training gentler.</small>
          </label>
          <label className="field">
            PPO epochs per rollout
            <input
              type="number"
              min={1}
              max={20}
              value={form.ppo_n_epochs}
              onChange={(e) => num("ppo_n_epochs", e.target.value)}
            />
            <small className="meta">Number of optimization passes over each collected rollout. Fewer passes reduce policy movement per update.</small>
          </label>
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
          <div className="section-title">Simulator</div>
          <label className="field">
            Policy observation inputs
            <select
              value={form.observation_profile}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  observation_profile: e.target.value as Settings["observation_profile"],
                }))
              }
            >
              <option value="simulator">Full simulator telemetry</option>
              <option value="official_sensors">Official allowed sensors</option>
            </select>
            <small className="meta">Official mode uses LiDAR, encoder-derived forward speed, IMU, actuator feedback, and prior actions; it excludes simulator-only lateral speed and pose.</small>
          </label>
          <label className="field">
            Simulator mode
            <select
              value={form.simulator_mode}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  simulator_mode: e.target.value as Settings["simulator_mode"],
                }))
              }
            >
              <option value="legacy">Legacy simulator</option>
              <option value="fixed_camera_on">Fixed-duration simulator · camera on</option>
              <option value="fixed_camera_off">Optimized simulator · camera off</option>
            </select>
          </label>
          <p className="meta run-stop-help">
            Optimized mode uses the validated fixed-duration player and removes unused
            camera readback. It keeps the full physics timestep and LiDAR scan. The
            default interval is 0.086 simulated seconds per action.
          </p>
          {form.simulator_mode !== "legacy" && (
            <label className="field">
              Simulated seconds per action
              <input
                type="number"
                min={0.001}
                step={0.001}
                value={form.action_interval_s ?? 0.086}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  setForm((f) => ({
                    ...f,
                    action_interval_s: Number.isFinite(n) && n > 0 ? n : f.action_interval_s,
                  }));
                }}
              />
            </label>
          )}
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
            Attempts per evaluation snapshot
            <input
              type="number"
              min={1}
              max={10}
              value={form.evaluation_runs_per_snapshot}
              onChange={(e) => num("evaluation_runs_per_snapshot", e.target.value)}
            />
            <small className="meta">
              Each attempt repeats the configured map spawn and ends on collision, stall, or 10 laps. The median selected score compares snapshots; mean and best are shown for context. Repeats measure simulator consistency, not generalization to other spawn points. More attempts make each evaluation take longer.
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
            A separate simulator evaluates each fixed policy snapshot in parallel for
            the configured number of attempts; each attempt ends on collision, stall,
            or 10 laps. The median score compares snapshots. Total attempt reward includes
            frontier progress, time and reverse costs. Stalling adds no separate
            terminal cost; collisions and other failures retain their configured costs.
            The selected score chooses the best checkpoint and drives exploration
            and plateau stopping. Evaluations do not overlap.
          </p>
          <label className="field">
            Reset after no frontier progress (centerline maps; 0 disables)
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
          <h3 className="reward-heading">Reward model</h3>
          <section className="reward-model">
            <div className="reward-equations" aria-label="Reward equations">
              <div className="reward-equation" role="math" aria-label="Frontier reward equals new frontier distance times base reward times one plus the square of average frontier speed divided by target speed, capped at one">
                <span className="math-var">r<sub>frontier</sub></span><span>=</span>
                <span>Δd · b · (1 + min(</span>
                <span className="math-fraction"><span>v̄<sub>f</sub></span><span>v<sub>target</sub></span></span>
                <span>, 1)<sup>2</sup>)</span>
              </div>
              <div className="reward-equation reward-equation-secondary" role="math" aria-label="Step reward equals frontier reward minus time, reverse, slip, steering, and terminal costs when a terminal event occurs">
                <span className="math-var">r<sub>step</sub></span><span>=</span>
                <span className="math-var">r<sub>frontier</sub></span><span>−</span>
                <span className="math-var">c<sub>t</sub></span><span>Δt</span><span>−</span>
                <span className="math-var">c<sub>rev</sub></span><span>d<sub>rev</sub></span><span>−</span>
                <span className="math-var">c<sub>slip</sub></span><span>|β|</span><span>−</span>
                <span className="math-var">c<sub>steer</sub></span><span>|Δδ|</span><span>−</span>
                <span className="math-var">I<sub>terminal</sub></span><span>·</span>
                <span className="math-var">C<sub>terminal</sub></span>
              </div>
              <div className="reward-equation reward-equation-secondary" role="math" aria-label="Crash cost equals fixed crash cost plus crash clawback fraction times accumulated positive frontier reward. Frontier stalls have no additional terminal cost; other failed endings use the configured failure cost.">
                <span className="math-var">C<sub>crash</sub></span><span>=</span>
                <span className="math-var">c<sub>crash</sub></span><span>+</span>
                <span className="math-fraction"><span>p<sub>crash</sub></span><span>100</span></span><span>R<sup>+</sup></span>
                <span className="math-spacer" />
                <span className="math-var">C<sub>stall</sub></span><span>=</span><span>0</span>
                <span className="math-spacer" />
                <span className="math-var">C<sub>other</sub></span><span>=</span>
                <span className="math-var">c<sub>other</sub></span><span>+</span>
                <span className="math-fraction"><span>p<sub>other</sub></span><span>100</span></span><span>R<sup>+</sup></span>
              </div>
              <div className="reward-notation">
                <span><strong>Δd</strong> new frontier metres</span>
                <span><strong>v̄<sub>f</sub></strong> episode-average frontier speed</span>
                <span><strong>R<sup>+</sup></strong> accumulated positive frontier reward</span>
              </div>
            </div>
            <div className="reward-examples" aria-label="Live reward examples">
              <span><small>0 m/s</small><strong>{rewardNumber(rewardAtPace(0))} / m</strong></span>
              <span><small>{rewardNumber(paceTarget / 2)} m/s</small><strong>{rewardNumber(rewardAtPace(paceTarget / 2))} / m</strong></span>
              <span><small>{rewardNumber(paceTarget)} m/s+</small><strong>{rewardNumber(rewardAtPace(paceTarget))} / m</strong></span>
            </div>
            <details className="reward-controls">
              <summary>Edit reward parameters</summary>
              <div className="reward-controls-grid">
                <label className="field">
                  <span className="reward-field-name"><i>b</i> · Base reward / frontier metre
                    <RewardTip text="Positive reward for each new metre the monotonic frontier advances. This is multiplied by the pace factor shown above." />
                  </span>
                  <input type="number" min={0} step="any" value={form.route_progress_scale}
                    onChange={(e) => num("route_progress_scale", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>v<sub>target</sub></i> · Pace target (m/s)
                    <RewardTip text="Episode-average frontier speed where the progress multiplier reaches its 2× cap. Faster paces do not increase the multiplier further." />
                  </span>
                  <input type="number" min={0.1} step="any" value={form.frontier_pace_target_mps}
                    onChange={(e) => num("frontier_pace_target_mps", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>c<sub>t</sub></i> · Time cost / second
                    <RewardTip text="Reward subtracted per simulated second, including while the car is moving." />
                  </span>
                  <input type="number" min={0} step="any" value={form.time_penalty_per_second}
                    onChange={(e) => num("time_penalty_per_second", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>c<sub>rev</sub></i> · Reverse cost / metre
                    <RewardTip text="Additional penalty per metre travelled backward relative to the car body. The frontier reward is also withheld for a new push while body velocity is backward." />
                  </span>
                  <input type="number" min={0} step="any" value={form.backward_speed_penalty_scale}
                    onChange={(e) => num("backward_speed_penalty_scale", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>c<sub>crash</sub></i> · Fixed crash cost
                    <RewardTip text="A fixed terminal penalty applied once when a collision ends the car's episode." />
                  </span>
                  <input type="number" min={0} step="any" value={form.collision_penalty_magnitude}
                    onChange={(e) => num("collision_penalty_magnitude", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>p<sub>crash</sub></i> · Crash clawback (%)
                    <RewardTip text="Percentage of the positive frontier reward accumulated in that car's life, subtracted once on collision." />
                  </span>
                  <input type="number" min={0} max={100} step="any" value={form.collision_reward_percent}
                    onChange={(e) => num("collision_reward_percent", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>c<sub>other</sub></i> · Other failure cost
                    <RewardTip text="Fixed cost for non-collision failed endings such as an idle timeout or step cap. Frontier-stall resets do not charge this cost; their elapsed time still incurs the time cost." />
                  </span>
                  <input type="number" min={0} step="any" value={form.episode_failure_penalty_magnitude}
                    onChange={(e) => num("episode_failure_penalty_magnitude", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>p<sub>other</sub></i> · Other failure clawback (%)
                    <RewardTip text="Percentage of accumulated positive frontier reward subtracted on other non-collision failed endings. Frontier-stall resets do not apply this clawback." />
                  </span>
                  <input type="number" min={0} max={100} step="any" value={form.episode_failure_reward_percent}
                    onChange={(e) => num("episode_failure_reward_percent", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>c<sub>slip</sub></i> · Slip cost
                    <RewardTip text="Penalty coefficient multiplied by the absolute slip angle in radians. Set to zero to disable." />
                  </span>
                  <input type="number" min={0} step="any" value={form.slip_penalty}
                    onChange={(e) => num("slip_penalty", e.target.value)} />
                </label>
                <label className="field">
                  <span className="reward-field-name"><i>c<sub>steer</sub></i> · Steering-change cost
                    <RewardTip text="Penalty coefficient multiplied by the absolute change in steering command between steps. Set to zero to disable." />
                  </span>
                  <input type="number" min={0} step="any" value={form.steer_jerk_penalty}
                    onChange={(e) => num("steer_jerk_penalty", e.target.value)} />
                </label>
              </div>
            </details>
          </section>
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
