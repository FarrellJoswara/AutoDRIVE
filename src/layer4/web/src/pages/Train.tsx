import { FormEvent, forwardRef, useEffect, useImperativeHandle, useMemo, useState } from "react";
import {
  createOfficialRun,
  getBestOfficialEvaluation,
  getOfficialRuns,
  getOfficialRunLiveTelemetry,
  getOfficialSettings,
  stopOfficialRun,
  type OfficialRunList,
  type OfficialChampion,
  type OfficialTrainSettings,
  type RunSummary,
  type TrainingPhaseTelemetry,
} from "../api";
import { setOfficialRunSelection } from "../store";

type NumericSetting = {
  [K in keyof OfficialTrainSettings]-?: OfficialTrainSettings[K] extends number ? K : never;
}[keyof OfficialTrainSettings];

export interface TrainPageHandle {
  start(): Promise<void>;
  stop(): Promise<void>;
}

interface TrainPageProps {
  onBusyChange?: (busy: boolean) => void;
  onReadyChange?: (ready: boolean) => void;
}

type RunFilter = "active" | "all" | "finished";

const ACTIVE_STATES = new Set(["starting", "running", "stopping"]);
const HISTORY_PAGE_SIZE = 8;

function isActive(run: RunSummary): boolean {
  return ACTIVE_STATES.has(run.state);
}

function runProgress(run: RunSummary): { step: number; total: number; percent: number } | null {
  if (run.kind === "evaluation") return null;
  const step = run.latest_metrics?.step ?? run.latest_telemetry?.step ?? run.latest_fleet?.step;
  const total = run.config?.total_timesteps;
  if (typeof step !== "number" || typeof total !== "number" || total <= 0) return null;
  return { step, total, percent: Math.min(100, Math.max(0, step / total * 100)) };
}

function Help({ text }: { text: string }) {
  return <span className="train-help" role="img" aria-label={text} title={text} style={{
    display: "inline-grid", placeItems: "center", width: 17, height: 17, marginLeft: 5,
    border: "1px solid currentColor", borderRadius: "50%", fontSize: 11, lineHeight: 1,
    cursor: "help", opacity: 0.78, verticalAlign: "middle",
  }}>i</span>;
}

function Field({ label, help, children }: { label: string; help: string; children: React.ReactNode }) {
  return <label className="field">
    <span>{label}<Help text={help} /></span>
    {children}
  </label>;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat().format(value);
}

export const TrainPage = forwardRef<TrainPageHandle, TrainPageProps>(function TrainPage(
  { onBusyChange, onReadyChange },
  ref,
) {
  const [form, setForm] = useState<OfficialTrainSettings | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [maxRuns, setMaxRuns] = useState(2);
  const [displayName, setDisplayName] = useState("");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<RunFilter>("all");
  const [showAllHistory, setShowAllHistory] = useState(false);
  const [bestOfficialModel, setBestOfficialModel] = useState<OfficialChampion | null>(null);
  const [selectedPhase, setSelectedPhase] = useState<TrainingPhaseTelemetry | null>(null);

  const activeRuns = useMemo(() => runs.filter(isActive), [runs]);
  const selectedRun = runs.find((run) => run.run_id === selectedRunId) ?? null;
  const canStart = form != null && activeRuns.length < maxRuns && !busy;

  useEffect(() => {
    setOfficialRunSelection(selectedRun);
  }, [selectedRunId, selectedRun?.state, selectedRun?.pid, selectedRun?.exit_code]);

  async function refreshRuns() {
    const result: OfficialRunList = await getOfficialRuns();
    setRuns(result.runs);
    setMaxRuns(result.max_concurrent_runs);
    setSelectedRunId((current) => {
      if (current && result.runs.some((run) => run.run_id === current)) return current;
      return result.runs.find(isActive)?.run_id ?? result.runs[0]?.run_id ?? null;
    });
  }

  useEffect(() => {
    let cancelled = false;
    Promise.all([getOfficialSettings(), getOfficialRuns()])
      .then(([settingsResult, runsResult]) => {
        if (cancelled) return;
        const configuredSettings = settingsResult.settings;
        setForm({
          ...configuredSettings,
          observation_profile: "official_sensors_camera",
          timeout_s: configuredSettings.simulator_backend === "windows_external"
            ? Math.max(180, configuredSettings.timeout_s)
            : configuredSettings.timeout_s,
        });
        setRuns(runsResult.runs);
        setMaxRuns(runsResult.max_concurrent_runs);
        setSelectedRunId(runsResult.runs.find(isActive)?.run_id ?? runsResult.runs[0]?.run_id ?? null);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setSelectedPhase(null);
    if (!selectedRunId) return () => { cancelled = true; };
    const refreshPhase = () => {
      void getOfficialRunLiveTelemetry(selectedRunId)
        .then((telemetry) => { if (!cancelled) setSelectedPhase(telemetry.training_phase ?? null); })
        .catch(() => undefined);
    };
    refreshPhase();
    const timer = window.setInterval(refreshPhase, 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [selectedRunId]);

  useEffect(() => {
    onReadyChange?.(!loading && form != null);
  }, [loading, form, onReadyChange]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void refreshRuns().catch(() => undefined);
      void getBestOfficialEvaluation().then((result) => setBestOfficialModel(result.best)).catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, []);

  function setNumber(key: NumericSetting, value: string) {
    const next = Number(value);
    if (!Number.isFinite(next)) return;
    setForm((current) => current ? { ...current, [key]: next } : current);
  }

  function setText(key: "resume", value: string) {
    setForm((current) => current ? { ...current, [key]: value.trim() === "" ? null : value } : current);
  }

  function setChoice<K extends "simulator_backend" | "simulator_display_mode" | "device" | "observation_profile" | "throttle_mode" | "negative_throttle_mode" | "steering_mode">(key: K, value: OfficialTrainSettings[K]) {
    setForm((current) => {
      if (!current) return current;
      const next = { ...current, [key]: value };
      if (key === "simulator_backend" && value === "windows_external") {
        next.timeout_s = Math.max(180, next.timeout_s);
      }
      return next;
    });
  }

  async function startRun(event?: FormEvent) {
    event?.preventDefault();
    if (!form || !canStart) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    onBusyChange?.(true);
    try {
      const created = await createOfficialRun({ ...form }, displayName);
      setSelectedRunId(created.run_id);
      setMessage(`Official run ${created.run_id} created with a settings snapshot.`);
      await refreshRuns();
      setSelectedRunId(created.run_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  }

  async function stopRun(runId = selectedRun?.run_id) {
    if (!runId) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    onBusyChange?.(true);
    try {
      await stopOfficialRun(runId);
      setMessage(`Stop requested for ${runId}.`);
      await refreshRuns();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  }

  useImperativeHandle(ref, () => ({ start: () => startRun(), stop: () => stopRun() }), [form, canStart, selectedRun]);

  const visibleRuns = useMemo(() => {
    const query = search.trim().toLowerCase();
    const filtered = runs.filter((run) => {
      const active = isActive(run);
      if (filter === "active" && !active) return false;
      if (filter === "finished" && active) return false;
      if (!query) return true;
      return [run.display_name, run.run_id, run.state, run.stop_reason]
        .some((value) => value?.toLowerCase().includes(query));
    });
    return filtered.sort((a, b) => Number(isActive(b)) - Number(isActive(a)));
  }, [runs, search, filter]);
  const activeVisible = visibleRuns.filter(isActive);
  const historyVisibleAll = visibleRuns.filter((run) => !isActive(run));
  const historyVisible = showAllHistory ? historyVisibleAll : historyVisibleAll.slice(0, HISTORY_PAGE_SIZE);

  if (loading) return <section className="panel"><h2>Official training</h2><p className="lede">Loading official training settings and runs…</p></section>;
  if (!form) return <section className="panel"><h2>Official training</h2><p className="error" role="alert">Could not load official settings: {error ?? "No settings returned."}</p></section>;

  return (
    <section className="panel official-train-panel">
      <header style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
        <h2 style={{ margin: 0 }}>Official training</h2>
        <span className="official-badge">AutoDRIVE</span>
        <span className="meta" aria-live="polite">{activeRuns.length}/{maxRuns} active · {Math.max(0, maxRuns - activeRuns.length)} slots available</span>
      </header>
      <p className="lede">Each run trains one model across its official simulator environments. Select a run to monitor it in Watch.</p>
      <section className="panel" aria-label="Best official model" style={{ marginTop: 12 }}>
        <h3 style={{ marginTop: 0 }}>Best officially evaluated model</h3>
        {bestOfficialModel ? <p>{bestOfficialModel.model_path} · median adjusted race time <strong>{bestOfficialModel.median_adjusted_race_time_s.toFixed(2)} s</strong> across {bestOfficialModel.attempts} valid attempts. Source: {bestOfficialModel.source}.</p>
          : <p className="meta">No checkpoint has completed three valid official evaluations yet.</p>}
      </section>

      <section aria-labelledby="runs-heading" style={{ marginTop: 16 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
          <h3 id="runs-heading" style={{ margin: 0 }}>Official runs <span className="meta">({runs.length})</span></h3>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <label className="field" style={{ margin: 0 }}>
              <span className="sr-only">Search runs</span>
              <input type="search" value={search} onChange={(event) => { setSearch(event.target.value); setShowAllHistory(false); }} placeholder="Search name or run ID" aria-label="Search runs by name or ID" />
            </label>
            <label className="field" style={{ margin: 0 }}>
              <span className="sr-only">Filter runs</span>
              <select value={filter} onChange={(event) => { setFilter(event.target.value as RunFilter); setShowAllHistory(false); }} aria-label="Filter runs">
                <option value="all">All runs</option>
                <option value="active">Active only</option>
                <option value="finished">History only</option>
              </select>
            </label>
          </div>
        </div>

        {activeVisible.length > 0 && <div style={{ marginBottom: 14 }}>
          <h4 style={{ margin: "8px 0" }}>Active now <span className="meta">{activeVisible.length}</span></h4>
          <div style={{ maxHeight: "62vh", overflow: "auto", border: "1px solid var(--border, #30443c)", borderRadius: 6 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 760, textAlign: "left" }}>
              <thead><tr>
                <th scope="col" style={thStyle}>Run</th><th scope="col" style={thStyle}>Status</th>
                <th scope="col" style={thStyle}>Progress</th><th scope="col" style={thStyle}>Started</th>
                <th scope="col" style={thStyle}>Actions</th>
              </tr></thead>
              <tbody>{activeVisible.map((run) => <RunRow key={run.run_id} run={run} selected={run.run_id === selectedRunId} busy={busy} onSelect={() => setSelectedRunId(run.run_id)} onStop={() => void stopRun(run.run_id)} />)}</tbody>
            </table>
          </div>
        </div>}

        {historyVisible.length > 0 && <div>
          <h4 style={{ margin: "8px 0" }}>Recent history <span className="meta">{historyVisibleAll.length}</span></h4>
          <div style={{ overflowX: "auto", border: "1px solid var(--border, #30443c)", borderRadius: 6 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 700, textAlign: "left" }}>
              <thead><tr>
                <th scope="col" style={thStyle}>Run</th><th scope="col" style={thStyle}>Status</th>
                <th scope="col" style={thStyle}>Steps</th><th scope="col" style={thStyle}>Finished</th>
                <th scope="col" style={thStyle}>Actions</th>
              </tr></thead>
              <tbody>{historyVisible.map((run) => <RunRow key={run.run_id} run={run} selected={run.run_id === selectedRunId} busy={busy} onSelect={() => setSelectedRunId(run.run_id)} onStop={() => void stopRun(run.run_id)} history />)}</tbody>
            </table>
          </div>
          {historyVisibleAll.length > HISTORY_PAGE_SIZE && <button type="button" className="btn" style={{ marginTop: 8 }} onClick={() => setShowAllHistory((show) => !show)}>
            {showAllHistory ? "Show recent only" : `Show all ${historyVisibleAll.length} history runs`}
          </button>}
        </div>}
        {visibleRuns.length === 0 && <p className="meta">{runs.length === 0 ? "No runs yet. Create the first run below." : "No runs match this search and filter."}</p>}
      </section>

      <details open className="official-settings-form" style={{ marginTop: 18, borderTop: "1px solid var(--border, #30443c)", paddingTop: 12 }}>
        <summary style={{ cursor: "pointer", fontWeight: 700, padding: "6px 0" }}>Create a training run <span className="meta">· {maxRuns - activeRuns.length} of {maxRuns} slots available</span></summary>
        <p className="meta">Settings are captured as a snapshot when the run starts. Changing this form will not change existing runs.</p>
        <form onSubmit={(event) => void startRun(event)}>
          <div className="grid" style={{ alignItems: "start" }}>
            <Field label="Run name" help="Optional label to identify this experiment in the run list.">
              <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="e.g. lower learning rate" />
            </Field>
            <Field label="Maximum training timesteps" help="Set to 0 for no step cap. If plateau stopping is enabled, the run ends after repeated completed-episode reward windows fail to improve. A positive value is a target: PPO finishes the current rollout, so it can exceed the target by up to one rollout.">
              <input type="number" min={0} value={form.total_timesteps} onChange={(event) => setNumber("total_timesteps", event.target.value)} />
            </Field>
            <Field label="Random seed" help="Seeds PPO and environment RNGs. The official simulator is free-running and does not expose a deterministic physics seed through this workflow.">
              <input type="number" value={form.seed} onChange={(event) => setNumber("seed", event.target.value)} />
            </Field>
            <Field label="Resume checkpoint" help="Optional checkpoint path. Leave blank to initialize a new policy; on resume, requested PPO settings are applied with fresh optimizer state.">
              <input value={form.resume ?? ""} onChange={(event) => setText("resume", event.target.value)} placeholder="Blank starts a new policy" />
            </Field>
          </div>
          {form.resume && <p className="meta">This run will resume the selected checkpoint and skip behavior cloning. Clear the resume field to start a fresh teacher-to-PPO run.</p>}

          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: "pointer", fontWeight: 650, padding: "6px 0" }}>Stopping rules and training objective</summary>
            <div className="grid" style={{ alignItems: "start", marginTop: 10 }}>
              <Field label="Stop when training plateaus" help="Compares the mean return of each completed-episode window. After the minimum steps, the run stops when the configured number of consecutive windows fail to beat the best mean by the improvement threshold. This uses training episode returns, not a separate deterministic evaluator.">
                <select value={form.stop_on_plateau ? "yes" : "no"} onChange={(event) => setForm((current) => current ? { ...current, stop_on_plateau: event.target.value === "yes" } : current)}>
                  <option value="yes">Enabled</option><option value="no">Disabled</option>
                </select>
              </Field>
              <Field label="Minimum steps before plateau checks" help="Prevents early stopping while the policy is still in its initial learning phase.">
                <input type="number" min={0} value={form.plateau_min_timesteps} onChange={(event) => setNumber("plateau_min_timesteps", event.target.value)} />
              </Field>
              <Field label="Episodes per comparison window" help="Each check averages this many completed training episodes. Higher values make plateau decisions less sensitive to one unusually good or bad episode.">
                <input type="number" min={2} max={100} value={form.plateau_window_episodes} onChange={(event) => setNumber("plateau_window_episodes", event.target.value)} />
              </Field>
              <Field label="Consecutive plateau windows" help="The run stops after this many comparison windows fail to improve the best window mean by the configured percentage.">
                <input type="number" min={1} max={100} value={form.plateau_patience} onChange={(event) => setNumber("plateau_patience", event.target.value)} />
              </Field>
              <Field label="Minimum improvement per window (%)" help="A window must beat the best previous window mean by at least this percentage to reset the plateau counter.">
                <input type="number" min={0} max={100} step="any" value={form.plateau_min_improvement_pct} onChange={(event) => setNumber("plateau_min_improvement_pct", event.target.value)} />
              </Field>
              <Field label="Maximum wall-clock duration (hours)" help="Optional overall run limit. Set to 0 to disable; the simulator's per-episode timeout is configured separately.">
                <input type="number" min={0} step="any" value={form.max_duration_hours} onChange={(event) => setNumber("max_duration_hours", event.target.value)} />
              </Field>
              <Field label="Time cost per simulator-clock second" help="Set to zero while learning to complete safe laps. Frontier progress has no speed multiplier, so cautious forward progress earns the same value per metre.">
                <input type="number" min={0} step="any" value={form.time_cost_per_simulated_second} onChange={(event) => setNumber("time_cost_per_simulated_second", event.target.value)} />
              </Field>
              <Field label="End a stalled life after (simulated seconds)" help="When the car makes no new frontier progress for this long, the training environment ends its life and resets it to spawn. This applies only to training; official lap evaluations keep running.">
                <input type="number" min={0} step="any" value={form.frontier_stagnation_s} onChange={(event) => setNumber("frontier_stagnation_s", event.target.value)} />
              </Field>
              <Field label="Reward per completed official lap" help="No separate lap bonus is used. A training life succeeds only when the official lap counter and route frontier both confirm the configured lap target.">
                <input type="number" min={0} max={0} value={0} disabled aria-label="Lap bonus disabled" />
              </Field>
              <Field label="Fixed cost per collision" help="A collision ends the life. The default 200-point cost exceeds the 100-point maximum progress reward, so even a crash after the full target remains negative while farther progress improves the score.">
                <input type="number" min={101} step="any" value={form.collision_penalty_magnitude} onChange={(event) => setNumber("collision_penalty_magnitude", event.target.value)} />
              </Field>
              <Field label="Collision clawback (%)" help="Defaults to zero so frontier progress remains valuable. The fixed collision cost is sized above the maximum progress reward, keeping a crash negative.">
                <input type="number" min={0} max={100} step="any" value={form.collision_reward_percent} onChange={(event) => setNumber("collision_reward_percent", event.target.value)} />
              </Field>
              <Field label="Reverse-motion cost per metre" help="Penalizes distance traveled backward relative to the car body. Reverse throttle used to brake while still moving forward is not penalized.">
                <input type="number" min={0} step="any" value={form.backward_speed_penalty_scale} onChange={(event) => setNumber("backward_speed_penalty_scale", event.target.value)} />
              </Field>
              <Field label="Non-collision failure cost" help="Applied once after a stall or watchdog timeout. The default 200-point cost exceeds the 100-point maximum progress reward, so farther progress improves the score without making failure profitable.">
                <input type="number" min={101} step="any" value={form.failed_episode_penalty} onChange={(event) => setNumber("failed_episode_penalty", event.target.value)} />
              </Field>
              <Field label="Failure clawback (%)" help="Defaults to zero so frontier progress remains valuable. The fixed failure cost is sized above the maximum progress reward.">
                <input type="number" min={0} max={100} step="any" value={form.episode_failure_reward_percent} onChange={(event) => setNumber("episode_failure_reward_percent", event.target.value)} />
              </Field>
            </div>
            <p className="meta">{form.total_timesteps === 0 ? (form.stop_on_plateau || form.max_duration_hours > 0 ? "No timestep limit; plateau and/or wall-clock rules end the run." : "No automatic end configured; stop the run manually.") : `Hard cap: ${formatCount(form.total_timesteps)} steps${form.stop_on_plateau ? " or earlier if the plateau rule triggers" : ""}.`}</p>
          </details>

          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: "pointer", fontWeight: 650, padding: "6px 0" }}>PPO and checkpoint settings</summary>
            <div className="grid" style={{ alignItems: "start", marginTop: 10 }}>
              <Field label="Official route-aware behavior cloning" help="New runs first collect sensor/action examples from a cautious route-following teacher in these same official simulator environments, then initialize PPO's actor from those examples. Resume runs skip this stage.">
                <select value={form.behavior_cloning_enabled ? "yes" : "no"} disabled={Boolean(form.resume)} onChange={(event) => setForm((current) => current ? { ...current, behavior_cloning_enabled: event.target.value === "yes" } : current)}>
                  <option value="yes">Enabled for new runs</option><option value="no">Start PPO directly</option>
                </select>
              </Field>
              <Field label="Teacher steps per environment" help="Number of official sensor/action examples collected from each simulator before PPO starts.">
                <input type="number" min={2} max={100000} value={form.behavior_cloning_steps_per_env} disabled={!form.behavior_cloning_enabled || Boolean(form.resume)} onChange={(event) => setNumber("behavior_cloning_steps_per_env", event.target.value)} />
              </Field>
              <Field label="Teacher target speed (m/s)" help="Maximum speed used by the route-aware pure-pursuit teacher. It slows further for tight corners.">
                <input type="number" min={0.5} max={10} step="any" value={form.behavior_teacher_max_speed_mps} disabled={!form.behavior_cloning_enabled || Boolean(form.resume)} onChange={(event) => setNumber("behavior_teacher_max_speed_mps", event.target.value)} />
              </Field>
              <Field label="Behavior-cloning epochs" help="Training passes over the collected teacher examples before PPO takes over.">
                <input type="number" min={1} max={100} value={form.behavior_cloning_epochs} disabled={!form.behavior_cloning_enabled || Boolean(form.resume)} onChange={(event) => setNumber("behavior_cloning_epochs", event.target.value)} />
              </Field>
              <Field label="Behavior-cloning batch size" help="Number of sensor/action examples used per cloning update.">
                <input type="number" min={8} max={4096} value={form.behavior_cloning_batch_size} disabled={!form.behavior_cloning_enabled || Boolean(form.resume)} onChange={(event) => setNumber("behavior_cloning_batch_size", event.target.value)} />
              </Field>
              <Field label="Behavior-cloning learning rate" help="Optimizer step size used only while fitting the actor to the route teacher.">
                <input type="number" min={0.000001} max={0.1} step="any" value={form.behavior_cloning_learning_rate} disabled={!form.behavior_cloning_enabled || Boolean(form.resume)} onChange={(event) => setNumber("behavior_cloning_learning_rate", event.target.value)} />
              </Field>
              <Field label="Automatic official evaluation" help="Runs a frozen checkpoint through one warm-up lap and ten scored laps on the official simulator. Three fresh simulator attempts are required before a result can promote the shared best model.">
                <select value={form.automatic_evaluation_enabled ? "yes" : "no"} onChange={(event) => setForm((current) => current ? { ...current, automatic_evaluation_enabled: event.target.value === "yes" } : current)}>
                  <option value="yes">Enabled</option><option value="no">Disabled</option>
                </select>
              </Field>
              <Field label="Evaluate every" help="Choose a run-relative transition step interval or completed episode interval. Episode counts are summed across all parallel environments.">
                <select value={form.automatic_evaluation_trigger} disabled={!form.automatic_evaluation_enabled} onChange={(event) => setForm((current) => current ? { ...current, automatic_evaluation_trigger: event.target.value as OfficialTrainSettings["automatic_evaluation_trigger"] } : current)}>
                  <option value="steps">Training steps</option><option value="episodes">Completed episodes</option>
                </select>
              </Field>
              <Field label={`Evaluation interval (${form.automatic_evaluation_trigger})`} help="A trigger saves a frozen policy snapshot and queues three independent official races. If evaluation is active, only the newest pending snapshot for this run is kept.">
                <input type="number" min={1} value={form.automatic_evaluation_interval} disabled={!form.automatic_evaluation_enabled} onChange={(event) => setNumber("automatic_evaluation_interval", event.target.value)} />
              </Field>
              <Field label="Checkpoint interval (steps)" help="How often the trainer writes a recoverable model checkpoint.">
                <input type="number" min={1} value={form.checkpoint_every} onChange={(event) => setNumber("checkpoint_every", event.target.value)} />
              </Field>
              <Field label="PPO steps per environment" help="Rollout length collected before each PPO update. This affects update frequency and must satisfy the backend's rollout constraints.">
                <input type="number" min={64} step={64} value={form.n_steps} onChange={(event) => setNumber("n_steps", event.target.value)} />
              </Field>
              <Field label="Learning rate" help="Optimizer step size used for each PPO update.">
                <input type="number" min={0} step="any" value={form.learning_rate} onChange={(event) => setNumber("learning_rate", event.target.value)} />
              </Field>
              <Field label="PPO epochs" help="Number of optimization passes over each collected rollout.">
                <input type="number" min={1} value={form.n_epochs} onChange={(event) => setNumber("n_epochs", event.target.value)} />
              </Field>
              <Field label="Discount (γ)" help="Weight assigned to future rewards. Higher values make the policy consider longer-term outcomes.">
                <input type="number" min={0} max={1} step="any" value={form.gamma} onChange={(event) => setNumber("gamma", event.target.value)} />
              </Field>
              <Field label="GAE trace (λ)" help="Controls the bias/variance trade-off in PPO's advantage estimates.">
                <input type="number" min={0} max={1} step="any" value={form.gae_lambda} onChange={(event) => setNumber("gae_lambda", event.target.value)} />
              </Field>
            </div>
          </details>

          <details style={{ marginTop: 8 }}>
            <summary style={{ cursor: "pointer", fontWeight: 650, padding: "6px 0" }}>Simulator, observations, and controls</summary>
            <div className="grid" style={{ alignItems: "start", marginTop: 10 }}>
              <Field label="Simulator graphics" help="Mission Control starts the official graphics-enabled Windows simulator and a native WSL PPO learner with the official ROS 2 bridge. Docker is not used on this run path.">
                <select value={form.simulator_backend} onChange={(event) => setChoice("simulator_backend", event.target.value as OfficialTrainSettings["simulator_backend"])}>
                  <option value="windows_external">Official Windows simulator (camera)</option>
                </select>
              </Field>
              <Field label="Simulator window" help="Offscreen mode runs the same official Windows simulator in Unity batch mode while keeping the GPU graphics device initialized. It omits -nographics so camera observations remain available, and it avoids opening simulator windows.">
                <select value={form.simulator_display_mode} onChange={(event) => setChoice("simulator_display_mode", event.target.value as OfficialTrainSettings["simulator_display_mode"])}>
                  <option value="gui">Show simulator windows</option>
                  <option value="offscreen">Offscreen rendering (camera on)</option>
                </select>
              </Field>
              <Field label="Parallel environments" help="Official simulator/Devkit pairs feeding one PPO model. More environments consume more CPU and memory; measure throughput before increasing.">
                <input type="number" min={1} max={8} value={form.n_envs} onChange={(event) => setNumber("n_envs", event.target.value)} />
              </Field>
              <Field label="PPO device" help="CUDA uses the WSL-visible NVIDIA GPU for PPO updates. Simulator rendering runs in the official Windows application.">
                <select value={form.device} onChange={(event) => setChoice("device", event.target.value as OfficialTrainSettings["device"])}>
                  <option value="cpu">CPU (default)</option><option value="cuda">NVIDIA GPU (CUDA)</option>
                </select>
              </Field>
              <Field label="Simulator startup timeout (s)" help="Maximum initial wait for official sensors. Individual control frames use a separate timeout of at most 5 seconds.">
                <input type="number" min={1} step="any" value={form.timeout_s} onChange={(event) => setNumber("timeout_s", event.target.value)} />
              </Field>
              <Field label="Per-episode simulator-clock watchdog (s)" help="Maximum elapsed episode time, including warm-up, measured from advancing ROS sensor timestamps when available. This is separate from the whole training run limit.">
                <input type="number" min={1} step="any" value={form.training_timeout_s} onChange={(event) => setNumber("training_timeout_s", event.target.value)} />
              </Field>
              <Field label="Observation profile" help="New official training runs always fuse the official front camera, LiDAR, and vehicle state as requested.">
                <select value="official_sensors_camera" disabled>
                  <option value="official_sensors_camera">Official sensors + front camera</option>
                </select>
              </Field>
              <Field label="Steering action scale" help="Scales the policy's steering output before it is sent to the simulator.">
                <input type="number" min={0} step="any" value={form.steering_action_scale} onChange={(event) => setNumber("steering_action_scale", event.target.value)} />
              </Field>
              <Field label="Straight throttle gain" help="Throttle adjustment applied when steering is near straight ahead.">
                <input type="number" min={0} step="any" value={form.straight_throttle_gain} onChange={(event) => setNumber("straight_throttle_gain", event.target.value)} />
              </Field>
              <Field label="Straight throttle steering threshold" help="Steering magnitude below which the straight throttle adjustment may apply.">
                <input type="number" min={0} step="any" value={form.straight_throttle_steering_threshold} onChange={(event) => setNumber("straight_throttle_steering_threshold", event.target.value)} />
              </Field>
              <Field label="Throttle mode" help="Choose whether the policy can request reverse throttle or only forward throttle.">
                <select value={form.throttle_mode} onChange={(event) => setChoice("throttle_mode", event.target.value as OfficialTrainSettings["throttle_mode"])}>
                  <option value="bidirectional">Bidirectional</option><option value="forward_only">Forward only</option>
                </select>
              </Field>
              <Field label="Negative throttle mode" help="Defines how negative policy throttle values are interpreted by the action adapter.">
                <select value={form.negative_throttle_mode} onChange={(event) => setChoice("negative_throttle_mode", event.target.value as OfficialTrainSettings["negative_throttle_mode"])}>
                  <option value="allow">Allow reverse</option><option value="zero">Clamp to zero</option><option value="positive_magnitude">Use positive magnitude</option>
                </select>
              </Field>
              <Field label="Steering mode" help="Use the normal steering direction or invert it for a simulator/control setup with reversed steering polarity.">
                <select value={form.steering_mode} onChange={(event) => setChoice("steering_mode", event.target.value as OfficialTrainSettings["steering_mode"])}>
                  <option value="normal">Normal</option><option value="invert">Invert</option>
                </select>
              </Field>
            </div>
          </details>

          <div className="actions" style={{ marginTop: 14 }}>
            <button type="submit" className="btn primary" disabled={!canStart}>{busy ? "Working…" : activeRuns.length >= maxRuns ? "Run limit reached" : "Start official run"}</button>
            <span className="meta">{activeRuns.length >= maxRuns ? "Stop or wait for a run to finish before starting another." : `${maxRuns - activeRuns.length} concurrent slot${maxRuns - activeRuns.length === 1 ? "" : "s"} available.`}</span>
          </div>
        </form>
      </details>

      {message && <p className="msg ok" role="status">{message}</p>}
      {error && <p className="msg err" role="alert">{error}</p>}
      {selectedPhase && <p className="meta" role="status">Selected run stage: {{ demonstration_collection: "Collecting official teacher examples", behavior_cloning: "Initializing PPO from teacher examples", rollout: "Collecting PPO rollout", ppo_update: "Updating PPO", stopped: "Training stopped" }[selectedPhase.phase] ?? selectedPhase.phase} · step {formatCount(selectedPhase.step)}.</p>}
      {selectedRun?.stop_reason && <p className="meta">Selected run result: {selectedRun.stop_reason}</p>}
      {selectedRun?.simulator_backend === "windows_external" && selectedRun.simulator_connections?.length ? <section className="panel" aria-label="Windows simulator connections" style={{ marginTop: 14 }}>
        <h3 style={{ marginTop: 0 }}>Official simulator connections</h3>
        <p className="meta">Mission Control starts one official Windows simulator per environment and connects it to its native WSL bridge. Camera frames and sensors feed the shared PPO learner.</p>
        <ol>{selectedRun.simulator_connections.map((connection) => <li key={connection.environment}>Environment {connection.environment}: <code>{connection.host}:{connection.port}</code> (ROS domain {connection.ros_domain_id})</li>)}</ol>
        <p className="meta">{selectedRun.config?.simulator_display_mode === "offscreen" ? "Offscreen simulator processes stop when the run stops." : "Simulator windows close when the run stops."}</p>
      </section> : null}
    </section>
  );
});

const thStyle: React.CSSProperties = { padding: "8px 10px", borderBottom: "1px solid var(--border, #30443c)", fontSize: "0.9em", whiteSpace: "nowrap" };
const tdStyle: React.CSSProperties = { padding: "8px 10px", borderBottom: "1px solid var(--border, #263a32)", verticalAlign: "middle" };

function RunRow({
  run,
  selected,
  busy,
  onSelect,
  onStop,
  history = false,
}: {
  run: RunSummary;
  selected: boolean;
  busy: boolean;
  onSelect: () => void;
  onStop: () => void;
  history?: boolean;
}) {
  const progress = runProgress(run);
  const started = run.started_at ? new Date(run.started_at).toLocaleString() : "—";
  const finished = run.finished_at ? new Date(run.finished_at).toLocaleString() : run.stop_reason || "—";
  return <tr aria-selected={selected} style={{ background: selected ? "color-mix(in srgb, var(--accent, #40d99a) 8%, transparent)" : undefined }}>
    <td style={tdStyle}>
      <strong>{run.display_name || run.run_id}</strong>
      <div className="meta" style={{ fontSize: "0.82em" }}>{run.run_id}</div>
      {(run.error || run.cleanup_error) && <div className="error" role="status" title={run.error || run.cleanup_error || undefined}>{run.error || run.cleanup_error}</div>}
    </td>
    <td style={tdStyle}><span className="badge" data-state={run.state}>{run.state}</span></td>
    <td style={{ ...tdStyle, minWidth: 170 }}>
      {progress ? <>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, fontVariantNumeric: "tabular-nums" }}><span>{formatCount(progress.step)} / {formatCount(progress.total)}</span><span>{progress.percent.toFixed(1)}%</span></div>
        <progress value={progress.percent} max={100} aria-label={`${run.display_name || run.run_id} training progress`} style={{ width: "100%", height: 6 }} />
      </> : <span className="meta">{(() => {
        const step = run.latest_metrics?.step ?? run.latest_telemetry?.step ?? run.latest_fleet?.step;
        if (typeof step === "number") return `${formatCount(step)} steps${run.config?.total_timesteps === 0 ? " · no step cap" : ""}`;
        return history ? "0 steps" : "Waiting for first update…";
      })()}</span>}
    </td>
    <td style={tdStyle}>{history ? finished : started}</td>
    <td style={{ ...tdStyle, whiteSpace: "nowrap" }}>
      <button type="button" className="btn" aria-pressed={selected} onClick={onSelect}>{selected ? "Selected for Watch" : "Watch this run"}</button>
      {!history && isActive(run) && <button type="button" className="btn danger" disabled={busy} onClick={onStop} style={{ marginLeft: 6 }}>Stop</button>}
    </td>
  </tr>;
}
