import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchMaps,
  getEvaluationHistory,
  getTrainStatus,
  getSettings,
  getPpoDiagnostics,
  type MetricsTelemetry,
  type PpoDiagnosticsUpdate,
  type EvaluationStatus,
  type EvaluationResult,
  type EvaluatorCarTelemetry,
  type MapCatalogEntry,
  type Settings,
} from "../api";
import { FleetCanvas } from "../fleet/FleetCanvas";
import { getMetricsHistoryHot, useHubStore } from "../store";
import { formatLapDuration } from "./lapTiming";

interface RewardSample {
  step: number;
  reward: number;
  complete: boolean;
  observedStep?: number;
}

const REWARD_HISTORY_LIMIT = 20_000;

type PpoPoint = { step: number; value: number };
type PpoMetricKey = keyof NonNullable<MetricsTelemetry["ppo"]>;

const PPO_METRIC_GROUPS: Array<{
  title: string;
  metrics: Array<{ key: PpoMetricKey; label: string; unit: string; help: string }>;
}> = [
  {
    title: "Policy update",
    metrics: [
      { key: "approx_kl", label: "Approx. KL", unit: "", help: "How far the updated policy moved from the policy that collected this rollout. Large spikes can indicate updates that are too aggressive; interpret alongside clip fraction and driving results." },
      { key: "clip_fraction", label: "Clip fraction", unit: "%", help: "Share of PPO optimization samples whose policy change was clipped. A persistently high value can mean updates are pushing hard against PPO’s trust-region limit." },
      { key: "policy_gradient_loss", label: "Policy gradient loss", unit: "", help: "The policy-optimization loss. Its sign and magnitude are not a driving score; use its trend with KL, clipping, and actual reward." },
    ],
  },
  {
    title: "Exploration",
    metrics: [
      { key: "std", label: "Action std", unit: "", help: "Standard deviation of the policy’s continuous action distribution. Higher values mean noisier sampled steering/throttle actions; it does not directly measure driving quality." },
      { key: "entropy_loss", label: "Entropy loss", unit: "", help: "Tracks policy randomness as used by SB3’s optimization objective. It is usually negative; changes are meaningful as a trend, not as a score." },
    ],
  },
  {
    title: "Value estimates",
    metrics: [
      { key: "explained_variance", label: "Explained variance", unit: "", help: "How well the value network predicts returns. Near 1 is strong prediction, around 0 is no better than predicting a constant, and below 0 is worse. This diagnoses the critic, not driving skill." },
      { key: "value_loss", label: "Value loss", unit: "", help: "Error in the value network’s return predictions. Its scale depends on reward magnitudes, so compare its trend within a run rather than against other runs." },
    ],
  },
  {
    title: "Optimization health",
    metrics: [
      { key: "fps", label: "Training throughput", unit: "steps/s", help: "Environment steps processed per second by training. This measures throughput, not how quickly the policy is learning." },
      { key: "learning_rate", label: "Learning rate", unit: "", help: "Optimizer step size used for this PPO update. It can remain constant or follow the configured schedule." },
      { key: "loss", label: "Total loss", unit: "", help: "Combined optimization loss reported by SB3. It is not episode reward or a policy-quality score; use the task and evaluator metrics to judge driving." },
    ],
  },
];

function formatPpoValue(key: PpoMetricKey, value: number): string {
  if (key === "fps") return Math.round(value).toLocaleString();
  if (key === "learning_rate") return value.toExponential(2);
  if (key === "clip_fraction") return `${(value * 100).toFixed(1)}%`;
  if (key === "explained_variance") return value.toFixed(3);
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(4);
}

function PpoMetricCard({
  metric,
  points,
}: {
  metric: (typeof PPO_METRIC_GROUPS)[number]["metrics"][number];
  points: PpoPoint[];
}) {
  const latest = points[points.length - 1];
  const min = Math.min(...points.map((point) => point.value));
  const max = Math.max(...points.map((point) => point.value));
  const span = max - min;
  const polyline = points.map((point, index) => {
    const x = points.length < 2 ? 50 : 3 + index / (points.length - 1) * 94;
    const y = 34 - (span > 1e-12 ? (point.value - min) / span * 28 : 14);
    return `${x},${y}`;
  }).join(" ");
  return (
    <article className="ppo-metric-card">
      <div className="ppo-metric-heading">
        <span>{metric.label}</span>
        <span className="reward-tip" tabIndex={0} aria-label={`About ${metric.label}`} data-tooltip={metric.help}>?</span>
      </div>
      <strong>{latest ? formatPpoValue(metric.key, latest.value) : "—"}</strong>
      {metric.unit && <span className="ppo-metric-unit">{metric.unit}</span>}
      {points.length > 1 ? (
        <svg className="ppo-sparkline" viewBox="0 0 100 40" role="img" aria-label={`${metric.label} trend over PPO updates`}>
          <line x1="0" x2="100" y1="34" y2="34" />
          <polyline points={polyline} />
          {points.map((point, index) => {
            if (index % Math.max(1, Math.ceil(points.length / 24)) !== 0 && index !== points.length - 1) return null;
            const x = points.length < 2 ? 50 : 3 + index / (points.length - 1) * 94;
            const y = 34 - (span > 1e-12 ? (point.value - min) / span * 28 : 14);
            return <circle key={`${point.step}-${index}`} cx={x} cy={y} r="1.35"><title>{`Step ${point.step.toLocaleString()}: ${formatPpoValue(metric.key, point.value)}`}</title></circle>;
          })}
        </svg>
      ) : <div className="ppo-sparkline-empty">{points.length ? "Trend begins with the next update" : "Waiting for PPO update"}</div>}
      <small>{points.length ? `step ${latest?.step.toLocaleString()} · ${points.length} updates` : "No samples yet"}</small>
    </article>
  );
}

function rewardHistoryStorageKey(runId: string, rolloutSize: number): string {
  return `aicar.watch.reward-history.v1:${runId}:${rolloutSize}`;
}

function loadRewardHistory(key: string): RewardSample[] {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((sample): sample is RewardSample =>
      sample != null &&
      Number.isFinite(sample.step) &&
      Number.isFinite(sample.reward) &&
      typeof sample.complete === "boolean",
    ).slice(-REWARD_HISTORY_LIMIT);
  } catch {
    return [];
  }
}

function downsampleChartPoints<T>(
  points: T[],
  getValue: (point: T) => number,
  maxPoints = 600,
): Array<{ point: T; index: number }> {
  if (points.length <= maxPoints) {
    return points.map((point, index) => ({ point, index }));
  }
  const bucketCount = Math.max(1, Math.floor((maxPoints - 2) / 2));
  const bucketSize = (points.length - 2) / bucketCount;
  const sampled: Array<{ point: T; index: number }> = [{ point: points[0], index: 0 }];
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = Math.floor(1 + bucket * bucketSize);
    const end = Math.min(points.length - 1, Math.floor(1 + (bucket + 1) * bucketSize));
    let minIndex = start;
    let maxIndex = start;
    for (let index = start + 1; index < end; index += 1) {
      const value = getValue(points[index]);
      if (value < getValue(points[minIndex])) minIndex = index;
      if (value > getValue(points[maxIndex])) maxIndex = index;
    }
    for (const index of minIndex === maxIndex
      ? [minIndex]
      : minIndex < maxIndex ? [minIndex, maxIndex] : [maxIndex, minIndex]) {
      if (sampled[sampled.length - 1].index !== index) sampled.push({ point: points[index], index });
    }
  }
  if (sampled[sampled.length - 1].index !== points.length - 1) {
    sampled.push({ point: points[points.length - 1], index: points.length - 1 });
  }
  return sampled;
}

function buildRewardSamples(
  history: MetricsTelemetry[],
  latest: MetricsTelemetry | null,
  rolloutSize: number,
  environmentCount: number,
  saved: RewardSample[],
): RewardSample[] {
  const runId = latest?.run_id ?? history[history.length - 1]?.run_id;
  if (!runId) return [];
  const runHistory = history.filter((sample) => sample.run_id === runId);
  if (latest && !runHistory.some((sample) => sample.step === latest.step)) runHistory.push(latest);
  const totals = new Map<number, number>();
  for (const sample of saved) {
    totals.set(Math.floor(sample.step / rolloutSize) - 1, sample.reward);
  }
  const savedCheckpoint = saved.reduce(
    (max, sample) => Math.max(max, sample.observedStep ?? 0),
    0,
  );
  // Older saved samples did not include their observation step. Preserve them
  // and continue from the hub's current snapshot instead of replacing a full
  // in-progress rollout with a partial estimate after refresh.
  let previousStep: number | null = saved.length
    ? savedCheckpoint || latest?.step || null
    : null;
  for (const sample of runHistory) {
    if (!Number.isFinite(sample.step) || !Number.isFinite(sample.reward)) continue;
    if (previousStep != null && sample.step <= previousStep) continue;
    const startStep = previousStep ?? Math.floor(Math.max(0, sample.step - 1) / rolloutSize) * rolloutSize;
    if (sample.step <= startStep) continue;
    // num_timesteps advances by environmentCount for each vector step, while
    // reward is the mean reward across those environments.
    const rewardPerGlobalStep = sample.reward / environmentCount;
    let cursor = startStep;
    while (cursor < sample.step) {
      const rolloutIndex = Math.floor(cursor / rolloutSize);
      const end = Math.min(sample.step, (rolloutIndex + 1) * rolloutSize);
      totals.set(rolloutIndex, (totals.get(rolloutIndex) ?? 0) + rewardPerGlobalStep * (end - cursor));
      cursor = end;
    }
    previousStep = sample.step;
  }
  const latestStep = Math.max(latest?.step ?? 0, runHistory[runHistory.length - 1]?.step ?? 0);
  return [...totals.entries()].map(([index, reward]) => ({
    step: (index + 1) * rolloutSize,
    reward,
    complete: latestStep >= (index + 1) * rolloutSize,
    observedStep: latestStep,
  }));
}

/**
 * Observe surface — underlay from Train-selected map_id; no map picker / Activate.
 */
export function WatchPage() {
  const { fleet, fleetAgeMs, metrics, status, trainingPhase } = useHubStore();
  const stale = fleetAgeMs > 500;
  const cars = useMemo(() => fleet?.cars ?? [], [fleet]);
  const [showMap, setShowMap] = useState(true);
  const [showFleet, setShowFleet] = useState(true);
  const [showFrontier, setShowFrontier] = useState(true);
  const [showCurrentProgress, setShowCurrentProgress] = useState(true);
  const [showLidar, setShowLidar] = useState(true);
  const [showDetails, setShowDetails] = useState(false);
  const [showTrainingProgress, setShowTrainingProgress] = useState(true);
  const [mapId, setMapId] = useState("none");
  const [maps, setMaps] = useState<MapCatalogEntry[]>([]);
  const [mapNote, setMapNote] = useState<string | null>(null);
  const [mapLoadErr, setMapLoadErr] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationStatus | null>(null);
  const [evaluationHistory, setEvaluationHistory] = useState<EvaluationResult[]>([]);
  const [persistedPpoUpdates, setPersistedPpoUpdates] = useState<PpoDiagnosticsUpdate[]>([]);
  const [persistedPpoRunId, setPersistedPpoRunId] = useState<string | null>(null);
  const [ppoHistoryError, setPpoHistoryError] = useState<string | null>(null);
  const [watchSettings, setWatchSettings] = useState<Settings | null>(null);
  const [rewardSamples, setRewardSamples] = useState<RewardSample[]>([]);
  const [selectedEnvId, setSelectedEnvId] = useState<number | null>(null);
  const mapsRef = useRef<MapCatalogEntry[]>([]);
  const lastAppliedMapIdRef = useRef<string | null>(null);
  const rolloutSize = Math.max(1, (watchSettings?.n_envs ?? 1) * (watchSettings?.ppo_n_steps ?? 1024));
  mapsRef.current = maps;

  const applyMapId = (mid: string, list: MapCatalogEntry[]) => {
    const id = mid.trim() || "none";
    if (lastAppliedMapIdRef.current === id) return;
    lastAppliedMapIdRef.current = id;
    setMapId(id);
    if (id === "none") {
      setMapNote(
        "Builtin track — no occupancy underlay (grid only). Set a mesh-ready map on Train."
      );
      setMapLoadErr(null);
      return;
    }
    const entry = list.find((m) => m.id === id);
    if (!entry) {
      setMapNote(`Train map "${id}" not in catalog.`);
      setMapLoadErr(`Unknown map ${id}`);
    } else if (!entry.yaml_url) {
      setMapNote(`Map "${id}" has no occupancy yaml — grid only.`);
      setMapLoadErr(null);
    } else {
      setMapNote(null);
      setMapLoadErr(null);
    }
  };

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      getSettings().catch(() => null),
      fetchMaps().catch(() => [] as MapCatalogEntry[]),
    ]).then(([settings, list]) => {
      if (cancelled) return;
      setWatchSettings(settings);
      setMaps(list);
      applyMapId(settings?.map_id ?? "none", list);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const history = getMetricsHistoryHot();
    const latest = metrics ?? history[history.length - 1] ?? null;
    if (!latest) {
      setRewardSamples([]);
      return;
    }
    const storageKey = rewardHistoryStorageKey(latest.run_id, rolloutSize);
    const samples = buildRewardSamples(
      history,
      latest,
      rolloutSize,
      Math.max(1, watchSettings?.n_envs ?? 1),
      loadRewardHistory(storageKey),
    ).slice(-REWARD_HISTORY_LIMIT);
    setRewardSamples(samples);
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(samples));
    } catch {
      // Keep the live graph working when browser storage is unavailable/full.
    }
  }, [metrics, rolloutSize, watchSettings?.n_envs]);

  const rewardTrend = rewardSamples;
  const rewardChartPoints = useMemo(
    () => downsampleChartPoints(rewardTrend, (point) => point.reward),
    [rewardTrend],
  );

  const rewardRange = useMemo(() => {
    if (rewardTrend.length === 0) return null;
    let min = Infinity;
    let max = -Infinity;
    for (const point of rewardTrend) {
      min = Math.min(min, point.reward);
      max = Math.max(max, point.reward);
    }
    return { min, max };
  }, [rewardTrend]);

  const ppoSeries = useMemo(() => {
    const history = getMetricsHistoryHot();
    const runId = persistedPpoRunId ?? metrics?.run_id ?? history[history.length - 1]?.run_id;
    if (metrics && metrics.run_id === runId && !history.some((sample) => sample.run_id === metrics.run_id && sample.step === metrics.step)) {
      history.push(metrics);
    }
    const updates = new Map<number, NonNullable<MetricsTelemetry["ppo"]>>(
      persistedPpoUpdates.map((update) => [update.step, update.values]),
    );
    for (const sample of history) {
      if (sample.run_id !== runId || !sample.ppo || !Number.isFinite(sample.ppo_step)) continue;
      const step = sample.ppo_step as number;
      updates.set(step, { ...updates.get(step), ...sample.ppo });
    }
    const ordered = [...updates.entries()].sort(([a], [b]) => a - b);
    const series = {} as Record<PpoMetricKey, PpoPoint[]>;
    for (const group of PPO_METRIC_GROUPS) {
      for (const item of group.metrics) {
        series[item.key] = ordered
          .map(([step, values]) => ({ step, value: values[item.key] }))
          .filter((point): point is PpoPoint => typeof point.value === "number" && Number.isFinite(point.value))
          .slice(-160);
      }
    }
    return { series, updateCount: ordered.length, latestStep: ordered[ordered.length - 1]?.[0] ?? null };
  }, [metrics, persistedPpoUpdates, persistedPpoRunId]);

  const evaluatorScores = useMemo(() => evaluationHistory
    .map((result, index) => ({
      index,
      step: result.timesteps,
      score: result.selection_score ?? (result.selection_metric === "total_reward"
        ? result.total_reward
        : result.selection_metric === "reward_per_simulated_second"
          ? result.reward_per_simulated_second
          : result.frontier_speed_mps),
      improved: result.improved ?? false,
    }))
    .filter((point): point is { index: number; step: number; score: number; improved: boolean } =>
      typeof point.score === "number" && Number.isFinite(point.score)), [evaluationHistory]);
  const evaluatorChartPoints = useMemo(
    () => downsampleChartPoints(evaluatorScores, (point) => point.score),
    [evaluatorScores],
  );
  const evaluatorRange = useMemo(() => {
    if (!evaluatorScores.length) return null;
    let min = Infinity;
    let max = -Infinity;
    for (const point of evaluatorScores) {
      min = Math.min(min, point.score);
      max = Math.max(max, point.score);
    }
    return { min, max };
  }, [evaluatorScores]);
  const evaluatorPolyline = evaluatorRange && evaluatorScores.length > 1
    ? evaluatorChartPoints.map(({ point, index }) => {
      const x = 64 + index / (evaluatorScores.length - 1) * 648;
      const span = evaluatorRange.max - evaluatorRange.min;
      const normalized = span > 1e-8 ? (point.score - evaluatorRange.min) / span : 0.5;
      return `${x.toFixed(1)},${(145 - normalized * 128).toFixed(1)}`;
    }).join(" ")
    : "";

  const evaluatorPatience = Math.max(
    1,
    evaluation?.plateau_patience ?? watchSettings?.plateau_patience ?? 1,
  );
  const staleEvaluations = Math.max(0, evaluation?.stale_evaluations ?? 0);
  const stalePercent = Math.min(100, staleEvaluations / evaluatorPatience * 100);
  const evaluatorTracking = evaluation != null || ["starting", "running", "stopping"].includes(status?.state ?? "");
  const evaluationInterval = watchSettings?.evaluation_every_timesteps ?? 0;
  const lastEvaluatedStep = evaluation?.latest_result?.timesteps ?? 0;
  const nextEvaluationThreshold = evaluationInterval > 0
    ? (Math.floor(lastEvaluatedStep / evaluationInterval) + 1) * evaluationInterval
    : null;
  // The current hub may predate the display-only PPO metadata endpoint. The
  // active Layer 3 default is 1024; newer hubs provide the value directly.
  const nextEvaluationSnapshotStep = evaluation?.state === "running" && evaluation.snapshot_timesteps != null
    ? evaluation.snapshot_timesteps
    : nextEvaluationThreshold == null
      ? null
      : Math.ceil(nextEvaluationThreshold / rolloutSize) * rolloutSize;
  const rewardPolyline = rewardRange && rewardTrend.length > 1
    ? rewardChartPoints.map(({ point, index }) => {
      const x = 64 + index / (rewardTrend.length - 1) * 648;
      const span = rewardRange.max - rewardRange.min;
      const normalized = span > 1e-8 ? (point.reward - rewardRange.min) / span : 0.5;
      const y = 145 - normalized * 128;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ")
    : "";

  useEffect(() => {
    let cancelled = false;
    let lastHistoryKey: string | null = null;
    let lastEvaluationKey: string | undefined;
    const syncHistory = (key: string) => {
      if (key === lastHistoryKey) return;
      lastHistoryKey = key;
      void getEvaluationHistory()
        .then((history) => { if (!cancelled) setEvaluationHistory(history); })
        .catch(() => undefined);
    };
    syncHistory("initial");
    const refreshEvaluation = () => {
      void getTrainStatus()
        .then((status) => {
          if (cancelled) return;
          const last = status.last_evaluation ?? null;
          const evaluationKey = last == null ? "none" : JSON.stringify(last);
          if (evaluationKey !== lastEvaluationKey) {
            lastEvaluationKey = evaluationKey;
            setEvaluation(last);
          }
          if (last?.state === "complete") syncHistory(last.updated_utc);
        })
        .catch(() => undefined);
    };
    refreshEvaluation();
    const timer = window.setInterval(refreshEvaluation, 500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  // TensorBoard event files are the durable source of PPO history. Re-read
  // them periodically so browser reloads and hub restarts retain the chart.
  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      if (document.visibilityState !== "visible") return;
      void getPpoDiagnostics()
        .then((history) => {
          if (cancelled) return;
          setPersistedPpoRunId(history.run_id);
          setPersistedPpoUpdates(history.updates);
          setPpoHistoryError(null);
        })
        .catch((error: Error) => setPpoHistoryError(error.message));
    };
    refresh();
    const timer = window.setInterval(refresh, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  // Follow Train map_id: event (immediate) + short poll + focus refresh
  useEffect(() => {
    let lastSettingsKey = "";
    const syncFromSettings = () => {
      void getSettings()
        .then((s) => {
          const settingsKey = JSON.stringify(s);
          if (settingsKey !== lastSettingsKey) {
            lastSettingsKey = settingsKey;
            setWatchSettings(s);
          }
          applyMapId(s.map_id ?? "none", mapsRef.current);
        })
        .catch(() => undefined);
    };

    const onMapEvent = (ev: Event) => {
      applyMapId(String((ev as CustomEvent).detail ?? "none"), mapsRef.current);
    };

    window.addEventListener("aicar-map-id", onMapEvent);
    window.addEventListener("focus", syncFromSettings);
    const timer = window.setInterval(syncFromSettings, 1000);
    return () => {
      window.removeEventListener("aicar-map-id", onMapEvent);
      window.removeEventListener("focus", syncFromSettings);
      window.clearInterval(timer);
    };
  }, []);

  const selectedMap = useMemo(
    () => maps.find((m) => m.id === mapId) ?? null,
    [maps, mapId]
  );

  const selected = useMemo(
    () => selectedEnvId == null ? null : cars.find((c) => c.env_id === selectedEnvId) ?? null,
    [cars, selectedEnvId]
  );
  const lapSupported = cars.filter((c) => c.lap_supported);
  const evaluatorCar: EvaluatorCarTelemetry | null =
    evaluation?.state === "running" ? evaluation.evaluator_car ?? null : null;
  const liveEvaluationResult = evaluation?.state === "running"
    ? evaluation.live_result ?? null
    : null;
  // Never present a previous completed evaluation as if it described the
  // snapshot currently being evaluated.
  const shownEvaluationResult = evaluation?.state === "running"
    ? liveEvaluationResult
    : evaluation?.latest_result ?? null;
  const completedEvaluationResult = evaluation?.state === "running"
    ? null
    : evaluation?.latest_result ?? null;
  const shownLapTimes = shownEvaluationResult?.lap_times_s ?? [];
  const shownTenLapTime = shownEvaluationResult?.best_10_lap_time_s ?? null;
  const selectionMetric = evaluation?.selection_metric ?? watchSettings?.evaluation_metric ?? "frontier_speed";
  const selectionMetricLabel = selectionMetric === "reward_per_simulated_second"
    ? "reward per simulated second"
    : selectionMetric === "total_reward"
      ? "total reward per attempt"
      : "frontier pace (m/s)";
  const currentTrainingStep = metrics?.step ?? fleet?.step ?? 0;
  const ppoIsUpdating = status?.state === "running" && trainingPhase?.phase === "ppo_update";
  const nextPpoStep = trainingPhase?.phase === "rollout"
    ? trainingPhase.step + rolloutSize
    : (Math.floor(currentTrainingStep / rolloutSize) + 1) * rolloutSize;
  const rolloutProgress = Math.max(0, Math.min(1,
    (currentTrainingStep - (nextPpoStep - rolloutSize)) / rolloutSize,
  ));

  return (
    <section className="panel fleet-panel">
      <h2>Watch</h2>

      {evaluatorTracking && (
        <section className="plateau-progress evaluator-plateau-top" aria-live="polite">
          <div className="plateau-progress-label">
            <span>Checks without qualifying improvement</span>
            <strong>{staleEvaluations}/{evaluatorPatience}</strong>
          </div>
          <div
            className="plateau-progress-track"
            role="progressbar"
            aria-label="Evaluator checks without qualifying improvement"
            aria-valuemin={0}
            aria-valuemax={evaluatorPatience}
            aria-valuenow={Math.min(staleEvaluations, evaluatorPatience)}
          >
            <div className="plateau-progress-fill" style={{ width: `${stalePercent}%` }} />
          </div>
          <p className="meta">
            A score needs to improve by at least {watchSettings?.plateau_min_improvement_pct ?? 1}% to reset the counter.
            {watchSettings?.plateau_min_timesteps != null && (
              <> Plateau stopping activates after {watchSettings.plateau_min_timesteps.toLocaleString()} steps.</>
            )}
          </p>
        </section>
      )}

      <div className="watch-metrics">
        <div className="watch-metrics-row">
          <span>
            <strong>step</strong> {metrics?.step ?? "—"}
          </span>
          <span>
            <strong>episode</strong> {metrics?.episode ?? "—"}
          </span>
          <span>
            <strong>laps</strong> {metrics?.completed_laps ?? 0}
          </span>
          <span>
            <strong>best lap</strong>{" "}
            {metrics?.best_lap_time_s != null
              ? `${metrics.best_lap_time_s.toFixed(2)} s`
              : "—"}
          </span>
          <span><strong>cars</strong> {cars.length}</span>
          <span className="meta">
            underlay <strong>{mapId}</strong>
          </span>
          <button
            type="button"
            className="btn"
            onClick={() => setShowDetails((v) => !v)}
          >
            {showDetails ? "Hide details" : "Details"}
          </button>
        </div>
        {showDetails && (
          <table className="live-table">
            <tbody>
              <tr>
                <td>loss</td>
                <td>{metrics?.loss != null ? metrics.loss.toFixed(6) : "—"}</td>
              </tr>
              <tr>
                <td>completed_laps</td>
                <td>{metrics?.completed_laps ?? 0}</td>
              </tr>
              <tr>
                <td>best_lap_time_s</td>
                <td>
                  {metrics?.best_lap_time_s != null
                    ? metrics.best_lap_time_s.toFixed(2)
                    : "—"}
                </td>
              </tr>
              <tr>
                <td>run_id</td>
                <td>{metrics?.run_id ?? "—"}</td>
              </tr>
              <tr>
                <td>checkpoint</td>
                <td>{metrics?.checkpoint ?? "—"}</td>
              </tr>
              <tr>
                <td>ts</td>
                <td>{metrics?.ts ?? "—"}</td>
              </tr>
            </tbody>
          </table>
        )}
      </div>

      <div className="fleet-toolbar">
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showMap}
            onChange={(e) => setShowMap(e.target.checked)}
          />
          Map
        </label>
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showFleet}
            onChange={(e) => setShowFleet(e.target.checked)}
          />
          Fleet
        </label>
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showFrontier}
            onChange={(e) => setShowFrontier(e.target.checked)}
          />
          Frontier
        </label>
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showCurrentProgress}
            onChange={(e) => setShowCurrentProgress(e.target.checked)}
          />
          Current position
        </label>
        <label className="check-inline">
          <input
            type="checkbox"
            checked={showLidar}
            disabled={selectedEnvId == null}
            onChange={(e) => setShowLidar(e.target.checked)}
          />
          LiDAR {selectedEnvId == null ? "(select a car)" : ""}
        </label>

        {cars.length > 0 && (
          <label className="field-inline">
            Focus
            <select
              value={selectedEnvId ?? "all"}
              onChange={(e) => setSelectedEnvId(e.target.value === "all" ? null : Number(e.target.value))}
            >
              <option value="all">All cars</option>
              {cars.map((c) => (
                <option key={c.env_id} value={c.env_id}>
                  env {c.env_id}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

  {mapNote && <p className="meta fleet-note">{mapNote}</p>}
  {mapLoadErr && (
    <p className="meta fleet-note" data-bad="1">
      {mapLoadErr}
    </p>
  )}
  {mapId !== "none" &&
    maps.some((m) => m.active && m.id !== mapId) && (
      <p className="meta fleet-note">
        Underlay is <strong>{mapId}</strong>; sims still run{" "}
        <strong>{maps.find((m) => m.active)?.id}</strong> until Start locks
        the Train map. Cars may sit off-track until then.
      </p>
    )}

      <div className="fleet-layout">
        <div className="fleet-canvas-stack">
          {status?.state === "running" && rolloutSize > 0 && (
            <div className="ppo-rollout-indicator" role="status" aria-live="polite">
              <div className="ppo-rollout-copy">
                <strong>{ppoIsUpdating ? "PPO updating · simulator paused" : `${Math.max(0, nextPpoStep - currentTrainingStep).toLocaleString()} steps until PPO update`}</strong>
                <span>{ppoIsUpdating
                  ? "Policy update in progress"
                  : `Step ${currentTrainingStep.toLocaleString()} of ${nextPpoStep.toLocaleString()} · ${rolloutSize.toLocaleString()} steps per rollout`}</span>
              </div>
              <div className="ppo-rollout-track" role="progressbar" aria-label="Progress to next PPO update" aria-valuemin={0} aria-valuemax={100} aria-valuenow={ppoIsUpdating ? 100 : Math.round(rolloutProgress * 100)}>
                <div className={ppoIsUpdating ? "ppo-rollout-fill updating" : "ppo-rollout-fill"} style={{ width: `${ppoIsUpdating ? 100 : rolloutProgress * 100}%` }} />
              </div>
            </div>
          )}
          <FleetCanvas
            showMap={showMap}
            showFleet={showFleet}
            showFrontier={showFrontier}
            showCurrentProgress={showCurrentProgress}
            showLidar={showLidar}
            selectedEnvId={selected?.env_id ?? -1}
            mapId={mapId}
            mapYamlUrl={selectedMap?.yaml_url ?? null}
            fleetPanel={fleet}
            evaluatorCar={evaluatorCar}
            evaluatorActive={evaluation?.state === "running"}
            evaluatorSnapshotTimesteps={evaluation?.snapshot_timesteps ?? null}
            trainingStep={metrics?.step}
            rolloutSize={rolloutSize}
          />
          <div className="canvas-legend" aria-hidden>
            <span className="leg-car">▸ car</span>
            <span className="leg-evaluator">◆ evaluator</span>
            <span className="leg-collision">red car · collision event</span>
            <span className="leg-current">━ current route position</span>
            <span className="leg-frontier">┄ best progress</span>
            <span className="leg-lap">┄ finish gate</span>
          </div>
        </div>

        <aside className="fleet-side">
          <h3>{selected ? `Env ${selected.env_id}` : "Fleet overview"}</h3>
          {!selected ? (
            <>
              {cars.length === 0 ? <p className="meta">No fleet sample yet. Start a train job with HUB_URL set.</p> : <>
                {lapSupported.length > 0 && (
                  <div className="fleet-lap-summary">
                    <span>Best 10-lap time</span>
                    <strong>{formatLapDuration(fleet?.best_10_lap_time_s)}</strong>
                  </div>
                )}
                {lapSupported.length === 0 ? <p className="meta">Lap tracking is unavailable on this map.</p> :
                  <div className="lap-list">
                    {lapSupported.map((car) => (
                      <details className="lap-disclosure" key={car.env_id}>
                        <summary>
                          <span>Env {car.env_id}</span>
                          <strong>{car.lap_count ?? 0} laps</strong>
                        </summary>
                        <dl className="lap-times">
                          <div><dt>Current lap</dt><dd>{car.lap_elapsed_s != null ? `${car.lap_elapsed_s.toFixed(2)} s` : "—"}</dd></div>
                          <div><dt>Last lap</dt><dd>{car.last_lap_time_s != null ? `${car.last_lap_time_s.toFixed(2)} s` : "—"}</dd></div>
                          <div><dt>Best lap</dt><dd>{car.best_lap_time_s != null ? `${car.best_lap_time_s.toFixed(2)} s` : "—"}</dd></div>
                        </dl>
                      </details>
                    ))}
                  </div>}
              </>}
              <details className="lap-disclosure evaluator-disclosure">
                <summary>
                  <span>Evaluator</span>
                  <strong>{evaluation?.state ?? "waiting"}</strong>
                </summary>
                <div className="evaluator-side" aria-live="polite">
              <div className="evaluator-heading">
                <span>{evaluation?.state ?? "waiting"}</span>
                {evaluation?.state === "running" && <span>Snapshot {evaluation.snapshot_timesteps?.toLocaleString() ?? "—"}</span>}
              </div>
              {liveEvaluationResult?.attempt_count != null && (
                <p className="meta evaluator-attempt-progress">
                  Attempt {liveEvaluationResult.attempt_index ?? 1} of {liveEvaluationResult.attempt_count} · this attempt is live; the snapshot score is summarized after all attempts finish.
                </p>
              )}
              {nextEvaluationSnapshotStep != null && evaluatorTracking && (
                <p className="meta evaluator-next-step">
                  {evaluation?.state === "running" ? "Evaluator snapshot" : "Next snapshot"} <strong>step {nextEvaluationSnapshotStep.toLocaleString()}</strong>
                </p>
              )}
              {shownEvaluationResult ? (
                <>
                  <div className="evaluator-grid">
                    <div><span>Frontier pace{liveEvaluationResult ? " · live" : ""}</span><strong>{shownEvaluationResult.frontier_speed_mps.toFixed(2)} m/s</strong></div>
                    <div><span>Frontier distance{liveEvaluationResult ? " · live" : ""}</span><strong>{shownEvaluationResult.frontier_distance_m.toFixed(1)} m</strong></div>
                    <div><span>Laps{liveEvaluationResult ? " · live" : ""}</span><strong>{shownEvaluationResult.laps_observed}/10</strong></div>
                    <div><span>Simulated time{liveEvaluationResult ? " · live" : ""}</span><strong>{shownEvaluationResult.simulated_seconds.toFixed(1)} s</strong></div>
                    <div><span>Reward per simulated second</span><strong>{shownEvaluationResult.reward_per_simulated_second.toFixed(2)}</strong></div>
                    <div><span>Total attempt reward{liveEvaluationResult ? " · live" : ""}</span><strong>{shownEvaluationResult.total_reward?.toFixed(2) ?? "—"}</strong></div>
                    {!liveEvaluationResult && <div><span>Failed episodes</span><strong>{evaluation?.latest_result?.failed_episodes ?? 0}</strong></div>}
                    <div><span>Best selected score · {selectionMetricLabel}</span><strong>{evaluation?.best_selection_score?.toFixed(3) ?? "—"}</strong></div>
                  </div>
                  {completedEvaluationResult?.evaluation_runs != null && (
                    <>
                      <div className="evaluator-aggregate-summary">
                        <strong>{completedEvaluationResult.evaluation_runs} attempts</strong>
                        <span>Median {completedEvaluationResult.selection_score_median?.toFixed(3) ?? completedEvaluationResult.selection_score?.toFixed(3) ?? "—"}</span>
                        <span>Mean {completedEvaluationResult.selection_score_mean?.toFixed(3) ?? "—"}</span>
                        <span>Best {completedEvaluationResult.selection_score_best?.toFixed(3) ?? "—"}</span>
                        <span>10-lap completions {completedEvaluationResult.successful_attempts ?? 0}/{completedEvaluationResult.evaluation_runs}</span>
                      </div>
                      {completedEvaluationResult.attempts && (
                        <details className="evaluator-attempts-details">
                          <summary>Individual attempts</summary>
                          <ol>
                            {completedEvaluationResult.attempts.map((attempt, index) => {
                              const score = selectionMetric === "frontier_speed"
                                ? attempt.frontier_speed_mps
                                : selectionMetric === "reward_per_simulated_second"
                                  ? attempt.reward_per_simulated_second
                                  : attempt.total_reward;
                              return <li key={index}>Attempt {index + 1}: score {score.toFixed(3)} · {attempt.laps_observed}/10 laps · {attempt.collisions} collisions</li>;
                            })}
                          </ol>
                        </details>
                      )}
                    </>
                  )}
                  <div className="evaluator-laps">
                    <div className="evaluator-laps-heading">
                      <strong>Lap times</strong>
                      <span>{shownLapTimes.length}/10 completed</span>
                    </div>
                    {shownLapTimes.length > 0 ? (
                      <ol className="evaluator-lap-times">
                        {shownLapTimes.map((lapTime, index) => (
                          <li key={`${index}-${lapTime}`}><span>Lap {index + 1}</span><strong>{lapTime.toFixed(2)} s</strong></li>
                        ))}
                      </ol>
                    ) : <p className="meta">No completed laps yet.</p>}
                    <div className="evaluator-ten-lap-time">
                      <span>Best 10-lap time</span>
                      <strong>{shownTenLapTime != null ? `${shownTenLapTime.toFixed(2)} s` : `Pending (${shownLapTimes.length}/10 laps)`}</strong>
                    </div>
                  </div>
                  {evaluation?.state === "running" && evaluation.latest_result && (
                    <p className="meta">{liveEvaluationResult ? "Live pace and distance update during this snapshot; checkpoint score updates when it finishes." : "Latest completed evaluation remains visible while this snapshot runs."}</p>
                  )}
                </>
              ) : <p className="meta">{evaluation?.state === "running" ? "Waiting for live evaluator telemetry." : "No evaluation result yet."}</p>}
                </div>
              </details>
            </>
          ) : (
            <dl className="fleet-stats">
              <div>
                <dt>Episode #</dt>
                <dd>{fleet?.episode ?? "—"}</dd>
              </div>
              <div>
                <dt>Steps</dt>
                <dd>{fleet?.step ?? "—"}</dd>
              </div>
              <div>
                <dt>Return</dt>
                <dd>
                  {selected.episode_return != null
                    ? selected.episode_return.toFixed(2)
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Speed</dt>
                <dd>
                  {selected.speed != null
                    ? `${selected.speed.toFixed(2)} m/s`
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Forward speed (signed)</dt>
                <dd>
                  {selected.v_long != null
                    ? `${selected.v_long.toFixed(2)} m/s`
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Throttle command</dt>
                <dd>
                  {selected.throttle_command != null
                    ? selected.throttle_command.toFixed(2)
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Steering command</dt>
                <dd>
                  {selected.steering_command != null
                    ? selected.steering_command.toFixed(2)
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Laps this episode</dt>
                <dd>
                  {selected.lap_supported
                    ? selected.lap_count ?? 0
                    : "unavailable"}
                </dd>
              </div>
              {selected.lap_supported && (
                <>
                  <div>
                    <dt>Current lap</dt>
                    <dd>
                      {selected.lap_elapsed_s != null
                        ? `${selected.lap_elapsed_s.toFixed(1)} s`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Last lap</dt>
                    <dd>
                      {selected.last_lap_time_s != null
                        ? `${selected.last_lap_time_s.toFixed(2)} s`
                        : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Best this episode</dt>
                    <dd>
                      {selected.best_lap_time_s != null
                        ? `${selected.best_lap_time_s.toFixed(2)} s`
                        : "—"}
                    </dd>
                  </div>
                </>
              )}
              {selected.frontier_progress_m != null && (
                <>
                  <div>
                    <dt>Frontier progress</dt>
                    <dd>{selected.frontier_progress_m.toFixed(2)} m</dd>
                  </div>
                  <div>
                    <dt>Frontier speed</dt>
                    <dd>{(selected.frontier_speed_mps ?? 0).toFixed(2)} m/s</dd>
                  </div>
                  <div>
                    <dt>Since last push</dt>
                    <dd>{(selected.time_since_frontier_push_s ?? 0).toFixed(1)} s</dd>
                  </div>
                </>
              )}
              {selected.current_progress_m != null && (
                <div>
                  <dt>Current route position</dt>
                  <dd>
                    {selected.current_progress_m.toFixed(2)} m
                    {selected.signed_route_delta_m != null &&
                      ` (${selected.current_route_speed_mps != null && selected.current_route_speed_mps >= 0 ? "+" : ""}${selected.current_route_speed_mps?.toFixed(2) ?? "—"} m/s route)`}
                  </dd>
                </div>
              )}
              {selected.reward_components && (
                <>
                  <div><dt>Reward · progress</dt><dd>{(selected.reward_components.route_progress ?? 0).toFixed(3)}</dd></div>
                  <div><dt>Reward · reverse gate</dt><dd>{(selected.reward_components.reverse_direction_gate ?? 0).toFixed(3)}</dd></div>
                  <div><dt>Reward · reverse</dt><dd>{(selected.reward_components.backward_motion ?? 0).toFixed(3)}</dd></div>
                  <div><dt>Reward · time</dt><dd>{(selected.reward_components.time_cost ?? 0).toFixed(3)}</dd></div>
                  <div><dt>Reward · collision</dt><dd>{(selected.reward_components.collision ?? 0).toFixed(1)}</dd></div>
                  <div><dt>Reward · failed episode</dt><dd>{(selected.reward_components.episode_failure ?? 0).toFixed(1)}</dd></div>
                  <div><dt>Reward · total</dt><dd>{(selected.reward_components.total ?? 0).toFixed(3)}</dd></div>
                </>
              )}
              <div>
                <dt>Pose (x,z)</dt>
                <dd>
                  {selected.pose
                    ? `${selected.pose[0].toFixed(2)}, ${selected.pose[1].toFixed(2)}`
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Yaw</dt>
                <dd>
                  {selected.yaw != null ? `${selected.yaw.toFixed(3)} rad` : "—"}
                </dd>
              </div>
              <div>
                <dt>Sample</dt>
                <dd data-stale={stale ? "1" : "0"}>
                  {stale ? "stale" : "fresh"}
                </dd>
              </div>
              <div>
                <dt>LiDAR beams</dt>
                <dd>{selected.lidar?.length ?? 0}</dd>
              </div>
            </dl>
          )}
        </aside>
      </div>

      <section className="training-progress-panel" aria-label="Training progress">
        <div className="training-progress-heading">
          <div>
            <h3>Training progress</h3>
          </div>
          <div className="training-progress-latest">
            <span>Current rollout reward</span>
            <strong>{rewardSamples.length ? rewardSamples[rewardSamples.length - 1].reward.toFixed(2) : "—"}</strong>
          </div>
          <button
            type="button"
            className="btn evaluator-toggle"
            aria-expanded={showTrainingProgress}
            onClick={() => setShowTrainingProgress((visible) => !visible)}
          >
            {showTrainingProgress ? "Hide" : "Show"}
          </button>
        </div>
        {showTrainingProgress && <>
        {rewardRange && rewardTrend.length > 0 ? (
          <>
            <svg className="reward-chart" viewBox="0 0 720 170" role="img" aria-label="Estimated total reward accumulated in each PPO rollout">
              {[17, 81, 145].map((y) => (
                <line key={y} x1="60" x2="712" y1={y} y2={y} className="reward-chart-grid" />
              ))}
              {[rewardRange.max, (rewardRange.max + rewardRange.min) / 2, rewardRange.min].map((value, index) => (
                <text key={index} x="53" y={[21, 85, 149][index]} textAnchor="end" className="reward-chart-axis-label">{value.toFixed(2)}</text>
              ))}
              <polyline points={rewardPolyline} className="reward-chart-line" />
              {rewardChartPoints.map(({ point, index }) => {
                const span = rewardRange.max - rewardRange.min;
                const normalized = span > 1e-8 ? (point.reward - rewardRange.min) / span : 0.5;
                return <circle key={point.step} cx={64 + index / Math.max(1, rewardTrend.length - 1) * 648} cy={145 - normalized * 128} r="3" className="reward-chart-point"><title>{`PPO rollout ending at step ${point.step.toLocaleString()} · estimated reward ${point.reward.toFixed(2)}${point.complete ? " · complete" : " · in progress"}`}</title></circle>;
              })}
            </svg>
            <div className="reward-chart-labels">
              <span>step {rewardTrend[0]?.step.toLocaleString() ?? "—"}</span>
              <span>range {rewardRange.min.toFixed(2)} to {rewardRange.max.toFixed(2)}</span>
              <span>{rewardTrend.filter((sample) => sample.complete).length} completed PPO rollouts</span>
              <span>through step {metrics?.step.toLocaleString() ?? "—"}</span>
            </div>
          </>
        ) : (
          <p className="meta reward-chart-empty">Waiting for reward samples from the current PPO rollout.</p>
        )}
        </>}
      </section>

      <section className="training-progress-panel ppo-diagnostics-panel" aria-label="PPO diagnostics">
        <div className="training-progress-heading">
          <div>
            <h3>PPO diagnostics</h3>
            <p className="meta">Saved TensorBoard optimizer and policy signals · refreshed every 10 seconds</p>
          </div>
          <span className="ppo-update-count">
            {ppoSeries.updateCount ? `${ppoSeries.updateCount} updates · through step ${ppoSeries.latestStep?.toLocaleString()}` : "Waiting for diagnostics"}
          </span>
        </div>
        <p className="ppo-diagnostics-note">
          These explain how PPO is learning; they are not driving scores. Compare them with rollout reward and evaluator results above.
          {ppoHistoryError && <> Saved history is unavailable: {ppoHistoryError}</>}
        </p>
        {PPO_METRIC_GROUPS.map((group) => (
          <div className="ppo-metric-group" key={group.title}>
            <h4>{group.title}</h4>
            <div className="ppo-metric-grid">
              {group.metrics.map((metric) => (
                <PpoMetricCard key={metric.key} metric={metric} points={ppoSeries.series[metric.key] ?? []} />
              ))}
            </div>
          </div>
        ))}
      </section>

      <section className="training-progress-panel evaluator-history-panel" aria-label="Evaluator performance">
        <div className="training-progress-heading">
          <div>
            <h3>Median snapshot score</h3>
            <p className="meta">Each point is the median selected score across the configured attempts · metric: {selectionMetricLabel}. The mean and best attempt are shown in the evaluator details above.</p>
          </div>
          <strong className="evaluator-history-count">{evaluatorScores.length} evaluations</strong>
        </div>
        {evaluatorScores.length > 1 && evaluatorRange ? (
          <>
            <svg className="reward-chart" viewBox="0 0 720 170" role="img" aria-label="Evaluator score across all completed evaluations">
              {[17, 81, 145].map((y) => <line key={y} x1="60" x2="712" y1={y} y2={y} className="reward-chart-grid" />)}
              {[evaluatorRange.max, (evaluatorRange.max + evaluatorRange.min) / 2, evaluatorRange.min].map((value, index) => (
                <text key={index} x="53" y={[21, 85, 149][index]} textAnchor="end" className="reward-chart-axis-label">{value.toFixed(2)}</text>
              ))}
              <polyline points={evaluatorPolyline} className="evaluator-chart-line" />
              {evaluatorChartPoints.map(({ point, index }) => {
                const span = evaluatorRange.max - evaluatorRange.min;
                const normalized = span > 1e-8 ? (point.score - evaluatorRange.min) / span : 0.5;
                return <circle key={`${point.index}-${point.step}`} cx={64 + index / (evaluatorScores.length - 1) * 648} cy={145 - normalized * 128} r="3.2" className={point.improved ? "evaluator-chart-point improved" : "evaluator-chart-point"}>
                  <title>{`Evaluation ${index + 1} · step ${point.step.toLocaleString()} · score ${point.score.toFixed(3)}${point.improved ? " · improved" : ""}`}</title>
                </circle>;
              })}
            </svg>
            <div className="reward-chart-labels"><span>Evaluation 1</span><span>{evaluatorScores.length} total saved runs</span><span>Evaluation {evaluatorScores.length}</span></div>
          </>
        ) : evaluatorScores.length === 1 ? (
          <p className="meta reward-chart-empty">First score: {evaluatorScores[0]?.score.toFixed(3)} at step {evaluatorScores[0]?.step.toLocaleString()}. The chart will connect it to later evaluations.</p>
        ) : (
          <p className="meta reward-chart-empty">No completed evaluator runs have been saved yet.</p>
        )}
      </section>
    </section>
  );
}
