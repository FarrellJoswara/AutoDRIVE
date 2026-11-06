import { useEffect, useRef, useState } from "react";
import { getLapHistory, type FleetCar, type LapHistoryResponse } from "../api";

function formatLapDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const roundedMilliseconds = Math.round(seconds * 1000);
  const minutes = Math.floor(roundedMilliseconds / 60_000);
  const remainder = (roundedMilliseconds % 60_000) / 1000;
  return `${minutes}:${remainder.toFixed(3).padStart(6, "0")}`;
}

export function LiveLapPanel({
  runId,
  car,
  age,
  disconnected,
}: {
  /** Actual training or replay session ID, never a model/checkpoint ID. */
  runId: string;
  car: FleetCar | null;
  age: { stale: boolean; text: string };
  disconnected: boolean;
}) {
  const [history, setHistory] = useState<LapHistoryResponse | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [bestPulse, setBestPulse] = useState(false);
  const previousCount = useRef<number | null>(null);
  const bestPulseTimer = useRef<number | null>(null);
  const envId = car?.env_id ?? null;
  const lapSession = car?.lap_session;
  const completedCount = lapSession?.completed_laps ?? null;
  const lastSplit = lapSession?.latest_lap_time_s ?? car?.last_lap_time_s ?? null;
  const sessionBest = lapSession?.best_lap_time_s ?? null;
  const completedDelta = lapSession?.last_lap_delta_s ?? null;
  const comparableSessions = history
    ? history.previous_sessions.filter((session) => {
      const currentTrack = history.track_id?.trim();
      const currentLayout = history.layout_id?.trim();
      const currentDirection = history.direction?.trim();
      const sessionTrack = session.track_id?.trim();
      const sessionLayout = session.layout_id?.trim();
      const sessionDirection = session.direction?.trim();
      return Boolean(session.run_id && session.run_id !== history.run_id
        && currentTrack && currentLayout && currentDirection
        && sessionTrack && sessionLayout && sessionDirection
        && sessionTrack === currentTrack
        && sessionLayout === currentLayout
        && sessionDirection === currentDirection);
    })
    : [];

  useEffect(() => {
    previousCount.current = null;
    setBestPulse(false);
    if (bestPulseTimer.current != null) window.clearTimeout(bestPulseTimer.current);
    bestPulseTimer.current = null;
    setHistory(null);
    setHistoryError(null);
    setDetailsOpen(false);
  }, [runId, envId]);

  useEffect(() => () => {
    if (bestPulseTimer.current != null) window.clearTimeout(bestPulseTimer.current);
  }, []);

  useEffect(() => {
    if (completedCount == null || completedCount <= 0) return;
    const count = completedCount;
    const previous = previousCount.current;
    previousCount.current = count;
    if (previous == null || count <= previous || completedDelta == null || completedDelta >= 0) return;
    setBestPulse(true);
    if (bestPulseTimer.current != null) window.clearTimeout(bestPulseTimer.current);
    bestPulseTimer.current = window.setTimeout(() => {
      bestPulseTimer.current = null;
      setBestPulse(false);
    }, 2200);
  }, [completedCount, completedDelta]);

  useEffect(() => {
    if (!detailsOpen || envId == null) return;
    let cancelled = false;
    setHistoryLoading(true);
    setHistoryError(null);
    void getLapHistory(runId, envId).then((result) => {
      if (!cancelled) setHistory(result);
    }).catch((reason: unknown) => {
      if (!cancelled) setHistoryError(`Session details unavailable: ${String(reason)}`);
    }).finally(() => {
      if (!cancelled) setHistoryLoading(false);
    });
    return () => { cancelled = true; };
  }, [detailsOpen, runId, envId, completedCount, historyRefresh]);

  const availability = disconnected
    ? "Disconnected · showing last received values"
    : age.stale
      ? `Stale · showing last received values (${age.text})`
      : age.text;
  return <section className="live-lap-panel" aria-label="Live lap timing" data-new-best={bestPulse}>
    <div className="live-lap-heading">
      <div><h3>Lap timing{car ? ` · Env ${car.env_id}` : ""}</h3><span className="meta" data-stale={disconnected || age.stale}>{availability}</span></div>
      {bestPulse && <span className="lap-best-pulse" role="status" aria-live="polite">New session best</span>}
    </div>
    {car ? <>
      <div className="live-lap-primary">
        <div><span>Latest lap</span><strong>{formatLapDuration(lastSplit)}</strong></div>
        <div className={bestPulse ? "is-new-best" : ""}><span>Session best</span><strong>{formatLapDuration(sessionBest)}</strong></div>
      </div>
      <div className="live-lap-secondary">
        <span>Current lap <strong>{lapSession?.current_lap_number ?? ((car.lap_count ?? completedCount ?? 0) + 1)}</strong></span>
        <span>Elapsed <strong>{formatLapDuration(lapSession?.current_lap_elapsed_s ?? car.lap_elapsed_s)}</strong></span>
        <span>Last lap delta <strong className={completedDelta == null ? "" : completedDelta < 0 ? "lap-delta-faster" : completedDelta > 0 ? "lap-delta-slower" : ""}>
          {completedDelta == null ? "—" : `${completedDelta > 0 ? "+" : ""}${completedDelta.toFixed(3)} s`}
        </strong></span>
      </div>
      <button className="live-lap-details-btn" type="button" onClick={() => setDetailsOpen((open) => !open)} aria-expanded={detailsOpen}>
        {detailsOpen ? "Hide session details" : "Session details · all laps and comparisons"}
      </button>
      {detailsOpen && <div className="lap-history-details">
        {historyLoading && <p className="meta">Loading lap history…</p>}
        {historyError && <><p className="error" role="alert">{historyError}</p><button className="live-lap-details-btn" type="button" onClick={() => setHistoryRefresh((value) => value + 1)}>Retry session details</button></>}
        {history && <>
          <p className="meta">Track {history.track_id ?? "unknown"} · layout {history.layout_id ?? "unknown"} · {history.direction ?? "unknown direction"} · session {history.run_id} · Env {history.env_id}</p>
          <h4>Current session</h4>
          {history.laps.length ? <ol className="lap-history-list">{history.laps.map((lap) => <li key={`${lap.episode}-${lap.lap_number}`}><span>Lap {lap.lap_number} · episode {lap.episode}</span><strong>{formatLapDuration(lap.time_s)}</strong></li>)}</ol> : <p className="meta">No completed laps in this session yet.</p>}
          <h4>Previous sessions · matching track, layout, and direction</h4>
          {comparableSessions.length ? comparableSessions.map((session) => <section className="previous-lap-session" key={session.run_id}>
            <p className="meta">{session.run_id} · {session.track_id} / {session.layout_id} / {session.direction}{session.completed_at ? ` · ${new Date(session.completed_at).toLocaleString()}` : ""}</p>
            <ol className="lap-history-list">{session.laps.map((lap) => <li key={`${lap.episode}-${lap.lap_number}`}><span>Lap {lap.lap_number} · episode {lap.episode}</span><strong>{formatLapDuration(lap.time_s)}</strong></li>)}</ol>
          </section>) : <p className="meta">No previous matching sessions.</p>}
        </>}
      </div>}
      {!lapSession && <p className="live-lap-note">Session timing is not available in this telemetry yet. The latest episode lap and live elapsed time are shown when reported.</p>}
    </> : <p className="meta">Waiting for car telemetry.</p>}
  </section>;
}
