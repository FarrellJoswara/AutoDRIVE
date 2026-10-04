import type { FleetCar } from "../api";

export interface TenLapMetrics {
  completedLaps: number;
  progressLaps: number;
  currentTimeS: number | null;
  bestCompletedTimeS: number | null;
}

/** Summarize the current episode's consecutive lap splits. */
export function tenLapMetrics(car: FleetCar): TenLapMetrics {
  const splits = (car.lap_times_s ?? []).filter(
    (time) => Number.isFinite(time) && time > 0,
  );
  const completedLaps = splits.length;
  const progressLaps = Math.min(10, completedLaps);
  const bestCompletedTimeS = completedLaps >= 10
    ? Math.min(...splits.slice(0, completedLaps - 9).map((_, start) =>
      splits.slice(start, start + 10).reduce((sum, lap) => sum + lap, 0),
    ))
    : null;

  let currentTimeS: number | null = null;
  if (car.lap_elapsed_s != null && Number.isFinite(car.lap_elapsed_s)) {
    const precedingLaps = completedLaps >= 10 ? splits.slice(-9) : splits;
    currentTimeS = precedingLaps.reduce((sum, lap) => sum + lap, car.lap_elapsed_s);
  } else if (completedLaps >= 10) {
    currentTimeS = splits.slice(-10).reduce((sum, lap) => sum + lap, 0);
  } else if (completedLaps > 0) {
    currentTimeS = splits.reduce((sum, lap) => sum + lap, 0);
  }

  return { completedLaps, progressLaps, currentTimeS, bestCompletedTimeS };
}

export function formatLapDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const minutes = Math.floor(seconds / 60);
  const remainder = (seconds % 60).toFixed(2).padStart(5, "0");
  return `${minutes}:${remainder}`;
}
