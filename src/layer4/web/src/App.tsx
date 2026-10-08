import { lazy, Suspense, useEffect, useState } from "react";
import { shutdownHub } from "./api";
import { TrainPage } from "./pages/Train";
import { startStore, useHubStore } from "./store";
import "./styles.css";

const WatchPage = lazy(() =>
  import("./pages/Watch").then((m) => ({ default: m.WatchPage }))
);
const ReplayPage = lazy(() =>
  import("./pages/Replay").then((m) => ({ default: m.ReplayPage }))
);

type Page = "train" | "watch" | "replay";

function trainingNotice(status: ReturnType<typeof useHubStore>["status"]) {
  if (!status || (status.state !== "exited" && status.state !== "error")) return null;
  const key = [
    status.state,
    status.started_at ?? "",
    status.exit_code ?? "",
    status.stop_reason ?? "",
    status.stopped_containers.join(","),
    status.error ?? "",
    status.cleanup_error ?? "",
  ].join(":");

  let title = "Training stopped";
  let detail = "The run has ended.";
  let level: "info" | "success" | "error" = "info";
  const terminatedBySignal = status.exit_code === -15;
  if (status.state === "error" || (status.exit_code != null && status.exit_code !== 0 && !terminatedBySignal)) {
    title = "Training failed";
    detail = status.error || `Train exited with code ${status.exit_code}.`;
    if (status.log_path) detail += ` Log: ${status.log_path}`;
    level = "error";
  } else if (terminatedBySignal) {
    title = status.stop_reason === "user_requested" ? "Training stopped" : "Training terminated";
    detail = status.stop_reason === "user_requested"
      ? "Training received a stop request and shut down. The final model may be available in the run folder."
      : "The process received SIGTERM (exit -15). This is a termination signal, so it does not identify its sender or indicate a Python exception.";
    if (status.log_path) detail += ` Log: ${status.log_path}`;
  } else if (status.stop_reason === "operator_stop" || status.stop_reason === "user_requested") {
    title = "Training stopped";
    detail = "Stopped by request. Check the run artifacts to confirm which checkpoints were saved.";
    level = "info";
  } else if (status.stop_reason === "evaluation_complete") {
    title = "Evaluation finished";
    detail = "Open Replay for the attempt result. Completion does not necessarily mean a valid race score.";
  } else if (status.stop_reason) {
    title = "Training finished";
    detail = status.stop_reason === "lap_target"
      ? "A car reached the lap target."
      : status.stop_reason === "max_duration"
        ? "The run duration limit was reached."
        : status.stop_reason === "progress_plateau"
          ? "Frontier improvement plateaued."
          : "The timestep limit was reached.";
    level = "success";
  }

  return { key, title, detail, level, status };
}

export function App() {
  const [page, setPage] = useState<Page>("train");
  const [shuttingDown, setShuttingDown] = useState(false);
  const [dismissedNotice, setDismissedNotice] = useState<string | null>(null);
  const { conn, status } = useHubStore();
  const notice = trainingNotice(status);

  useEffect(() => {
    startStore();
  }, []);

  async function onShutdown() {
    if (shuttingDown) return;
    const ok = window.confirm(
      "Shut down Mission Control?\n\nThis stops the hub process. Training (if running) will be stopped first."
    );
    if (!ok) return;
    setShuttingDown(true);
    try {
      await shutdownHub();
    } catch {
      // Hub may die before the response is fully read — treat as success.
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="brand">
          AiCar <span>Mission Control</span>
        </h1>
        <div className="header-actions">
          <div className="conn" data-state={conn}>
            {conn}
          </div>
          <button
            type="button"
            className="btn danger shutdown-btn"
            disabled={shuttingDown}
            onClick={() => void onShutdown()}
          >
            {shuttingDown ? "Shutting down…" : "Shut down"}
          </button>
        </div>
      </header>

      <nav className="nav">
        {(
          [
            ["train", "Train"],
            ["watch", "Watch"],
            ["replay", "Replay"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            data-active={page === id}
            onClick={() => setPage(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      {notice && dismissedNotice !== notice.key && (
        <aside
          className="global-notice"
          data-level={notice.level}
          role={notice.level === "error" ? "alert" : "status"}
          aria-live={notice.level === "error" ? "assertive" : "polite"}
        >
          <div className="global-notice-copy">
            <strong>{notice.title}</strong>
            <span>{notice.detail}</span>
            {notice.status.stopped_containers.length > 0 && (
              <span>
                Last run stopped simulator containers ({notice.status.stopped_containers.join(", ")}).
                They will restart when you start training again.
              </span>
            )}
            {notice.status.cleanup_error && (
              <span>Simulator cleanup issue: {notice.status.cleanup_error}</span>
            )}
          </div>
          <button
            type="button"
            className="global-notice-dismiss"
            aria-label="Dismiss notification"
            onClick={() => setDismissedNotice(notice.key)}
          >
            ×
          </button>
        </aside>
      )}

      <div hidden={page !== "train"}>
        <TrainPage />
      </div>
      {page === "watch" && (
        <Suspense
          fallback={
            <section className="panel">
              <h2>Watch</h2>
              <p className="lede">Loading…</p>
            </section>
          }
        >
          <WatchPage />
        </Suspense>
      )}
      {page === "replay" && (
        <Suspense fallback={<section className="panel"><h2>Replay</h2><p className="lede">Loading…</p></section>}>
          <ReplayPage />
        </Suspense>
      )}
    </div>
  );
}
