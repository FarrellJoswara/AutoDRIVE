import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { shutdownHub } from "./api";
import { TrainPage, type TrainPageHandle } from "./pages/Train";
import { MapsPage } from "./pages/Maps";
import { startStore, useHubStore } from "./store";
import "./styles.css";

const WatchPage = lazy(() =>
  import("./pages/Watch").then((m) => ({ default: m.WatchPage }))
);
const ReplayPage = lazy(() =>
  import("./pages/Replay").then((m) => ({ default: m.ReplayPage }))
);

type Page = "train" | "maps" | "watch" | "replay";

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
  if (status.state === "error" || (status.exit_code != null && status.exit_code !== 0)) {
    title = "Training failed";
    detail = status.error || `Train exited with code ${status.exit_code}.`;
    if (status.log_path) detail += ` Log: ${status.log_path}`;
    level = "error";
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
  const [trainBusy, setTrainBusy] = useState(false);
  const [trainReady, setTrainReady] = useState(false);
  const [dismissedNotice, setDismissedNotice] = useState<string | null>(null);
  const { conn, status } = useHubStore();
  const notice = trainingNotice(status);
  const trainRef = useRef<TrainPageHandle>(null);
  const running =
    status?.state === "running" ||
    status?.state === "starting" ||
    status?.state === "stopping";

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
          <button
            type="button"
            className="btn primary header-run-btn"
            disabled={!trainReady || trainBusy || running}
            onClick={() => void trainRef.current?.start()}
          >
            Start
          </button>
          <button
            type="button"
            className="btn danger header-run-btn"
            disabled={!trainReady || trainBusy || !running || status?.state === "stopping"}
            onClick={() => void trainRef.current?.stop()}
          >
            Stop
          </button>
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
            ["maps", "Maps"],
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
        <TrainPage
          ref={trainRef}
          onBusyChange={setTrainBusy}
          onReadyChange={setTrainReady}
        />
      </div>
      {page === "maps" && <MapsPage />}
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
