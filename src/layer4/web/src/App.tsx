import { lazy, Suspense, useEffect, useState } from "react";
import { shutdownHub } from "./api";
import { TrainPage } from "./pages/Train";
import { MapsPage } from "./pages/Maps";
import { startStore, useHubStore } from "./store";
import "./styles.css";

const WatchPage = lazy(() =>
  import("./pages/Watch").then((m) => ({ default: m.WatchPage }))
);

type Page = "train" | "maps" | "watch";

export function App() {
  const [page, setPage] = useState<Page>("train");
  const [shuttingDown, setShuttingDown] = useState(false);
  const { conn } = useHubStore();

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
            ["maps", "Maps"],
            ["watch", "Watch"],
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

      {page === "train" && <TrainPage />}
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
    </div>
  );
}
