import { useState } from "react";
import { getSettings, startTrain, stopTrain } from "../api";
import { useHubStore } from "../store";

export function TrainPage() {
  const { status } = useHubStore();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const state = status?.state ?? "idle";
  const running = state === "running" || state === "starting" || state === "stopping";

  async function onStart() {
    setErr(null);
    setBusy(true);
    try {
      const settings = await getSettings();
      await startTrain(settings);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  async function onStop() {
    setErr(null);
    setBusy(true);
    try {
      await stopTrain();
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <h2>Train</h2>
      <p className="lede">
        Starts <code>python -m src.layer3.train</code> with the saved Settings argv.
        One job at a time.
      </p>

      <p>
        Status{" "}
        <span className="badge" data-state={state}>
          {state}
        </span>
      </p>

      <div className="meta">
        <div>
          <strong>pid</strong> {status?.pid ?? "—"}
        </div>
        <div>
          <strong>started</strong> {status?.started_at ?? "—"}
        </div>
        <div>
          <strong>exit</strong> {status?.exit_code ?? "—"}
        </div>
        <div>
          <strong>log</strong> {status?.log_path ?? "—"}
        </div>
        <div>
          <strong>hub</strong> {status?.hub_url ?? "—"}
        </div>
        {status?.error && (
          <div>
            <strong>error</strong> {status.error}
          </div>
        )}
        {status?.cleanup_error && (
          <div>
            <strong>Docker cleanup error</strong> {status.cleanup_error}
          </div>
        )}
        {status?.stopped_containers && status.stopped_containers.length > 0 && (
          <div>
            <strong>stopped containers</strong>{" "}
            {status.stopped_containers.join(", ")}
          </div>
        )}
        {status?.argv && status.argv.length > 0 && (
          <div style={{ marginTop: "0.5rem", wordBreak: "break-all" }}>
            <strong>argv</strong> {status.argv.join(" ")}
          </div>
        )}
      </div>

      <div className="actions">
        <button
          type="button"
          className="primary"
          disabled={busy || running}
          onClick={() => void onStart()}
        >
          Start
        </button>
        <button
          type="button"
          className="danger"
          disabled={busy || !running || state === "stopping"}
          onClick={() => void onStop()}
        >
          Stop
        </button>
      </div>
      {err && <p className="msg err">{err}</p>}
    </section>
  );
}
