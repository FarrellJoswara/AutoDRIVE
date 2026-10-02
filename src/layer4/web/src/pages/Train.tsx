import { FormEvent, useEffect, useState } from "react";
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

const empty: Settings = {
  n_envs: 1,
  base_port: 4567,
  timesteps: 50000,
  out: null,
  run_name: null,
  seed: 0,
  device: "auto",
  resume: null,
  headless: true,
  auto_launch: true,
  connect_timeout: 90,
  frame_skip: 4,
  max_episode_steps: 1000,
  stagnation_speed_threshold: 0.15,
  stagnation_steps: 50,
  frontier_stagnation_seconds: 5,
  forward_scale: 1,
  route_progress_scale: 10,
  collision_penalty: -5,
  slip_penalty: 0.2,
  steer_jerk_penalty: 0.05,
  telemetry_every_n: 200,
  fleet_hz: 15,
  lidar_display_beams: 120,
  telemetry_lidar_max_envs: 4,
  docker_mode: false,
  stop_sims_on_train_exit: true,
  stop_stack_on_train_exit: false,
  map_id: "none",
};

type NumKey = {
  [K in keyof Settings]: Settings[K] extends number ? K : never;
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

export function TrainPage() {
  const { status } = useHubStore();
  const [form, setForm] = useState<Settings>(empty);
  const [mapChoices, setMapChoices] = useState<MapCatalogEntry[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const state = status?.state ?? "idle";
  const running =
    state === "running" || state === "starting" || state === "stopping";
  const stoppedSims =
    !!status?.stopped_containers && status.stopped_containers.length > 0;
  const selectedTrainMap =
    mapChoices.find((m) => m.id === form.map_id) ?? null;

  useEffect(() => {
    let cancelled = false;
    Promise.all([getSettings(), fetchMaps().catch(() => [] as MapCatalogEntry[])])
      .then(([s, maps]) => {
        if (cancelled) return;
        const choices = runnableMaps(maps);
        setMapChoices(choices);
        const merged = { ...empty, ...s };
        if (!choices.some((m) => m.id === merged.map_id)) {
          merged.map_id = "none";
        }
        setForm(merged);
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
      setForm({ ...empty, ...res.settings });
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
    try {
      const res = await putSettings(form);
      const saved = { ...empty, ...res.settings };
      setForm(saved);
      await startTrain(saved);
      setMsg(`Started — map locked: ${saved.map_id || "none"}`);
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

  if (loading) {
    return (
      <section className="panel">
        <h2>Train</h2>
        <p className="lede">Loading…</p>
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
        <div className="actions" style={{ marginTop: 0 }}>
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
      </div>

      <p className="lede">
        Start locks in the map below (activates physics if needed), then runs{" "}
        <code>python -m src.layer3.train</code>. Prep meshes on the Maps tab.
      </p>

      {state === "exited" && stoppedSims && (
        <p className="msg err">
          Last run stopped compose sims ({status?.stopped_containers?.join(", ")}
          ). Hit Start again — the hub will restart them before training.
        </p>
      )}
      {state === "exited" && status?.exit_code != null && status.exit_code !== 0 && (
        <p className="msg err">
          Train exited with code {status.exit_code}
          {status.log_path ? ` — see ${status.log_path}` : ""}.
        </p>
      )}

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
                    const next = { ...f, map_id };
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

          <div className="section-title">Train / job</div>
          <label className="field">
            n_envs
            <input
              type="number"
              min={1}
              value={form.n_envs}
              onChange={(e) => num("n_envs", e.target.value)}
            />
          </label>
          <label className="field">
            base_port
            <input
              type="number"
              value={form.base_port}
              onChange={(e) => num("base_port", e.target.value)}
            />
          </label>
          <label className="field">
            timesteps
            <input
              type="number"
              min={1}
              value={form.timesteps}
              onChange={(e) => num("timesteps", e.target.value)}
            />
          </label>
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
            max_episode_steps
            <input
              type="number"
              min={0}
              value={form.max_episode_steps}
              onChange={(e) => num("max_episode_steps", e.target.value)}
            />
          </label>
          <label className="field">
            stagnation_speed_threshold
            <input
              type="number"
              step="any"
              value={form.stagnation_speed_threshold}
              onChange={(e) => num("stagnation_speed_threshold", e.target.value)}
            />
          </label>
          <label className="field">
            stagnation_steps
            <input
              type="number"
              value={form.stagnation_steps}
              onChange={(e) => num("stagnation_steps", e.target.value)}
            />
          </label>
          <label className="field">
            frontier_stagnation_seconds
            <input
              type="number"
              min={0}
              step="any"
              value={form.frontier_stagnation_seconds}
              onChange={(e) => num("frontier_stagnation_seconds", e.target.value)}
            />
          </label>
          <label className="field">
            forward_scale
            <input
              type="number"
              step="any"
              value={form.forward_scale}
              onChange={(e) => num("forward_scale", e.target.value)}
            />
          </label>
          <label className="field">
            route_progress_scale
            <input
              type="number"
              min={0}
              step="any"
              value={form.route_progress_scale}
              onChange={(e) => num("route_progress_scale", e.target.value)}
            />
          </label>
          <label className="field">
            collision_penalty
            <input
              type="number"
              step="any"
              value={form.collision_penalty}
              onChange={(e) => num("collision_penalty", e.target.value)}
            />
          </label>
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
}
