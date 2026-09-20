import { FormEvent, useEffect, useState } from "react";
import { Settings, getSettings, putSettings } from "../api";

const empty: Settings = {
  n_envs: 1,
  base_port: 4567,
  timesteps: 10000,
  out: null,
  run_name: null,
  seed: 0,
  device: "auto",
  resume: null,
  headless: true,
  auto_launch: true,
  connect_timeout: 90,
  frame_skip: 1,
  max_episode_steps: 0,
  stagnation_speed_threshold: 0.15,
  stagnation_steps: 200,
  forward_scale: 1,
  collision_penalty: 0,
  slip_penalty: 0,
  steer_jerk_penalty: 0,
  telemetry_every_n: 200,
  fleet_hz: 15,
  lidar_display_beams: 120,
  telemetry_lidar_max_envs: 4,
  docker_mode: false,
  stop_sims_on_train_exit: true,
  stop_stack_on_train_exit: false,
};

type NumKey = {
  [K in keyof Settings]: Settings[K] extends number ? K : never;
}[keyof Settings];

export function SettingsPage() {
  const [form, setForm] = useState<Settings>(empty);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getSettings()
      .then((s) => {
        if (!cancelled) setForm({ ...empty, ...s });
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

  if (loading) {
    return (
      <section className="panel">
        <h2>Settings</h2>
        <p className="lede">Loading…</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>Settings</h2>
      <p className="lede">
        Maps 1:1 to <code>python -m src.layer3.train</code> flags. Hub-only fields
        control telemetry publish rates.
      </p>
      <form onSubmit={onSave}>
        <div className="grid">
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
              onChange={(e) => setForm((f) => ({ ...f, headless: e.target.checked }))}
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
            forward_scale
            <input
              type="number"
              step="any"
              value={form.forward_scale}
              onChange={(e) => num("forward_scale", e.target.value)}
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
        {msg && <p className="msg ok">{msg}</p>}
        {err && <p className="msg err">{err}</p>}
      </form>
    </section>
  );
}
