# HANDOFF — F1TENTH RL stack (read this first)

You and a friend are picking this up cold. This file is the maze map.
Everything lives under `ADSS Toolkit/autodrive_py/rl/` unless noted.
Working directory for commands: **`ADSS Toolkit/autodrive_py`** (parent of `rl/`).
Branch: **`rl/phase-1-research`**.

---

## What this project is

Gym-side **PPO** for solo F1TENTH / RoboRacer time-attack, aimed at beating a pinned **Follow-the-Gap (FTG)** baseline on sealed maps, then (later) AutoDRIVE Bridge / Jetson.

| Layer | Role |
| ----- | ---- |
| `racing_env` + `observation` | Gymnasium env: LiDAR + proprio (contracts **2.0.0**, obs_dim **186**) |
| `train_ppo` | SB3 PPO, checkpoints, `live_status`, race-best promote, early-stop |
| `control_ui` | Ugly stdlib HTTP panel — Start / Continue / Stop / Watch / TB |
| `watch` | Separate OpenCV process — lag-behind twins, **not** live Subproc poses |
| `map_pack` | train_ok / validation / holdout seals |
| `eval_*` / `compare_models` | Official race scores (`adjusted_time = lap + 10·collisions`) |

Frozen interface: [`contracts.md`](contracts.md). Longer plan: [`PLAN.md`](PLAN.md). Campaign process: [`CAMPAIGN_GOVERNANCE.md`](CAMPAIGN_GOVERNANCE.md).

---

## DO NOT TOUCH (sacred)

| Item | Why |
| ---- | --- |
| **`overnight_soak_20260917_082739`** | Protected overnight soak. **Never kill** without a deliberate Continue restart of the *same* `run_id`. |
| Early-stop floors | `select_timeout` ≥ **220**, `race_eval_every` ≥ **50k**, warmup ≥ **2**, `min_timesteps` ≥ **100k**. Do not lower defaults to “ship faster.” |
| Contracts **2.0.0** | Obs / action / λ=10 score. Refuse-load on mismatch. |
| Holdout / validation maps | Never train on sealed holdouts or the pinned validation map without explicit allow + board. |
| Dual-writer / Stop path | Start/Continue must **never** call kill-tree. Only Stop kills. `ui_selftest` mocks Stop while soak is live. |

If UI banner shows a short A/B smoke (~few k timesteps) while overnight is still alive → multi-run latch. See [Multi-run reality](#multi-run-reality).

---

## How to run (operator)

Activate venv once:

```powershell
cd "ADSS Toolkit/autodrive_py"
.\rl\.venv\Scripts\Activate.ps1
# first time: py -3.13 -m venv rl/.venv && pip install -r rl/requirements.txt
```

### Control UI

```powershell
.\rl\start_ui.ps1
# or: python -m rl.control_ui
# open http://127.0.0.1:7860/
```

Buttons: **Start training**, **Continue**, **Stop**, **Open Watch**, TensorBoard. Knobs: map, timesteps, `n_envs`, overnight preset, early-stop patience.

### Train (headless, no UI)

```powershell
.\rl\start_train.ps1
# or: python -m rl.train_ppo --map map0 --timesteps 100000 --device auto --n-envs 8
```

- Log: `rl/logs/<run_id>.log`
- Models: `rl/models/<run_id>/` (`best_model.zip`, `latest_model.zip`, `checkpoints/`, `train.lock`, `config.json`)
- Live trail: `rl/runs/<run_id>/live_status.json`

### Watch (separate process)

```powershell
python -m rl.watch --follow --map map0 --n-envs 8 --no-beams
# pin a run:
python -m rl.watch --follow --run_id overnight_soak_20260917_082739
```

Watch **lags** training on purpose. Overlay KPIs are **unofficial** — promote from eval only.

### Stop / Continue

- **Stop** in Control UI (or Ctrl+C in the train window) — kills the train process tree. **Does not** delete models.
- **Continue** resumes the focused / pinned `run_id` from the last complete checkpoint (`reset_num_timesteps=False`).
- After a crash or UI restart: set pin → Continue (see [Resume after crash](#resume-after-crash)).

---

## Where overnight soak lives

| Artifact | Path |
| -------- | ---- |
| Protected `run_id` | `overnight_soak_20260917_082739` |
| Lock | `rl/models/overnight_soak_20260917_082739/train.lock` |
| Status | `rl/runs/overnight_soak_20260917_082739/live_status.json` |
| Log | `rl/logs/overnight_soak_20260917_082739.log` |
| Operator pin | `rl/logs/CURRENT_RUN.txt` (often this run_id) |
| Narrative | [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) § Protected training run |

**Why not kill:** hours of resume-safe progress (`--resume`, same run_id). Killing without Continue loses the live process; wrong Start can dual-write or latch the UI onto a disposable smoke.

Quick alive check (do not kill):

```powershell
Get-Content "rl\runs\overnight_soak_20260917_082739\live_status.json"
# expect phase learning|validating, rising timesteps
```

---

## Key files map

| File | Job |
| ---- | --- |
| [`control_ui.py`](control_ui.py) | HTTP panel; Start/Continue/Stop; multi-run ranking via `CURRENT_RUN` + live timesteps |
| [`ui_ops.py`](ui_ops.py) | Testable operator helpers (locks, presets, banner, Continue argv) |
| [`train_ppo.py`](train_ppo.py) | PPO train loop, RaceBest, early-stop, locks |
| [`racing_env.py`](racing_env.py) | Gym env + reward / stall / collision |
| [`live_status.py`](live_status.py) | Atomic JSON trail; `find_latest_status` / `pick_status_for_operator` |
| [`watch.py`](watch.py) + [`watch_kpi.py`](watch_kpi.py) | Lag-behind viewer + race KPI overlay |
| [`map_pack.py`](map_pack.py) + `maps/map_pack.json` | Roles + seals |
| [`auto_train.py`](auto_train.py) | Thin outer chain (dry-run default); **refuses** live overnight |
| [`eval_protocol.py`](eval_protocol.py) / [`eval_cli.py`](eval_cli.py) | Official protocol |
| `start_ui.ps1` / `start_train.ps1` | One-click launchers |

Campaign / history (do not delete — organize via index):

| Doc | Role |
| --- | ---- |
| [`CAMPAIGN_GOVERNANCE.md`](CAMPAIGN_GOVERNANCE.md) | Ship gate + seats |
| [`CAMPAIGN_BOARD.md`](CAMPAIGN_BOARD.md) | Append-only ship permission |
| [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md) | Operator narrative + protected run |
| [`CAMPAIGN_UI.md`](CAMPAIGN_UI.md) | UI chrome inbox |
| [`CAMPAIGN_BUGFIX.md`](CAMPAIGN_BUGFIX.md) | Stable-core bugs |
| [`CAMPAIGN_SUGGESTIONS.md`](CAMPAIGN_SUGGESTIONS.md) | Soft ideas (not mandates) |
| [`PLAN.md`](PLAN.md) / [`PLAN_PROGRESS.md`](PLAN_PROGRESS.md) | Longer roadmap |
| [`ideas_review/`](ideas_review/) | Personality deep-dives |
| [`IDEAS_TRIMMED.md`](IDEAS_TRIMMED.md) / [`IDEAS_MEGA.md`](IDEAS_MEGA.md) | Idea archaeology |

---

## Glossary — phases & locks

### `live_status.json` phases

| Phase | Meaning | If it looks stuck… |
| ----- | ------- | ------------------ |
| `starting` | Spawn in progress; timesteps may be blank | Wait; check log |
| `learning` | PPO rolling | Timesteps / `steps_per_sec` should move |
| `validating` | Mid-train race eval (can take **minutes**; ~220s select timeout) | **Not hung.** Timesteps freeze on purpose. |
| `early_stopped` | Patience exhausted — clean exit | Continue only if you intend a new budget |
| (missing / stale) | Trail not updating | Check process + `train.lock`; file age >90s → TRAINSTALE in Watch |

### Locks & pins

| Thing | Path / meaning |
| ----- | -------------- |
| `train.lock` | `models/<run_id>/train.lock` — live PID claim. Start/Continue refuse if another live lock exists. |
| `CURRENT_RUN.txt` | `logs/CURRENT_RUN.txt` — operator pin so UI ranks this run over short smokes. |
| Dual-writer refuse | Two writers on one `run_id` = refuse. Never “fix” by killing overnight from Start. |

Sacred overnight preset knobs (UI “Overnight”): patience **5**, min_improve **0.5**, eval every **50k**, warmup **2**, min_ts **100k**, select **220**, budget **OFF** (unlimited safety ceiling).

---

## Multi-run reality

Operators often have **several** live `train.lock`s (overnight + disposable A/B smokes).

Symptoms if the UI latches wrong:

- Banner shows a **new** `run_id` at ~few thousand timesteps while overnight is still at hundreds of k.
- Watch / Continue / Stop feel aimed at the wrong process.

What to check:

1. `rl/logs/CURRENT_RUN.txt` — should be the run you care about (usually overnight).
2. `find_live_run_locks` / Models list — multiple `[locked pid=…]`.
3. Prefer Focus / pin (when UI exposes it) or rewrite `CURRENT_RUN.txt`, then refresh status.
4. Ranking preference (already in Control): **pin → highest live timesteps → `--resume`/`--unlimited` → leaf PID**.

Do **not** “simplify” by killing multi-train support or sweeping overnight with an unmocked Stop during selftest.

---

## Resume after crash

1. Confirm soak / target still has `models/<run_id>/` with checkpoints.
2. Pin: write `run_id` into `rl/logs/CURRENT_RUN.txt` (one line).
3. Clear **stale** lock only if PID is dead (`clear_stale_lock` / UI helper) — never clear a live overnight lock.
4. Control UI → **Continue** (same maps / n_envs / sacred knobs as before when possible).
5. Confirm `live_status` phase returns to `learning`/`validating` and timesteps advance from the checkpoint (not 0).

If the process died mid-validate: Continue from last complete `ppo_*_steps.zip`; learning was not lost solely because phase said `validating`.

---

## Test commands (matter for handoff)

From `ADSS Toolkit/autodrive_py` with `rl/.venv`:

```powershell
python -m rl.ui_selftest          # Control helpers + HTTP; Stop mocked — safe with overnight live
python -m rl._bugfix_p0_checks    # Atomic zip + exclusive train.lock
python -m rl.test_watch_overlay   # Headless Watch KPI smoke (~45s)
python -m rl.test_status_pick     # Multi-run live_status ranking (pin / timesteps)
python -m rl.test_map_pack        # Pack roles / seals
```

Auto-train refuse path (idle / dry-run only — never against live overnight):

```powershell
python -m rl.auto_train --dry-run
```

---

## If X looks wrong → check Y

| Symptom | Check |
| ------- | ----- |
| Banner stuck on tiny timesteps | Multi-run latch → `CURRENT_RUN.txt` + live locks |
| Phase=`validating` forever-looking | Mid-train eval; wait; read `msg` / log for map + timeout |
| Start refused | Live `train.lock` or holdout map — intentional |
| Continue won’t start | Stale lock with dead PID, or missing checkpoint |
| Watch map mismatch banner | Watched map ∉ run `config.json` maps / hash changed |
| `ep_rew_mean` looks great, race score bad | Reward ≠ race score; trust official `adjusted_time` |
| GPU “idle” | Expected (CPU VecEnv); not a mandate to AMP/compile |
| Selftest / Stop killed overnight | Bug — Stop must be mocked in tests; restore via Continue |

---

## Doc index (start here → history)

1. **You are here:** [`HANDOFF.md`](HANDOFF.md)
2. Day-to-day operator + quickstart: [`README.md`](README.md)
3. Tonight’s campaign narrative: [`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md)
4. May I ship?: [`CAMPAIGN_BOARD.md`](CAMPAIGN_BOARD.md) + [`CAMPAIGN_GOVERNANCE.md`](CAMPAIGN_GOVERNANCE.md)
5. Roadmap: [`PLAN.md`](PLAN.md) → [`PLAN_PROGRESS.md`](PLAN_PROGRESS.md)
6. Contracts: [`contracts.md`](contracts.md)
7. Soft / UI / bugfix inboxes: `CAMPAIGN_SUGGESTIONS.md`, `CAMPAIGN_UI.md`, `CAMPAIGN_BUGFIX.md`
8. Personality archives: `ideas_review/`, `IDEAS_TRIMMED.md`, `CAMPAIGN_CRITIQUE.md`, `CAMPAIGN_RESEARCH.md`

History is append-only for board/suggestions — **organize with links, don’t delete.**
