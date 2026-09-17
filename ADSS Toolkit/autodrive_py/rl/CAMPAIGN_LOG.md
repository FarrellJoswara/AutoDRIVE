# F1TENTH RL Continuous Improvement Campaign

**Start:** 2026-09-17 03:54 America/Chicago  
**Deadline:** ~2026-09-17 12:54 America/Chicago (9 hours) or credits exhausted  
**Branch:** `rl/phase-1-research`

---

## Protected training run

| Field | Value |
| ----- | ----- |
| **run_id** | `overnight_soak_20260917_082739` |
| **Status at start** | ACTIVE — DO NOT KILL without restart |
| **Command** | `train_ppo --resume overnight_soak_20260917_082739 --timesteps 50000000 --n-envs 8 --vec-env subproc --unlimited-timesteps --early-stop-patience 4 --early-stop-min-improve 0.5 --early-stop-warmup-evals 2 --early-stop-min-timesteps 100000 --race-eval-every 50000 --select-timeout 220 --map map0` |
| **live_status (~03:56)** | timesteps=170496, steps/s≈252, n_envs=8, phase=learning, vec=subproc |
| **GPU (~03:56)** | RTX 3060 ~20–40% util, ~1.6 GiB / 12 GiB VRAM — **CPU-bound on env rollouts** |
| **PIDs** | 31472, 34256 (train_ppo); GPU compute also shows 34256, 21824 |

**Rule:** Improve code while this train runs. If a fix requires stopping train, restart Continue/Start on the same run_id afterward.

---

## Critique (user ideas) — before architecture choices

| Idea | Verdict | Safer approach |
| ---- | ------- | -------------- |
| Intuitive UI for all params | **Keep** — high value; users shouldn't need an agent | Sectioned panels + tooltips + presets; expose knobs already in `train_ppo`, don't invent new hyper APIs mid-run |
| Drastic UI upgrade | **Keep, tempered** — monospace dump is unusable overnight | Tkinter (already used) with labeled frames; **no** Electron/web frameworks on Windows |
| GPU ≫ CPU | **Keep, but diagnose first** | Env step is CPU; GPU only does PPO update. ROI: more n_envs / faster lidar / larger batch during update — not "move RacingEnv to GPU" |
| Autogen maps + autotrain | **Keep, incremental** | `map_pack` + `trackgen` exist; wire continuous mode to cycle train_safe maps |
| Continuous until plateau → new run | **Keep — scaffold first** | Outer loop after early-stop; do **not** change overnight early-stop math |
| Faster validating | **Keep, careful** | Fast progress probe ≠ race score. Official eval stays honest; mid-train can use fewer eps / progress-only probe labeled clearly |
| Giant occupancy grids / exotic sensors | **Defer** | Would slow CPU further; contracts frozen at v2.0.0 |

---

## Timeline

### 03:54–04:00 — Snapshot (A)

- Confirmed train running: `overnight_soak_20260917_082739`
- Wrote this log; `CURRENT_RUN.txt` stale (`20260916_224705_...`) — will refresh to protected run without touching train process
- Next: profile bottlenecks (B), continuous scaffold (C), fast val (D), UI panels (E), push (F)

---

## Shipped / Tests

_(appended as work lands)_
