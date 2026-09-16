# F1TENTH RL Stack

This directory holds a multi-phase reinforcement-learning stack for solo F1TENTH / RoboRacer time-attack in AutoDRIVE: research and frozen interfaces first, then procedural tracks, a classical Follow-the-Gap baseline in `f1tenth_gym`, PPO training with a model registry, an AutoDRIVE Bridge path for live sim, and a cross-backend eval CLI. Each phase is independently shippable with its own pass criteria; later phases soft-consume earlier artifacts when present but always ship fallbacks so missing pieces do not block verification.

**Branch / worktree:** All RL work happens on `rl/phase-1-research` in an isolated worktree. The upstream `AutoDRIVE-Devkit` tree is left untouched.

**Frozen interface:** [contracts.md](contracts.md) is the Phase 1 interface lock (observation, action, reward, metrics). Phases 3–6 must honor it or bump `contracts_version`.

## Phase index

| Phase | Goal | Depends on | Pass criteria |
| ----- | ---- | ---------- | ------------- |
| **1 — Research & contracts** | Capture competition research and freeze obs/action/reward/metrics so later phases agree | Nothing | `research_notes.md` and `contracts.md` exist; docs only — no runtime |
| **2 — Track generation** | Autogenerate race maps for training/generalization | Nothing (Phase 1 docs optional) | `generate --num_maps 3` writes 3 valid map+yaml+centerline sets; no gym/sim required |
| **3 — Gym FTG** | Classical Follow-the-Gap races in `f1tenth_gym` with competition-shaped metrics | Soft: Phase 2 maps (else demo map) | One command runs ≥1 gym episode and prints metrics; no PPO/AutoDRIVE |
| **4 — Gym PPO + registry** | Train LiDAR PPO; save comparable `best_model.zip` + leaderboard | Soft: Phase 2/3; hard: gym deps | Smoke train produces `run_id/` with model+metrics; `compare_models` shows ≥1 row; no AutoDRIVE |
| **5 — AutoDRIVE Bridge** | Drive AutoDRIVE Simulator live with FTG over Bridge | Nothing from 2–4 (sim on port 4567 for live pass) | Offline: Bridge imports + Reset in API; with sim: FTG issues throttle/steering |
| **6 — Cross-eval** | One CLI scores FTG or PPO on gym and/or AutoDRIVE | Soft: Phases 3–5 artifacts; degraded mode always | Each backend/policy path works when deps exist; missing pieces → clear error, not a crash |

## Status

| Phase | Status | Notes |
| ----- | ------ | ----- |
| **1** | **Complete** | Deliverables: [research_notes.md](research_notes.md) + [contracts.md](contracts.md). Pass = docs exist; **no runtime**. |
| **2–6** | Not started | — |
