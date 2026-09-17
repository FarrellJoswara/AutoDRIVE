# Campaign minimalist — 9h window

**Lens:** Default is refuse. KEEP only what moves **official adjusted_time vs pinned FTG**, **overnight continue integrity**, or **crash rate down**.  
**Grounded in:** `CAMPAIGN_CRITIQUE.md`, `CAMPAIGN_LOG.md`, `PLAN.md` (Sep 2026). No code.

---

## Complexity traps

### Continuous-train

Unlimited + patience without warmup / min_timesteps / sparse-eval grace → EarlyStop at ~40–100k. Dual Start or Continue on a live `run_id` dual-writes and corrupts zips. Soft-stop ≠ hard kill — document; do not invent Discord reward dials or outer “until plateau forever” loops that touch overnight early-stop math. Outer continuous scaffold may exist later; **do not change** select_timeout, eval floor, warmup, or `phase=early_stopped` preservation while chasing autopilot.

**500k / 5M “serious” defaults before crash rate ~0** farm death, not skill.

### Fancy UI

Watch KPIs, ghost, mismatch banners, map gen, model timeline, presets — largely shipped. More pixels ≠ better policy. Watch stays **unofficial**. Coach chrome, banners, seasons/XP, Electron/web second UI, “drastic upgrade” skins — comfort theater while beat-FTG on holdouts is unclaimed.

Operator value is **Start / Stop / Continue / refuse dual-writer / honest status** — not a prettier panel.

### GPU chasing

Gym step is **CPU-sim + VecEnv** bound. Live soak: RTX ~20–40% util, VRAM barely warm — expect idle GPU. CUDA / AMP / `torch.compile` / TensorRT / “move env to GPU” before logging steps/sec knee is theater. Tune `n_envs` and Dummy vs Subproc; coach may flag low steps/sec — **never** “buy GPU.”

---

## What to refuse (this campaign)

| Urge | Refuse because |
| ---- | -------------- |
| Lower mid-train select timeout / drop eval floor / strip warmup·min_ts | Regresses overnight EarlyStop that already burned hours |
| Dual Start / Continue while lock live / incomplete ckpt | Corrupts `models/<run_id>/` |
| Train sealed holdouts without loud allow | Voids claims; map3 never default |
| Promote on `ep_rew_mean` / partial zip | Lies about race skill |
| Mid-run reward Discord / algo zoo / residual+BC before clean laps | Cosplay; serialization: metric → overnight → collision first |
| GPU / AMP / compile as the “speed” project | Env-bound; measure steps/sec first |
| Watch aesthetics, second UI, map-regen in hot loop | Does not move official Δt |
| Bridge / `:4567` / Jetson / camera | Phase 3; gated on holdout beat-FTG |
| Unlimited continuous without budget discipline | EarlyStop traps + dual-writer risk |

If a feature fights contracts `2.0.0`, holdout seals, overnight early-stop constants, or dual-writer refuse: **cut the feature**, not the guard.

---

## KEEP (≤15)

1. **Protect overnight early-stop** — select_timeout ≥220, auto eval floor ≥50k, warmup≥2, min_ts≥100k, Overnight preset, preserve `early_stopped`.
2. **Continue-train soak** — Stop → Continue same `run_id` from last **complete** ckpt; refuse while locked/live.
3. **Dual-writer refuse** — `train.lock` + busy flag; one writer per `run_id`.
4. **Contracts refuse-load** — obs-dim / version mismatch fails closed everywhere (eval, resume, Watch, Continue).
5. **Holdout / seal refuse** — Start without allow dies on sealed maps; `verify` before official append; UI allow stays loud.
6. **Promote by race score only** — `race_score_key` / adjusted_time on validation·official; never `ep_rew_mean`; atomic zip only.
7. **Collision-first path** that visibly drops crash rate before speed unlock / longer budgets.
8. **Honest official FTG vs PPO row** — same protocol, non-null adjusted_time; lose honestly if needed.
9. **Multi-map train-safe only** — fingerprint pack hashes; never train-on-holdout by accident.
10. **Throughput honesty** — log steps/sec + vec truth; tune workers to the knee, not GPU %.
11. **Soft-stop vs hard-kill copy** — one sentence; no new stop machinery unless Continue soak fails.
12. **One DR *or* clearance assist** — only after crash rate drops and one serious overnight survives; not both.
13. **Shorter serious runs until metric truth** — crash-first before 500k cosplay.
14. **Pinned FTG protocol frozen** — λ=10; same seeds/episodes/timeout; do not retune FTG mid-campaign to “win.”
15. **Operator minimum** — sectioned knobs + presets that expose existing `train_ppo` flags; Tkinter only; no new hyper APIs mid-run.

---

**Monk close:** Protect overnight + contracts + holdouts → prove Continue → drop collisions → publish honest FTGΔ. Refuse GPU theater, Watch chrome, bridge, and continuous-train without discipline. Until sealed beat-FTG, every new subsystem is a cut.
