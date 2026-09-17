# Campaign reliability — continuous / UI / auto-map vs overnight

> Persona: reliability / chaos engineer.  
> Inputs: `CAMPAIGN_LOG.md`, `CAMPAIGN_CRITIQUE.md`, `PLAN_PROGRESS` overnight fix/soak, `ideas_review/08_reliability.md`, skim of `control_ui` / `ui_ops` / `train_ppo` / `live_status` / `map_pack`.  
> **No implementation** — failure-mode brief + mandatory safeguards. Code notes only if critical.

**North star this window:** overnight continue integrity + sealed holdouts + honest early-stop. Continuous mode, UI redesign, and auto map-train are optional accelerators that must not reintroduce the Sep-17 false EarlyStop (~40k–100k) or dual-write corruption.

---

## Verdict in one line

Scaffold continuous / prettier UI / map cycling **outside** the overnight early-stop math; treat dual-writer, holdout gates, and `phase=early_stopped` honesty as hard refuses — if a feature fights them, cut the feature.

---

## What already burned overnight (do not regress)

| Killer | Mechanism | Shipped antidote (lock these) |
| --- | --- | --- |
| Dense mid-train eval | Unlimited auto `race-eval-every` → 10k | Floor **≥50k**, unlimited ref **500k** |
| Permanent DNF mid-train | `select-timeout=60` ≪ lap ~180–210s | **`MID_TRAIN_SELECT_TIMEOUT_S ≥ 220`** |
| Patience from first stall | No warmup / no min-ts grace | Warmup **≥2**, min_ts **≥100k** when unlimited+patience |
| Soft patience under unlimited | patience 3 | Floor raise to **≥5** under `--unlimited-timesteps` |
| Lying banner | `LiveStatusCallback` overwrote `early_stopped` with `learning` | Preserve terminal phase + post-`learn` rewrite |
| Console abort | Unicode `→` on Windows cp1252 | ASCII-safe prints |

Overnight preset must stay: **budget OFF**, patience **5**, eval **50k**, warmup **2**, min_ts **100k**, select **220**. UI build stamp still advertises overnight-earlystop; soak: `python -m rl._overnight_soak_smoke`.

Protected campaign rule: improve code while train runs; if stop required, **Continue same `run_id`** — never Double-Start a second writer.

---

## How the three campaign ideas break overnight

### 1. Continuous mode (plateau → new run / outer loop)

**Intent (campaign):** outer loop *after* early-stop; do **not** retune overnight early-stop math.

| Failure | How it shows up overnight | Severity |
| --- | --- | --- |
| Outer loop “fixes” early-stop by restarting forever | False EarlyStop every ~warmup+patience×eval → infinite short runs; leaderboard / disk fill; looks “alive” while never learning | **P0** |
| Auto-Continue / auto-Start on same `run_id` while lock/PID still live | Dual zip writers; truncated `best_model`; Continue from incomplete | **P0** |
| Auto-Start **new** `run_id` without clearing ownership | Two trains, one UI banner; Stop kills wrong tree or neither | **P0** |
| Soft-stop vs hard-kill confusion | Continuous thinks process dead → Start while Subproc workers still write | **P0** |
| Outer loop drops warmup / min_ts / select-timeout | Replays Sep-17 bug under a new wrapper name | **P0** |
| Continuous treats EarlyStop as crash and “repairs” by lowering patience / eval interval | Operator “help” that re-arms the overnight killer | **P0** |
| Plateau detection uses `ep_rew_mean` or train map score | Restarts on reward noise; promotes wrong brain | **P1** |
| No budget / max-run / wall-clock cap on outer loop | Unlimited×continuous = disk and zombie farm | **P1** |
| Continues into holdout/validation because argv rebuilt from UI picker | Sealed train; voided official claims | **P0** |

**Safer shape:** one process finishes (budget or `early_stopped`) → lock released → **only then** spawn next job with frozen overnight argv template + new `run_id` (or explicit Continue). Outer loop owns scheduling; inner `train_ppo` early-stop constants stay sacred.

### 2. UI redesign (panels / presets / “intuitive” knobs)

**Intent:** expose existing knobs; Tkinter/sectioned HTML already in use — no second stack mid-run.

| Failure | How it shows up overnight | Severity |
| --- | --- | --- |
| Overnight preset loses warmup / min_ts / select_timeout in form POST | Start looks “Overnight” but argv omits grace → early death | **P0** |
| Budget toggle UX inverted or default ON | “Overnight” silently becomes 500k budget stop (or patience-0 refuse) | **P1** |
| Allow-sealed checkbox default / sticky / quiet | Train on map2/map3/map4; claims void with green UI | **P0** |
| Preview busy_reason uses cached external scan (`force=False`) while Start uses live | Operator clicks Start because preview looked idle → kill path / refuse race | **P1** |
| Start/Continue call `_kill_train_tree()` after busy check | Busy **false-negative** → kills protected overnight then spawns | **P0** |
| Two browser tabs / two `start_ui` instances | Dual Start; lock helps but UI `_train_busy` is per-process | **P0** |
| Delete / Generate / map picker while train holds artifacts | Operator clobber; Continue map list drift | **P1** |
| Banner heuristics demote `early_stopped` / `validating` | “Idle/Stale” after clean EarlyStop → human Restart wrong | **P1** |
| Soft-stop copy missing | Operator Force-kills mid-atomic save | **P1** |

**Safer shape:** redesign = layout + tooltips + presets wiring to **same** `ui_ops.PRESETS` / `resolve_stop_budget` / `start_guard`. Any new Start path must pass `ui_selftest` overnight constant checks and refuse Double-Start in plain English.

### 3. Autogen maps + autotrain (cycle train_safe)

**Intent:** wire `map_pack` + `trackgen` to continuous cycling of **train_ok** maps only.

| Failure | How it shows up overnight | Severity |
| --- | --- | --- |
| Cycle includes sealed / validation pin | Holdout train; selection map contaminated | **P0** |
| Generate with `--seal` / pin-validation in hot loop | Regenerates or re-pins → seal break or silent protocol drift | **P0** |
| Overwrite existing YAML/centerline of a live train map | Mid-run geometry swap; fail-loud missing → silent wrong world | **P0** |
| Multi-map train without `pack_fingerprint` in `config.json` | Unattributable overnight; can’t reproduce | **P1** |
| Autogen while `verify_pack` dirty | Train proceeds on broken seals | **P0** |
| Autotrain “helpfully” passes `--allow-holdout` | Checkbox at machine speed | **P0** |
| Map gen UI shares process / GIL with train spawn path | Start latency / missed busy window | **P2** |

**Safer shape:** generate offline or under exclusive mapgen lock; only enqueue `train_safe_maps()` ids; refuse Start if `verify_pack` fails; fingerprint every run; never touch sealed content hashes.

---

## Dual writers

**Invariant:** at most one live writer under `models/<run_id>/` (checkpoints, `best_model`, lock, sibling `runs/<run_id>/live_status`).

| Guard today | Gap / chaos |
| --- | --- |
| `train.lock` + `acquire_run_lock` in `train_ppo` | Stale lock after kill — Continue clears; must not clear **live** owner |
| UI `_train_busy` (owned proc + external scan) | External scan can miss / lag; preview cache can lie |
| Start/Continue refuse when busy | Then **`_kill_train_tree()`** — unsafe if busy was wrong |
| Stop clears stale lock after verified death | Soft-stop incomplete → Continue refused (good) or zombie still writes (bad) |

**Mandatory:** refuse Double-Start / Continue-on-live before any kill; kill only on explicit Stop or after PID+lock prove dead; continuous outer loop must wait for lock release + exit code, not for banner Idle alone.

---

## Early-stop (sacred math)

Do **not** change for continuous / UI / map features:

1. `MID_TRAIN_SELECT_TIMEOUT_S ≥ 220`
2. `AUTO_RACE_EVAL_EVERY_FLOOR ≥ 50_000`
3. Warmup evals ≥ 2 when patience > 0 (auto)
4. Min timesteps ≥ 100_000 when unlimited + patience (auto)
5. Unlimited patience floor ≥ 5
6. Patience never ticks on baseline / warmup / pre-min-ts / unscorable
7. Promote + stop decisions use **validation** race score — never `ep_rew_mean`
8. `phase=early_stopped` must win over LiveStatus default `learning`
9. Overnight preset: budget OFF + patience > 0 + grace knobs wired through Start **and** Continue

Continuous mode may **react** to `early_stopped` (schedule next run). It must not **redefine** what counts as a plateau.

---

## Holdouts / validation pin

| Rule | Why |
| --- | --- |
| `assert_train_safe` on every train argv (CLI + UI) | Blocks map2/map4 sealed + map3 validation |
| Protocol outranks stale manifest for `is_holdout` | Manifest wipe must not unseal |
| UI allow-sealed is loud + voids official claims | Override is poison, not convenience |
| `assert_seals_intact` before official leaderboard append | Tamper → no claim |
| `pack_fingerprint` in `config.json` | Overnight attributable |
| Auto map cycle ⊆ `train_safe_maps()` only | No “almost all maps” shortcuts |
| Mid-run map YAML swap fails loud | Wrong world ≠ policy bug |

---

## Mandatory safeguards (ship / keep before continuous ships)

### Control plane
1. **Single-writer:** live lock + PID → refuse Start/Continue; never clear live locks.
2. **Busy before kill:** `_kill_train_tree` only after Stop or proven-dead; continuous must not Start-kill.
3. **Continue = last complete ckpt**, same `run_id`, contracts precheck, maps from prior `config.json`.
4. **Soft-stop ≠ dead:** banner + copy; outer loop waits for exit + lock gone.
5. **Dual-UI / dual-tab:** document refuse; prefer one Control UI per machine for overnight.

### Early-stop / overnight
6. **Regression lock** constants above; any touch to `RaceBestModelCallback` / `ui_ops` → re-run `_overnight_soak_smoke` + `ui_selftest`.
7. **Preset + Continue argv parity:** warmup / min_ts / select_timeout must survive UI redesign POST paths.
8. **Terminal phase honesty:** `early_stopped` / `validating` not clobbered by status writers.

### Maps / claims
9. **Holdout refuse** without flags; UI preview REFUSED in red without checkbox.
10. **No generate-over-seal**; clash refuse already in generate_maps_op — keep.
11. **Official path:** seals intact + same protocol as FTG; Watch stays unofficial.

### Continuous-specific (before enabling)
12. **Budget discipline:** max consecutive runs, max wall time, max disk under `models/`.
13. **Frozen overnight argv template** for child jobs (no “smart” patience decay).
14. **New `run_id` after EarlyStop** (default) unless operator explicitly Continues; never fork two writers on one id.
15. **Chaos acceptance:** Double-Start refuse, Continue-while-locked refuse, holdout argv exit ≠ 0, soak no `early_stopped` before warmup+patience.

---

## Critical code notes (found while skimming — not patched here)

1. **Validation heartbeat vs terminal phase (race).**  
   `RaceBestModelCallback._stop_heartbeat` sets an Event but does **not join** the daemon pulse thread. A in-flight `_write_status(phase="validating")` can land **after** `phase=early_stopped`. Post-`learn` rewrite usually repairs; a crash in the window leaves a lying `validating`/`learning` freeze (same class of lie as the Sep-17 banner bug). Continuous auto-restart keyed only on phase is unsafe until join-or-generation-token exists.

2. **Start/Continue → `_kill_train_tree()` after `_train_busy()`.**  
   Correct when busy detection is perfect; a false-negative external scan **kills the protected overnight** then spawns. Continuous mode amplifies this. Safeguard: never kill inside Start/Continue; only Stop (or require lock+PID consensus twice).

No other code changes requested; treat (1)(2) as blockers before unattended continuous.

---

## Chaos drills (gate continuous / UI / auto-map)

| # | Drill | Pass |
| --- | --- | --- |
| 1 | Overnight preset → >100k steps without EarlyStop from first DNF plateau | Survives grace |
| 2 | `_overnight_soak_smoke` | ≥ warmup+patience val→learn cycles; no illegal early_stopped |
| 3 | Start while train alive | Plain-English refuse; one writer |
| 4 | Stop → lock cleared → Continue same `run_id` | timesteps rise; contracts OK |
| 5 | Continue while lock live | Refuse |
| 6 | `train_ppo --map map2` / `map3` without allow | exit ≠ 0 |
| 7 | Continuous (when built): after `early_stopped`, wait lock release, then **one** next job | No overlap PIDs on same id |
| 8 | Autogen N maps then autotrain | Only train_ok ids; verify_pack green; fingerprint present |
| 9 | Kill mid-checkpoint | Resume ignores incomplete; no promote |
| 10 | Truncate / stall `live_status` | Stale/Crashed, not Learning |

---

## Cut / defer (reliability lens)

| Cut now | Why |
| --- | --- |
| Continuous without budget + lock wait | Replays overnight death as a loop |
| UI that invents new stop/early-stop semantics | Regresses sacred math |
| Auto `--allow-holdout` / train-on-validation | Poisons selection + claims |
| Map regen in train hot loop | Seal + geometry races |
| GPU/Watch chrome as “overnight fix” | Doesn’t move survival or FTGΔ |

| Defer | Until |
| --- | --- |
| Fancy outer-loop curriculum scheduler | Continue soak + dual-writer proven |
| Second UI / web stack | Single control plane boringly solid |
| Discord/hot reward dials | Fingerprint death |

---

## Orchestrator one-liner

**Keep overnight early-stop + dual-writer + holdout gates frozen; scaffold continuous/UI/auto-map only as a scheduler that refuses to touch those guards — and fix heartbeat join + Start-kill-before-spawn before any unattended outer loop.**
