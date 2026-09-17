# PLAN progress

Companion to [`PLAN.md`](PLAN.md). Agents append status under owned sections; do not rewrite others' work.

---

## Watch — race KPIs shipped

**Status:** PLAN Phase 0 #11 + Phase 2 #7/#13 (Watch half) **done**. All five handoff TODOs implemented and smoked headlessly. No `control_ui` / `train_ppo` / `racing_env` edits.

### Shipped

1. **Race-KPI overlay** — bottom-left block, race score first, `ep_rew` last and only inside the train row:
   - `PPO  adj 41.3 | lap 31.3 | col 1 | laps 3/8` (current episode)
   - `FTG  last 2 ep: adj 38.9 | col 0 | crash none` (ghost)
   - `delta +2.4s vs FTG ghost -> FTG ahead`
   - `PPO  last 10 ep: adj ... | col ... | crash wall 5 stall 1` (rolling window)
   - `train ts 49152 | 812 steps/s | crash eps 34 | rate 42% | rew 12.5` (from `live_status`)
   - `adjusted_time = lap + 10·collisions`, identical to `metrics_io.make_metrics`. No lap → `DQ~<timeout proxy>` (same rule as `eval_protocol._adjusted_or_dq`), never a silent blank.
   - Session totals use a **rolling 10-episode window**: an hours-long Watch session must not inflate collisions forever.
2. **Lag-behind label** — grey sub-line under the centered label: `lag-behind replay of latest_model.zip (8s old) - not the training cars`; FTG variant while waiting for weights; standalone says `offline rollout - ... (not a training run)`.
3. **Map mismatch banner** — full-width red bar. `MAP MISMATCH` (watched map ∉ run's `maps`), `MAP CHANGED` (same stem, different `map_file_hash` → regenerated), `CONTRACTS MISMATCH` (config `contracts_version` ≠ frozen), `TRAIN STALE` (no `live_status` for >90 s). Priority: contracts > map > stale. Without `config.json` it falls back to `map<N>` tokens in the `run_id`; unknown → quiet (no false alarms). Map hashing runs only when the followed run changes, not per frame.
4. **Crash / stall cues** — X = wall, circle = stall/timeout, triangle = TTC, stamped where the twin died and labelled with the tag. Repeats within 0.8 m collapse to one `wall x3` mark so the FTG graveyard cannot bury the map. Dead twins stay on screen dimmed; the **last frame of each episode is held ~450 ms** (crashes used to land between sparse redraws and were never drawn).
5. **FTG ghost** — hollow white car with its own episode lifecycle (`GhostRunner`), so there is always a real classical time on screen even when every PPO twin dies in turn 1. Auto-hidden while twins are themselves on FTG; `--no-ghost` to disable.

Also: twin envs now mirror the run's `ttc_truncate` / `spawn_jitter` / `n_lidar` from `config.json`, so twins fail the way training does; `viewer._draw_outlined_text` switched from an 8-way halo (which smeared shut into black blobs under OpenCV 5) to a translucent plate + 1 px shadow.

### Files touched

| File | This turn |
| ---- | --------- |
| `watch.py` | KPI/ghost/marker plumbing, lag-behind + banner copy, held end-of-episode frame, `--no-ghost`, env flags from run config |
| `viewer.py` | Warn banner, sub-label, KPI block, crash markers, ghost/dimmed cars, readable text |
| `watch_kpi.py` | **New tiny helper** — `FleetKpi` / `KpiTally`, `adjusted_time`, `beat_line`, `train_status_line`, map & contracts warnings |
| `test_watch_overlay.py` | **New** — headless smoke (stubs OpenCV GUI, writes frames to `logs/watch_overlay/`) |

### Tests

`rl/.venv` → `python -m rl.test_watch_overlay` (also runs under pytest): **12/12 pass, ~45 s.**

- KPI math vs leaderboard formula, DQ proxy, rolling window, beat-FTG delta, marker merge, banner matrix (mismatch / changed hash / run-id fallback / contracts), train row ordering.
- Real-env render check of every overlay layer at once → `logs/watch_overlay/overlay_layers.png`.
- End-to-end `watch.main()` standalone (FTG twins) → `standalone_last.png`.
- End-to-end `watch.main() --follow` against the newest contracts-v2 run on `--map map3` (deliberate mismatch): weights load, ghost runs, red banner asserted → `follow_last.png`. Skips itself if no v2 run with `latest_model.zip` exists.

### Notes for other tracks

- Watch numbers are **unofficial** by construction (jittered spawns, stochastic twins) and say so; promotion still belongs to `eval_cli` / `compare_models`.
- **Integration (closed):** ghost / follow twin envs default to `protocol_timeout_s()` (400 under official_v2). CLI `--timeout-s` overrides for short smokes. Overlay DQ proxy uses the same helper.
- `README.md` still describes the old Watch overlay (`timesteps / ep_rew / n_envs`); left untouched on purpose since docs are outside Watch ownership.

---

## Maps

**Status:** **Shipped.** Cached map generation, roles, sealed holdouts (hash-pinned), and a validation pin are live behind `rl/map_pack.py`. Covers PLAN Phase 0 #1 (map pack manifest), Phase 2 training #1 (seal holdouts), and hard-serialization rule #9. Siblings must switch their gates to these APIs (see TODOs).

### Current pack (`maps/map_pack.json` — tracked in git; `maps/map*/` is gitignored)

| Map | Role | Seal hash | Notes |
| --- | ---- | --------- | ----- |
| `map0` | train_ok | — | pinned in `eval_protocol.yaml` |
| `map1` | train_ok | — | |
| `map2` | **holdout (sealed)** | `87b5545bd611ddb7` | protocol holdout H; seal cannot be lifted from the pack |
| `map3` | **validation (pinned)** | — | new; for `best_model` selection only, never trained on |
| `map4` | **holdout (sealed)** | `ce0a9c950b1d2303` | new second holdout H2 |

The manifest carries two hashes per map: `map_hash` (yaml+image, matches `eval_protocol.map_file_hash` in official rows) and `content_hash` (yaml+image+`centerline.csv`+`start_pose.txt`). Seals pin `content_hash`, so editing a centerline — which silently changes lap times — breaks the seal.

### API for siblings (no `control_ui` / `watch` / `train_ppo` edits were made)

```python
from .map_pack import (
    ensure_maps,          # cached generation; returns {"created": [...], "cached": [...]}
    list_maps,            # rows: id, role, sealed, train_safe, hashes, label ("map2 [HOLDOUT]")
    train_safe_maps,      # ids a trainer may use (excludes sealed + validation pin)
    holdout_maps, validation_map, map_info,
    is_holdout,           # sealed in pack OR in frozen eval_protocol.yaml
    is_train_safe,
    assert_train_safe,    # raises HoldoutViolation; allow_holdout / allow_validation escapes
    verify_pack, assert_seals_intact,   # re-hash; SealBroken on drift
    pack_fingerprint,     # {validation_map, train_ok, holdouts:{id:hash}, map_hashes} for config.json
    HoldoutViolation, SealBroken,
)
```

All accept `maps_root=` (defaults to `rl/maps`) and an optional pre-loaded `pack=` to avoid re-reads. `load_pack()` never hashes, so UI refreshes stay cheap; hashing happens only on generate / seal / verify.

### CLI

```text
python -m rl.map_pack generate --num 6 [--seal --reason "..."] [--validation] [--role train_ok] [--scale 0.25]
python -m rl.map_pack list [--train-safe|--holdout] [--ids|--json]     # --ids = comma-separated for scripts
python -m rl.map_pack seal map5 --reason "holdout H3"
python -m rl.map_pack unseal map5 --force        # refused for protocol-sealed maps, always
python -m rl.map_pack pin-validation map3
python -m rl.map_pack verify [--json]            # exit 1 if any seal/pin is broken
python -m rl.map_pack is-holdout map2            # exit 0 = holdout, 3 = not (scriptable)
python -m rl.map_pack fingerprint
python -m rl.trackgen --num_maps 6 [--scale 0.25] [--pgm] [--force]   # auto-registers in the pack
```

### Guarantees (each covered by the smoke)

- **Cached:** `generate --num N` only draws missing ids; re-running is a no-op. Map `<prefix><i>` is reproducible from `(seed, i)` alone (per-index RNG), so growing a pack never shifts existing geometry.
- **Sealed:** sealing pins `content_hash`; `verify` / `assert_seals_intact` report `SEAL BROKEN` on any drift. `--force` regeneration and `unseal` are both refused for sealed maps; protocol-sealed maps can never be unsealed via the pack.
- **Validation pin is not train-safe:** `train_safe_maps()` excludes it and `assert_train_safe` raises on it, so selection stays unbiased. `verify` warns if the pin is missing, sealed, or listed in the protocol's `train_ok_maps`.
- **Protocol outranks the manifest:** `is_holdout` returns True for `eval_protocol.yaml`-sealed maps even if `map_pack.json` is missing or stale.

### Files touched

| File | This turn |
| ---- | --------- |
| `map_pack.py` | **New** — manifest, roles, seals, train gate, fingerprint, CLI |
| `trackgen.py` | Per-index deterministic RNG (`GEN_VERSION = 2`), `generate_map` / `generate_map_specs`, cached `skip_existing`, `--scale`, opt-in `--pgm`, auto-register; `generate_maps()` signature kept |
| `test_map_pack.py` | **New** — 21-check smoke in a temp pack (`python -m rl.test_map_pack`) |
| `maps/map_pack.json` | **New** — pack manifest (map3 validation, map4 sealed) |
| `maps/map3/`, `maps/map4/` | **New** — generated, 0.1 MB each |

### Tests

- `python -m rl.test_map_pack` → **21/21 PASS in ~2 s** (cached generation, roles, gate + overrides, force/unseal refusals, fingerprint, clean verify, byte-identical regeneration, centerline-tamper seal break).
- `python -m rl.map_pack verify` on the real pack → **PACK OK** (5 maps, 2 seals intact).
- Gym load smoke: `map3` / `map4` build `RacingEnv` and step with `obs_dim=186`; FTG on all five maps is identical (`lap=None`, `cols=1`, ~110 steps) — the new maps are no regression.
- Re-sealing `map2` reproduced hash `87b5545bd611ddb7`, confirming seals are content-derived, not timestamped.

### Notes for other tracks

1. **FTG completes no lap on any map** (map0–map4: 1 collision at ~110 steps, `lap_time=None`). Until that's fixed the pinned FTG row is a `TIMEOUT + 10·cols` DQ proxy, so "beat FTG" is not yet a real comparison. Owner: FTG / eval track, not maps.
2. **New maps are png-only and 400× smaller** (0.1 MB vs ~40 MB): `trackgen` now writes the ROS `.pgm` twin only with `--pgm`. `racing_env.load_map` follows `image:` (and falls back to the png twin), and the UI thumbnailer already prefers png. Pass `--pgm` if the bridge/ROS path ever needs it.
3. **Raster size is the real per-env cost, not disk:** at the legacy `--scale 1.0` a map is ~300 m across → ~47 M occupancy cells (~45 MB per `RacingEnv`, so ~700 MB at `n_envs=16`). `--scale 0.25` gives an ~80 m track with a ~5 m corridor and 3.5 M cells (13× less), which is also closer to true F1TENTH dimensions. Not applied to `map0`–`map4` because it changes geometry and would require re-pinning the FTG baseline — worth a deliberate decision by the training track.

### TODOs (siblings — small, API is ready)

1. **`train_ppo.py`**: replace the `eval_protocol.is_holdout_map` loop with `assert_train_safe(map_ids, allow_holdout=args.allow_holdout)` — this also blocks training on the validation pin, which the current check misses. Add `pack_fingerprint()` into `config.json` (PLAN Phase 0 #4).
2. **`train_ppo.py`**: point the `RaceBestModelCallback` eval maps at `validation_map()` instead of the train maps, so `best_model` is not selected on trained geometry.
3. **`control_ui.py`**: ~~`ui_ops.map_labels` → `map_pack.list_maps()` + generate via `ensure_maps`~~ **done** (Control UI track).
4. **`eval_protocol.yaml`** (eval track): consider adding `map4` as a second sealed holdout and recording `map3` as the validation map so the protocol and pack agree in one place.
5. **Anyone pinning official results**: call `assert_seals_intact()` before appending an official leaderboard row, so a regenerated holdout can't silently invalidate a claim.

---

## Race core

**Status:** Phase 0 training track **shipped** (honest metric / best_model / FTG pin / refuse-load). Phase 1 train generalization **wired** (jitter, multi-map, collision-first/speed-gate/TTC flags, atomic ckpts, resume). Beat-FTG gym gate (Phase 2) is **not** claimed — PPO has not beaten the pinned FTG finisher yet.

### Shipped (this track + prior Opus turn)

1. **`map_pack.assert_train_safe`** in `train_ppo` — blocks sealed holdouts **and** the validation pin unless `--allow-holdout` / `--allow-validation`. Pack `verify` refuses broken seals before train. `pack_fingerprint()` lands in `config.json`.
2. **`best_model` by race score** — `RaceBestModelCallback` promotes on **validation map** (`map3`), never train geometry. Ranking via `race_score_key` (finishers ≻ DNFs ≻ progress). Mid-train budget `--select-timeout` (default **60 s**); official still uses protocol `timeout_s=400`. Mid-train eval writes `live_status` `phase=validating` (+ heartbeat) so the UI does not look hung; early-stop exits 0 with `phase=early_stopped`. Unlimited auto `--race-eval-every` floor raised to **25k** (was 10k).
3. **Official eval protocol `official_v2`** — `timeout_s=400` (laps need ~180–210 s at 6 m/s); maps map0 + sealed map2 + sealed map4; `eval_cli --official` appends `kind=official` after `assert_seals_intact`.
4. **FTG completes laps** — retuned `ftg.py` (forward-arc gap, proportional steer, front braking). map0: **lap≈205 s, 0 cols**. Pinned row `ftg_official_v2` adjusted_time≈198.2 (3 maps × 1 seed smoke pin). Watch’s “FTG DNF ~110 steps” was the **default env `TIMEOUT_S=60`**, not a broken planner — ghost/eval must pass `timeout_s≥400` to see a finish.
5. **Fingerprint + refuse** — `config_fingerprint`; `compat.load_ppo_refusing_mismatch` / config contracts refuse on eval + resume. Atomic `atomic_save_sb3` for best/latest.
6. **Jitter / multi-map / curriculum flags** — spawn jitter on by default; `--map map0,map1` rotates across `n_envs`; `--collision-first` / `--speed-gate` / `--ttc-truncate` / `--lidar-dr` wired through env.
7. **`compare_models`** — sorts with `race_score_key`; `--official` filters to current `protocol_id` so DNF proxy rows (e.g. official_v1 adj=80) cannot outrank a finisher.

### Files (race-core)

| File | Role |
| ---- | ---- |
| `train_ppo.py` | assert_train_safe, validation selection, fingerprint, resume, RaceBestModelCallback, `--select-timeout` |
| `racing_env.py` | jitter, TTC, collision-first, speed-gate, high-water lap gate |
| `ftg.py` / `run_ftg.py` | classical baseline that finishes under protocol timeout |
| `eval_protocol.py` / `.yaml` / `eval_cli.py` / `eval_official.py` | official_v2 protocol + runners |
| `metrics_io.py` / `compare_models.py` / `compat.py` / `contracts.py` | board, race key, refuse-load, scoring_rev |
| `live_status.py` | steps/sec + crash_rate_estimate |
| `checkpoint_io.py` | (if present) shared atomic helpers via metrics_io |

### Tests / smokes (`rl/.venv` from `autodrive_py`)

- `python -m rl._p0_checks` → **11/11 PASS** (v1 refuse, atomic zip, jitter, multi-map assign, lap anti-hack, TTC).
- `python -m rl._ftg_lap 400 1.0 map0` → lap=205.05, cols=0, progress=99%.
- `train_ppo --map map3` / `--map map2` → exit 2 refuse validation / holdout.
- `train_ppo --run_id smoke_racecore_p0` (2k steps) → best_model on map3, fingerprint written.
- `train_ppo --map map0,map1` multi-map smoke → OK.
- `compare_models --official --recommend` → only `ftg_official_v2` under protocol=official_v2.

### Notes for siblings

- **Watch:** ~~FTG ghost DNFs under default `TIMEOUT_S=60`~~ — **closed in Integration:** ghost/follow use `protocol_timeout_s()` (400); `--timeout-s` for smoke overrides.
- **Control UI:** ~~`ui_ops.race_candidate` sorted raw `adjusted_time`~~ — **closed in Integration:** same `row_race_score_key` + `protocol_id` filter as `compare_models`.
- **Maps TODOs 1–2** (assert_train_safe + validation selection + pack fingerprint): **done** in `train_ppo`.

### Still open (not P0 blockers)

- Full 5-seed official FTG re-pin (current pin is 1 seed × 3 maps).
- PPO beat-FTG on map0 + holdout (Phase 2 gate).
- Optional: raise train-episode `TIMEOUT_S` only if you want lap-completion shaping during learn (today progress shaping + 60 s is intentional).

---

## Control UI — operator P0 shipped

**Status:** PLAN Phase 0 #8–#10 + Phase 1 #7/#9 + Phase 2 #9–#11 (operator half) **done**. Helpers in `ui_ops.py`; stdlib HTTP panel in `control_ui.py`. No `watch` / `train_ppo` / `racing_env` / `map_pack` edits.

### Shipped

1. **Start/Stop honesty** — `_train_busy` refuses Double-Start (owned proc or external `train_ppo`); Stop kills the train tree + UI-spawned TensorBoard, verifies survivors, clears stale `train.lock` so Continue can resume. Plain-English messages when nothing was running / kill incomplete.
2. **Dual-writer refuse** — live `models/<run_id>/train.lock` blocks Start (new run collision) and Continue; stale locks cleared after Stop / before Continue. Contracts precheck on Continue/Load.
3. **Status banner** — Idle / Learning / **Validating** / **EarlyStop** / Saving (latest_model mtime <4s) / Stopping / Stale / Crashed from PID + `live_status` age / frozen timesteps / exit code / `phase` + `msg`.
4. **Continue-train** — `continue_train_argv` → `train_ppo --resume <run_id> --run_id <same>` from last complete checkpoint; reuses the run's map list from `config.json`.
5. **Model load / delete** — Models table: Load best/latest → Watch (prechecked); Continue; Delete (refuses live run / live lock / path traversal). Race-candidate slot from official leaderboard via `race_score_key` + current `protocol_id` (same as `compare_models`).
6. **Map generate + labels** — Generate N/seed via `map_pack.ensure_maps` (created vs cached); picker labels `[HOLDOUT]` / `[VALIDATION]` from `map_pack.list_maps`; Start + preview refuse sealed/pinned unless override checkbox.
7. **Presets + Start preview** — Debug / Quick / Overnight fill timesteps+n_envs and refresh the preview card (map role, vec, run path, holdout/busy blocks).
8. **Coach / health stub** — crash-rate / steps/sec / too-many-workers / open-Watch tip; demotes `ep_rew` in copy.
9. **Launchers** — `start_ui.ps1` documents the panel; `start_train.ps1` defaults 100k timesteps, VecEnv=auto (subproc when n_envs>1), `-Resume` / `-AllowHoldout` pass-through.
10. **Early-stop vs timesteps copy** — Timesteps labeled as max practice budget / safety cap; Early-stop patience has hover tooltips + one-line hint: stop on learning plateau (patience > 0), patience 0 = only timesteps / manual Stop.

### Files touched

| File | Role |
| ---- | ---- |
| `ui_ops.py` | Testable helpers: presets, banner, start_guard, continue argv, map labels/generate, models timeline/delete/precheck, coach, race_candidate |
| `control_ui.py` | HTTP UI wiring + Start/Stop/Continue/Load/Delete/Gen maps actions |
| `ui_selftest.py` | Offline smoke (helpers + local HTTP refusal paths) |
| `start_ui.ps1` / `start_train.ps1` | Launcher copy + resume/holdout/vec defaults aligned with UI |

### Tests

`rl/.venv` → `python -m rl.ui_selftest` (from `autodrive_py`): **~80/80 PASS in ~5 s.**

- Holdout/validation labels + `start_guard` refuse/override
- Map pack generate + cache + protocol-prefix refuse
- Model timeline / contracts precheck / delete + lock honesty
- Continue `--resume` argv + dual-writer busy/lock refuse (unit + HTTP)
- Banner matrix (incl. Saving) + coach heuristics
- HTTP: status/preview/models + Start/Continue/Delete/Load/Generate refusal paths (no real training)

### Notes for other tracks

- **Saving** banner is a soft mtime heuristic; mid-train validation now sets hard `phase=validating` / early-stop sets `phase=early_stopped` (train track).
- Continue/resume still depends on `train_ppo --resume` + atomic checkpoints (race-core). UI only builds argv and refuses unsafe loads.
- Map Start gate uses pack roles; race-core should still call `assert_train_safe` inside `train_ppo` so CLI can't bypass the UI checkbox.
- **Integration:** `race_candidate` now shares ranking with `compare_models` (see ## Integration).

---

## Integration

**Status:** Cross-track close-out after four parallel PLAN tracks (Watch / maps / race-core / Control UI). Ranking + timeout alignment fixed; smokes run under `rl/.venv`.

### Closed this pass

| Gap | Fix |
| --- | --- |
| Control UI “recommended” ranked raw `adjusted_time` (DNF proxy could beat finisher) | `ui_ops.race_candidate` uses `row_race_score_key` + current `protocol_id` filter (same as `compare_models --official --recommend`); UI copy says `race_score_key` |
| Watch FTG ghost / follow used train `TIMEOUT_S=60` → false DNF | Ghost + twin envs default to `eval_protocol.protocol_timeout_s()` (400 under official_v2); DQ overlay proxy matches; `--timeout-s` for short smokes |
| Shared ranking helper | `eval_protocol.row_race_score_key` + `protocol_timeout_s`; `compare_models` / `eval_official` re-export |

### Smokes (this pass)

| Command | Result |
| ------- | ------ |
| `python -m rl.ui_selftest` | **PASS** (~7 s) |
| `python -m rl.test_map_pack` | **PASS** (~3 s) |
| `python -m rl.test_watch_overlay` | **12/12 PASS** (~24 s; e2e uses `--timeout-s 20`) |
| Quick import `train_ppo` / `eval_protocol` / `compare_models` + `row_race_score_key` finisher≻DNF | **OK**; `protocol_timeout_s()==400`; `race_candidate` None until an official PPO row exists |

### P0 shipped (tracks + integration)

- Official protocol `official_v2` (timeout 400, sealed maps, FTG pin that finishes)
- `best_model` by `race_score_key` on validation map; refuse-load / fingerprint
- Map pack seals + `assert_train_safe`; UI holdout/validation gates
- Watch race-KPI overlay + ghost; Control UI Start/Stop/Continue/Load/Delete/presets/coach
- Recommended race candidate aligned with compare ranking

### Recent fix (2026-09-17)

**Random mid-train “pause” / climbing status age / mysterious stop:** caused by `RaceBestModelCallback` blocking `learn()` during validation (auto every 10k with unlimited+patience) while `live_status` froze, then early-stop exiting 0 with no UI banner. Fixed: `phase=validating` + heartbeat, mid-train timeout default 60s, auto eval floor 25k, `phase=early_stopped` + EarlyStop banner, clean exit messaging.

### Deferred

| Priority | Item |
| -------- | ---- |
| **P1** | PPO beat-FTG on map0 + holdout (Phase 2 gate) — not claimed |
| **P1** | Full 5-seed official FTG re-pin (current pin is 1 seed × 3 maps smoke) |
| **P1** | Overnight continue soak after UI Stop; multi-map overnight train |
| **P2** | Raise train-episode `TIMEOUT_S` only if lap-completion shaping is desired (60 s intentional today) |
| **P2** | `live_status.phase=saving` for hard Saving banner; README Watch overlay copy refresh |
| **P3** | Aesthetic Watch / seasons / bridge / Jetson / algorithm zoo (PLAN non-goals) |

---

## Overnight bug — diagnosis

**Verdict (2026-09-17 ~03:12 local):** Not a crash during eval. Unlimited + patience=3 + auto `race-eval-every=10000` early-stopped after a few mid-train DNF validations on `map3`. UI “Validating” was the blocking race-eval; process then exited 0. **Not overnight-safe.** Diagnosis only — no fix in this pass.

### Primary evidence (most recent overnight-style runs)

| Run / log | Mode | What happened | Wall / steps |
| --- | --- | --- | --- |
| **`logs/20260917_025009_ppo_gym_gen231_0_hard.log`** + `runs/20260917_025009_…/live_status.json` | `--unlimited-timesteps`, patience=3, **auto eval every 10k** | Early-stop @ ts=70035 | ~231 s train |
| **`logs/20260917_024749_ppo_gym_gen231_0_hard.log`** + `runs/20260917_024749_…/live_status.json` | same | Early-stop @ ts=40000 (≈ **first rollout**) | ~77 s train |
| `logs/20260917_030128_…` (newer, **not** the overnight failure mode) | budget **on**, 100k, **no** early-stop | Final validation @ 114688 then normal end | ~627 s — expected budget stop |
| `logs/20260917_024131_…` / `024152_…` | unlimited startup | **Crash before learn:** `UnicodeEncodeError` on `\u2192` in NOTE print (cp1252) | exit non-zero |

`config.json` for both early-stop runs: `unlimited_timesteps: true`, `early_stop_patience: 3`, `early_stop_min_improve: 0.5`, **`race_eval_every: 10000`**, `validation_map: map3`, mid-train select timeout 60 s.

### Exact sequence — `20260917_024749` (clearest “stopped while Validating”)

Startup quotes:

```text
NOTE: --unlimited-timesteps -> safety ceiling 50,000,000 timesteps (early-stop is the real exit; ceiling is hard abort only)
NOTE: --early-stop-patience=3 -> auto --race-eval-every=10000
Early stop: patience=3 validation race-eval(s) without meaningful improvement (min_improve=0.5s adj / DNF progress; maps=['map3'])
```

1. **First validation ~ts=10000** → promotes `best_model`, **DNF** `progress=0.0019`, `cols=0`. First eval is always “meaningful” → patience counter **reset to 0**. `best_model_meta.json` timestamps this at `2026-09-17T07:48:19Z` with `adjusted_time: 90.0` (timeout proxy; still DNF).
2. **Evals ~20k, ~30k, ~40k** → no further promote; progress never beats the min-improve gate vs the first DNF ref.
3. **Stop @ ts=40000** (patience 3/3). Quote:

```text
Early stop: no meaningful validation race-score improvement (min_improve=0.5s adj / DNF progress) for 3 eval(s) @ ts=40000 (best_key=(1, 90.0, -0.001939969636416449, 0))
```

`best_key` still pinned to the **first** eval. With `n_envs=20`, `n_steps=2048`, one rollout ≈ 40960 steps — **early-stop fired at the end of the first rollout**, before meaningful PPO learning.

Frozen `live_status.json`: `phase: "learning"`, `timesteps: 40000` (not `early_stopped`) — see bug #5.

### Exact sequence — `20260917_025009` (same bug, one DNF jump then stall)

1. **ts≈10005** — first eval promotes DNF `progress=0.0019` → patience **0/3**.
2. **ts≈20k / 30k** — no promote (misses → patience climbs).
3. **ts≈40020** — promote DNF `progress=0.4729` (≫ +1% vs 0.0019) → **meaningful**, patience **reset to 0**. Meta: `best_model_meta.json` @ `2026-09-17T07:52:06Z`.
4. **ts≈50k / 60k / 70k** — no further meaningful DNF progress (still DNF; `adjusted_time` stuck at 90.0 with 0 cols) → patience **3/3**.
5. **Stop @ ts=70035**. Quote:

```text
Early stop: no meaningful validation race-score improvement (min_improve=0.5s adj / DNF progress) for 3 eval(s) @ ts=70035 (best_key=(1, 90.0, -0.4728804591932554, 0))
```

Only **2** SB3 rollouts completed (`live_status` `rollouts: 2`, `ep_rew_mean≈40.5`). Final smoke metrics still `dnf: True`, `mean_progress_frac: 0.4728…`. No Traceback; clean early-stop exit (exit 0). Log does **not** contain `Training ended via early-stop` / `NOTE: … clean exit` (those prints were absent in the binary that ran, or not reached).

### Precise bugs (for fix siblings)

1. **Eval every too soon (primary overnight killer)** — Unlimited auto resolved to **`race-eval-every=10000`** (`NOTE: … auto --race-eval-every=10000`; configs agree). Patience=3 ⇒ stop by **~40k** if no second meaningful gain, or **~70k** after one DNF jump. Source now claims floor **25k** / ref **250k** (`ui_ops.AUTO_RACE_EVAL_EVERY_FLOOR`), but **these failing runs still used 10k**. Even 25k×3 is only ~100k steps — still not overnight.
2. **DNF plateau makes min-improve nearly impossible after the first jump** — Mid-train select timeout **60 s** (`MID_TRAIN_SELECT_TIMEOUT_S`); laps need ~180–210 s → **validation never finishes**. Among DNFs, meaningful = +`max(0.5%, min_improve×2%)` progress (default **+1% lap**) **or** −0.5 s adj proxy. With 0 collisions, adj stays **90.0** forever → **only progress** can reset patience. After ~47% in 60 s, +1% every 10k steps fails → counter burns out.
3. **Not “first eval always no improve”** — First eval correctly counts as meaningful (`is_meaningful_race_improvement(..., ref=None) → True`) and resets patience. The failure is **post-first** stalls under (1)+(2).
4. **Patience is fine mechanically; cadence × DNF gate is wrong for overnight** — `race_eval_patience_update` trips on the 3rd consecutive non-meaningful eval as designed; design is unsafe when evals are dense and scores stay DNF.
5. **`live_status` left as `phase: "learning"` after early-stop** — Both early-stop runs freeze on LiveStatus-shaped `phase: "learning"` at the stop timestep, **not** `early_stopped`. `LiveStatusCallback._on_training_end` always writes `phase: "learning"`. UI can show **Validating → gone/Idle/Stale** instead of **EarlyStop**, which matches “Validating then stopped.”
6. **Separate launcher crash (not the Validating stop)** — `024131` / `024152`: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'` on Windows cp1252 while printing unlimited NOTE — train never starts.

### Non-causes (ruled out)

- Crash / exception during the successful early-stop runs (no Traceback; metrics written).
- Patience starting at 3 without a first reset (first eval does reset).
- Most recent `030128` “Validating then stop” — that was **final** validation after hitting the **100k budget** (`train_timesteps: 100000`), not early-stop (`patience` unused; live_status `msg: "learning: final validation done"`).

### Pointers for fix track

- Raise / rethink unlimited auto `race-eval-every` (and/or require min steps before patience counts).
- Soften DNF early-stop (ignore patience while all-DNF, lower progress threshold, or longer mid-train timeout if finishers are required to “improve”).
- Ensure `phase=early_stopped` wins over LiveStatus on training_end (and survives post-`learn`).
- Replace Unicode arrows in prints for Windows consoles.
| **P0** | Overnight early-stop false plateau — see **## Overnight bug — early-stop audit** (fix sibling; do not ship overnight with current defaults) |

---

## Overnight bug — early-stop audit

**Status:** Audit only (2026-09-17). **Do not implement here** — fix sibling owns the patch. Symptom: UI shows `Validating` then `EarlyStop` after a few mid-train races; unlimited / overnight runs die at ~75k–100k instead of learning overnight.

**Scope:** `train_ppo.py` (`RaceBestModelCallback`, `is_meaningful_race_improvement`, `race_eval_patience_update`, `_race_eval_maps`) + `ui_ops.py` constants (`MID_TRAIN_SELECT_TIMEOUT_S=60`, `AUTO_RACE_EVAL_EVERY_FLOOR=25_000`, `UNLIMITED_EVAL_REFERENCE_TS=250_000`) + `eval_protocol.race_score_key`.

### Checklist answers (user questions)

| Question | Verdict |
| -------- | ------- |
| Does first validation count as “no improvement” and start patience at 1? | **No.** `patience_ref_metrics is None` → `is_meaningful_race_improvement` returns `True`; counter stays `0`. Covered by `ui_selftest.test_meaningful_improvement`. |
| Does patience hit N on first eval? | **No** for any `patience >= 1`. Stop needs N consecutive *non*-meaningful evals *after* a ref exists. |
| Is `best_key` / best score initialized so first eval can never improve? | **No.** `best_key = None` → first eval always promotes; first eval also always resets patience ref. |
| Mid-train timeout 60s → permanent DNF → never meaningful improve → instant early stop? | **Yes — primary overnight bug** (see below). Laps need ~180–210 s; mid-train budget is 60 s → finishers essentially impossible; DNF→finish never resets patience. |
| `race_score_key` / `min_improve` broken for early DNF policies? | **Yes — interaction bugs** under all-DNF + 60 s (adj-before-progress; 1% progress bar; discrete adj). |
| Unlimited + auto eval every 25k stopping after 1–3 evals? | **Yes with typical patience 3–5.** Timeline: first eval @ 25k (baseline), then stop at `25k + patience×25k` if no *meaningful* DNF progress (e.g. patience 3 → **~100k**). |

### What is *not* broken

- First-eval / `best_key=None` / `patience_ref=None` init path is correct.
- `race_eval_patience_update`: `improved → (0, False)`; miss → `n+1`, stop iff `n >= patience`.
- Tiny `race_score_key` gains still promote `best_model` without resetting patience (intentional split).
- `_last_eval_ts = 0` correctly skips eval until `eval_freq` (avoids untrained step-1 race).

### Concrete bugs (code-level)

#### BUG-1 — Permanent mid-train DNF makes “meaningful” improvement almost unreachable (P0)

`MID_TRAIN_SELECT_TIMEOUT_S = 60` (`ui_ops` / `--select-timeout` default). Official / real laps need ~180–210 s (`protocol_timeout_s()==400`). `_race_eval_maps(..., timeout_s=60)` therefore almost always returns `mean_lap_time=None`, `dnf=True`, `adjusted_time = 60 + 10·cols`.

`is_meaningful_race_improvement` then never takes the easy path `DNF → finish`. Under both-DNF it requires either:

- `mean_progress_frac` up by `max(0.005, min_improve * 0.02)` → **+1% of lap at default `min_improve=0.5`**, or
- `adjusted_time` down by `≥ min_improve` seconds.

Once the policy mostly survives without wall hits, `cols≈0` and `adj` sticks at **60** forever. Only the **+1% progress** gate remains. Early overnight learning often moves ≪1% per 25k steps (or stalls at the same turn) → every eval after the first is a miss → early-stop after `patience` validations.

**Reproduce shape:** unlimited + patience 3 + min_improve 0.5 + auto eval 25k → Validating @ 25/50/75/100k → EarlyStop @ ~100k.

#### BUG-2 — Patience vs promotion diverge under all-DNF (P0)

`_race_and_maybe_promote`: any strict `key < best_key` saves `best_model`; patience only resets when `meaningful`. For DNFs with equal `adj`, `race_score_key` is `(1, adj, -progress, cols)` — so **+0.3% progress promotes best_model** but **does not** reset patience (`min_prog=0.01`). Overnight can keep improving the zip while the stop clock still runs out.

#### BUG-3 — `race_score_key` ranks DNF `adj` before progress; mid-train adj is mostly crash count (P1)

Key order: finisher flag → `adjusted_time` → `-progress` → cols. For 60 s DNFs, `adj = 60 + 10·cols` is discrete. A **clean timeout at ~2% progress** (`adj=60`) sorts **better** than a **wall crash at ~40% progress** (`adj=70`). That can:

- promote a timid / stuck policy over one that covers more track then dies;
- count crash→timeout as “meaningful” (`adj` −10 ≥ 0.5) even when progress **drops**, resetting patience on a worse driver.

Progress is only a tiebreaker when `cols` (hence `adj`) matches — wrong signal for a budget that cannot finish laps.

#### BUG-4 — Auto interval + suggested patience = stop after 1–3 post-baseline evals (P0 product)

Unlimited auto: `max(25_000, UNLIMITED_EVAL_REFERENCE_TS // 10)` = **25k**. UI copy suggests patience **3–5**. With BUG-1, that is intentionally “a few validations then exit,” not overnight-safe. No grace / warmup before patience counts (first ref at 25k is often near-zero / stall-truncated progress via `STALL_TIMEOUT_S=8`).

#### BUG-5 — Fixed eval seed + short horizon → identical miss streaks (P1)

`_race_eval_maps` always uses `seed=self.seed + 10_000`. Deterministic. If weights barely move in 25k steps, metrics repeat → guaranteed non-meaningful streak → stop in exactly `patience` evals. Correct for a true plateau; catastrophic when the metric cannot see learning under 60 s.

#### Non-bugs / red herrings

- First eval does **not** increment `_no_improve`.
- `best_key` is **not** initialized to a sentinel that blocks the first improve.
- Hitting patience on eval #1 alone does **not** happen unless `patience <= 0` (disabled) or logic elsewhere is wrong — observed “stop after validating” is eval #2..#(1+patience).

### Recommended fixes (for fix sibling — pick a coherent set)

1. **Decouple mid-train timeout from early-stop usefulness (preferred)**  
   - Raise default `--select-timeout` toward something that can finish (e.g. **≥220** or protocol **400**), **or**  
   - Keep a short timeout for UI responsiveness but score early-stop on **progress_frac (and maybe crash rate) only** while `mean_lap_time is None` / timeout ≪ protocol.  
   - Document: mid-train will not see finishers until timeout ≥ lap time.

2. **All-DNF meaningful gate** (`is_meaningful_race_improvement`)  
   - Lower DNF progress epsilon (e.g. `max(0.001, min_improve * 0.002)` → 0.1% at default), **or**  
   - When both DNF, treat any strict `race_score_key` progress improvement as meaningful if `min_improve` is in “seconds” mode but no finisher exists yet.  
   - Optionally: if `new_key < best_key` and still all-DNF, reset patience (align stop with promotion until first mid-train finish).

3. **Grace before early-stop**  
   - Ignore patience until `num_timesteps >= grace` (e.g. **250k–500k**) or until `N_eval >= 2` after a minimum train budget. First eval still promotes `best_model`.

4. **Overnight defaults**  
   - Unlimited auto: raise `UNLIMITED_EVAL_REFERENCE_TS` (e.g. 1M → eval every 100k) and/or floor to **100k**.  
   - UI: for budget-off / overnight preset, default patience **≥8–10** or force grace; warn if `select_timeout < ~200` and patience ≤5.

5. **DNF ranking for mid-train** (if timeout stays short)  
   - Temporary select key: `(1, -progress, cols)` or progress-first while `timeout_s < protocol_timeout_s()`, so far-then-crash beats idle timeout. Keep official `race_score_key` for leaderboard / `--official-eval`.

6. **Tests to add** (fix sibling)  
   - Simulated sequence: all-DNF adj=60, progress 0.10 → 0.103 → 0.106 under default min_improve must **not** trip patience=3 if grace/progress fix chosen.  
   - 60 s vs 400 s: assert mid-train metrics stay DNF at 60; document expected gate.  
   - crash@0.4 vs timeout@0.02 ranking under select timeout.

### Suggested minimal patch set (if sibling wants smallest overnight-safe change)

1. `MID_TRAIN_SELECT_TIMEOUT_S = 240` (or 400) **and**  
2. early-stop grace `max(0, num_timesteps) < 200_000` → never increment `_no_improve` **and**  
3. auto eval floor `100_000` when unlimited.

Do **not** only raise patience in the UI — BUG-1 still false-plateaus; user just waits longer between identical DNF misses.

### Code anchors

- Callback / patience: `train_ppo.py` `RaceBestModelCallback._on_step`, `_race_and_maybe_promote` (~364–485)  
- Meaningful gate: `is_meaningful_race_improvement` (~182–236)  
- DNF adj proxy: `_race_eval_maps` (~171–173)  
- Constants: `ui_ops.py` `MID_TRAIN_SELECT_TIMEOUT_S`, `AUTO_RACE_EVAL_EVERY_FLOOR`, `UNLIMITED_EVAL_REFERENCE_TS`  
- Rank key: `eval_protocol.race_score_key` (~295–319)

---

## Overnight bug — fix

**Status (2026-09-17):** Shipped. Diagnosis + audit confirmed clean EarlyStop at ~40k/70k with auto eval **10k**, mid-train **60s** permanent DNF, and `live_status` stuck on `phase=learning`. Fixes below make overnight early-stop safe.

### Root cause (confirmed)

Unlimited + patience=3 + auto `race-eval-every=10000` + select-timeout=60s → a few DNF validations → EarlyStop exit 0 long before learning. First-eval baseline logic was already correct.

### Shipped fixes

| Fix | Detail |
| --- | --- |
| Mid-train timeout | `MID_TRAIN_SELECT_TIMEOUT_S = **220**` (laps ~180–210s can finish); official protocol still 400s |
| Auto eval floor | `AUTO_RACE_EVAL_EVERY_FLOOR = **50_000**`, unlimited ref **500k** → auto every ≥50k (was 10k/25k) |
| Warmup / grace | `--early-stop-warmup-evals` (auto 2 when patience>0), `--early-stop-min-timesteps` (auto **100k** when unlimited+patience). Patience never ticks during baseline / warmup / pre-min-ts |
| Unlimited patience floor | patience &lt;5 raised to **5** under `--unlimited-timesteps` |
| DNF ranking | `race_score_key`: among DNFs **progress ≻ cols ≻ adj** (far-then-crash beats idle timeout) |
| DNF meaningful | any strict key improve resets patience (aligns with best_model promotion) |
| Unscorable skip | failed / empty validation → warn, **no** early-stop tick |
| `phase=early_stopped` | RaceBest writes it; LiveStatus `_on_training_end` **preserves** terminal phase; post-`learn` rewrite |
| Overnight UI preset | budget OFF, patience 5, eval 50k, min_improve 0.5, warmup 2, min_ts 100k, select_timeout 220 |

### CLI

```text
--early-stop-warmup-evals N   # -1=auto
--early-stop-min-timesteps N  # -1=auto (100k when unlimited+patience)
--select-timeout 220          # default
```

### Overnight recommended UI settings

1. Click **Overnight (early-stop)** preset (or: budget OFF, patience **5**, eval every **50000**, min improve **0.5**).
2. Leave mid-train timeout at default **220s** (do not use 60).
3. Warmup/min-ts auto-applied by train when unlimited; preset also sends them on Start.

### Tests / smoke

- `python -m rl.ui_selftest` — **PASS** (patience state machine: baseline/warmup/grace/miss/stop/unscorable; DNF progress-first ranking; overnight preset constants; UI build `overnight-earlystop-20260917`).
- Smoke `smoke_earlystop_warmup`: patience=3, warmup_evals=4, min_timesteps=100k, race-eval-every=1024, select-timeout=8s, timesteps=8192 → **8 mid-train validations**, completed full budget, `early_stopped: false`, `no_improve` stayed 0 (grace). Proves first few Validating cycles no longer kill the run.

---

## Overnight bug — soak

**Status (2026-09-17):** **PASS** — reliability soak against post-fix `train_ppo` / `ui_ops` (sibling 4108dec0).

### Code re-verify (claims vs tree)

| Claim | Confirmed |
| ----- | --------- |
| Mid-train select-timeout **220s** | `ui_ops.MID_TRAIN_SELECT_TIMEOUT_S = 220.0` |
| Unlimited auto eval floor **50k** | `AUTO_RACE_EVAL_EVERY_FLOOR = 50_000`, ref `500_000` |
| Warmup **2** evals / min timesteps **100k** | `DEFAULT_EARLY_STOP_WARMUP_EVALS=2`, `DEFAULT_EARLY_STOP_MIN_TIMESTEPS=100_000`; CLI `--early-stop-warmup-evals` / `--early-stop-min-timesteps`; `early_stop_patience_tick` |
| DNF progress-first | `eval_protocol.race_score_key` → `(1, -progress, cols, adj)`; far-then-crash ≺ idle timeout |
| `early_stopped` preserved | `LiveStatusCallback` `preserve_terminal_phase=True` on training_end |
| Overnight UI preset | budget OFF, patience 5, eval 50k, warmup 2, min_ts 100k, select_timeout 220 |

Unit: `early_stop_patience_tick` baseline/warmup/miss/stop + min-ts grace + DNF ranking — **OK**. `python -m rl.ui_selftest` — **all checks passed**.

### Soak run (aggressive multi-eval)

| | |
| - | - |
| Run | `overnight_soak_20260917_082739` |
| Env | DummyVecEnv, `n_envs=2`, CPU, map0 train / map3 validation |
| Early-stop | patience **3**, warmup_evals **2**, min_timesteps **0** (force patience window on), min_improve 50 |
| Cadence | `--race-eval-every 2048`, `--select-timeout 5` (short wall for soak; production default remains 220) |
| Result | **PASS** ~79 s |

**Evidence**

- `live_status` phase sequence: `learning ↔ validating` × **4** full cycles; process still alive after 4th (killed by harness after PASS gate).
- Log: `warmup_evals=2`; Validation race-eval @ 2048 / 4096 / 6144 / **8192**.
- At start of 4th validating, status showed `eval_count=3`, `no_improve=1` — patience only ticking **after** warmup (not on first cycles).
- `early_stopped` **never** observed; illegal early-stop before `warmup+patience` (=5) would have failed the harness.
- Prior soak `overnight_soak_20260917_082232`: also **PASS** (3 validating→learning).

Artifacts: `rl/logs/overnight_soak_20260917_082739.log`, `…_soak_result.json`.

### Overnight recipe (user)

1. **Restart Control UI** so it loads `overnight-earlystop-20260917`: stop old UI → `.\start_ui.ps1` → open http://127.0.0.1:7860/ → **Ctrl+F5**.
2. Click **Overnight (early-stop)** preset (budget OFF, patience 5, eval every 50k, warmup 2, min_ts 100k, select-timeout 220).
3. Pick a **train_ok** map (not map2/map4 holdout, not map3 validation).
4. **Start** a new run, or **Continue** an existing overnight run (same settings).
5. Leave it: expect occasional **Validating** then back to **Learning**; **EarlyStop** only after warmup + patience plateau — not after the first Validating.

Do **not** set mid-train timeout back to 60s; do not drop patience to 1–2 with eval every ≤10k.


