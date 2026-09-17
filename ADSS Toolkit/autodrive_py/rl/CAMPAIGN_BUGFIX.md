# CAMPAIGN_BUGFIX — Stable-core audit (Bugfix / Perf hunter)

**Scope:** contracts · racing_env · live_status · checkpoint/atomic saves (`train_ppo` + `metrics_io`) · map_pack · observation packing · Subproc workers  
**Skip:** UI reskin · Control chrome · anything board-marked **DEFER** (n_envs knee, continuous outer, GPU AMP, embed Watch)  
**Overnight:** observe-only `overnight_soak_20260917_082739` — do not kill  
**Tick:** 2026-09-17 ~04:11 America/Chicago  
**Status:** STANDING — resume each campaign tick

---

## P0 — crash / data loss / dual-writer / status truth

| ID | Where | Bug | Suggested fix | Status |
| -- | ----- | --- | ------------- | ------ |
| **P0-1** | `metrics_io.py` `atomic_save_sb3` (~old L148–158) | On Windows lock, `final.replace(bak)` failure fell through to **`final.unlink()`**, destroying the only good `best_model`/`latest_model` before the new zip was in place. Overnight promote / mid-rollout latest save → **weight loss**. | Never unlink `final`. Move-aside → replace; on failure restore from `.bak` or keep prior zip; raise without deleting. | **FIXED** this tick + `_bugfix_p0_checks` |
| **P0-2** | `metrics_io.py` `acquire_run_lock` (~old L320–354) | Lock write used `atomic_write_json` (overwrite), **not** `O_CREAT\|O_EXCL`. Two Continues / Start+CLI racing the same `run_id` could both “acquire” → dual SB3 writers → corrupt zips. | Exclusive create; stale (dead PID) unlink once + retry; lose race → refuse. | **FIXED** this tick + tests (**B0.2**) |
| **P0-3** | `live_status.py` `LiveStatusCallback._write` | Mid-trail always forced `phase=learning` unless `preserve_terminal_phase` on `training_end`. CallbackList still runs siblings after RaceBest returns False; any later/earlier trail could **clobber `early_stopped` / strip `validating` fields**. | If disk phase is terminal or `validating`, do not rewrite learning payload (clock-only refresh for terminal). | **FIXED** this tick + tests |
| **P0-4** | `metrics_io.py` `find_last_complete_checkpoint` | After a failed atomic replace, only `.bak` may remain; Continue said “no complete checkpoint”. | Prefer zip then accept `latest_model.zip.bak` / `best_model.zip.bak`. | **FIXED** this tick |
| **P0-5** | `racing_env.resolve_map_yaml` + `map_pack.assert_train_safe` | Soft resolve **silently fell back** to any `map*.yaml` / demo when id missing; train gate only blocked holdout/validation — typo could train wrong map while claiming requested id (**B0.1** / **B1.5**). | `strict=True` on train/eval + `assert_resolved_map_id`; `assert_train_safe` refuses missing yaml + non-`train_safe` ids. Soft fallback kept for Watch/demo only. | **FIXED** this tick + `_bugfix_p0_checks` |

---

## P1 — correctness / integrity (ship next ticks)

| ID | Where | Bug / risk | Suggested fix |
| -- | ----- | ---------- | ------------- |
| **P1-1** | `train_ppo.py` ~1112 `CheckpointCallback` | SB3 checkpoint writes are **not** routed through `atomic_save_sb3`. Kill/disk-full mid-write → `ppo_*_steps.zip` that passes size>1024 but fails load. | Wrap with atomic helper or post-save rename from `*_tmpsave.zip`; skip incomplete in `find_last_complete_checkpoint` (zip test open). |
| **P1-2** | `train_ppo.py` ~586 `best_model_meta.json` | Non-atomic `.write_text` beside atomic zip → torn meta / zip desync on crash. | **FIXED:** `atomic_write_json` for meta (+ `_bugfix_p0_checks`). |
| **P1-3** | `map_pack.py` `load_pack` ~215–220 | Corrupt `map_pack.json` silently → empty pack + disk adopt as `train_ok` → may **train on holdout** until protocol seals force role. | Fail loud or backup+refuse; never silent empty on JSON error if file non-empty. |
| **P1-4** | `metrics_io.py` `atomic_write_json` ~119–124 | No Windows lock retry (unlike `live_status.write_live_status`). UI reading `config.json` can raise mid-train. | Same retry/fallback pattern as live_status (non-fatal for config optional paths). |
| **P1-5** | `live_status.py` / Monitor | Crash-rate estimate depends on Monitor `info_keywords`; Subproc + Dummy parity OK today, but any env thunk without those keywords → **silent 0 crash_rate**. | Assert keywords in `_make_env` smoke; document required keys. |
| **P1-6** | `racing_env.py` `_apply_spawn_jitter` ~361–371 | After 24 rejects, returns **base pose** even if base were occupied (bad map start). | If base colliding, raise / resample from centerline. |
| **P1-7** | Subproc fallback `train_ppo.py` | Fallback to Dummy used to log WARNING only; operators could believe `n_envs` parallel while serial. | **FIXED (fail-loud):** ERROR log + `vec_env_fallback` in config/`live_status`; `--require-subproc` hard-refuses Dummy. Fallback kept as safety net. |

---

## P2 — polish / edge

| ID | Where | Note | Suggested fix |
| -- | ----- | ---- | ------------- |
| **P2-1** | `acquire_run_lock` PID check | Windows PID reuse can treat unrelated process as lock holder (false refuse) or rarely miss. | Store `CreateTime` / start time in lock; compare. |
| **P2-2** | `observation.py` `downsample_lidar` | Linear interp on angle-unsorted raw scans if bridge order ≠ gym. | Document FOV order contract; refuse len mismatch when bridge-connected. |
| **P2-3** | `contracts.py` / scoring | `SCORING_REV` independent of ABI — good; ensure all board writers stamp rev. | Grep append paths each tick. |
| **P2-4** | `map_pack.save_pack` | `os.replace` OK; no WinError 5 retry. | Retry like live_status if operators edit manifest during verify. |

---

## Perf opportunities (stable cores only — not DEFER knee/AMP)

| ID | Where | Opportunity | Risk / gate |
| -- | ----- | ----------- | ----------- |
| **PERF-1** | `racing_env.cast_lidar` | Hot path; pure Python nested loops. Numba/Cython or precompute angle table + vectorized grid sample. | Must not change ray semantics (contracts). Ablate FPS before/after. |
| **PERF-2** | `project_centerline_s` | Per-step local window already; cache `seg_len`/`cum` on env (done). Consider tighter window when on-track. | Too-tight window → progress stall / false re-anchor. |
| **PERF-3** | Subproc workers | Each worker loads full occupancy PNG. Share memmap / smaller maps for train_ok. | Correctness of occ queries; don’t touch overnight knobs mid-run. |
| **PERF-4** | `LiveStatusCallback` latest zip | Saving latest every rollout is disk-heavy (atomic rename OK). | Already gated by `--save-latest-every-rollouts`; default audit vs soak I/O. |
| **PERF-5** | Mid-train `_race_eval_maps` | Blocks `learn()` for `select_timeout` (~220s). Fast probe is separate. | **Do not** shorten sacred select timeout; only reduce frequency via existing flags. |
| **PERF-6** | Observation pack | `np.concatenate` per step allocates. Preallocate obs buffer on env. | Tiny win vs LiDAR cast; safe micro-opt. |

**Explicitly out of scope this hunt:** n_envs knee coach, `torch.compile`/AMP, continuous outer train, UI reskin.

---

## Fixes landed this tick

1. `metrics_io.atomic_save_sb3` — no destroy-on-lock; restore `.bak` on failed replace.  
2. `metrics_io.acquire_run_lock` — `O_EXCL` exclusive create + stale clear + race refuse (**B0.2**).  
3. `metrics_io.find_last_complete_checkpoint` — recover `*.zip.bak`.  
4. `live_status.LiveStatusCallback._write` — never clobber `early_stopped` / `validating`.  
5. **B0.1 / P0-5:** `resolve_map_yaml(..., strict=True)` + `assert_resolved_map_id` on train/eval; `assert_train_safe` refuses missing yaml + non-train_safe.  
6. **P1-7:** Subproc→Dummy ERROR + `vec_env_fallback` flag in live_status/config; `--require-subproc`.  
7. Tests: `python -m rl._bugfix_p0_checks` via `rl/.venv` (B0.1 + B0.2 + status).
8. **P1-2:** RaceBest `best_model_meta.json` via `atomic_write_json` (support-loop ~04:25).

**Safe for overnight:** no train process restart; library-only. Next Continue/new train picks up lock+save+map-gate fixes. Live overnight process keeps old code in memory until it exits.

---

## Next tick (STANDING resume)

1. Re-scan for regressions around Continue + dual-writer (`ui_selftest` dual-writer section).  
2. Ship **P1-1** atomic CheckpointCallback path if free hands.  
3. ~~Ship **P1-2** atomic `best_model_meta.json`.~~ **DONE** (~04:25 support loop).  
4. **P1-3** map_pack corrupt-manifest refuse.  
5. Optional PERF-6 obs buffer if cast_lidar still dominates after measuring.  
6. Keep overnight observe-only.

---

## Test commands

```text
cd "ADSS Toolkit/autodrive_py"
.\rl\.venv\Scripts\python.exe -m rl._bugfix_p0_checks
.\rl\.venv\Scripts\python.exe -m rl._p0_checks
```
