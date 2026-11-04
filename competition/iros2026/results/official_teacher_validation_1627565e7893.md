# Official teacher validation: run 1627565e7893

**Result: passed.** The official Windows simulator reports an episode win after **10 consecutive clean laps**, exactly meeting the 10-lap goal. The episode summary records **zero collisions**, no stagnation, and termination reason `training_goal_reached`. The active scene was verified as the IROS 2026 competition track.

- Run: `1627565e7893` (Competition teacher ten laps with control watchdog)
- Started: 2026-10-10T21:27:27.183145+00:00; completion phase recorded: 2026-10-10T22:00:57.584293+00:00
- Runtime/backend: official / Windows external simulator
- Route: `competition/iros2026/tracks/competition/official_centerline.csv`, length 45.802092 m; active `SRL 2026 IROS Track` verified with scene SHA-256 `a5d11e435857e2ead32ec0eb065dbbd011c41fa80d1bf8a187177014e014b674`
- Official episode diagnostics: lap count 10, episode_won `True`, frontier progress 503.0152 m, route net progress 457.2148 m (9.9824 route lengths by net displacement; 0.8061 m short of ten centerline lengths by projection-derived net progress), 5549 forward-progress steps and zero backward, reverse-motion, or negative-throttle steps
- BC only: 48,862 transitions and 8 epochs; validation loss 0.021343 → 0.000118; checkpoint exists at `logs/rl/official_run_1627565e7893/behavior_cloned_policy.zip`
- Camera samples: dataset array shape [48862, 90, 160, 3] (`uint8`); sampled frames have nonzero pixel variation (frame 0 standard deviation 83.47; mean absolute pixel change from frame 1 to 2 is 0.263).

## Teacher timing

The teacher trace has 48,862 records (steps 1–48862). `step_duration_s`: mean 33.8 ms, median 31.7 ms, p95 41.4 ms, p99 50.0 ms, max 1207.5 ms; 83 records exceed 100 ms. `control_gap_s`: mean 32.8 ms, median 31.6 ms, p95 42.1 ms, p99 49.8 ms, max 272.7 ms; 33 gaps exceed 100 ms. Maximum adjacent logged position delta was 0.1808 m. The trace has no timestamps to assign causes to timing outliers.

The run began before the later async progress callback and preview-worker join changes on disk; those changes were not loaded by this process. `config.json` also has a stale teacher description mentioning Porto, while `track_profile.json` records the verified competition scene and `official_centerline.csv` route.

Evidence is stored in `logs/rl/official_run_1627565e7893`: `behavior_cloning_summary.json`, `latest_training_phase.json`, `stop_reason.json`, `track_profile.json`, and `teacher_trace.jsonl`.
