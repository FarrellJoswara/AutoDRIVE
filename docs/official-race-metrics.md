# Official PPO race metrics journal

Official PPO training appends durable outcome records to
`<run output directory>/race_metrics.jsonl`. Each line is one JSON object with
`schema_version`, a stable `event_id`, `event_type`, `run_id`, and UTC write
time. Records are appended, flushed, and `fsync`ed individually. On recovery,
the trainer scans valid existing rows to restore deduplication state, episode
indices, cumulative counters, and checkpoint-interval metrics; an incomplete
trailing line is retained and separated from new rows.

## Event rows

### `lap`

One row is written for each official lap-counter transition, including a lap
whose timing is invalid. It records global PPO `timestep`, `env_id`,
`episode_index`, scored or warm-up `lap_index`, `official_lap_count`, `phase`,
`lap_time_s`, `timing_valid`, `timing_missing_reason`, and the timing update
generation when available. Collision state and faults are captured at the
transition: `collision_event`, `race_collisions`, `raw_collision_count`, and a
`faults` object with disqualification, missing timing count, timing validity,
command expiry count, neutral fallback count, and termination reason.

### `episode`

Written once when the environment supplies its terminal race outcome. It
contains the environment's `clean_10_lap_episode` and `episode_won` flags,
termination reason, completed and required lap counts, all valid
`race_lap_times_s`, timing validity and missing count, race and raw collision
counts, race duration, route frontier distance and target, SB3 episode return,
length and duration, and lap-time mean, median, and minimum for that episode.

### `checkpoint`

Written after the post-optimizer checkpoint is saved. It identifies the
checkpoint path, SHA-256, PPO timestep and optimizer update. `interval` reports
race-lap sample, valid, and invalid counts plus mean, median, and minimum time
since the prior checkpoint row. `cumulative` reports run totals for lap events,
valid laps, episodes, clean ten-lap episodes, collision episodes, and collision
count. Rolling ten-lap fields include the number of valid within-episode
windows, best ten-lap total and per-lap mean, mean ten-lap total/window mean,
and the best window's times. Windows must contain ten valid, consecutively
indexed race laps from one episode; warm-up, invalid, and episode-boundary laps
are excluded.

The file is training-derived and independent of automatic evaluation. It does
not change PPO reward, policy observations, stopping criteria, or checkpoint
selection.
