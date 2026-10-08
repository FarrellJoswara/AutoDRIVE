
### Negative-throttle-to-zero diagnostic — 2026-10-07

- Starting checkpoint: `logs/rl/lidar_speed_consistency_finetune_20261007/ckpt/ppo_80000_steps.zip` (SHA-256 `33C37807A951990597C7247F8650178D791DD47AECD082969473707F55BE6FFC`).
- Official API base digest: `sha256:4ce4334657feb4c6760aa61f23a76f8962bf80e8e746e2547b082985a95a46a2`; official simulator digest: `sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`; evaluator image `aicar-iros2026:latest`, digest `sha256:0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`.
- Single factor: map negative policy throttle to zero/coast (`negative_throttle_mode=zero`). Fixed settings: PPO, `official_sensors`, bidirectional throttle range, normal steering, steering action scale 0.945, straight throttle gain 1.0, threshold 0.15, official 1 warmup + 10 scored lap flow, 300 s failsafe, 150,000-step failsafe, 500-step trace. Fresh simulator/API processes, Docker network alias `api`.
- Result: warmup 66.045 s; 114 warmup collisions; then disqualified after 11 scored collisions before the first scored lap. Control cadence 19.65 Hz; LiDAR about 40 Hz. No valid race time; no promotion.
- A prior matched `allow` result on the same source checkpoint completed warmup in 194.547 s with 374 warmup collisions, then was also disqualified before a scored lap. Single-attempt spread is large; this pair does not establish a reliable winning mapping. Keep policy output mapping unchanged pending repeated comparisons.
- One initial attempt was invalid because the manually launched simulator lacked the managed Docker `api` network alias and received no ROS sensors. A short reproduction showed DNS lookup failures; the managed-equivalent alias was added for the valid run. The invalid attempt was excluded. No product code or official evaluation conditions changed.
- Raw result: `official_ppo80k_source_zero_attempt1_20261007.json`.
