# Official PPO continuation: four epochs — 2026-10-07

## Hypothesis

The prior official continuation used one PPO epoch per rollout and showed a
very small update (approximate KL 0.00085, clip fraction 0.0078) after 4,096
steps. Increasing only `n_epochs` from 1 to 4 may make each collected rollout
produce a more consequential learning update. Race performance, not training
reward, determines whether this helps.

## Starting point and baseline

- Starting checkpoint:
  `logs/rl/lidar_speed_consistency_finetune_20261007/ckpt/ppo_80000_steps.zip`
- SHA-256:
  `33C37807A951990597C7247F8650178D791DD47AECD082969473707F55BE6FFC`
- This preserved checkpoint was already evaluated with official images and
  permitted policy inputs. Two available attempts recorded first scored laps
  of 13.926 s and 14.576 s, then reached 11 scored collisions and were
  disqualified. There are zero valid 10-lap attempts in the full saved results
  set, so these are diagnostic baselines only—not a champion time.
- Official Devkit base image:
  `autodriveecosystem/autodrive_roboracer_api:2026-iros-compete`,
  `sha256:4ce4334657feb4c6760aa61f23a76f8962bf80e8e746e2547b082985a95a46a2`.
- Official simulator image:
  `autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete`,
  `sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`.
- Existing PPO policy/evaluator runtime image used by the run manager:
  `aicar-iros2026:latest`,
  `sha256:0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`.

## Controlled settings

- Requested additional timesteps: 20,480; plateau stopping disabled for this
  bounded experiment; checkpoint every 4,096 steps.
- PPO: `n_steps=1024`, `learning_rate=0.00001`, `n_epochs=4` (the sole
  experimental change), `gamma=0.99`, `gae_lambda=0.95`; batch size 64,
  clip range 0.2, entropy coefficient 0.01, value coefficient 0.5,
  max gradient norm 0.5.
- Official race flow: one warm-up lap, 10 scored laps, official checkpoint
  collision resets, terminate after more than 10 scored collisions; 600 s
  training episode watchdog.
- Observation/action: `official_sensors`; permitted sensor observations only;
  bidirectional throttle, reverse allowed, normal steering, steering scale
  0.945, straight throttle gain 1.0, steering threshold 0.15.
- Training reward remains fixed: 1 reward cost per simulated second, +100 per
  completed warm-up/scored lap, escalating collision penalty of 10 times the
  collision number, and -1000 for disqualification or watchdog failure.
- This checkpoint is a starting weight source from earlier local training;
  every training step and every performance judgment in this experiment uses
  the pinned official simulator/API runtime. No custom-simulator result is used
  as a score or promotion criterion.

## Run and result

- Managed run ID: `9dec8907c63c`, display name
  `official-ppo80k-epochs4-20261007`; status was `running` at start.
- Before collecting the first rollout, the run printed and verified effective
  PPO values: `n_steps=1024`, rollout buffer 1024, `n_epochs=4`, learning rate
  and optimizer LR `1e-5`, `gamma=0.99`, `gae_lambda=0.95`, batch size 64,
  clip 0.2, entropy coefficient 0.01, value coefficient 0.5, and grad norm
  0.5. The checkpoint policy architecture loaded as `lidar_cnn` with
  `LidarStateExtractor`.
- Exact runtime image IDs observed on the created containers: API/evaluator
  `aicar-iros2026` (`sha256:0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`)
  and official simulator
  `autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete`
  (`sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`).
- Watch telemetry connected to the selected run and reported the initial
  official rollout at 80,000 steps. PPO diagnostics and race evaluation results
  will be added after they are available. The run has its own output directory
  and checkpoints and does not replace the starting checkpoint.
- Progress snapshot: at 84,600 total steps the managed run was still `running`,
  with no errors and Watch telemetry current. Four PPO updates completed;
  `explained_variance` progressed from -0.289 at update 1 to 0.088 at update 4,
  while the fourth update's approximate KL was 0.00150 and clip fraction
  0.00317. The trainer reported six episodes, two completed laps, no 10-lap
  attempt, and a best lap of 16.5725 s. This is training telemetry, not an
  official evaluation result, and does not establish improvement.
- First run checkpoint:
  `logs/rl/official_run_9dec8907c63c/checkpoints/official_ppo_84096_steps.zip`,
  54,919,358 bytes, SHA-256
  `B3726EC29AC36DA64F1C8B6044091BD90DF3155C2CA3B8767DF94438416F17FF`.
  It is preserved in the run directory and has not been promoted.
- Any candidate promotion requires repeated complete official 10-lap
  evaluations and a better median adjusted time without reduced completion
  consistency.
