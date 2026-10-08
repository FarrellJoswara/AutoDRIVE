
### Official PPO learning-rate continuation from 80k — 2026-10-07

## Hypothesis

The prior four-epoch update used learning rate `1e-5`; at its latest rollout the approximate KL was about `0.00183` and clip fraction about `0.00464`, indicating a small policy change. Test whether changing only the learning rate to `1e-4` creates a more useful update while keeping the official environment and all other PPO, reward, observation, and action settings fixed.

## Starting checkpoint and baseline

- Starting checkpoint: `logs/rl/lidar_speed_consistency_finetune_20261007/ckpt/ppo_80000_steps.zip`; SHA-256 `33C37807A951990597C7247F8650178D791DD47AECD082969473707F55BE6FFC`.
- Fresh official `allow` attempt on those exact bytes: 194.547 s warm-up, then 11 scored collisions before a scored lap. Result: `official_ppo80k_epochs4_source_allow_attempt1_20261007.json`.
- Four-epoch continuation checkpoint at 84,096: SHA-256 `B3726EC29AC36DA64F1C8B6044091BD90DF3155C2CA3B8767DF94438416F17FF`; one official attempt timed out at 300 s with zero completed laps and zero collisions. Result: `official_ppo80k_epochs4_candidate84096_allow_attempt1_20261007.json`.
- Four-epoch continuation checkpoint at 88,192: SHA-256 `46F96A4F950EBB52FD606034BB60DC46C77D2F1F1EC2A2AA7CF92A5F69B0D155`; one official attempt timed out at 300 s with zero completed laps and zero collisions. Result: `official_ppo80k_epochs4_candidate88192_allow_attempt1_20261007.json`.
- A provenance audit found that the earlier 13.926/14.576 s laps in older reports came from a different model version despite reusing the same checkpoint filename: the stored action traces differ, and those reports predate the current checkpoint's write time. Those laps are withdrawn as baseline evidence for this experiment.
- Preserved parent checkpoint `logs/rl/lidar_speed_finetune_bidirectional_lr1e5_exploration030_from_action50_20261007/best_evaluated_model.zip`, SHA-256 `2D160983A8FA99FDDF7451768B49DA261EB7E715142E72325E93FC47AE6807DC`, was separately tested: zero-reverse mapping completed one scored lap in 15.7467 s before DQ; allowing reverse produced a 15.7327 s scored lap but was DQ. Both were invalid races. These support retaining bidirectional/allow controls but are not a valid score or promotion.

## Controlled run

- Managed official run ID: `e3370f5ed16c`, display name `official-lr1e4-from80k-20261007`.
- Requested additional timesteps: 20,480; checkpoints every 4,096. Plateau stopping off for this bounded experiment.
- PPO: `n_steps=1024`, learning rate `1e-4` (sole changed factor), `n_epochs=4`, gamma `.99`, GAE `.95`, batch 64, clip `.2`, entropy `.01`, value `.5`, gradient norm `.5`.
- Reward: time cost 1 per simulated second, +100 per warm-up/scored lap, escalating collision cost `10 * collision_number`, -1000 on DQ/watchdog.
- Policy interface: official sensor observation, bidirectional throttle, negative throttle allowed, normal steering, steering scale .945, straight-throttle gain 1.0, threshold .15.
- Runtime images: API/evaluator `aicar-iros2026:latest` digest `sha256:0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`; official simulator digest `sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`.
- Before the first rollout, the official trainer verified requested/effective learning rate `0.0001`, optimizer LR `0.0001`, four epochs, 1,024-step buffer, and the remaining PPO values. Watch telemetry connected to run `e3370f5ed16c`.
- The bounded run stopped at 92,288 and all artifacts were preserved. The continuation result is recorded below; no checkpoint has been promoted.

### Progress checkpoint — 84,096 total steps

- Saved checkpoint: `logs/rl/official_run_e3370f5ed16c/checkpoints/official_ppo_84096_steps.zip`, SHA-256 `E1AADF494C56BEA2D4A9720A39F4D71FA236B92679CDFF46417AEF6157DE1AFF`.
- Four PPO updates after the resume. Approximate KL values observed: 0.0360, 0.0150, 0.0328; clip fractions: 0.214, 0.074, 0.190; explained variance: -0.020, 0.051, 0.185. Updates are now materially larger than the `1e-5` trial; no divergence/error has appeared.
- At 84,800 global steps the run remained active, had seven episodes and no completed lap yet. The live car had zero current-episode collisions and positive body-forward speed at the last frame; this is only a snapshot, not a performance result.
- Continue to the bounded run end while monitoring stability. Evaluate only saved official-policy checkpoints in fresh official simulator processes. No promotion decision yet.

### 92,288-step checkpoint evaluation and continuation decision

- The learning-rate run was stopped cleanly after 12 PPO updates at its preserved 92,288-step checkpoint (SHA-256 `D0C3DE7CA6458C5314059B0503EE37DC623EE985E03E3A54F486A8BD371F5796`); the source checkpoint remains untouched and all run checkpoints are retained.
- PPO stayed numerically stable. Later updates had approximate KL around 0.0106–0.0196, clip fraction 0.105–0.142, and explained variance from 0.04 to 0.24. The logged episode mean return stayed near -1.61e3 and 18 episodes had produced zero completed laps at the checkpoint.
- One fresh official evaluation of the 92,288-step checkpoint completed warm-up in 171.77 s with 284 warm-up collisions, then reached 11 scored collisions before completing any scored lap (race disqualification at 178.89 s). Raw report: `official_lr1e4_92288_allow_attempt1_20261007.json`.
- Compared with the exact 80k source's one `allow` attempt (194.55 s warm-up, 374 warm-up collisions, then 11 scored collisions before a race lap), this is a modest warm-up-only improvement but still no race lap and no valid score. One trial each is not enough to claim a performance gain. Continue only a short same-settings extension to check whether the improvement grows; do not promote this policy.

### Bounded extension run from 92,288

- Managed official run: `8e83a8cd22b2`, display name `official-lr1e4-extension-from92288-20261007`.
- Starting checkpoint: `logs/rl/official_run_e3370f5ed16c/checkpoints/official_ppo_92288_steps.zip`, SHA-256 `D0C3DE7CA6458C5314059B0503EE37DC623EE985E03E3A54F486A8BD371F5796`.
- Additional budget: 8,192 official steps, checkpoint every 4,096; same PPO/reward/action settings as the prior `1e-4` run, with a fresh optimizer on resume as specified by the official resume path.
- Effective values will be verified by the new run log before rollout. This is a bounded continuation to establish whether the modest warm-up-only signal persists; it is not a promotion run.

### Bounded extension comparison — rejected

- The managed continuation `8e83a8cd22b2` started from the untouched 92,288-step checkpoint (SHA-256 `D0C3DE7CA6458C5314059B0503EE37DC623EE985E03E3A54F486A8BD371F5796`) and trained 8,192 more official steps with the same LR `1e-4`, four PPO epochs, 1,024-step rollout, reward weights, observations, and action mapping. Only policy age changed. Requested/effective PPO configuration verified before rollout. It completed at global step 100,400, with no completed laps over 11 training episodes; final diagnostics were approx-KL `0.0131`, clip fraction `0.123`, explained variance `0.237`, and action std `0.301`. Both checkpoints were preserved: 96,384 steps SHA-256 `61B817D3E7D975B35C3292030EF9C93BA1F892BB9EE53275991AF469D87F2E82`; 100,480 steps SHA-256 `105157CB1A390B119E69B75EA83436E8001A80657324C8968023CB3DD1E7AE9F`.
- Compared each checkpoint in three fresh, deterministic official simulator processes, with the same official API/evaluator digest `sha256:0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`, simulator digest `sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`, Porto setup, bidirectional throttle, reverse allowed, steering scale `.945`, straight-throttle gain `1.0`, threshold `.15`, 500-step trace cap, 300-second wall guard, and 150,000-step failsafe. All six attempts were disqualified after 11 scored collisions, with zero scored laps and therefore no valid race time.
- The source 92,288-step checkpoint warm-up times were `171.769`, `33.226`, and `34.267` s (median `34.267` s); warm-up collisions were `284`, `44`, and `43` (median `44`). The 100,480-step candidate warm-up times were `45.703`, `79.283`, and `209.465` s (median `79.283` s); warm-up collisions were `54`, `103`, and `292` (median `103`). Repetition exposes high run-to-run variance, but the candidate's median is worse on both warm-up indicators. Reject the continuation; preserve it for provenance, and use 92,288 only as the next experiment's incumbent starting checkpoint, not as a verified race champion.
- Raw results: `official_lr1e4_92288_allow_attempt1_20261007.json` through `official_lr1e4_92288_allow_attempt3_20261007.json`, and `official_lr1e4_100480_allow_attempt1_20261007.json` through `official_lr1e4_100480_allow_attempt3_20261007.json`. All use the unmodified official simulator image; no custom-simulator outcome was included.
- The intermediate 96,384-step checkpoint (SHA-256 `61B817D3E7D975B35C3292030EF9C93BA1F892BB9EE53275991AF469D87F2E82`) was also screened in two fresh official processes. Both attempts were disqualified at 11 scored collisions with zero scored laps; warm-up was 55.414 s / 92 contacts and 256.828 s / 516 contacts. The two-run medians (156.121 s / 304 contacts) were clearly worse than the 92,288-step source's three-run medians, so no third screen was warranted. Raw reports: `official_lr1e4_96384_allow_attempt1_20261007.json` and `official_lr1e4_96384_allow_attempt2_20261007.json`.
