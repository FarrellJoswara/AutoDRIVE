# Official checkpoint screen: 5e-5 continuation at 96,384 steps — 2026-10-07

## Question
Does the 96,384-step checkpoint from the 5e-5 official PPO continuation show a repeatable improvement over its 92,288-step parent in the official simulator, sufficient to justify continuing this branch?

## Checkpoint and conditions
- Candidate: `logs/rl/official_run_0e55d523cea8/checkpoints/official_ppo_96384_steps.zip`
- Candidate SHA-256: `8AA21434EBE2748E96E1036AFE6D87A1CA33E785C329DCFEDC4FE2B7C86521C3`
- Parent: `logs/rl/official_run_e3370f5ed16c/checkpoints/official_ppo_92288_steps.zip` (the preceding official PPO checkpoint)
- API/evaluator image: `aicar-iros2026:latest`, sha256 `0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`
- Simulator image: `autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete`, sha256 `749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`
- Deterministic PPO, official sensor observations, 1 warm-up lap plus 10 scored laps; bidirectional throttle, reverse allowed, normal steering, steering scale 0.945, straight gain 1.0, threshold 0.15; same settings for all candidate attempts.

## Results
Candidate raw reports:
- `official_lr5e5_96384_attempt1_20261008.json`: warm-up 57.230 s / 85 collisions; 0 scored laps; DQ at 12 scored collisions.
- `official_lr5e5_96384_attempt2_20261007.json`: warm-up 26.719 s / 31 collisions; 0 scored laps; DQ at 11 scored collisions.
- `official_lr5e5_96384_attempt3_20261007.json`: warm-up 27.121 s / 26 collisions; 0 scored laps; DQ at 12 scored collisions.

Candidate median warm-up: 27.121 s and 31 collisions. All three attempts were disqualified and produced no valid race time.

Parent comparison reports from the preceding experiment:
- `official_lr1e4_92288_allow_attempt1_20261007.json`: warm-up 171.769 s / 284 collisions; 0 scored laps; DQ at 11 scored collisions.
- `official_lr1e4_92288_allow_attempt2_20261007.json`: warm-up 33.226 s / 44 collisions; 0 scored laps; DQ at 11 scored collisions.
- `official_lr1e4_92288_allow_attempt3_20261007.json`: warm-up 34.267 s / 43 collisions; 0 scored laps; DQ at 11 scored collisions.

Parent median warm-up: 34.267 s and 44 collisions. This suggests improved warm-up behavior on the candidate, but it is not a verified race improvement: both checkpoints failed every scored-race attempt, and no valid lap-time distribution exists. Preserve both; do not call either a champion.

## Decision
Continue from the 96,384-step candidate for one bounded, same-settings 8,192-step official PPO extension. The measured warm-up improvement is enough to justify this controlled continuation, but not promotion. Re-evaluate the resulting checkpoint in three fresh official simulator processes before deciding whether to continue or change one factor.
