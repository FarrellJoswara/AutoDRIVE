# Training and Evaluation Changes

Implementation status: completed in the current working tree. Training episodes now represent car lives; laps and lap times are diagnostics only.

## Episode and lap behavior

- Training always disables lap-based termination. Cars continue through multiple laps until collision, frontier stall, or an explicitly configured safety cutoff.
- Lap count and lap times remain telemetry. A 10-lap time is reported only when the evaluator observes ten consecutive completed laps in one life; it is not required to evaluate or rank a policy.

## Reward design

- New high-water frontier distance is the only positive driving reward. Time costs 5 reward units per simulated second, so making the same frontier progress faster improves reward rate.
- No lap-crossing or lap-completion bonus is paid.
- Collision settings are backend-owned: fixed cost defaults to 100, plus 20% of positive frontier reward earned in that car's current life. Both settings accept positive values; Layer 2 applies the negative sign.
- Non-collision early episode endings receive a separate default cost of 100. Collision termination remains configurable and enabled by default.

## Deterministic evaluation and exploration

- Layer 3 runs a deterministic evaluation on its own simulator for a fixed 30 simulated seconds every 50,000 training steps by default. It records frontier distance and pace, reward per simulated second, crashes, failed episodes, lap count and lap times when available.
- Evaluation reward per simulated second is the checkpoint and plateau score. It includes frontier progress, the time cost, reversing, and failure penalties; raw frontier pace remains a separately reported metric.
- The best evaluated checkpoint is preserved and selected as the final model when an evaluation has completed.
- Action standard deviation is clamped to backend-configured bounds (default 0.2–0.8). It is scaled down by 0.9 after an improvement and raised by 1.1 after each three stale evaluations.
- Plateau stopping uses consecutive deterministic evaluations and minimum training steps. It no longer waits for any lap count.
- Docker allocates one extra simulator for evaluation, including at the maximum of 16 training environments.

Validation: focused Python tests and the Train UI production build pass. A live PPO training/evaluation run has not been started as part of this implementation.
