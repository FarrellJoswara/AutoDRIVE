# Porto replay regression · November 11, 2026

The exact 1,195,886-step checkpoint went from nine clean laps and a 24.768 s
median flying lap to three crashes before its first lap. Both evaluations
record the same model SHA-256, simulator executable SHA-256, and effective
manager settings. The three later attempts also started at the same pose
and issued nearly identical initial commands to the earlier attempts.

| Measurement | Earlier clean replay | Later replay of the same checkpoint |
| --- | ---: | ---: |
| Completed laps across three attempts | 9 | 0 |
| Median action interval per attempt | 59.8–60.4 ms | 72.7–76.8 ms |
| Median ROS sensor stamp interval per attempt | 59.0–59.8 ms | 67.1–71.1 ms |
| Mean camera frame age per attempt | 12.2–12.6 ms | 26.7–29.7 ms |
| Mean policy inference time per attempt | 12.5–13.8 ms | 6.1–7.0 ms |
| Control-gap faults | 0 | 0 |

In one failed replay, the car reached roughly the same part of the corner
after 186 policy decisions versus 215 in the clean attempt. Its lateral
offset there was about 35 cm versus 16 cm. All three failed attempts ended
around 30–33 m of recorded frontier progress, in the same early section.
The car was steering with fewer opportunities to correct and older camera
observations. That is a plausible mechanism for missing the turn. The exact
cause of the timing shift is still open. Policy inference actually got
faster. The evaluator only flags control gaps above one second, so its zero
fault count is compatible with a sustained change of 13–17 ms.

The simulator launcher recorded different Xorg display numbers. Across the
available Porto diagnostics, display `:8` had a median action interval of
60.5 ms over 21 attempts and display `:2` had 75.6 ms over 67 attempts.
The launcher allocates a new dummy Xorg display and the runs span different
models and times. These numbers identify groups worth comparing and do not
establish that the display number caused the timing change.

Both PPO trials completed their two audited updates. All three Porto
attempts for each trial ended in collision, including the trial with the
halved pace bonus and zero time penalty. Because the unchanged parent also
failed, the new trials cannot isolate the effect of the reward or PPO
changes. Separately, the previous unlimited run ended on an NFS quota error
during a metrics write. Quota was freed before the trials. The comparison
finished without a qualifying checkpoint and there is currently no active
training or replay.

The next controlled test is to hold the parent checkpoint, track, simulator
build, and evaluation protocol fixed. Measure sensor arrival, action publish,
and simulator command application separately, then compare a replay near
the earlier 60 ms cadence with one near 75 ms. If the outcome follows cadence,
stabilize it or train against the observed timing range before comparing
reward settings again.

Evidence: [clean attempt](../logs/rl/official_run_2de825038a08/evaluation.json),
[failed attempt](../logs/rl/official_run_6b686ed46208/evaluation.json),
[fresh parent control](../logs/rl/ppo_pivot_production_20261111/parent_control_diagnostic/result.json),
[trial results](../logs/rl/ppo_pivot_20261111/report.json).

## November 12 timing test

The same checkpoint failed all three instrumented Porto attempts
(`03fff842fa844ed4`). Relay and evaluator timestamps used the same
host monotonic clock. Across the three attempts, the median policy step
took 63–65 ms inside the environment and 7–8 ms elapsed between steps.
The relay saw simulator sensor packets 65–68 ms apart. Its round trip
to the Devkit took 13–14 ms. About 52–54 ms elapsed between the relay
enqueuing the response and receiving the next simulator packet.

The response-to-packet interval includes the socket transport and the
simulator's work. The relay measures enqueue completion, so the interval
does not identify when the simulator actually applied the command. The
earlier clean replay lacks this instrumentation, so the current interval
identifies where the loop spends time and cannot establish which stage
grew relative to the clean replay.

The saved effective manager settings and runtime hashes match across the
original clean and failed replays. The audit found no recorded change in
the policy, the simulator binary, or the ROS and relay launch arguments
that explains the slowdown. Separate hub instances and runtime scheduling
remain possible factors. The change in the allocated Xorg display number
is an observation from the logs; each run starts a new dummy display.

The second three-attempt replay (`2f8d0b8d9a9c426a`) disabled detailed
policy traces and kept relay timing enabled. It still crashed three
times. Median control intervals were 70.5–76.0 ms, compared with
70.7–73.4 ms with tracing enabled. The simulator response-to-packet
interval stayed at 53–54 ms. The slowdown and collisions reproduced
with either trace setting. Relay timing capture was disabled after
the test. [Published timing summary](porto-timing-20261112-summary.json)
