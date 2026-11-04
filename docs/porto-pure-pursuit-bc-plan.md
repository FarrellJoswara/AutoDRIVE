# Porto pure pursuit and behavior cloning: Luna execution plan

Status: planned; simulator driving, fitting, and evaluation have not started.

## Objective and scope

Make the existing pure-pursuit teacher drive clean laps on the official Porto practice simulator, collect demonstrations, behavior-clone a sensor policy, and demonstrate that the student can drive Porto independently. PPO is a later phase and must not start as part of this plan.

All execution subagents use `gpt-6-luna`. The coordinating agent owns integration and advancing the gates. Runtime operations are serial: one agent owns the simulator and actuator commands at a time. Agents may inspect independent code in parallel, but shared-file edits must be assigned to one owner.

## Confirmed starting point

- `src/layer3/expert.py` contains `CenterlineExpert`: pure pursuit using official IPS pose and route geometry, conservative speed control, and front LiDAR clearance.
- `src/layer3/official_train.py` already couples teacher collection, collection acceptance, cloning, and a BC-only exit. `src/layer3/behavior_cloning.py` supplies collection and fitting functions.
- The official Porto runtime is `autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice`. The existing official Windows binary is the competition release. Check actual available runtime state again during execution.
- `competition/iros2026/official_centerline.csv` is the aligned official Porto practice route, approximately 31.12 m long. Its metadata is `competition/iros2026/official_frontier.json`. Do not substitute the repository-map coordinates directly for this aligned route.
- `official_tracks.py` supports explicit practice selection through `AICAR_OFFICIAL_TRACK_PROFILE=practice`.
- The Porto smoke result proves 20 action steps and movement, not a completed lap.
- The BC diagnostic evaluator currently restricts the route to the competition frontier and needs a Porto diagnostic path.
- Saved Mission Control settings select an unrelated competition PPO resume. Use a new Porto configuration and output directory; preserve existing artifacts and campaign state. Inspect active processes before allocating runtime resources. Respect existing admission checks; if they prevent a separate Porto run, report the concrete conflict instead of silently bypassing them.

## Task 1 — Luna runtime agent: explicit Porto configuration

Primary ownership: a new Porto launch/configuration entry point, `official_tracks.py` if needed, and runtime documentation. Coordinate changes to train/evaluation files with their owners.

1. Check the official practice image, API image, ports, ROS domain, GPU availability, and existing simulator/trainer processes.
2. Prepare an isolated Porto preset: practice profile, one environment, fresh initialization, BC enabled, BC-only exit, automatic competition evaluation disabled, and no campaign tags or resume checkpoint.
3. Use the official Docker practice runtime and the documented API bridge. Verify renderer and camera output before choosing `official_sensors_camera`. Do not silently switch observation profiles to work around broken camera rendering. Do not use `-nographics` for camera observations. If rendering is unavailable, record the failure and identify the compatible official practice runtime needed.
4. Pin and record simulator image identity, route hash, spawn alignment, observation profile, action transforms, and effective configuration. Start the teacher speed target explicitly at 0.3 m/s.
5. Use identity action transforms: bidirectional throttle, negative-throttle handling `allow`, normal steering, steering scale 1, and throttle gain 1. The teacher itself commands nonnegative throttle.

Gate: fresh telemetry, usable sensor frames, confirmed practice route and spawn, and isolated runtime ready. Report actual runtime evidence rather than image names alone.

## Task 2 — Luna controller agent: teacher-only driving

Primary ownership: `src/layer3/expert.py` and a new teacher-only entry point, with collector changes coordinated through Task 3.

1. Expose a teacher-only mode that uses the same official environment and route as BC collection without constructing or fitting a PPO policy.
2. Save pose, route projection, lateral error, action, lap events, collision counters, and timing diagnostics. Record step duration, control gaps, pose age, command expiry, and speed source where available.
3. Validate steering sign and coordinate conversion at low speed on Porto. Check route direction and seam crossing. Set and document a lateral-error bound from the route corridor and vehicle clearance before accepting runs.
4. Tune lookahead and speed only when driving evidence identifies a problem. Preserve failed traces and record effective parameters for each attempt.
5. Stop and neutralize controls on collision, stale telemetry, command expiry, sustained stall, operator stop, or a bounded step/wall-time guard. Control observations/actions still occur in steps; this plan does not introduce a modified simulator or assume lockstep physics.

Gate: three consecutive complete clean Porto laps in each of two fresh simulator starts, zero collisions or stalls, forward route traversal, valid corridor tracking, and no telemetry/control faults. Cross-check simulator lap counters against route traversal. A guard timeout or partial lap is a failed attempt.

## Task 3 — Luna dataset agent: accepted Porto demonstrations

Primary ownership: `src/layer3/behavior_cloning.py`; own any shared collector edits needed by Task 2.

1. Reuse the accepted controller configuration. Collect to a completed-lap goal with an explicit step/wall-time cap, rather than assuming the default 4,000 transitions is enough. Start with ten clean laps total across at least two fresh episodes; increase data only if student evaluation warrants it.
2. Preserve episode boundaries and lap boundaries. Configure the official episode lap goal to match collection; the existing acceptance function requires a successful episode, not merely crossing a lap line mid-episode.
3. Reject a collection attempt containing collision, failed episode, stall, telemetry/control fault, or incomplete required lap goal. Preserve rejected attempts separately with their reasons.
4. Save only `lidar`, `state`, `camera` when enabled, and `actions` in the student archive. Keep privileged pose, route progress, and diagnostics in separate trace files.
5. Record source runtime/route identities, controller parameters, seed, action mapping, observation shapes, timing statistics, lap boundaries, and collection acceptance evidence.

Gate: accepted clean episodes, correct observation/action pairing and array shapes, finite data, and no privileged fields in the student archive.

## Task 4 — Luna BC agent: fit and export the student

Primary ownership: `src/layer3/official_train.py` integration and fitting/export code, with edits to `behavior_cloning.py` handed off from Task 3.

1. Fit from the accepted dataset using the existing policy architecture and actor cloning implementation. Begin with eight epochs, batch size 256, and learning rate 0.001; record changes if validation requires tuning.
2. Hold out a whole fresh episode or complete laps, preserving chronology. Avoid distributing adjacent frames across training and validation.
3. Record held-out loss and per-action error. Ensure the camera is normalized exactly once and deployed preprocessing matches collection.
4. Save `behavior_cloned_policy.zip`, observation/action metadata, dataset and checkpoint hashes, effective configuration, and fitting summary in the isolated Porto run directory.
5. Exit before any PPO rollout or optimizer update. Treat loss as a fitting diagnostic; driving acceptance belongs to Task 5.

Gate: reproducibly loadable checkpoint with complete Porto provenance and a sensor-only observation contract.

## Task 5 — Luna evaluation agent: independent Porto student laps

Primary ownership: `src/layer3/official_evaluate.py` and the minimum CLI/service support needed for a practice diagnostic.

1. Add explicit verified practice-track diagnostic support. Keep competitive evaluation rules separate; Porto diagnostic results must not promote a competition best checkpoint.
2. Launch a fresh official Porto simulator per attempt with the same runtime, observation preprocessing, action mapping, and route identity as collection.
3. Run deterministic student actions with teacher control disabled and no learning. Privileged pose may support diagnostics, but must never enter policy inputs or correct its actions.
4. Use the same three-lap target in three fresh attempts. Record laps, collisions, stalls, route coverage, lap times, timing faults, checkpoint hash, and termination reason for each attempt.
5. If the student fails, diagnose the saved trace: controller/route errors return to Task 2; missing coverage returns to Task 3; fitting errors return to Task 4. Retain a bounded budget for each attempt. Repeated runtime failures pause dependent work with a concrete blocker.

Gate: all three fresh attempts complete three clean laps independently with zero collisions, stalls, or telemetry/control faults. Low imitation loss alone never satisfies this gate.

## Scheduling, verification, and deliverables

- Task 1 configuration work, Task 2 entry-point work, and Task 5 evaluator implementation may proceed in parallel with distinct file owners. Runtime driving begins after Task 1 passes. Tasks 3, 4, and the actual Task 5 evaluations then run in order.
- Add focused tests for new route/profile selection, teacher-only behavior, collection termination/acceptance, and practice diagnostic isolation. Run the affected checks and actual simulator gates; avoid unrelated test expansion.
- Use a dedicated run directory under `logs/rl/porto_bc_<run-id>/` for configuration, teacher traces, accepted/rejected collection reports, demonstrations, fitting summary, checkpoint, and independent student results.
- Finish with a short evidence report linking the teacher lap results, dataset, checkpoint, and student evaluation. State failures and uncompleted gates explicitly.
- After the student gate passes, prepare a separate Porto PPO handoff using this checkpoint and its metadata. Determine conservative action noise from Porto action scales; do not inherit the old competition campaign or its exploration profile by default. PPO launch is outside this plan.
