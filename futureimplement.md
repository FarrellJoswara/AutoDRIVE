# Official training frontier and evaluation work

## Implemented

- Official PPO lives start at the documented AutoDRIVE spawn and use restricted IPS x/y only for Layer 2 frontier reward and episode control. The coordinates are excluded from policy observations and actions.
- `competition/iros2026/official_centerline.csv` is a bounded, closed route reference derived from a completed official simulator lap trace. It is documented as a demonstrated path, not surveyed track geometry, and each training reset validates the expected spawn before initializing the frontier.
- New high-water frontier distance earns continuous progress reward, and each completed official lap earns a 100-point training milestone. Simulated time costs 5 reward units/second by default. A new collision ends the life and costs 100 plus 50% of positive frontier and lap reward; frontier stagnation ends the life after 5 simulated seconds, and applies the configured non-collision failure penalty.
- Layer 4 exposes the frontier, lap milestone, collision clawback, and stagnation settings; it migrates old reward settings and packages the frontier data with the official training image.

## Still outstanding

- A valid, collision-compliant official ten-lap result remains outstanding. Current automatic evaluations complete one scored lap before reaching the 11-collision disqualification threshold; no checkpoint is in the official best registry.
- The frontier is based on a successful recorded vehicle path. Further long-run training must confirm that its corridor projection remains useful across collisions, resets, and parallel environments.
- Parallel throughput, sustained timing stability, and checkpoint recovery continue to require runtime monitoring; Python tests do not establish the official simulator workflow.
