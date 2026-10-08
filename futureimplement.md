# Official training frontier and evaluation work

## Implemented

- Official PPO lives start at the documented AutoDRIVE spawn and use restricted IPS x/y only for Layer 2 frontier reward and episode control. The coordinates are excluded from policy observations and actions.
- `competition/iros2026/official_centerline.csv` is a bounded, closed route reference derived from a completed official simulator lap trace. It is documented as a demonstrated path, not surveyed track geometry, and each training reset validates the expected spawn before initializing the frontier.
- Only new high-water frontier distance earns positive driving reward. Simulated time costs 5 reward units/second by default; lap crossings add no reward and do not end an episode. A new collision ends the life and costs 100 plus 20% of its positive frontier return. A stalled frontier resets after 10 simulated seconds with time cost only; other watchdog failures cost 100.
- Layer 4 exposes the current objective, migrates old reward settings, and packages the frontier data with the official training image.

## Still outstanding

- Layer 3 still stops on training episode-return plateau. The planned separate deterministic 30-second official evaluation every 50,000 steps, reward-rate checkpoint selection, and adaptive action-standard-deviation schedule are not implemented.
- The frontier is based on a successful recorded vehicle path. Long-run official simulator training must confirm that its 2 m corridor projects consistently across collisions, resets, and parallel environments.
- Parallel throughput, sustained timing stability, checkpoint recovery, and a valid ten-lap official result remain runtime acceptance checks; Python tests do not establish them.
