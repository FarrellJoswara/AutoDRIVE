# IROS 2026 practice-track frontier

`official_centerline.csv` is the training-only route reference for the Porto
practice track. It starts from the repository's Porto map centerline, aligns
that map's mesh to the active `Porto Track` mesh in the official
`2026-iros-practice` simulator image, then converts the route into the ROS IPS
frame. The registration and spawn measurements are recorded in
`official_frontier.json`. The mesh registration has 0.091 m median and 0.246 m
90th-percentile residual; the live initial IPS pose is `(0.801, 3.1583)` and is
0.088 m from the aligned route. Its tangent differs from the official spawn
heading by 0.59 degrees. Training checks the nominal spawn `(0.8,
3.1583)` after each reset before initializing the frontier.

The route keeps the Porto map's per-point corridor widths and has a 31.117 m
loop. It is a map-derived centerline, not a trajectory from the policy. Layer 2
uses restricted IPS x/y only for reward and life boundaries; those values stay
out of PPO observations and actions. The independent competition evaluator
continues to use the official `2026-iros-compete` image and official race
metrics, and does not load this practice-track route.

## Live verification

On 2026-10-08, the Layer 2 training environment was connected to the official
`2026-iros-practice` simulator image. Reset returned IPS `(0.8005, 3.1583)`;
20 low-throttle, zero-steering steps then advanced the frontier by 2.083 m with
no collision, and the last step reported positive frontier reward. The focused
run record is in `results/porto_frontier_practice_smoke_20261008.json`. This
verifies reset anchoring and reward projection under the official practice
runtime; it does not establish a completed lap, ten-lap qualification, or PPO
performance.
