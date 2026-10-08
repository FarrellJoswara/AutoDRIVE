# IROS 2026 official AutoDRIVE route frontier

`official_centerline.csv` is a training-only route-projection reference derived
from the restricted IPS positions in `logs/rl/official_run_e73d099ec8fd/evaluation.json`.
That official simulator evaluation completed a lap; the route file follows its
first complete approximately 61 m lap and uses a 2 m corridor on either side.
It is a demonstrated car trajectory, not a surveyed geometric track centerline.
The recorded official spawn is stored in `official_frontier.json` and checked on
every training reset before a new frontier is initialized.

Layer 2 uses restricted IPS x/y only to compute reward and life boundaries.
These values are excluded from PPO observations and actions. The independent
official ten-lap evaluator does not load this route and continues to score the
simulator's official lap times and collisions.
