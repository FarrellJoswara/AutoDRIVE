# Official IROS 2026 competition route

This route matches the active **SRL 2026 IROS Track** in the official
`2026-iros` Windows **compete** release. It is not the 31.12 m Porto practice
route at `competition/iros2026/official_centerline.csv`. The competition loop
is approximately 45.80 m, with a much longer initial straight.

Source: <https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-iros-2026/>

The release archive hash, installed scene hash, and mesh-to-IPS transform are
recorded in `official_frontier.json`. Geometry was inspected read-only; no
simulator binary, scene, physics, or sensor assets were changed. The route is
an extracted teacher reference, not a centerline supplied by the organizers.

## Runtime selection

`src/layer3/official_tracks.py` selects competition for `windows_external` or
a `compete` image, and practice otherwise. `AICAR_OFFICIAL_TRACK_PROFILE` can
explicitly select `competition` or `practice`. Windows training verifies the
installed `Data/level0` hash against that selection and fails on a mismatch.
Each run saves its selection and verification in `track_profile.json`.

The teacher projects measured official IPS XY onto this route and follows a
short pure-pursuit target. Neither map coordinates nor privileged pose are
student inputs: BC examples contain camera, LiDAR, vehicle state, and actions.
Front LiDAR limits throttle; it does not move the steering target sideways.

The matching Watch map is `simulator/maps/iros2026_official/occupancy`.
`scripts/export_official_competition_map.py` rebuilds its raster from a
read-only OBJ export of the active official wall mesh.

## Acceptance

A successful teacher must complete ten laps in a single official episode,
with zero collisions and sufficient measured route distance. Collection ends
at the first collision, and the BC gate rejects that dataset. Unit tests and
the synthetic bicycle-model test do not establish official lap completion.
See the run's `behavior_cloning_summary.json` and `teacher_trace.jsonl` for
actual evidence. A successful teacher also does not establish that the cloned
student can complete ten laps; that requires a separate policy evaluation.

Camera previews and progress publication run outside the driving loop with
bounded outstanding work. Layer 1 releases throttle after 250 ms without a new
policy command, preserving steering while coasting. This cannot protect
against suspension of the entire host, simulator, or ROS process. Timing gaps
are recorded in the teacher trace for diagnosis.
