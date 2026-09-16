# AutoDRIVE F1TENTH / RoboRacer RL — Phase 1 Research Notes

**Purpose.** Phase 1 research artifact for our AutoDRIVE F1TENTH / RoboRacer reinforcement-learning stack. It records competition format, legal sensors, what winning teams actually ship, and the design locks that later phases must honor. **Docs only — no code.** These notes inform Phases 2–6; they do not implement anything.

---

## Competition format

RoboRacer **Sim Racing League** (AutoDRIVE) is **solo time-attack**: each team races the clock alone on the same track. It is **not** head-to-head multi-car racing.

| Stage | What happens |
| --- | --- |
| **Qualification** | Complete clean autonomous laps on the **practice** track released ahead of time (typically 10 consecutive race laps within collision tolerance). Speed matters less than finishing without DQ. |
| **Finals** | **Time-attack** on a **previously unseen** competition track. Ranked by adjusted race time. |

**Race structure (12 laps total):**

1. **Warm-up lap** — time and collisions ignored (startup / connect buffer).
2. **10 race laps** — only these count for race time and collision penalties.
3. **Cool-down lap** — time and collisions ignored; finishing it is optional.

**Scoring (official results match this closed form):**

- `adjusted_race_time = race_time + 10 × collision_count`
- Each counted collision adds **10 s** to the cumulative penalty (1st → +10 s total, 2nd → +20 s total, …).
- **>10 collisions** in a single racing event → **disqualification**.
- On collision, the vehicle is reset to the last checkpoint; the lap timer does **not** reset.

Warm-up / cool-down collisions and times are excluded from scoring.

---

## Sensors (AutoDRIVE Technical Guide)

Topic roles below follow the competition Technical Guide (`/autodrive/roboracer_1/...` ROS 2 streams; older docs may say `f1tenth_1`). **Restricted** streams may be used for training, logging, and reward shaping **offline**; they must **not** feed the race-time policy.

### INPUT — legal at race inference

| Stream | Role |
| --- | --- |
| `front_camera` | RGB image |
| `lidar` | 2D LaserScan (primary exteroception for our stack) |
| `imu` | IMU |
| `left_encoder` / `right_encoder` | Wheel joint state |
| `throttle` / `steering` | Actuator feedback (current commanded/applied values) |

**Outputs (actuators):** `throttle_command`, `steering_command`.

### RESTRICTED — training / debug only (not race inference)

| Stream | Role |
| --- | --- |
| `ips` | Indoor positioning / ground-truth pose |
| `lap_count` | Lap counter |
| `lap_time` | Current lap elapsed |
| `last_lap_time` | Previous lap time |
| `best_lap_time` | Best lap so far |
| `collision_count` | Collision counter |
| `speed` | Ground-truth speed |
| `reset_command` | Sim reset trigger |
| `tf` | Transform tree / pose ground truth |

**Camera note:** `front_camera` **exists and is legal** at race time. We still choose **LiDAR-primary** for Phases 1–6 (sample efficiency, gym parity, simpler obs). Camera remains a later optional modality, not a Phase 1–6 requirement.

---

## What top teams do

### ICRA 2025 Sim Racing finals (0 collisions for podium)

| Rank | Team | Race time | Collisions | Adjusted |
| --- | --- | --- | --- | --- |
| 1 | **VAUL** (Université Laval) | 111.46 s | 0 | 111.46 s |
| 2 | **Autoware Aces** | 122.16 s | 0 | 122.16 s |
| 3 | **Kanka** (U. Minnesota) | 129.28 s | 0 | 129.28 s |

Dominant pattern among winners: **classical autonomy stacks** — map / localize → optimize a **raceline** → track with **Pure Pursuit** (or equivalent geometric/MPC control), often with a full ROS 2 pipeline (VAUL particle-filter localization + PP; Autoware Aces via Autoware; Kanka classical stack). Zero collisions beat raw top speed with crashes (see Autoware Aces quals: fast lap but 10 collisions → poor adjusted time).

### IROS 2024 Sim Racing finals (0 collisions for podium)

| Rank | Team | Race time | Collisions | Adjusted |
| --- | --- | --- | --- | --- |
| 1 | **TURTLEBOT** | 141.59 s | 0 | 141.59 s |
| 2 | **IDEA_LAB** | 186.58 s | 0 | 186.58 s |
| 3 | **KU F1TENTH** | 244.88 s | 0 | 244.88 s |

Same lesson: finish clean on the unseen track.

### Competitive RL (qualification / research track)

Where RL appears competitively in this format, common choices are:

- **LiDAR** observations (downsampled scans), not vision-first.
- **PPO** (or similar on-policy RL).
- **Discrete / MultiDiscrete** action spaces over throttle and steering bins (easier credit assignment than continuous for sparse collision signals).

Classical map→raceline pipelines still own the finals podium; RL is viable if it generalizes to unseen tracks and stays near **zero collisions**.

### Useful open references

- VAUL AutoDRIVE workspace: [vaul-ulaval/autodrive_roboracer_ws](https://github.com/vaul-ulaval/autodrive_roboracer_ws)
- Competition pages: ICRA 2025 / IROS 2024 Sim Racing League (AutoDRIVE Ecosystem site) — see Sources.

---

## Design implications for our stack

| Implication | Why |
| --- | --- |
| **Collision-first over raw speed** | +10 s per crash (and DQ after 10) dominates ranking; podium teams all had **0** collisions in finals. |
| **No opponent / multi-agent training** | Wrong format — solo time-attack only. |
| **No human-demo behavior cloning** as a Phase 1–6 pillar | Winners use classical stacks or learned policies from interaction; we do not depend on demonstration datasets. |
| **Train in `f1tenth_gym` on procedural tracks; eval / transfer on AutoDRIVE** | Gym supports fast multi-map training; AutoDRIVE Unity tracks are not easily generated from Python. |
| **LiDAR-primary; camera later optional** | Legal camera exists; we still lock LiDAR for Phases 1–6. |
| **Adjusted score** | `adjusted_score = lap_time + 10 * collisions` (same 10 s/collision rule as race time + penalties). |

Shared locks for all later phases: solo time-attack, LiDAR-primary obs, no restricted topics in the race policy, adjusted-time metric above.

---

## Phase independence reminder

**Phase 1 is documentation only** (`research_notes.md` + `contracts.md`). No runtime, no deps, no sim.

Later phases **soft-depend** on these notes and contracts (they should honor the locks) but each phase remains a shippable unit with its own pass criteria and fallbacks. Phase 1 does not block building Phase 2–6 artifacts independently once the contracts are frozen.

---

## Sources / links

### Competition — format, rules, results

- [RoboRacer Sim Racing League @ ICRA 2025](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-icra-2025/) — format, quals/finals tables
- [RoboRacer Sim Racing League @ IROS 2024](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-iros-2024/) — finals: TURTLEBOT, IDEA_LAB, KU F1TENTH
- [Competition Rules (2025)](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-rules-2025/) — solo time-attack, warm-up / 10 race / cool-down, 10 s/collision, DQ >10
- [Competition Rules (2024)](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-rules-2024/) — same core scoring structure
- [Technical Guide (2025)](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2025/) — Input vs Restricted ROS 2 topics, sensor specs

### Teams / stacks

- [vaul-ulaval/autodrive_roboracer_ws](https://github.com/vaul-ulaval/autodrive_roboracer_ws) — VAUL AutoDRIVE Sim Racing environment
- [VAUL robots (F1TENTH: map / particle filter / Pure Pursuit)](https://vaul.fsg.ulaval.ca/en/robots/)
- Autoware write-up on Sim Racing access: [autoware.org — sim racing](https://autoware.org/how-sim-racing-is-expanding-access-to-autonomy/)

### Training substrate (later phases)

- [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) — gym training env + track tooling we plan to vendor
- AutoDRIVE Devkit / Simulator (this repo + AutoDRIVE Ecosystem containers) — race-time transfer / eval target
