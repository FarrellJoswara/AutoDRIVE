# Layer 3 — Learning to drive (plan / README)

**Status:** Implemented (v1). Build plan / checklist: [`LAYER3.md`](../../LAYER3.md).

Layer 3 is the **brain trainer**: it uses Layer 2’s Gym env and teaches a neural net to pick throttle + steering.

---

## Plain English: what problem are we solving?

Every step, the car gives us two piles of numbers:

1. **LiDAR** — 1081 distance readings in a fan around the car (“how far is stuff in each direction?”).
2. **State** — 9 normalized sensor/control values (body speeds, yaw rate, acceleration, actuator feedback, and previous commands).

We need the computer to turn that into:

```text
[throttle, steering]
```

A **neural net** does that. While **training**, we also keep score (reward from Layer 2). If the car went forward, that’s usually good. The trainer (PPO) nudges the net so “good” actions happen more often.

---

## “1D-CNN on LiDAR + MLP on state, then fuse” — what that actually means

Think of the net as a **two-input gadget** that outputs one summary, then another gadget that picks the pedals.

### Input A — LiDAR (1081 numbers in a row)

Imagine standing in a circle and measuring distance every fraction of a degree. That’s a **line of 1081 numbers**, left → right around the car, including both edges of the 270° scan.

A **1D-CNN** is a small sliding window that looks at **nearby beams together** (e.g. “these 5 beams next to each other look like a wall / a gap”). It slides along the strip and builds a shorter summary that means something like “shape of the free space.”

Why not dump all 1081 into a normal dense net? You *can*, but then every beam is mixed with every other beam with no built-in idea that beam 50 is next to beam 51. The CNN is a cheat sheet: “neighbors matter.”

### Input B — State (9 numbers)

Speed, sideways slip, last controls, etc. There’s no “strip of space” here — just a small bag of facts. A plain **MLP** (stack of fully connected layers: numbers in → mix → numbers out) is enough.

### “Fuse”

We now have:

- a summary vector from the LiDAR CNN  
- a summary vector from the state MLP  

**Fuse** = glue them into **one longer list of numbers** (e.g. 256 floats). That list is the car’s situation in a form the rest of the net likes.

```text
  lidar[1081] ──▶ 1D-CNN ──▶ summary_A ─┐
                                         ├─▶ glue ──▶ one feature vector (e.g. 256)
  state[9]    ──▶ MLP    ──▶ summary_B ─┘
                                              │
                                              ▼
                                    “what should I do?” heads
                                    (throttle + steering)
```

That glue module is what we call the **feature extractor** (`extractors.py`).

---

## Is Layer 3 “just the extractor”?

**No.**

| Piece | Where | What it does |
|-------|--------|--------------|
| Feature extractor | `src/layer3/extractors.py` | LiDAR CNN + state MLP → one feature vector |
| Env / VecEnv factory | `src/layer3/envs.py` | 1 env, or **N envs via SubprocVecEnv** |
| Demonstration driver | `src/layer3/expert.py` | Optional map-aware teacher for action labels; disabled by default |
| Behavior cloning | `src/layer3/behavior_cloning.py` | Warm-start PPO actor from LiDAR/state demonstrations |
| **Training** | `src/layer3/train.py` | PPO training, deterministic evaluation, best-checkpoint selection, evaluation plateau and bounded exploration |
| **Evaluation** | `src/layer3/evaluate.py` | Fixed simulated-time deterministic frontier, pace, crash, stall and lap diagnostics |
| **Playback** | `src/layer3/play.py` | Load `.zip`, drive — **no learning** (always 1 env) |
| Thin CLI | `scripts/demo.py train` / `play` | Flags → calls into `layer3` |
| Extractor unit test | `scripts/test_layer3_extractor.py` | Fake tensors; no Unity |

PPO’s update math is inside **Stable-Baselines3**. We wire envs + net + `learn()`.

### Many sims at once

Faster training = **N** cars in **N processes**, still **one** brain:

```text
Env0 @ port 4567 ─┐
Env1 @ port 4568 ─┼─▶ SubprocVecEnv ─▶ PPO updates ONE net
Env2 @ port 4569 ─┘
```

Implemented in `envs.py` / `train.py`. Full detail: [`LAYER3.md` §5](../../LAYER3.md).

---

## What training actually does (no jargon)

1. Start one or more Gym envs (each talks to its own sim via Layers 1–2).
2. Optional: on closed centerline maps, a cautious teacher can label LiDAR/state samples with throttle and steering. This warm-up is disabled by default. Route pose is used for labels only; it is not added to the policy observation.
3. If teacher labels were collected, behavior cloning can initialize the PPO actor from them. Otherwise PPO starts from its own initialization (or the checkpoint selected with Resume).
4. A lap crossing never ends an episode. Cars continue until a collision, stalled frontier, or configured safety termination; lap counts and times remain measurements.
5. A separate simulator evaluates an immutable policy snapshot in the background for the configured number of attempts, while the training simulators continue collecting experience. Each attempt ends on collision, frontier stall, or ten completed laps. The median selected score drives checkpoint selection, exploration changes, and plateau stopping; per-attempt scores, the mean, and the best attempt are retained for inspection. If an evaluation batch is still running at the next interval, one request for the latest policy is queued; snapshots never overlap. Attempts currently repeat the configured map spawn, so they measure simulator consistency rather than generalization across spawn conditions.
6. PPO exploration standard deviation is clamped to settings-defined bounds, reduced after evaluation improvements, and raised modestly after repeated stale evaluations. The Unity pause callback resumes physics when training exits.
7. `python -m src.layer3.evaluate --model logs/rl/<run>/final_model.zip --map-id porto` runs the same failure-or-ten-lap evaluation outside training.

Playback skips 3–4: load zip → only look → act (single car).

---

## Layer 3 files

```text
src/layer3/
  __init__.py
  extractors.py    # CNN + MLP + fuse
  envs.py          # make_env / make_vec_env (Dummy vs Subproc)
  expert.py        # centerline teacher for action labels
  behavior_cloning.py # demonstrations + actor warm start
  train.py         # curriculum + PPO + checkpoints
  evaluate.py      # repeatable policy evaluation
  play.py          # load + drive
  README.md        # this file

scripts/demo.py                  # thin CLI → layer3.train / layer3.play
scripts/test_layer3_extractor.py
```

---

## How to use (after it’s built)

```bash
# teach — smoke (1 env)
python scripts/demo.py train --n-envs 1 --timesteps 100000 --out logs/rl/run1

# teach — parallel sims
python scripts/demo.py train --n-envs 2 --timesteps 100000 --out logs/rl/run2

# watch / eval (always 1 env, no learning)
python scripts/demo.py play --model logs/rl/run2/final_model.zip --steps 2000
```

Reward math stays in Layer 2 (`src/layer2/rewards.py`). Layer 3 should not reinvent scoring.

---

## Related

- Root plan + checklist: [`LAYER3.md`](../../LAYER3.md)  
- Layer 2 env/rewards: [`../layer2/README.md`](../layer2/README.md)  
- Layer 1 driver: [`../layer1/README.md`](../layer1/README.md)
