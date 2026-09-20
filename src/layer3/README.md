# Layer 3 — Learning to drive (plan / README)

**Status:** Implemented (v1). Build plan / checklist: [`LAYER3.md`](../../LAYER3.md).

Layer 3 is the **brain trainer**: it uses Layer 2’s Gym env and teaches a neural net to pick throttle + steering.

---

## Plain English: what problem are we solving?

Every step, the car gives us two piles of numbers:

1. **LiDAR** — 1080 distance readings in a fan around the car (“how far is stuff in each direction?”).
2. **State** — 8 other numbers (speed, slip, last throttle/steer, etc.).

We need the computer to turn that into:

```text
[throttle, steering]
```

A **neural net** does that. While **training**, we also keep score (reward from Layer 2). If the car went forward, that’s usually good. The trainer (PPO) nudges the net so “good” actions happen more often.

---

## “1D-CNN on LiDAR + MLP on state, then fuse” — what that actually means

Think of the net as a **two-input gadget** that outputs one summary, then another gadget that picks the pedals.

### Input A — LiDAR (1080 numbers in a row)

Imagine standing in a circle and measuring distance every fraction of a degree. That’s a **line of 1080 numbers**, left → right around the car.

A **1D-CNN** is a small sliding window that looks at **nearby beams together** (e.g. “these 5 beams next to each other look like a wall / a gap”). It slides along the strip and builds a shorter summary that means something like “shape of the free space.”

Why not dump all 1080 into a normal dense net? You *can*, but then every beam is mixed with every other beam with no built-in idea that beam 50 is next to beam 51. The CNN is a cheat sheet: “neighbors matter.”

### Input B — State (8 numbers)

Speed, sideways slip, last controls, etc. There’s no “strip of space” here — just a small bag of facts. A plain **MLP** (stack of fully connected layers: numbers in → mix → numbers out) is enough.

### “Fuse”

We now have:

- a summary vector from the LiDAR CNN  
- a summary vector from the state MLP  

**Fuse** = glue them into **one longer list of numbers** (e.g. 256 floats). That list is the car’s situation in a form the rest of the net likes.

```text
  lidar[1080] ──▶ 1D-CNN ──▶ summary_A ─┐
                                         ├─▶ glue ──▶ one feature vector (e.g. 256)
  state[8]    ──▶ MLP    ──▶ summary_B ─┘
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
| **Training** | `src/layer3/train.py` | PPO `learn()`, checkpoints, save `.zip` |
| **Playback** | `src/layer3/play.py` | Load `.zip`, drive — **no learning** (always 1 env) |
| Thin CLI | `scripts/demo.py train` / `play` | Flags → calls into `layer3` |
| Extractor unit test | `scripts/test_layer3_extractor.py` | Fake tensors; no Unity |

PPO’s update math is inside **Stable-Baselines3**. We wire envs + net + `learn()`.

### Many sims at once (planned)

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
2. Build a PPO model that **includes** our extractor.  
3. Repeat many times: look → act → get reward → remember (N envs do this in parallel when `n_envs>1`).  
4. Every so often, PPO updates the net weights from those memories.  
5. Save a **checkpoint zip** (the “trained brain”).

Playback skips 3–4: load zip → only look → act (single car).

---

## Files we will create (Python)

```text
src/layer3/
  __init__.py
  extractors.py    # CNN + MLP + fuse
  envs.py          # make_env / make_vec_env (Dummy vs Subproc)
  train.py         # learn + save
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

# teach — parallel sims (planned)
python scripts/demo.py train --n-envs 2 --base-port 4567 --timesteps 100000 --out logs/rl/run2

# watch / eval (always 1 env, no learning)
python scripts/demo.py play --model logs/rl/run2/final_model.zip --steps 2000
```

Reward math stays in Layer 2 (`src/layer2/rewards.py`). Layer 3 should not reinvent scoring.

---

## Related

- Root plan + checklist: [`LAYER3.md`](../../LAYER3.md)  
- Layer 2 env/rewards: [`../layer2/README.md`](../layer2/README.md)  
- Layer 1 driver: [`../layer1/README.md`](../layer1/README.md)
