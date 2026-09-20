# Layer 3 — PPO Training Plan

**Status:** Implemented (v1) — extractor + train/play + SubprocVecEnv path.  
**Depends on:** Layer 1 (`src/layer1/`) verified · Layer 2 (`src/layer2/`) implemented  
**Package:** `src/layer3/`

**Start here if RL is new:** [`src/layer3/README.md`](src/layer3/README.md) (plain English).  
Skim “Locked decisions” and “Build order” if you just want to implement.

---

## 1. What Layer 3 is

Layer 3 does **not** talk to Unity or Socket.IO. It only calls Gymnasium (`reset` / `step` / `close`).

A **policy** (neural net) maps `obs → action`. **PPO** (Stable-Baselines3) collects experience and updates that net. We save a `.zip` and can **play** it without learning.

```text
Unity ←→ Layer 1 (Racer) ←→ Layer 2 (AutoDriveEnv) ←→ Layer 3 (PPO / policy)
```

### Where code lives (locked)

| What | Where |
|------|--------|
| Feature extractor (CNN+MLP+fuse) | `src/layer3/extractors.py` |
| Env factory + VecEnv wiring | `src/layer3/envs.py` |
| **Training** (`learn`, checkpoints) | `src/layer3/train.py` |
| **Playback** (load zip, drive) | `src/layer3/play.py` |
| Package exports | `src/layer3/__init__.py` |
| Thin CLI | `scripts/demo.py train` / `play` → call into `layer3` |
| Extractor unit test | `scripts/test_layer3_extractor.py` |

PPO math stays inside SB3. We configure and run it.

---

## 2. Concepts (short)

Full CNN/MLP/fuse explainer: [`src/layer3/README.md`](src/layer3/README.md).

| Term | Meaning |
|------|---------|
| **Observation** | Dict: `lidar` (1080,), `state` (8,) |
| **Action** | `[throttle, steering]` ∈ [-1, 1] |
| **Feature extractor** | LiDAR 1D-CNN + state MLP → one fused vector |
| **PPO** | Training algorithm (SB3) |
| **VecEnv** | Wrapper that steps **N** envs together for one PPO |
| **SubprocVecEnv** | VecEnv where each env runs in its **own process** |
| **Checkpoint** | Saved brain (`.zip`) |

### One policy, N cars

N parallel envs collect data for **one** shared policy — not N separate models.

```text
Env0 + Sim0 (port 4567) ─┐
Env1 + Sim1 (port 4568) ─┼─▶ SubprocVecEnv ─▶ PPO updates ONE net
Env2 + Sim2 (port 4569) ─┘
```

---

## 3. Locked decisions

| Topic | Decision |
|-------|----------|
| Algorithm | **PPO** (SB3) |
| Policy | **`MultiInputPolicy`** + custom `LidarStateExtractor` |
| Extractor | 1D-CNN on `lidar` + MLP on `state` → fuse |
| Train / play modules | **`src/layer3/train.py`** and **`play.py`** (logic); `scripts/demo.py` thin CLI only |
| Parallelism | **Plan includes `SubprocVecEnv` + N sims** (see §5). Smoke first with `n_envs=1`, then scale. |
| `n_envs` default (real train) | **2** to start; CLI `--n-envs`; raise toward 4 if VRAM/CPU allow |
| Ports | `base_port + i` (default base `4567`) — one unique port per env |
| Sim launch (host v1) | Each worker **`auto_launch=True`** headless Unity (N processes on one machine) |
| Sim launch (Docker later) | Optional: sims already up → `auto_launch=False`, connect by port |
| Play | **Always 1 env** (no VecEnv needed to evaluate) |
| Device | CUDA if available, else CPU |
| Artifacts | `logs/rl/<run_id>/` — ckpt, tb, `final_model.zip`, `config.json` |
| Env knobs | Layer 2 defaults unless CLI overrides |
| Out of scope | UI · waypoint rewards · custom PPO · Docker A/B *implementation* (design must stay compatible) |

---

## 4. Target layout

```text
src/layer3/
├── __init__.py        # exports
├── extractors.py      # LidarStateExtractor
├── envs.py            # make_env, make_vec_env (Dummy vs Subproc)
├── train.py           # train loop + CLI main()
├── play.py            # rollout + CLI main()
└── README.md          # plain English + how-to

scripts/demo.py        # train | play | layer1 | layer2 | check-env
scripts/test_layer3_extractor.py

logs/rl/<run>/         # gitignored contents
```

---

## 5. SubprocVecEnv + N sims (planned implementation)

### What we implement

In **`src/layer3/envs.py`** + **`train.py`**:

1. **`make_env(port, rank, seed, ...)`** — returns a **thunk** `() -> AutoDriveEnv` (SB3 requires picklable env factories for subprocesses).
2. **`make_vec_env(n_envs, base_port, ...)`**
   - `n_envs == 1` → `DummyVecEnv` (same process; easiest smoke) **or** still Subproc with 1 worker (prefer Dummy for `n_envs=1`).
   - `n_envs >= 2` → **`SubprocVecEnv`** with `n_envs` factories on ports `base_port .. base_port+n_envs-1`.
3. **`train.py`** builds `PPO(..., env=vec_env)`, then `learn(...)`.
4. On exit: `vec_env.close()` so every worker kills its `Racer` / Unity.

### Why Subproc (not threads)

- Each env owns a **Unity process + Socket.IO server**.
- Parallel rollouts need **processes** (`SubprocVecEnv`). Threads + GIL are the wrong tool for N physics clients.
- Layer 1’s gevent thread stays **inside** each worker for I/O only.

### Sketch (illustrative)

```python
# envs.py (concept)
def make_env(port: int, seed: int, **env_kwargs):
    def _init():
        env = AutoDriveEnv(port=port, headless=True, **env_kwargs)
        env.reset(seed=seed)
        return env
    return _init

def make_vec_env(n_envs: int, base_port: int = 4567, seed: int = 0, **env_kwargs):
    env_fns = [
        make_env(port=base_port + i, seed=seed + i, **env_kwargs)
        for i in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)  # one process per AutoDriveEnv + Unity
```

```python
# train.py (concept)
vec_env = make_vec_env(n_envs=args.n_envs, base_port=args.base_port, ...)
model = PPO("MultiInputPolicy", vec_env, policy_kwargs={...}, ...)
model.learn(total_timesteps=args.timesteps)
model.save(out / "final_model")
vec_env.close()
```

### Host constraints (document in README when shipping)

| Constraint | Mitigation |
|------------|------------|
| N Unity on one GPU | Start `--n-envs 2`; watch VRAM/CPU; don’t jump to 8 |
| Port collisions | Unique `base_port + i`; fail fast if bind fails |
| Windows + spawn | Env factory must be **importable/picklable** (module-level functions, not lambdas closing over weird state) |
| Slow connect | Per-env `connect_timeout`; staggered launch optional if N sims thundering-herd |
| Play vs train | Play uses **single** `AutoDriveEnv`; train uses VecEnv |

### Relation to Docker A×N

| Mode | Who starts Unity |
|------|------------------|
| **Host / early Docker (combined)** | Each Subproc worker: `AutoDriveEnv(auto_launch=True)` |
| **Later: A×N + B** | Compose starts N sim containers; workers use `auto_launch=False` and connect to assigned ports |

**SubprocVecEnv code stays in Layer 3 either way.** Docker only changes *how* sims appear on those ports — not the VecEnv API.

### Hyperparams with N envs

| Knob | Notes with VecEnv |
|------|-------------------|
| `n_envs` | CLI; 1 smoke, 2 default “real”, 4 if machine allows |
| `n_steps` | Rollout **per env** before update; total samples/update ≈ `n_steps * n_envs` |
| `batch_size` | Must divide `n_steps * n_envs` cleanly (SB3) |

Starting points still: `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`, `gamma=0.99` — retune once N>1 if needed.

---

## 6. What each file contains

### `extractors.py` — `LidarStateExtractor`

- SB3 `BaseFeaturesExtractor`
- LiDAR → Conv1d stack; state → MLP; concat → `features_dim` (256)

### `envs.py`

- `make_env` / `make_vec_env`
- Chooses `DummyVecEnv` vs `SubprocVecEnv`
- Central place for port map + seed offsets

### `train.py`

- `make_model(vec_env, ...)` — PPO + extractor + `policy_kwargs`
- `train(...)` — callbacks, `learn`, save zip + `config.json` (include `n_envs`, ports)
- `main()` — `--n-envs`, `--base-port`, `--timesteps`, `--out`, `--device`, `--seed`, `--resume`

### `play.py`

- Single env; `PPO.load`; `predict` loop; `--model`, `--port`, `--steps`
- No SubprocVecEnv

### `scripts/demo.py`

```text
demo.py train → src.layer3.train.main
demo.py play  → src.layer3.play.main
```

---

## 7. Build order (checklist)

### Phase A — Single-env smoke (prove learning stack)

1. [ ] `src/layer3/` package: `__init__.py`, `extractors.py`, `envs.py` (`n_envs=1` / Dummy), `train.py`, `play.py`
2. [ ] `scripts/test_layer3_extractor.py`
3. [ ] Wire `demo.py train` / `play`
4. [ ] Host: `train --n-envs 1 --timesteps 5000` → zip; `play` loads it
5. [ ] Update root README status; gitignore `logs/rl/` contents

**Exit:** extractor test + train/play smoke pass.

### Phase B — SubprocVecEnv + N sims (planned; do this as part of Layer 3)

1. [ ] Implement `SubprocVecEnv` path in `envs.py` for `n_envs >= 2`
2. [ ] Verify picklable env factory on **Windows**
3. [ ] `train --n-envs 2 --base-port 4567` — two headless Unities, one policy
4. [ ] Confirm `vec_env.close()` kills both sims
5. [ ] Document VRAM guidance; optional `--n-envs 4` trial
6. [ ] Write `n_envs` / ports into `config.json`

**Exit:** stable multi-env train for ≥ tens of thousands of steps without port/zombie leaks.

### Phase C — Polish

1. [ ] Resume-from-checkpoint  
2. [ ] Reward / stagnation CLI pass-through  
3. [ ] Failure-mode notes (disconnect, NaNs)  
4. [ ] Hook for Docker `auto_launch=False` + external sims (no full Docker rewrite required yet)

---

## 8. Roadmap fit

```text
Layer 3 Phase A     → 1 env train/play on host
Layer 3 Phase B     → SubprocVecEnv + N local sims   ← planned in this doc
Docker A×N + B      → same train.py; sims provided by compose
UI                  → monitor/launch; not required to train
```

If learning stalls, tune Layer 2 rewards / truncation before rewriting VecEnv.

---

## 9. Dependencies

`gymnasium`, `torch`, `stable-baselines3` (already in `requirements.txt`).

---

## 10. Acceptance tests

| # | Check |
|---|--------|
| 1 | Extractor unit test passes |
| 2 | `train --n-envs 1 --timesteps 5000` writes a zip |
| 3 | `play --model <zip>` runs |
| 4 | `train --n-envs 2` runs; two ports in use; clean shutdown |
| 5 | Docs updated (`src/layer3/README.md`, root README) |

---

## 11. Open / fuzzy

1. Staggered Unity launch if N≥4 thrashes the GPU at startup.  
2. Exact Conv channel sizes (start 32→64).  
3. State normalization — still default **off**.  
4. Whether `n_envs=1` uses Dummy only or always Subproc — **Dummy for 1, Subproc for ≥2**.

---

## Related docs

- [`src/layer3/README.md`](src/layer3/README.md) — plain English  
- [`src/layer1/README.md`](src/layer1/README.md) · [`src/layer2/README.md`](src/layer2/README.md)  
- [`PLAN.md`](PLAN.md) · [`README.md`](README.md)
