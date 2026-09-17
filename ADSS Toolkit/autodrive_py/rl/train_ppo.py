"""Phase 4: PPO training on RacingEnv + honest best_model / resume / fingerprint."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .compat import ContractsMismatch, load_ppo_refusing_mismatch, refuse_model_obs
from .contracts import (
    CONTRACTS_VERSION,
    N_LIDAR_DEFAULT,
    SPEED_MAX_MPS,
    STALL_TIMEOUT_S,
    STEERING_BINS,
    THROTTLE_BINS,
    TIMEOUT_S,
    obs_dim,
)
from .eval_protocol import map_file_hash, race_score_key
from .map_pack import (
    HoldoutViolation,
    assert_train_safe,
    pack_fingerprint,
    validation_map,
    verify_pack,
)
from .metrics_io import (
    acquire_run_lock,
    atomic_save_sb3,
    atomic_write_json,
    config_fingerprint,
    find_last_complete_checkpoint,
    git_sha_short,
    make_metrics,
    release_run_lock,
    write_run_artifacts,
)
from .racing_env import RacingEnv, resolve_map_yaml
from .ui_ops import (
    AUTO_RACE_EVAL_EVERY_FLOOR,
    DEFAULT_EARLY_STOP_MIN_TIMESTEPS,
    DEFAULT_EARLY_STOP_PATIENCE_UNLIMITED,
    DEFAULT_EARLY_STOP_WARMUP_EVALS,
    MID_TRAIN_SELECT_TIMEOUT_S,
    UNLIMITED_EVAL_REFERENCE_TS,
    UNLIMITED_TIMESTEPS_SAFETY,
)


def resolve_device(requested: str) -> str:
    import torch

    req = (requested or "auto").lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return req


def _parse_net_arch(s: str) -> list[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        return [256, 256]
    return [int(p) for p in parts]


def _parse_maps(map_arg: str) -> list[str]:
    """Comma-separated map ids, or single id."""
    parts = [p.strip() for p in str(map_arg).split(",") if p.strip()]
    return parts or ["map0"]


def _make_env(
    map_yaml: Path,
    n_lidar: int,
    seed: int,
    rank: int,
    *,
    spawn_jitter: bool,
    collision_first: bool,
    speed_gate: bool,
    ttc_truncate: bool,
    lidar_dr: bool,
):
    def _thunk():
        from stable_baselines3.common.monitor import Monitor

        env = RacingEnv(
            map_yaml=map_yaml,
            n_lidar=n_lidar,
            seed=seed + rank,
            spawn_jitter=spawn_jitter,
            collision_first=collision_first,
            speed_gate=speed_gate,
            ttc_truncate=ttc_truncate,
            lidar_dr=lidar_dr,
        )
        return Monitor(
            env,
            info_keywords=("collision", "stall", "ttc", "crash_tag", "progress_frac"),
        )

    return _thunk


def _eval_policy(env: RacingEnv, model, episodes: int = 3, *, seed0: int = 0):
    lap_times = []
    collisions = 0
    returns = []
    progress = []
    for i in range(episodes):
        obs, info = env.reset(seed=seed0 + i)
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            if info.get("collision"):
                collisions += 1
            if info.get("lap_time") is not None:
                lap_times.append(float(info["lap_time"]))
            done = terminated or truncated
        returns.append(ep_ret)
        progress.append(env.progress_frac)
    mean_lap = float(sum(lap_times) / len(lap_times)) if lap_times else None
    return mean_lap, collisions, float(np.mean(returns)), float(np.mean(progress))


def _race_eval_maps(
    model,
    map_yamls: list[Path],
    *,
    n_lidar: int,
    episodes: int,
    seed: int,
    timeout_s: float = TIMEOUT_S,
) -> dict:
    """Quick race score on given maps (used for best_model promotion during train).

    Runs on a short budget by default, so most candidates DNF and are ranked by
    distance covered. The true race score comes from ``--official-eval``.
    """
    all_laps: list[float] = []
    cols = 0
    rets: list[float] = []
    progs: list[float] = []
    for my in map_yamls:
        env = RacingEnv(map_yaml=my, n_lidar=n_lidar, seed=seed, timeout_s=timeout_s)
        mean_lap, c, mean_ret, mean_prog = _eval_policy(env, model, episodes=episodes, seed0=seed)
        cols += c
        rets.append(mean_ret)
        progs.append(mean_prog)
        if mean_lap is not None:
            all_laps.append(mean_lap)
    mean_lap = float(sum(all_laps) / len(all_laps)) if all_laps else None
    metrics = make_metrics(
        run_id="train_race_eval",
        policy="ppo",
        backend="gym",
        mean_lap_time=mean_lap,
        total_collisions=cols,
        n_episodes=episodes * len(map_yamls),
        tracks_eval=[p.stem for p in map_yamls],
        kind="train_eval",
        mean_return=float(np.mean(rets)) if rets else None,
        mean_progress_frac=float(np.mean(progs)) if progs else None,
        dnf=mean_lap is None,
    )
    if metrics.get("adjusted_time") is None:
        # DNF proxy so promotion can still rank; race_score_key keeps DNFs behind finishers.
        metrics["adjusted_time"] = float(timeout_s) + 10.0 * int(cols)
    return metrics


# Default early-stop "meaningful" threshold: 0.5 s of adjusted_time for finishers.
# Among DNFs, any strict race_score_key improvement resets patience (progress-first
# ranking) so mid-train learning is visible even before a finished lap.
DEFAULT_EARLY_STOP_MIN_IMPROVE = 0.5


def is_meaningful_race_improvement(
    new_metrics: dict,
    ref_metrics: dict | None,
    *,
    min_improve: float = DEFAULT_EARLY_STOP_MIN_IMPROVE,
) -> bool:
    """Whether new validation metrics beat the patience reference enough to reset early-stop.

    Ranking uses ``race_score_key`` (finishers ≻ DNFs; DNFs progress-first) — never
    ep_rew. ``min_improve`` is in **adjusted_time seconds** for finishers:

    - ``min_improve <= 0``: any strict ``race_score_key`` improvement counts (legacy).
    - DNF → finish: always meaningful.
    - Both finishers: need ``ref_adj - new_adj >= min_improve`` seconds.
    - Both DNF: any strict ``race_score_key`` improvement is meaningful (aligns
      patience with ``best_model`` promotion; progress-first key so far-then-crash
      beats idle timeout).

    ``best_model`` promotion can still happen on tiny key improvements; only the
    early-stop patience counter uses this gate (vs the last *meaningful* ref).
    """
    if ref_metrics is None:
        return True
    new_key = race_score_key(new_metrics)
    ref_key = race_score_key(ref_metrics)
    if new_key >= ref_key:
        return False
    min_i = float(min_improve)
    if min_i <= 0.0:
        return True

    new_fin = new_metrics.get("mean_lap_time") is not None
    ref_fin = ref_metrics.get("mean_lap_time") is not None
    if new_fin and not ref_fin:
        return True
    if new_fin and ref_fin:
        try:
            return float(ref_metrics["adjusted_time"]) - float(new_metrics["adjusted_time"]) >= min_i
        except (TypeError, ValueError, KeyError):
            return True

    # Both DNF: progress-first key already decided new_key < ref_key → meaningful.
    return True


def race_eval_patience_update(
    *,
    improved: bool,
    no_improve: int,
    patience: int,
) -> tuple[int, bool]:
    """Advance early-stop counter after one validation race eval (no warmup).

    Returns ``(new_no_improve_count, should_stop)``. Prefer
    ``early_stop_patience_tick`` when warmup / min-timesteps grace is needed.
    """
    n, stop, _event = early_stop_patience_tick(
        improved=improved,
        no_improve=no_improve,
        patience=patience,
        eval_count=10**9,
        timesteps=10**18,
        warmup_evals=0,
        min_timesteps=0,
        scorable=True,
        is_baseline=False,
    )
    return n, stop


def early_stop_patience_tick(
    *,
    improved: bool,
    no_improve: int,
    patience: int,
    eval_count: int,
    timesteps: int,
    warmup_evals: int = 0,
    min_timesteps: int = 0,
    scorable: bool = True,
    is_baseline: bool = False,
) -> tuple[int, bool, str]:
    """Pure early-stop state transition after one validation race-eval.

    Returns ``(new_no_improve, should_stop, event)`` where event is one of:
    ``disabled``, ``unscorable``, ``baseline``, ``warmup``, ``grace_timesteps``,
    ``improved``, ``miss``, ``stop``.

    Patience never increments during baseline, warmup (``eval_count <= warmup_evals``),
    or before ``min_timesteps``. Unscorable maps never early-stop.
    """
    if int(patience) <= 0:
        return 0, False, "disabled"
    if not scorable:
        return int(no_improve), False, "unscorable"
    if is_baseline:
        return 0, False, "baseline"
    if int(warmup_evals) > 0 and int(eval_count) <= int(warmup_evals):
        return (0 if improved else int(no_improve)), False, "warmup"
    if int(min_timesteps) > 0 and int(timesteps) < int(min_timesteps):
        return (0 if improved else int(no_improve)), False, "grace_timesteps"
    if improved:
        return 0, False, "improved"
    n = int(no_improve) + 1
    if n >= int(patience):
        return n, True, "stop"
    return n, False, "miss"


def RaceBestModelCallback(*args, **kwargs):
    """Promote best_model.zip by race score (never by ep_rew).

    Optional early-stop: after ``patience`` consecutive *post-warmup* race-evals
    with no *meaningful* improvement, ``_on_step`` returns False so PPO stops.
    Warmup / min-timesteps grace never increments patience. Unscorable validation
    skips the patience tick (warning only).

    Mid-train evals block ``model.learn()`` — write ``live_status`` phase
    ``validating`` (+ heartbeat) so the UI does not look hung.
    """
    import threading

    from stable_baselines3.common.callbacks import BaseCallback

    from .live_status import is_terminal_phase, write_live_status

    class _CB(BaseCallback):
        def __init__(
            self,
            run_dir: Path,
            eval_maps: list[Path],
            *,
            n_lidar: int,
            eval_episodes: int,
            eval_freq: int,
            seed: int,
            timeout_s: float = MID_TRAIN_SELECT_TIMEOUT_S,
            patience: int = 0,
            min_improve: float = DEFAULT_EARLY_STOP_MIN_IMPROVE,
            warmup_evals: int = 0,
            min_timesteps: int = 0,
            status_path: Path | None = None,
            status_run_id: str | None = None,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            self.run_dir = Path(run_dir)
            self.eval_maps = list(eval_maps)
            self.n_lidar = int(n_lidar)
            self.eval_episodes = max(1, int(eval_episodes))
            self.eval_freq = max(0, int(eval_freq))
            self.seed = int(seed)
            self.timeout_s = float(timeout_s)
            self.patience = max(0, int(patience))
            self.min_improve = max(0.0, float(min_improve))
            self.warmup_evals = max(0, int(warmup_evals))
            self.min_timesteps = max(0, int(min_timesteps))
            self.status_path = Path(status_path) if status_path else None
            self.status_run_id = status_run_id
            self.best_key = None
            self.best_metrics: dict | None = None
            # Last metrics that reset patience (may lag best_model on tiny gains).
            self.patience_ref_metrics: dict | None = None
            self._no_improve = 0
            self._eval_count = 0
            self.early_stopped = False
            self.early_stop_msg: str | None = None
            # Start at 0, not -inf: racing the untrained policy on step 1 burns an
            # eval and pins best_model to weights that predate the first update.
            self._last_eval_ts = 0
            self._hb_stop: threading.Event | None = None
            self._hb_thread: threading.Thread | None = None
            # Generation token: late pulses after stop/join must not write.
            self._hb_gen = 0

        def _status_base(self) -> dict:
            return {
                "run_id": self.status_run_id,
                "timesteps": int(self.num_timesteps),
                "unix_time": time.time(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "patience": self.patience,
                "no_improve": int(self._no_improve),
                "eval_count": int(self._eval_count),
                "warmup_evals": self.warmup_evals,
                "early_stop_min_timesteps": self.min_timesteps,
                "select_timeout_s": self.timeout_s,
                "eval_maps": [p.stem for p in self.eval_maps],
            }

        def _write_status(self, *, phase: str, msg: str) -> None:
            if self.status_path is None:
                return
            # Never let validating/learning heartbeat clobber a terminal phase.
            if self.early_stopped and not is_terminal_phase(phase):
                return
            if not is_terminal_phase(phase):
                # Also honor a terminal phase already on disk (join race window).
                try:
                    from .live_status import read_live_status

                    prev = read_live_status(self.status_path)
                except OSError:
                    prev = None
                if prev and is_terminal_phase(prev.get("phase")):
                    return
            payload = self._status_base()
            payload["phase"] = phase
            payload["msg"] = msg
            try:
                write_live_status(self.status_path, payload)
            except OSError as exc:
                if self.verbose:
                    print(f"[race-eval] live_status write skipped: {exc}")

        def _start_heartbeat(self, msg: str) -> None:
            """Refresh unix_time while validation blocks learn() so status age stays fresh."""
            self._stop_heartbeat()
            if self.status_path is None or self.early_stopped:
                return
            stop = threading.Event()
            self._hb_stop = stop
            self._hb_gen += 1
            gen = self._hb_gen

            def _pulse() -> None:
                while not stop.wait(5.0):
                    if gen != self._hb_gen or self.early_stopped or stop.is_set():
                        return
                    self._write_status(phase="validating", msg=msg)

            th = threading.Thread(target=_pulse, name="race-eval-hb", daemon=True)
            self._hb_thread = th
            th.start()

        def _stop_heartbeat(self) -> None:
            """Signal the pulse thread and join it before any terminal status write."""
            stop = self._hb_stop
            th = self._hb_thread
            self._hb_stop = None
            self._hb_thread = None
            # Bump gen so any in-flight pulse that slipped past wait() bails out.
            self._hb_gen += 1
            if stop is not None:
                stop.set()
            if th is not None and th.is_alive():
                # Pulse uses stop.wait(5.0) — join slightly longer than one interval.
                th.join(timeout=6.0)

        def _on_step(self) -> bool:
            if self.eval_freq <= 0:
                return True
            ts = int(self.num_timesteps)
            if ts - self._last_eval_ts < self.eval_freq:
                return True
            self._last_eval_ts = ts
            improved, scorable, is_baseline = self._race_and_maybe_promote()
            self._eval_count += 1
            self._no_improve, should_stop, event = early_stop_patience_tick(
                improved=improved,
                no_improve=self._no_improve,
                patience=self.patience,
                eval_count=self._eval_count,
                timesteps=ts,
                warmup_evals=self.warmup_evals,
                min_timesteps=self.min_timesteps,
                scorable=scorable,
                is_baseline=is_baseline,
            )
            if event == "unscorable":
                print(
                    "WARNING: validation map unscorable — skipping early-stop "
                    f"patience tick @ ts={ts} (continuing train)"
                )
                self._write_status(
                    phase="learning",
                    msg="learning: validation unscorable — early-stop tick skipped",
                )
                return True
            if should_stop:
                # Join heartbeat BEFORE writing terminal phase so a late
                # validating pulse cannot land after early_stopped.
                self._stop_heartbeat()
                self.early_stopped = True
                self.early_stop_msg = (
                    f"early-stop: no meaningful improvement "
                    f"(min_improve={self.min_improve:g}s adj / DNF progress) for "
                    f"{self.patience} eval(s) after warmup "
                    f"(warmup_evals={self.warmup_evals}, "
                    f"min_timesteps={self.min_timesteps}) @ ts={ts}"
                )
                print(
                    f"Early stop: no meaningful validation race-score improvement "
                    f"(min_improve={self.min_improve:g}s adj / DNF progress) for "
                    f"{self.patience} eval(s) @ ts={ts} "
                    f"(best_key={self.best_key})"
                )
                print(f"NOTE: {self.early_stop_msg} - clean exit (not a crash)")
                self._write_status(phase="early_stopped", msg=self.early_stop_msg)
                return False
            if self.patience > 0:
                if event in ("baseline", "warmup", "grace_timesteps"):
                    resume_msg = (
                        f"learning: validation {event} — patience not counting yet "
                        f"(evals={self._eval_count}, warmup={self.warmup_evals}, "
                        f"min_ts={self.min_timesteps}; "
                        f"no_improve stays {self._no_improve}/{self.patience})"
                    )
                elif improved:
                    resume_msg = (
                        f"learning: validation improved — patience reset "
                        f"(0/{self.patience})"
                    )
                else:
                    resume_msg = (
                        f"learning: validation no meaningful improve — "
                        f"patience {self._no_improve}/{self.patience}"
                    )
                self._write_status(phase="learning", msg=resume_msg)
            else:
                self._write_status(
                    phase="learning",
                    msg="learning: mid-train validation done",
                )
            return True

        def _on_training_end(self) -> None:
            # Join first — never leave a validating pulse racing training_end.
            self._stop_heartbeat()
            # Final promotion pass only if we did not already stop mid-run.
            if self.early_stopped:
                msg = self.early_stop_msg or "early-stop: no meaningful improvement"
                self._write_status(phase="early_stopped", msg=msg)
                return
            self._race_and_maybe_promote()
            self._write_status(phase="learning", msg="learning: final validation done")

        def _race_and_maybe_promote(self) -> tuple[bool, bool, bool]:
            """Run validation race eval; promote best_model if race_score_key improves.

            Returns ``(meaningful, scorable, is_baseline)``.
            """
            if not self.eval_maps:
                print("WARNING: no validation maps for race-eval — unscorable")
                return False, False, False
            maps_label = ",".join(p.stem for p in self.eval_maps) or "?"
            val_msg = (
                f"racing validation map ({maps_label}) — "
                f"timeout={self.timeout_s:g}s x {self.eval_episodes} ep; "
                f"timesteps paused (not stuck)"
            )
            print(
                f"Validation race-eval @ ts={self.num_timesteps} "
                f"maps={[p.stem for p in self.eval_maps]} "
                f"timeout_s={self.timeout_s:g} episodes={self.eval_episodes}"
            )
            self._write_status(phase="validating", msg=val_msg)
            self._start_heartbeat(val_msg)
            is_baseline = self.patience_ref_metrics is None
            try:
                try:
                    metrics = _race_eval_maps(
                        self.model,
                        self.eval_maps,
                        n_lidar=self.n_lidar,
                        episodes=self.eval_episodes,
                        seed=self.seed + 10_000,
                        timeout_s=self.timeout_s,
                    )
                except Exception as exc:
                    print(f"WARNING: validation race-eval failed ({exc}) — unscorable")
                    return False, False, is_baseline
            finally:
                self._stop_heartbeat()
            if not isinstance(metrics, dict):
                return False, False, is_baseline
            key = race_score_key(metrics)
            meaningful = is_meaningful_race_improvement(
                metrics,
                self.patience_ref_metrics,
                min_improve=self.min_improve,
            )
            # Promote on any strict key improvement (keep true best weights).
            if self.best_key is None or key < self.best_key:
                self.best_key = key
                self.best_metrics = metrics
                atomic_save_sb3(self.model, self.run_dir / "best_model")
                finished = metrics.get("mean_lap_time") is not None
                meta = {
                    # Say plainly which number won, so nobody reads a DNF pick as a lap time.
                    "selected_by": (
                        "adjusted_time" if finished else "progress_frac (DNF, no scored lap)"
                    ),
                    "dnf": not finished,
                    "timesteps": int(self.num_timesteps),
                    "metrics": metrics,
                }
                (self.run_dir / "best_model_meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                prog = metrics.get("mean_progress_frac")
                print(
                    f"best_model promoted @ ts={self.num_timesteps} "
                    f"{'adjusted_time=' + str(metrics.get('adjusted_time')) if finished else 'DNF'} "
                    f"progress={prog if prog is None else round(float(prog), 4)} "
                    f"cols={metrics.get('total_collisions')}"
                )
            if meaningful:
                self.patience_ref_metrics = dict(metrics)
            return meaningful, True, is_baseline

    return _CB(*args, **kwargs)


# Default short timeout for optional NON-OFFICIAL progress probes.
# Must stay well below MID_TRAIN_SELECT_TIMEOUT_S so nobody mistakes it for race score.
DEFAULT_FAST_PROBE_TIMEOUT_S = 30.0


def FastProbeCallback(*args, **kwargs):
    """Optional mid-train progress probe — status-only, never official.

    Writes ``probe_*`` fields into live_status. Does **not** promote
    ``best_model``, does **not** tick early-stop patience, does **not**
    write ``kind=official`` / train_eval board rows. Only the full
    ``RaceBestModelCallback`` select-timeout (~220s) path counts for
    promote / patience.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    from .live_status import is_terminal_phase, read_live_status, write_live_status

    class _Probe(BaseCallback):
        def __init__(
            self,
            eval_maps: list[Path],
            *,
            n_lidar: int,
            probe_every: int,
            seed: int,
            timeout_s: float = DEFAULT_FAST_PROBE_TIMEOUT_S,
            status_path: Path | None = None,
            status_run_id: str | None = None,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            self.eval_maps = list(eval_maps)
            self.n_lidar = int(n_lidar)
            self.probe_every = max(0, int(probe_every))
            self.seed = int(seed)
            self.timeout_s = float(timeout_s)
            self.status_path = Path(status_path) if status_path else None
            self.status_run_id = status_run_id
            self._last_probe_ts = 0

        def _on_step(self) -> bool:
            if self.probe_every <= 0 or not self.eval_maps:
                return True
            ts = int(self.num_timesteps)
            if ts - self._last_probe_ts < self.probe_every:
                return True
            self._last_probe_ts = ts
            if self.status_path is not None:
                try:
                    prev = read_live_status(self.status_path)
                except OSError:
                    prev = None
                if prev and (
                    is_terminal_phase(prev.get("phase"))
                    or prev.get("phase") == "validating"
                ):
                    # Never interrupt / clobber full race-eval or early_stopped.
                    return True
            print(
                f"[fast-probe] NON-OFFICIAL progress probe @ ts={ts} "
                f"timeout={self.timeout_s:g}s — not race score, not patience, not promote"
            )
            try:
                metrics = _race_eval_maps(
                    self.model,
                    self.eval_maps,
                    n_lidar=self.n_lidar,
                    episodes=1,
                    seed=self.seed + 20_000,
                    timeout_s=self.timeout_s,
                )
            except Exception as exc:
                print(f"[fast-probe] skipped ({exc})")
                return True
            # Force honesty: never look like a finished race score under short timeout.
            metrics = dict(metrics or {})
            metrics["kind"] = "smoke"
            metrics["official"] = False
            metrics["probe_label"] = (
                "NON-OFFICIAL progress probe — not race score / not early-stop / not FTGΔ"
            )
            prog = metrics.get("mean_progress_frac")
            cols = metrics.get("total_collisions")
            print(
                f"[fast-probe] progress={prog if prog is None else round(float(prog), 4)} "
                f"cols={cols} (status-only; RaceBest 220s still owns patience)"
            )
            if self.status_path is None:
                return True
            try:
                prev = read_live_status(self.status_path) or {}
            except OSError:
                prev = {}
            if is_terminal_phase(prev.get("phase")) or prev.get("phase") == "validating":
                return True
            payload = dict(prev)
            payload.update(
                {
                    "run_id": self.status_run_id or prev.get("run_id"),
                    "timesteps": ts,
                    "unix_time": time.time(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "phase": prev.get("phase") or "learning",
                    "probe_kind": "smoke",
                    "probe_official": False,
                    "probe_label": metrics["probe_label"],
                    "probe_timeout_s": self.timeout_s,
                    "probe_mean_progress_frac": prog,
                    "probe_total_collisions": cols,
                    "progress_probe_kind": "smoke",
                    "mean_progress_frac_estimate": prog,
                    "msg": (
                        prev.get("msg")
                        if prev.get("phase") == "validating"
                        else (
                            f"learning: NON-OFFICIAL probe progress="
                            f"{prog if prog is None else round(float(prog), 4)} "
                            f"(patience still uses full {MID_TRAIN_SELECT_TIMEOUT_S:g}s eval)"
                        )
                    ),
                }
            )
            try:
                write_live_status(self.status_path, payload)
            except OSError as exc:
                if self.verbose:
                    print(f"[fast-probe] live_status write skipped: {exc}")
            return True

    return _Probe(*args, **kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gym PPO train (honest best_model / resume)")
    parser.add_argument(
        "--map",
        type=str,
        default="map0",
        help="Map id or comma-separated list for multi-map n_envs diversity",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2048,
        help="Use ~2048 for smoke; start_train.ps1 defaults higher (50k-100k serious). "
        "With --unlimited-timesteps this is ignored (safety ceiling applied instead).",
    )
    parser.add_argument(
        "--unlimited-timesteps",
        action="store_true",
        help="Do not stop solely on the timesteps budget. Uses a hard safety ceiling "
        f"of {UNLIMITED_TIMESTEPS_SAFETY:,} steps; requires --early-stop-patience > 0.",
    )
    parser.add_argument("--n_lidar", type=int, default=N_LIDAR_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", type=str, choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--net-arch", type=str, default="256,256")
    parser.add_argument("--tb", action="store_true", default=True)
    parser.add_argument("--no-tb", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--status-every-rollouts", type=int, default=1)
    parser.add_argument("--status-every-steps", type=int, default=512)
    parser.add_argument("--no-live-status", action="store_true")
    parser.add_argument("--save-latest-every-rollouts", type=int, default=1)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from run_id or path to zip; prefers last complete checkpoint",
    )
    parser.add_argument(
        "--fork-resume",
        action="store_true",
        help="When resuming, allocate a new run_id (fork) instead of continuing same id",
    )
    parser.add_argument("--spawn-jitter", action="store_true", default=True)
    parser.add_argument("--no-spawn-jitter", action="store_true")
    parser.add_argument("--collision-first", action="store_true")
    parser.add_argument("--speed-gate", action="store_true")
    parser.add_argument("--ttc-truncate", action="store_true", default=True)
    parser.add_argument("--no-ttc-truncate", action="store_true")
    parser.add_argument("--lidar-dr", action="store_true")
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Allow sealed holdout maps in the train map list (default: refuse)",
    )
    parser.add_argument(
        "--allow-validation",
        action="store_true",
        help="Allow the pinned validation map in the train list (default: refuse)",
    )
    parser.add_argument(
        "--race-eval-every",
        type=int,
        default=0,
        help="Promote best_model by adjusted_time every N timesteps (0=end only). "
        "Auto-set when --early-stop-patience > 0 and this is left at 0.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop train after N consecutive validation race-evals with no "
        "*meaningful* race_score_key improvement (0=disabled). See "
        "--early-stop-min-improve. Uses adjusted_time/progress, never ep_rew. "
        "Requires mid-train evals (--race-eval-every, or auto).",
    )
    parser.add_argument(
        "--early-stop-min-improve",
        type=float,
        default=DEFAULT_EARLY_STOP_MIN_IMPROVE,
        help="How much better validation score must get to reset early-stop "
        f"patience (default {DEFAULT_EARLY_STOP_MIN_IMPROVE}). Finishers: seconds "
        "of adjusted_time (lap+10*cols). DNFs: any strict race_score_key "
        "improvement (progress-first). 0 = any key improvement (legacy).",
    )
    parser.add_argument(
        "--early-stop-warmup-evals",
        type=int,
        default=-1,
        help="First N validation race-evals never increment early-stop patience "
        f"(-1 = auto: {DEFAULT_EARLY_STOP_WARMUP_EVALS} when patience>0 else 0). "
        "Baseline establishing is always free; use 2+ so the first miss after "
        "baseline cannot start the stop clock alone.",
    )
    parser.add_argument(
        "--early-stop-min-timesteps",
        type=int,
        default=-1,
        help="Do not increment early-stop patience until num_timesteps reaches N "
        f"(-1 = auto: {DEFAULT_EARLY_STOP_MIN_TIMESTEPS} when unlimited+patience "
        "else 0). Prevents EarlyStop before meaningful learning.",
    )
    parser.add_argument(
        "--select-timeout",
        type=float,
        default=MID_TRAIN_SELECT_TIMEOUT_S,
        help="Sim-time budget (s) for mid-train best_model / early-stop race evals "
        f"(default {MID_TRAIN_SELECT_TIMEOUT_S:g}). Must be >=~180-220 so a real "
        "lap can finish; 60s forced permanent DNF and false early-stop. "
        "Official scoring still uses eval_protocol.yaml timeout_s (400). "
        "ONLY these full evals tick early-stop patience / promote best_model.",
    )
    parser.add_argument(
        "--fast-probe-every",
        type=int,
        default=0,
        help="Optional NON-OFFICIAL progress probe every N timesteps (0=off). "
        "Writes live_status probe_* only — never promotes, never ticks patience, "
        "never kind=official. Leave off for overnight.",
    )
    parser.add_argument(
        "--fast-probe-timeout",
        type=float,
        default=DEFAULT_FAST_PROBE_TIMEOUT_S,
        help=f"Sim-time budget (s) for --fast-probe-every (default {DEFAULT_FAST_PROBE_TIMEOUT_S:g}). "
        "Short on purpose — progress signal only, not race score.",
    )
    parser.add_argument(
        "--official-eval",
        action="store_true",
        help="After train, run official protocol eval and append kind=official row",
    )
    args = parser.parse_args(argv)

    early_stop_patience = max(0, int(args.early_stop_patience))
    early_stop_min_improve = max(0.0, float(args.early_stop_min_improve))
    unlimited_timesteps = bool(args.unlimited_timesteps)
    if unlimited_timesteps:
        if early_stop_patience <= 0:
            parser.error(
                "--unlimited-timesteps requires --early-stop-patience > 0 "
                "(otherwise training never stops except Ctrl+C / Stop)"
            )
        if early_stop_patience < DEFAULT_EARLY_STOP_PATIENCE_UNLIMITED:
            print(
                f"NOTE: raising --early-stop-patience {early_stop_patience} -> "
                f"{DEFAULT_EARLY_STOP_PATIENCE_UNLIMITED} for overnight-safe unlimited"
            )
            early_stop_patience = DEFAULT_EARLY_STOP_PATIENCE_UNLIMITED
        print(
            f"NOTE: --unlimited-timesteps -> safety ceiling "
            f"{UNLIMITED_TIMESTEPS_SAFETY:,} timesteps "
            f"(early-stop is the real exit; ceiling is hard abort only)"
        )
        args.timesteps = UNLIMITED_TIMESTEPS_SAFETY

    # Warmup / grace defaults: free when patience off; overnight-safe when on.
    if int(args.early_stop_warmup_evals) < 0:
        early_stop_warmup_evals = (
            DEFAULT_EARLY_STOP_WARMUP_EVALS if early_stop_patience > 0 else 0
        )
    else:
        early_stop_warmup_evals = max(0, int(args.early_stop_warmup_evals))
    if int(args.early_stop_min_timesteps) < 0:
        early_stop_min_timesteps = (
            DEFAULT_EARLY_STOP_MIN_TIMESTEPS
            if unlimited_timesteps and early_stop_patience > 0
            else 0
        )
    else:
        early_stop_min_timesteps = max(0, int(args.early_stop_min_timesteps))

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .live_status import LiveStatusCallback

    spawn_jitter = bool(args.spawn_jitter) and not bool(args.no_spawn_jitter)
    ttc_truncate = bool(args.ttc_truncate) and not bool(args.no_ttc_truncate)

    device = resolve_device(args.device)
    n_envs = max(1, int(args.n_envs))
    net_arch = _parse_net_arch(args.net_arch)
    n_steps = max(64, int(args.n_steps))
    batch_size = max(32, int(args.batch_size))
    n_epochs = max(1, int(args.n_epochs))

    buffer_size = n_steps * n_envs
    if batch_size > buffer_size:
        batch_size = buffer_size
        print(f"NOTE: batch_size clamped to buffer size {batch_size} (n_steps*n_envs)")

    print(f"torch={torch.__version__} device={device}", end="")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    maps_root = Path(__file__).resolve().parent / "maps"
    models_root = Path(__file__).resolve().parent / "models"
    runs_root = Path(__file__).resolve().parent / "runs"
    map_ids = _parse_maps(args.map)

    # The map pack — not the protocol alone — decides what a trainer may touch:
    # it also guards the validation pin used to promote best_model.
    try:
        assert_train_safe(
            map_ids,
            maps_root=maps_root,
            allow_holdout=bool(args.allow_holdout),
            allow_validation=bool(args.allow_validation),
        )
    except HoldoutViolation as exc:
        print(f"ERROR: {exc}")
        return 2

    pack_report = verify_pack(maps_root)
    if not pack_report.get("ok"):
        broken = [m["id"] for m in pack_report.get("maps", []) if m.get("sealed") and not m.get("ok")]
        if broken:
            print(f"ERROR: broken seal on {', '.join(broken)} - re-pin before training.")
            for issue in pack_report.get("issues", []):
                print(f"  - {issue}")
            return 2
        print("WARNING: map pack verify reported issues:")
        for issue in pack_report.get("issues", []):
            print(f"  - {issue}")

    map_yamls = [resolve_map_yaml(m, maps_root) for m in map_ids]
    map_hashes = {p.stem: map_file_hash(p) for p in map_yamls}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    resume_path: Path | None = None
    resume_run_dir: Path | None = None
    reset_num_timesteps = True

    if args.resume:
        r = Path(args.resume)
        if r.suffix == ".zip" and r.is_file():
            resume_path = r
            resume_run_dir = r.parent
        elif (models_root / args.resume).is_dir():
            resume_run_dir = models_root / args.resume
            resume_path = find_last_complete_checkpoint(resume_run_dir)
        elif r.is_dir():
            resume_run_dir = r
            resume_path = find_last_complete_checkpoint(r)
        else:
            print(f"ERROR: --resume not found: {args.resume}")
            return 2
        if resume_path is None:
            print(f"ERROR: no complete checkpoint under {resume_run_dir}")
            return 2
        print(f"Resume weights: {resume_path}")
        reset_num_timesteps = False

    if args.run_id:
        run_id = args.run_id
    elif resume_run_dir is not None and not args.fork_resume:
        run_id = resume_run_dir.name
    else:
        run_id = f"{stamp}_ppo_gym_{map_yamls[0].stem}"

    run_dir_early = models_root / run_id
    run_dir_early.mkdir(parents=True, exist_ok=True)
    try:
        acquire_run_lock(run_dir_early)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    env_fns = []
    for i in range(n_envs):
        my = map_yamls[i % len(map_yamls)]
        env_fns.append(
            _make_env(
                my,
                args.n_lidar,
                args.seed,
                i,
                spawn_jitter=spawn_jitter,
                collision_first=bool(args.collision_first),
                speed_gate=bool(args.speed_gate),
                ttc_truncate=ttc_truncate,
                lidar_dr=bool(args.lidar_dr),
            )
        )

    vec_kind = args.vec_env
    vec_env_requested = vec_kind
    if vec_kind == "subproc" and n_envs > 1:
        try:
            vec_env = SubprocVecEnv(env_fns)
        except Exception as exc:
            print(f"WARNING: SubprocVecEnv failed ({exc}); falling back to DummyVecEnv")
            vec_kind = "dummy"
            vec_env = DummyVecEnv(env_fns)
    else:
        if n_envs == 1:
            vec_kind = "dummy"
        vec_env = DummyVecEnv(env_fns)
    print(
        f"vec_env_requested={vec_env_requested} vec_env_active={vec_kind} "
        f"n_envs={n_envs} maps={[p.stem for p in map_yamls]} "
        f"jitter={spawn_jitter} collision_first={args.collision_first} "
        f"speed_gate={args.speed_gate} ttc={ttc_truncate} lidar_dr={args.lidar_dr}"
    )

    tb_log = None if args.no_tb else str(runs_root / run_id)
    if tb_log:
        Path(tb_log).mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard: tensorboard --logdir \"{runs_root}\"")

    policy_kwargs = dict(net_arch=net_arch)

    if resume_path is not None:
        try:
            from .compat import refuse_obs_dim

            # Load with env attached so n_envs can differ from the original run.
            model = PPO.load(str(resume_path), env=vec_env, device=device)
            refuse_model_obs(model, n_lidar=args.n_lidar, label=str(resume_path))
            cfg_path = (resume_run_dir or resume_path.parent) / "config.json"
            if cfg_path.is_file():
                import json as _json

                try:
                    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                    from .compat import refuse_config_contracts

                    refuse_config_contracts(cfg, label=str(cfg_path))
                except ContractsMismatch:
                    raise
                except Exception:
                    pass
        except ContractsMismatch as exc:
            print(f"ERROR: refuse resume - {exc}")
            release_run_lock(run_dir_early)
            vec_env.close()
            return 2
        if tb_log:
            model.tensorboard_log = tb_log
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=3e-4,
            gamma=0.99,
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
        )
        refuse_model_obs(model, n_lidar=args.n_lidar, label="new PPO")

    status_dir = runs_root / run_id
    status_dir.mkdir(parents=True, exist_ok=True)
    callbacks = []

    # Atomic-ish checkpoints: SB3 writes complete zip; we still mark via CheckpointCallback
    if args.checkpoint_every and args.checkpoint_every > 0:
        ckpt_dir = run_dir_early / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, int(args.checkpoint_every) // max(1, n_envs)),
                save_path=str(ckpt_dir),
                name_prefix="ppo",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
        print(f"Checkpoints every ~{args.checkpoint_every} steps -> {ckpt_dir}")

    # Select best_model on the pinned validation map, never on trained geometry.
    val_id = validation_map(maps_root)
    if val_id and not args.allow_validation:
        select_maps = [resolve_map_yaml(val_id, maps_root)]
        print(f"best_model selection map: {val_id} (validation pin, not trained on)")
    else:
        select_maps = map_yamls
        print(
            "WARNING: no validation pin available; selecting best_model on train maps "
            f"{[p.stem for p in map_yamls]} (selection is biased)"
        )

    race_eval_every = int(args.race_eval_every)
    # early_stop_patience already resolved above (unlimited-timesteps guard).
    if early_stop_patience > 0 and race_eval_every <= 0:
        # Need periodic validation evals for patience to mean anything.
        # When unlimited, don't scale off the 50M safety ceiling — use a nominal
        # reference so race-evals still fire often enough for early-stop
        # without pausing every ~10k steps (that killed overnight at ~40k).
        ref_ts = (
            UNLIMITED_EVAL_REFERENCE_TS
            if unlimited_timesteps
            else int(args.timesteps)
        )
        race_eval_every = max(AUTO_RACE_EVAL_EVERY_FLOOR, int(ref_ts) // 10)
        print(
            f"NOTE: --early-stop-patience={early_stop_patience} -> "
            f"auto --race-eval-every={race_eval_every} "
            f"(raise --race-eval-every for fewer mid-train pauses)"
        )
    select_timeout = float(args.select_timeout)
    if early_stop_patience > 0:
        print(
            f"Early stop: patience={early_stop_patience} validation race-eval(s) "
            f"without meaningful improvement "
            f"(min_improve={early_stop_min_improve:g}s adj / DNF progress; "
            f"warmup_evals={early_stop_warmup_evals}; "
            f"min_timesteps={early_stop_min_timesteps}; "
            f"maps={[p.stem for p in select_maps]}; "
            f"mid-train timeout={select_timeout:g}s)"
        )

    status_path: Path | None = None
    if not args.no_live_status:
        latest_path = None
        save_latest_every = int(args.save_latest_every_rollouts)
        if save_latest_every > 0:
            latest_path = run_dir_early / "latest_model.zip"
        status_path = status_dir / "live_status.json"
        # LiveStatus before RaceBest so early-stop / validating status written
        # by RaceBestModelCallback wins on training_end.
        callbacks.append(
            LiveStatusCallback(
                status_path=status_path,
                run_id=run_id,
                every_rollouts=max(1, int(args.status_every_rollouts)),
                every_steps=max(0, int(args.status_every_steps)),
                latest_model_path=latest_path,
                save_latest_every_rollouts=max(1, save_latest_every) if latest_path else 1,
                n_envs=n_envs,
                vec_env_active=vec_kind,
            )
        )
        print(f"Live status -> {status_path}")
        if latest_path:
            print(f"Latest weights every {save_latest_every} rollout(s) -> {latest_path}")

    race_cb = RaceBestModelCallback(
        run_dir_early,
        select_maps,
        n_lidar=args.n_lidar,
        eval_episodes=max(1, int(args.eval_episodes)),
        eval_freq=race_eval_every,
        seed=args.seed,
        timeout_s=select_timeout,
        patience=early_stop_patience,
        min_improve=early_stop_min_improve,
        warmup_evals=early_stop_warmup_evals,
        min_timesteps=early_stop_min_timesteps,
        status_path=status_path,
        status_run_id=run_id,
    )
    callbacks.append(race_cb)

    fast_probe_every = max(0, int(args.fast_probe_every))
    fast_probe_timeout = float(args.fast_probe_timeout)
    if fast_probe_every > 0:
        print(
            f"NOTE: fast progress probe every {fast_probe_every} ts "
            f"(timeout={fast_probe_timeout:g}s) — NON-OFFICIAL; "
            f"patience/promote still use select-timeout={select_timeout:g}s only"
        )
        callbacks.append(
            FastProbeCallback(
                select_maps,
                n_lidar=args.n_lidar,
                probe_every=fast_probe_every,
                seed=args.seed,
                timeout_s=fast_probe_timeout,
                status_path=status_path,
                status_run_id=run_id,
            )
        )

    # Write config.json BEFORE learn so an early crash still leaves Continue with maps/contracts.
    early_config = {
        "contracts_version": CONTRACTS_VERSION,
        "run_id": run_id,
        "policy": "ppo",
        "backend": "gym",
        "n_lidar": args.n_lidar,
        "action_space": "MultiDiscrete([4, 11])",
        "throttle_bins": THROTTLE_BINS.tolist(),
        "steering_bins": STEERING_BINS.tolist(),
        "obs_include_speed": True,
        "obs_include_imu": True,
        "imu_dim": 3,
        "obs_dim": obs_dim(args.n_lidar),
        "speed_max_mps": SPEED_MAX_MPS,
        "timeout_s": TIMEOUT_S,
        "stall_timeout_s": STALL_TIMEOUT_S,
        "timesteps": args.timesteps,
        "unlimited_timesteps": unlimited_timesteps,
        "map": str(map_yamls[0]),
        "maps": [str(p) for p in map_yamls],
        "map_hashes": map_hashes,
        "device": device,
        "n_envs": n_envs,
        "vec_env": vec_kind,
        "vec_env_requested": vec_env_requested,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "net_arch": net_arch,
        "seed": args.seed,
        "spawn_jitter": spawn_jitter,
        "collision_first": bool(args.collision_first),
        "speed_gate": bool(args.speed_gate),
        "ttc_truncate": ttc_truncate,
        "lidar_dr": bool(args.lidar_dr),
        "git_sha": git_sha_short(Path(__file__).resolve().parents[2]),
        "resume_from": str(resume_path) if resume_path else None,
        "best_model_selection": "adjusted_time",
        "map_pack": pack_fingerprint(maps_root),
        "validation_map": val_id,
        "selection_maps": [p.stem for p in select_maps],
        "allow_holdout": bool(args.allow_holdout),
        "allow_validation": bool(args.allow_validation),
        "race_eval_every": race_eval_every,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_improve": early_stop_min_improve,
        "early_stop_warmup_evals": early_stop_warmup_evals,
        "early_stop_min_timesteps": early_stop_min_timesteps,
        "select_timeout_s": select_timeout,
        "fast_probe_every": fast_probe_every,
        "fast_probe_timeout_s": fast_probe_timeout if fast_probe_every > 0 else 0,
        "phase": "training",
    }
    early_config["fingerprint"] = config_fingerprint(early_config)
    try:
        atomic_write_json(run_dir_early / "config.json", early_config)
        print(f"config.json (early) -> {run_dir_early / 'config.json'}")
    except OSError as exc:
        print(f"WARNING: early config.json write failed: {exc}")

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=int(args.timesteps),
            progress_bar=False,
            tb_log_name="ppo",
            callback=callbacks or None,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        train_s = time.time() - t0
        vec_env.close()
        release_run_lock(run_dir_early)

    if getattr(race_cb, "early_stopped", False):
        stop_msg = (
            getattr(race_cb, "early_stop_msg", None)
            or "early-stop: no meaningful improvement"
        )
        print(f"Training ended via early-stop (exit 0): {stop_msg}")
        if status_path is not None:
            from .live_status import write_live_status

            try:
                write_live_status(
                    status_path,
                    {
                        "run_id": run_id,
                        "phase": "early_stopped",
                        "msg": stop_msg,
                        "timesteps": int(getattr(model, "num_timesteps", 0) or 0),
                        "unix_time": time.time(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except OSError as exc:
                print(f"WARNING: early-stop live_status write failed: {exc}")

    # Ensure latest exists
    atomic_save_sb3(model, run_dir_early / "latest_model")

    # Prefer race-selected best; if missing (no laps), leave latest only — do not rename lie.
    best_zip = run_dir_early / "best_model.zip"
    best_meta_path = run_dir_early / "best_model_meta.json"
    if best_zip.is_file() and best_meta_path.is_file():
        try:
            best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
            race_metrics = best_meta.get("metrics") or {}
        except (OSError, json.JSONDecodeError):
            race_metrics = {}
        mean_lap = race_metrics.get("mean_lap_time")
        collisions = int(race_metrics.get("total_collisions") or 0)
        mean_ret = race_metrics.get("mean_return")
        mean_prog = race_metrics.get("mean_progress_frac")
    else:
        # Fallback smoke eval for metrics.json (kind=smoke) — does NOT invent best_model.
        eval_env = RacingEnv(
            map_yaml=map_yamls[0],
            n_lidar=args.n_lidar,
            seed=args.seed + 10_000,
            spawn_jitter=False,
        )
        mean_lap, collisions, mean_ret, mean_prog = _eval_policy(
            eval_env, model, episodes=args.eval_episodes
        )
        print("NOTE: best_model.zip not race-promoted (null/incomplete race score).")

    metrics = make_metrics(
        run_id=run_id,
        policy="ppo",
        backend="gym",
        mean_lap_time=mean_lap if isinstance(mean_lap, (int, float)) or mean_lap is None else None,
        total_collisions=int(collisions),
        n_episodes=args.eval_episodes * len(map_yamls),
        tracks_eval=[p.stem for p in map_yamls],
        kind="smoke",
        mean_return=mean_ret,
        n_lidar=args.n_lidar,
        timeout_s=TIMEOUT_S,
        seed=args.seed,
        train_timesteps=args.timesteps,
        train_seconds=train_s,
        device=device,
        steps_per_sec=(float(args.timesteps) / train_s) if train_s > 0 else None,
        best_model_by="adjusted_time" if best_zip.is_file() else None,
        mean_progress_frac=mean_prog,
        dnf=mean_lap is None,
    )
    config = {
        "contracts_version": CONTRACTS_VERSION,
        "run_id": run_id,
        "policy": "ppo",
        "backend": "gym",
        "n_lidar": args.n_lidar,
        "action_space": "MultiDiscrete([4, 11])",
        "throttle_bins": THROTTLE_BINS.tolist(),
        "steering_bins": STEERING_BINS.tolist(),
        "obs_include_speed": True,
        "obs_include_imu": True,
        "imu_dim": 3,
        "obs_dim": obs_dim(args.n_lidar),
        "speed_max_mps": SPEED_MAX_MPS,
        "timeout_s": TIMEOUT_S,
        "stall_timeout_s": STALL_TIMEOUT_S,
        "timesteps": args.timesteps,
        "unlimited_timesteps": unlimited_timesteps,
        "map": str(map_yamls[0]),
        "maps": [str(p) for p in map_yamls],
        "map_hashes": map_hashes,
        "device": device,
        "n_envs": n_envs,
        "vec_env": vec_kind,
        "vec_env_requested": vec_env_requested,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "net_arch": net_arch,
        "seed": args.seed,
        "spawn_jitter": spawn_jitter,
        "collision_first": bool(args.collision_first),
        "speed_gate": bool(args.speed_gate),
        "ttc_truncate": ttc_truncate,
        "lidar_dr": bool(args.lidar_dr),
        "git_sha": git_sha_short(Path(__file__).resolve().parents[2]),
        "resume_from": str(resume_path) if resume_path else None,
        "best_model_selection": "adjusted_time",
        "map_pack": pack_fingerprint(maps_root),
        "validation_map": val_id,
        "selection_maps": [p.stem for p in select_maps],
        "allow_holdout": bool(args.allow_holdout),
        "allow_validation": bool(args.allow_validation),
        "race_eval_every": race_eval_every,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_improve": early_stop_min_improve,
        "early_stop_warmup_evals": early_stop_warmup_evals,
        "early_stop_min_timesteps": early_stop_min_timesteps,
        "select_timeout_s": select_timeout,
        "early_stopped": bool(getattr(race_cb, "early_stopped", False)),
    }
    config["fingerprint"] = config_fingerprint(config)
    write_run_artifacts(
        models_root,
        run_id,
        config,
        metrics,
        train_tracks=[p.stem for p in map_yamls],
        append_board=bool(metrics.get("adjusted_time") is not None),
    )
    print(f"config fingerprint={config['fingerprint']}")
    if best_zip.is_file():
        print(f"best_model (race-selected): {best_zip}")
    print(f"latest_model: {run_dir_early / 'latest_model.zip'}")
    print(metrics)

    if args.official_eval and best_zip.is_file():
        from .eval_protocol import eval_ppo_protocol
        from .metrics_io import append_leaderboard

        print("Running official protocol eval on best_model...")
        try:
            off = eval_ppo_protocol(best_zip, n_lidar=args.n_lidar, run_id=f"{run_id}_official")
            append_leaderboard(models_root / "leaderboard.csv", off)
            (run_dir_early / "official_metrics.json").write_text(
                json.dumps(off, indent=2), encoding="utf-8"
            )
            print(json.dumps({k: off.get(k) for k in ("adjusted_time", "total_collisions", "kind", "protocol_id")}, indent=2))
        except ContractsMismatch as exc:
            print(f"ERROR official eval refused: {exc}")
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
