"""Behavior-clone PPO's actor from sensorimotor teacher demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


BC_INITIAL_ACTION_STD = 0.2


def initialize_behavior_cloned_action_std(model: Any, std: float = BC_INITIAL_ACTION_STD) -> float:
    """Set PPO sampling noise to the teacher-compatible range after cloning.

    Behavior cloning fits the policy mean. Leaving SB3's default unit Gaussian
    deviation intact makes PPO sample actions far outside the teacher's narrow
    steering/throttle range before its first update, which can immediately
    destroy the cloned driving behavior.
    """
    import math
    import torch

    value = float(std)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("behavior-cloned action standard deviation must be finite and positive")
    log_std = getattr(model.policy, "log_std", None)
    if log_std is None or not torch.is_tensor(log_std):
        raise ValueError("behavior-cloned policy must expose a tensor log_std")
    with torch.no_grad():
        log_std.fill_(math.log(value))
    return float(log_std.detach().exp().mean().cpu())


def save_behavior_cloned_checkpoint(model: Any, output_dir: Path) -> Path:
    """Save the cloned policy as a standalone, official-runtime checkpoint."""
    target = Path(output_dir) / "behavior_cloned_policy.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target.with_suffix("")))
    if not target.is_file():
        raise RuntimeError("behavior cloning did not produce its checkpoint")
    return target


def collect_centerline_demonstrations(
    vec_env: Any,
    expert: Any,
    *,
    steps: int,
    output_path: Path,
    should_stop: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, list[Optional[float]]], None]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Collect observations/actions; map pose is used for labels only."""
    observations = vec_env.reset()
    lidar_rows = []
    camera_rows = []
    state_rows = []
    action_rows = []
    successful_episodes = 0
    completed_episodes = 0
    termination_counts: Dict[str, int] = {}
    termination_examples = []
    n_envs = int(vec_env.num_envs)
    # Keep compact, non-identifying diagnostics so a saved BC run can prove
    # whether the teacher advanced along the route and whether it ever asked
    # for reverse. Do not persist privileged pose/yaw samples in the dataset.
    initial_route_progress = [None] * n_envs
    last_route_progress = [None] * n_envs
    route_forward_steps = [0] * n_envs
    route_backward_steps = [0] * n_envs
    reverse_motion_steps = [0] * n_envs
    negative_throttle_steps = [0] * n_envs

    total_steps = max(0, int(steps))
    report_every = max(1, total_steps // 10)
    for step_index in range(total_steps):
        if should_stop is not None and should_stop():
            break
        current_infos = vec_env.env_method("get_current_info")
        for env_index, info in enumerate(current_infos):
            progress = info.get("current_progress_m")
            if progress is not None:
                progress = float(progress)
                if initial_route_progress[env_index] is None:
                    initial_route_progress[env_index] = progress
                last_route_progress[env_index] = progress
        actions = expert.action_batch(current_infos)
        for env_index, action in enumerate(actions):
            if float(action[0]) < -1e-6:
                negative_throttle_steps[env_index] += 1
        for env_index in range(n_envs):
            lidar_rows.append(np.asarray(observations["lidar"][env_index], dtype=np.float32))
            state_rows.append(np.asarray(observations["state"][env_index], dtype=np.float32))
            if "camera" in observations:
                camera_rows.append(np.asarray(observations["camera"][env_index], dtype=np.uint8).copy())
            action_rows.append(actions[env_index])
        observations, _, dones, infos = vec_env.step(actions)
        for env_index, (done, info) in enumerate(zip(dones, infos)):
            progress = info.get("current_progress_m")
            if progress is not None:
                progress = float(progress)
                previous = last_route_progress[env_index]
                if previous is None:
                    initial_route_progress[env_index] = progress
                elif progress - previous > 0.02:
                    route_forward_steps[env_index] += 1
                elif progress - previous < -0.02:
                    route_backward_steps[env_index] += 1
                last_route_progress[env_index] = progress
            if float(info.get("v_long", 0.0) or 0.0) < -0.15:
                reverse_motion_steps[env_index] += 1
            if done:
                completed_episodes += 1
                if bool(info.get("episode_won", False)):
                    successful_episodes += 1
                reason = str(
                    info.get("termination_reason") or info.get("truncate_reason") or "unknown"
                )
                termination_counts[reason] = termination_counts.get(reason, 0) + 1
                if len(termination_examples) < 8:
                    termination_examples.append({
                        "env_id": env_index,
                        "reason": reason,
                        "lap_count": int(info.get("lap_count", 0) or 0),
                        "frontier_progress_m": info.get("frontier_progress_m"),
                    })
        if (step_index + 1) % report_every == 0 or step_index + 1 == total_steps:
            print(
                f"expert collection: {step_index + 1}/{total_steps} steps per env; "
                f"clean laps={successful_episodes}",
                flush=True,
            )
            if progress_callback is not None:
                progress_callback(step_index + 1, total_steps, list(last_route_progress))

    dataset = {
        "lidar": np.asarray(lidar_rows, dtype=np.float32),
        "state": np.asarray(state_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.float32),
    }
    if camera_rows:
        dataset["camera"] = np.asarray(camera_rows, dtype=np.uint8)
        if not np.any(dataset["camera"]):
            raise RuntimeError(
                "all collected camera frames are black; refusing to train a camera policy "
                "from a graphics-disabled or unrendered simulator stream"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    route_diagnostics = [
        {
            "env_id": env_index,
            "start_progress_m": initial_route_progress[env_index],
            "end_progress_m": last_route_progress[env_index],
            "net_progress_m": (
                None
                if initial_route_progress[env_index] is None
                or last_route_progress[env_index] is None
                else round(
                    last_route_progress[env_index] - initial_route_progress[env_index],
                    4,
                )
            ),
            "forward_progress_steps": route_forward_steps[env_index],
            "backward_progress_steps": route_backward_steps[env_index],
            "reverse_motion_steps": reverse_motion_steps[env_index],
            "negative_throttle_steps": negative_throttle_steps[env_index],
        }
        for env_index in range(n_envs)
    ]
    return dataset, {
        "transitions": int(len(action_rows)),
        "completed_episodes": completed_episodes,
        "successful_episodes": successful_episodes,
        "termination_counts": termination_counts,
        "termination_examples": termination_examples,
        "route_direction_diagnostics": route_diagnostics,
    }


def collect_policy_demonstrations(
    vec_env: Any,
    teacher: Any,
    *,
    steps: int,
    output_path: Path,
    behavior_policy: Any = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Record sensor observations paired with a privileged PPO teacher's actions.

    The running environments return the student observations. A separate
    Layer 2 diagnostic method supplies the teacher's simulator-profile input;
    only the student observations and normalized action labels are persisted.
    If ``behavior_policy`` is provided, it controls the car while the teacher
    still supplies labels, implementing a DAgger-style on-policy data round.
    """
    observations = vec_env.reset()
    lidar_rows = []
    camera_rows = []
    state_rows = []
    action_rows = []
    total_steps = max(0, int(steps))
    report_every = max(1, total_steps // 10)
    n_envs = int(vec_env.num_envs)

    for step_index in range(total_steps):
        teacher_observations = vec_env.env_method(
            "get_privileged_teacher_observation"
        )
        teacher_batch = {
            key: np.stack([row[key] for row in teacher_observations]).astype(
                np.float32, copy=False
            )
            for key in ("lidar", "state")
        }
        labels, _ = teacher.predict(teacher_batch, deterministic=True)
        labels = np.asarray(labels, dtype=np.float32).reshape(n_envs, 2)
        if behavior_policy is None:
            actions = labels
        else:
            actions, _ = behavior_policy.predict(observations, deterministic=True)
            actions = np.asarray(actions, dtype=np.float32).reshape(n_envs, 2)
        lidar_rows.extend(
            np.asarray(observations["lidar"], dtype=np.float32).copy()
        )
        state_rows.extend(
            np.asarray(observations["state"], dtype=np.float32).copy()
        )
        if "camera" in observations:
            camera_rows.extend(np.asarray(observations["camera"], dtype=np.uint8).copy())
        action_rows.extend(labels.copy())
        observations, _, _, _ = vec_env.step(actions)
        if (step_index + 1) % report_every == 0 or step_index + 1 == total_steps:
            print(
                f"policy distillation collection: {step_index + 1}/{total_steps} "
                f"steps per env; {len(action_rows)} sensor/action pairs",
                flush=True,
            )

    dataset = {
        "lidar": np.asarray(lidar_rows, dtype=np.float32),
        "state": np.asarray(state_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.float32),
    }
    if camera_rows:
        dataset["camera"] = np.asarray(camera_rows, dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    return dataset, {"transitions": int(len(action_rows)), "environments": n_envs}


def behavior_clone_actor(
    model: Any,
    dataset: Dict[str, np.ndarray],
    *,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, float]:
    """Fit the PPO actor's mean action to teacher labels, with a held-out split."""
    import torch
    import torch.nn.functional as F

    count = int(len(dataset["actions"]))
    if count < 2:
        raise ValueError("at least two expert transitions are required for behavior cloning")

    # Keep neighboring time samples together: random frame splits leak nearly
    # identical consecutive observations across train and validation and make
    # the reported imitation error look far better than real rollout transfer.
    validation_count = max(1, int(round(count * 0.1)))
    validation_indices = np.arange(count - validation_count, count)
    training_indices = np.arange(0, count - validation_count)
    if len(training_indices) == 0:
        training_indices, validation_indices = np.arange(count), np.arange(1)
    rng = np.random.default_rng(seed)

    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    device = policy.device

    def evaluate(indices: np.ndarray) -> float:
        losses = []
        policy.set_training_mode(False)
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                obs, _ = policy.obs_to_tensor({key: value[batch] for key, value in dataset.items() if key != "actions"})
                actions = torch.as_tensor(dataset["actions"][batch], device=device)
                predicted = policy.get_distribution(obs).distribution.mean
                losses.append(float(F.smooth_l1_loss(predicted, actions).item()))
        policy.set_training_mode(True)
        return float(np.mean(losses)) if losses else 0.0

    initial_validation_loss = evaluate(validation_indices)
    training_losses = []
    stopped_early = False
    for _ in range(max(1, int(epochs))):
        rng.shuffle(training_indices)
        for start in range(0, len(training_indices), batch_size):
            if should_stop is not None and should_stop():
                stopped_early = True
                break
            batch = training_indices[start:start + batch_size]
            obs, _ = policy.obs_to_tensor({key: value[batch] for key, value in dataset.items() if key != "actions"})
            actions = torch.as_tensor(dataset["actions"][batch], device=device)
            predicted = policy.get_distribution(obs).distribution.mean
            loss = F.smooth_l1_loss(predicted, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            training_losses.append(float(loss.item()))
        if stopped_early:
            break

    final_validation_loss = evaluate(validation_indices)
    policy.set_training_mode(False)
    return {
        "transitions": float(count),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "mean_training_loss": float(np.mean(training_losses)) if training_losses else 0.0,
        "epochs": float(max(1, int(epochs))),
        "stopped_early": float(stopped_early),
    }
