"""Behavior-clone PPO's actor from sensorimotor teacher demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def collect_centerline_demonstrations(
    vec_env: Any,
    expert: Any,
    *,
    steps: int,
    output_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Collect observations/actions; map pose is used for labels only."""
    observations = vec_env.reset()
    lidar_rows = []
    state_rows = []
    action_rows = []
    position_rows = []
    yaw_rows = []
    progress_rows = []
    speed_rows = []
    successful_episodes = 0
    completed_episodes = 0
    termination_counts: Dict[str, int] = {}
    termination_examples = []
    n_envs = int(vec_env.num_envs)

    total_steps = max(0, int(steps))
    report_every = max(1, total_steps // 10)
    for step_index in range(total_steps):
        current_infos = vec_env.env_method("get_current_info")
        actions = expert.action_batch(current_infos)
        for env_index in range(n_envs):
            lidar_rows.append(np.asarray(observations["lidar"][env_index], dtype=np.float32))
            state_rows.append(np.asarray(observations["state"][env_index], dtype=np.float32))
            action_rows.append(actions[env_index])
            info = current_infos[env_index] if isinstance(current_infos[env_index], dict) else {}
            position = info.get("position") or (0.0, 0.0, 0.0)
            position_rows.append(np.asarray(position, dtype=np.float32))
            yaw_rows.append(float(info.get("yaw", 0.0) or 0.0))
            progress_rows.append(float(info.get("current_progress_m", 0.0) or 0.0))
            speed_rows.append(float(info.get("true_speed", 0.0) or 0.0))
        observations, _, dones, infos = vec_env.step(actions)
        for env_index, (done, info) in enumerate(zip(dones, infos)):
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
                        "current_progress_m": info.get("current_progress_m"),
                        "position": list(info.get("position") or ()),
                        "yaw": info.get("yaw"),
                        "speed": info.get("true_speed"),
                    })
        if (step_index + 1) % report_every == 0 or step_index + 1 == total_steps:
            print(
                f"expert collection: {step_index + 1}/{total_steps} steps per env; "
                f"clean laps={successful_episodes}",
                flush=True,
            )

    dataset = {
        "lidar": np.asarray(lidar_rows, dtype=np.float32),
        "state": np.asarray(state_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.float32),
        "positions": np.asarray(position_rows, dtype=np.float32),
        "yaws": np.asarray(yaw_rows, dtype=np.float32),
        "route_progress_m": np.asarray(progress_rows, dtype=np.float32),
        "speed_mps": np.asarray(speed_rows, dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    return dataset, {
        "transitions": int(len(action_rows)),
        "completed_episodes": completed_episodes,
        "successful_episodes": successful_episodes,
        "termination_counts": termination_counts,
        "termination_examples": termination_examples,
    }


def behavior_clone_actor(
    model: Any,
    dataset: Dict[str, np.ndarray],
    *,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> Dict[str, float]:
    """Fit the PPO actor's mean action to teacher labels, with a held-out split."""
    import torch
    import torch.nn.functional as F

    count = int(len(dataset["actions"]))
    if count < 2:
        raise ValueError("at least two expert transitions are required for behavior cloning")

    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    validation_count = max(1, int(round(count * 0.1)))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    if len(training_indices) == 0:
        training_indices, validation_indices = order, order[:1]

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
                obs, _ = policy.obs_to_tensor({
                    "lidar": dataset["lidar"][batch],
                    "state": dataset["state"][batch],
                })
                actions = torch.as_tensor(dataset["actions"][batch], device=device)
                predicted = policy.get_distribution(obs).distribution.mean
                losses.append(float(F.smooth_l1_loss(predicted, actions).item()))
        policy.set_training_mode(True)
        return float(np.mean(losses)) if losses else 0.0

    initial_validation_loss = evaluate(validation_indices)
    training_losses = []
    for _ in range(max(1, int(epochs))):
        rng.shuffle(training_indices)
        for start in range(0, len(training_indices), batch_size):
            batch = training_indices[start:start + batch_size]
            obs, _ = policy.obs_to_tensor({
                "lidar": dataset["lidar"][batch],
                "state": dataset["state"][batch],
            })
            actions = torch.as_tensor(dataset["actions"][batch], device=device)
            predicted = policy.get_distribution(obs).distribution.mean
            loss = F.smooth_l1_loss(predicted, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            training_losses.append(float(loss.item()))

    final_validation_loss = evaluate(validation_indices)
    policy.set_training_mode(False)
    return {
        "transitions": float(count),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "mean_training_loss": float(np.mean(training_losses)) if training_losses else 0.0,
        "epochs": float(max(1, int(epochs))),
    }
