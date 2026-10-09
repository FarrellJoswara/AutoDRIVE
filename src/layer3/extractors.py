"""Custom SB3 features extractor: 1D-CNN on lidar + MLP on state."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def transfer_lidar_policy_to_camera(source: dict, target: dict) -> dict:
    """Copy an unpooled lidar/state PPO policy into a camera-fused policy.

    The camera branch starts with zero influence on the merge layer, so the
    initial action distribution exactly matches the champion while PPO can
    subsequently learn camera features from policy gradients.
    """
    result = {key: value.clone() for key, value in target.items()}
    copied = set()
    for key, value in source.items():
        if key.endswith("merge.0.weight"):
            continue
        if key in result and result[key].shape == value.shape:
            result[key] = value.clone()
            copied.add(key)
    camera_width = 64 * 4 * 6
    merge_keys = [key for key in source if key.endswith("merge.0.weight")]
    if not merge_keys:
        raise ValueError("source policy has no merge-layer weights")
    for merge_key in merge_keys:
        if merge_key not in target:
            raise ValueError(f"target policy is missing {merge_key}")
        source_merge = source[merge_key]
        target_merge = target[merge_key]
        state_key = merge_key.removesuffix("merge.0.weight") + "state_net.2.weight"
        if state_key not in source:
            raise ValueError(f"source policy is missing {state_key}")
        state_width = int(source[state_key].shape[0])
        lidar_width = int(source_merge.shape[1] - state_width)
        if target_merge.shape[1] != lidar_width + camera_width + state_width:
            raise ValueError(
                f"unexpected target merge width {target_merge.shape[1]} for "
                f"lidar={lidar_width}, camera={camera_width}, state={state_width}"
            )
        if source_merge.shape[0] != target_merge.shape[0]:
            raise ValueError("source and target merge output widths differ")
        result[merge_key][:, :lidar_width] = source_merge[:, :lidar_width]
        result[merge_key][:, lidar_width:lidar_width + camera_width] = 0
        result[merge_key][:, -state_width:] = source_merge[:, -state_width:]
        copied.add(merge_key)
    missing = [
        key for key in source
        if key not in copied and key not in result
    ]
    if missing:
        raise ValueError(f"source policy has unmapped parameters: {missing[:5]}")
    return result


class LidarStateExtractor(BaseFeaturesExtractor):
    """
    Encode Dict obs {"lidar": (1081,), "state": (9,)} into one feature vector.

    LiDAR uses a 1D-CNN (neighboring beams). State uses a small MLP. Both are
    concatenated and projected to ``features_dim``.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        n_lidar = int(observation_space.spaces["lidar"].shape[0])
        n_state = int(observation_space.spaces["state"].shape[0])

        self.lidar_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            lidar_out_dim = int(self.lidar_net(torch.zeros(1, 1, n_lidar)).shape[1])

        self.state_net = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.merge = nn.Sequential(
            nn.Linear(lidar_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        lidar = observations["lidar"].float().unsqueeze(1)  # (B, 1, L)
        state = observations["state"].float()
        fused = torch.cat([self.lidar_net(lidar), self.state_net(state)], dim=1)
        return self.merge(fused)


class PooledLidarStateExtractor(BaseFeaturesExtractor):
    """Encode a LiDAR scan while retaining local obstacle proximity by sector.

    The legacy extractor flattens 271 x 64 convolution activations into a dense
    projection. This version pools them to 64 ordered scan sectors first. Both
    average and maximum pooling preserve broad geometry and narrow close returns,
    while reducing the projection input from 17,408 to 8,256 values.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        n_lidar = int(observation_space.spaces["lidar"].shape[0])
        n_state = int(observation_space.spaces["state"].shape[0])
        self.lidar_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.lidar_average = nn.AdaptiveAvgPool1d(64)
        self.lidar_nearest = nn.AdaptiveMaxPool1d(64)
        with torch.no_grad():
            features = self.lidar_net(torch.zeros(1, 1, n_lidar))
            average = self.lidar_average(features)
            nearest = self.lidar_nearest(features)
            lidar_out_dim = int(average.shape[1] * average.shape[2] + nearest.shape[1] * nearest.shape[2])

        self.state_net = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.merge = nn.Sequential(
            nn.Linear(lidar_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        lidar = observations["lidar"].float().unsqueeze(1)
        state = observations["state"].float()
        features = self.lidar_net(lidar)
        pooled = torch.cat(
            [
                self.lidar_average(features).flatten(start_dim=1),
                self.lidar_nearest(features).flatten(start_dim=1),
            ],
            dim=1,
        )
        return self.merge(torch.cat([pooled, self.state_net(state)], dim=1))


class TemporalLidarStateExtractor(BaseFeaturesExtractor):
    """Encode four ordered official LiDAR frames plus the current sensor state."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        lidar_shape = observation_space.spaces["lidar"].shape
        if len(lidar_shape) != 2:
            raise ValueError("temporal_lidar_cnn requires lidar shape (frames, beams)")
        n_frames, n_beams = map(int, lidar_shape)
        n_state = int(observation_space.spaces["state"].shape[0])
        self.lidar_net = nn.Sequential(
            nn.Conv1d(n_frames, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.lidar_average = nn.AdaptiveAvgPool1d(64)
        self.lidar_nearest = nn.AdaptiveMaxPool1d(64)
        self.state_net = nn.Sequential(
            nn.Linear(n_state, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.merge = nn.Sequential(nn.Linear(64 * 64 * 2 + 64, features_dim), nn.ReLU())

    def forward(self, observations: dict) -> torch.Tensor:
        lidar = observations["lidar"].float()
        state = observations["state"].float()
        features = self.lidar_net(lidar)
        pooled = torch.cat(
            [self.lidar_average(features).flatten(1), self.lidar_nearest(features).flatten(1)],
            dim=1,
        )
        return self.merge(torch.cat([pooled, self.state_net(state)], dim=1))


class LidarCameraStateExtractor(BaseFeaturesExtractor):
    """Fuse official LiDAR, front RGB camera, and proprioceptive state."""

    def __init__(
        self, observation_space: gym.spaces.Dict, features_dim: int = 256,
        lidar_pooling: bool = True,
    ):
        super().__init__(observation_space, features_dim=features_dim)
        n_lidar = int(observation_space.spaces["lidar"].shape[0])
        n_state = int(observation_space.spaces["state"].shape[0])
        camera_shape = observation_space.spaces["camera"].shape
        if len(camera_shape) != 3 or 3 not in (camera_shape[0], camera_shape[2]):
            raise ValueError("camera observation must be RGB HxWx3")
        self.lidar_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            *([nn.AdaptiveAvgPool1d(64)] if lidar_pooling else []), nn.Flatten(),
        )
        self.lidar_pooling = bool(lidar_pooling)
        self.camera_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
        )
        self.state_net = nn.Sequential(
            nn.Linear(n_state, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.merge = nn.Sequential(
            nn.Linear(
                (64 * 64 if self.lidar_pooling else int(self.lidar_net(torch.zeros(1, 1, n_lidar)).shape[1]))
                + 64 * 4 * 6 + 64,
                features_dim,
            ), nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        lidar = observations["lidar"].float().unsqueeze(1)
        camera = observations["camera"].float()
        if camera.shape[-1] == 3:
            camera = camera.permute(0, 3, 1, 2)
        camera = camera / 255.0
        state = observations["state"].float()
        return self.merge(torch.cat(
            [self.lidar_net(lidar), self.camera_net(camera), self.state_net(state)], dim=1
        ))


def policy_kwargs_for_architecture(architecture: str) -> dict:
    """Return SB3 policy kwargs for a named, checkpoint-compatible network."""
    extractors = {
        "lidar_cnn": LidarStateExtractor,
        "lidar_cnn_pooled": PooledLidarStateExtractor,
        "temporal_lidar_cnn": TemporalLidarStateExtractor,
        "lidar_camera_cnn": LidarCameraStateExtractor,
        "lidar_camera_raw_cnn": LidarCameraStateExtractor,
    }
    try:
        extractor = extractors[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Unknown policy architecture {architecture!r}; expected one of {tuple(extractors)}"
        ) from exc
    return {
        "features_extractor_class": extractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            **({"lidar_pooling": False} if architecture == "lidar_camera_raw_cnn" else {}),
        },
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
    }
