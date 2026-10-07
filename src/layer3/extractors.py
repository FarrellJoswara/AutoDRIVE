"""Custom SB3 features extractor: 1D-CNN on lidar + MLP on state."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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


def policy_kwargs_for_architecture(architecture: str) -> dict:
    """Return SB3 policy kwargs for a named, checkpoint-compatible network."""
    extractors = {
        "lidar_cnn": LidarStateExtractor,
        "lidar_cnn_pooled": PooledLidarStateExtractor,
        "temporal_lidar_cnn": TemporalLidarStateExtractor,
    }
    try:
        extractor = extractors[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Unknown policy architecture {architecture!r}; expected one of {tuple(extractors)}"
        ) from exc
    return {
        "features_extractor_class": extractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
    }
