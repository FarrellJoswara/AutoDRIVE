"""Custom SB3 features extractor: 1D-CNN on lidar + MLP on state."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarStateExtractor(BaseFeaturesExtractor):
    """
    Encode Dict obs {"lidar": (1080,), "state": (8,)} into one feature vector.

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
