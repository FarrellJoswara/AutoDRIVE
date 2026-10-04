"""Unit test LidarStateExtractor shapes (no Unity)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.layer2.spaces import LIDAR_BEAMS, STATE_DIM, make_observation_space
from src.layer3.extractors import LidarStateExtractor


def test_lidar_state_extractor_forward_shape():
    space = make_observation_space()
    extractor = LidarStateExtractor(space, features_dim=256)
    batch = 4
    obs = {
        "lidar": torch.rand(batch, LIDAR_BEAMS),
        "state": torch.randn(batch, STATE_DIM),
    }
    out = extractor(obs)
    assert out.shape == (batch, 256)
    assert torch.isfinite(out).all()


def test_lidar_state_extractor_features_dim_matches():
    space = make_observation_space()
    extractor = LidarStateExtractor(space, features_dim=128)
    out = extractor({"lidar": torch.zeros(1, LIDAR_BEAMS), "state": torch.zeros(1, STATE_DIM)})
    assert out.shape == (1, 128)
    assert extractor.features_dim == 128
