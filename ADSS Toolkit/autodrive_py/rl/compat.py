"""Contracts / observation compatibility checks (hard-refuse mismatches)."""

from __future__ import annotations

from pathlib import Path

from .contracts import CONTRACTS_VERSION, N_LIDAR_DEFAULT, obs_dim


class ContractsMismatch(RuntimeError):
    """Raised when a checkpoint/config does not match the frozen contracts."""


def expected_obs_dim(n_lidar: int = N_LIDAR_DEFAULT) -> int:
    return obs_dim(int(n_lidar))


def refuse_obs_dim(got: int, *, n_lidar: int = N_LIDAR_DEFAULT, label: str = "model") -> None:
    expected = expected_obs_dim(n_lidar)
    if int(got) != expected:
        raise ContractsMismatch(
            f"{label} obs_dim={got} but contracts {CONTRACTS_VERSION} expects {expected} "
            f"(n_lidar={n_lidar}). Quarantine/retrain; v1 LiDAR-only zips are refused."
        )


def refuse_model_obs(model, *, n_lidar: int = N_LIDAR_DEFAULT, label: str = "model") -> None:
    """Hard-refuse SB3 (or any) policy whose observation_space width mismatches contracts."""
    space = getattr(model, "observation_space", None)
    if space is None or not hasattr(space, "shape") or not space.shape:
        raise ContractsMismatch(f"{label}: missing observation_space")
    refuse_obs_dim(int(space.shape[-1]), n_lidar=n_lidar, label=label)


def refuse_config_contracts(config: dict | None, *, label: str = "config") -> None:
    if not config:
        raise ContractsMismatch(f"{label}: missing config.json")
    ver = str(config.get("contracts_version") or "")
    if ver and ver != CONTRACTS_VERSION:
        raise ContractsMismatch(
            f"{label} contracts_version={ver!r} != frozen {CONTRACTS_VERSION}"
        )
    if "obs_dim" in config:
        refuse_obs_dim(int(config["obs_dim"]), n_lidar=int(config.get("n_lidar", N_LIDAR_DEFAULT)), label=label)


def load_ppo_refusing_mismatch(
    model_path: str | Path,
    *,
    n_lidar: int = N_LIDAR_DEFAULT,
    device: str = "auto",
):
    """``PPO.load`` + hard refuse on obs/contracts mismatch."""
    from stable_baselines3 import PPO

    path = Path(model_path)
    label = str(path)
    model = PPO.load(str(path), device=device)
    refuse_model_obs(model, n_lidar=n_lidar, label=label)
    # Sibling config.json when present
    cfg_path = path.parent / "config.json"
    if cfg_path.is_file():
        import json

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = None
        if cfg:
            refuse_config_contracts(cfg, label=str(cfg_path))
    return model
