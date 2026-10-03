"""Load ROS occupancy grids into a world-metre mask."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OccupancyGrid:
    """Known obstacle and known free masks; threshold-band cells stay unknown."""

    occupied: np.ndarray  # (H, W) bool, row 0 = image top
    free: np.ndarray  # (H, W) bool; unknown cells are false in both masks
    resolution: float
    origin_x: float
    origin_z: float  # ROS yaml origin[1] → Unity Z in this stack
    negate: int


def parse_ros_yaml(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = re.sub(r"#.*$", "", raw).strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            parts = [p.strip() for p in val[1:-1].split(",")]
            out[key] = [float(p) for p in parts]
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    if "image" not in out or "resolution" not in out or "origin" not in out:
        raise ValueError("yaml missing image/resolution/origin")
    return out


def _load_gray(path: Path) -> np.ndarray:
    """Return HxW uint8 grayscale (row 0 = top of image)."""
    if path.suffix.lower() == ".pgm":
        return _load_pgm(path)
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _load_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data.startswith(b"P5"):
        return _load_pgm_p5(data)
    if data.startswith(b"P2"):
        return _load_pgm_p2(data)
    # Pillow often handles both
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _load_pgm_p5(data: bytes) -> np.ndarray:
    i = 0
    n = len(data)

    def skip_ws_comments() -> None:
        nonlocal i
        while i < n:
            c = data[i]
            if c == 0x23:  # #
                while i < n and data[i] not in (0x0A, 0x0D):
                    i += 1
                continue
            if c <= 0x20:
                i += 1
                continue
            break

    def read_token() -> str:
        nonlocal i
        skip_ws_comments()
        start = i
        while i < n and data[i] > 0x20:
            i += 1
        return data[start:i].decode("ascii")

    magic = read_token()
    if magic != "P5":
        raise ValueError(f"unsupported PGM {magic}")
    width = int(read_token())
    height = int(read_token())
    maxval = int(read_token())
    if i < n and data[i] <= 0x20:
        i += 1
    need = width * height
    pixels = np.frombuffer(data, dtype=np.uint8, count=need, offset=i)
    if maxval != 255:
        pixels = (pixels.astype(np.float32) * (255.0 / maxval)).astype(np.uint8)
    return pixels.reshape((height, width))


def _load_pgm_p2(data: bytes) -> np.ndarray:
    text = data.decode("ascii", errors="ignore")
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tokens.extend(line.split())
    if not tokens or tokens[0] != "P2":
        raise ValueError("invalid P2 PGM")
    width = int(tokens[1])
    height = int(tokens[2])
    maxval = int(tokens[3])
    vals = np.array([int(t) for t in tokens[4 : 4 + width * height]], dtype=np.float32)
    if maxval != 255:
        vals = vals * (255.0 / maxval)
    return vals.astype(np.uint8).reshape((height, width))


def load_occupancy(occupancy_dir: Path, yaml_path: Path) -> OccupancyGrid:
    meta = parse_ros_yaml(yaml_path.read_text(encoding="utf-8"))
    image_path = occupancy_dir / str(meta["image"])
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    gray = _load_gray(image_path)
    resolution = float(meta["resolution"])
    origin = meta["origin"]
    origin_x = float(origin[0])
    origin_z = float(origin[1])
    negate = int(meta.get("negate", 0) or 0)
    occ_thresh = float(meta.get("occupied_thresh", 0.65) or 0.65)
    free_thresh = float(meta.get("free_thresh", 0.196) or 0.196)
    # ROS map_server: negate first, then convert grayscale to occupancy
    # probability. Values between thresholds are unknown, not free.
    vals = gray.astype(np.float32) / 255.0
    if negate:
        vals = 1.0 - vals
    occupancy = 1.0 - vals
    occupied = occupancy > occ_thresh
    free = occupancy < free_thresh
    return OccupancyGrid(
        occupied=occupied,
        free=free,
        resolution=resolution,
        origin_x=origin_x,
        origin_z=origin_z,
        negate=negate,
    )


def pixel_to_world(grid: OccupancyGrid, col: float, row: float) -> Tuple[float, float]:
    """Image pixel (col, row; row 0 = top) → world (x, z) metres."""
    h = grid.occupied.shape[0]
    x = grid.origin_x + col * grid.resolution
    # ROS: row 0 is top = origin_z + height*res
    z = grid.origin_z + (h - row) * grid.resolution
    return x, z
