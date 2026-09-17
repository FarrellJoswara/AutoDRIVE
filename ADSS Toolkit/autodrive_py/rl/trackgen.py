"""
Procedural F1TENTH-style track generation (Phase 2).

Adapted from f1tenth_gym random_trackgen.py (CarRacing-style), MIT License
Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

Outputs per map under maps/<name>/:
  <name>.png, <name>.yaml, centerline.csv, start_pose.txt  (+ <name>.pgm with --pgm)

Map ``<prefix><i>`` is reproducible from ``(seed, i)`` alone, so a pack can be
grown or repaired one map at a time without shifting the other maps' geometry.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import shapely.geometry as shp


WIDTH = 10.0  # half-track buffer (pixels in generator space → meters after scale)

# Bumped when generated geometry/outputs change: v1 = shared RNG stream + pgm
# image, v2 = per-index RNG + png image. Recorded in the map pack manifest.
GEN_VERSION = 2


def create_track(rng: np.random.Generator):
    CHECKPOINTS = 16
    SCALE = 6.0
    TRACK_RAD = 900 / SCALE
    TRACK_DETAIL_STEP = 21 / SCALE
    TRACK_TURN_RATE = 0.31
    start_alpha = 0.0

    checkpoints = []
    for c in range(CHECKPOINTS):
        alpha = 2 * math.pi * c / CHECKPOINTS + float(rng.uniform(0, 2 * math.pi / CHECKPOINTS))
        rad = float(rng.uniform(TRACK_RAD / 3, TRACK_RAD))
        if c == 0:
            alpha = 0
            rad = 1.5 * TRACK_RAD
        if c == CHECKPOINTS - 1:
            alpha = 2 * math.pi * c / CHECKPOINTS
            start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
            rad = 1.5 * TRACK_RAD
        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

    x, y, beta = 1.5 * TRACK_RAD, 0.0, 0.0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi
        while True:
            failed = True
            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break
            if not failed:
                break
            alpha -= 2 * math.pi
            continue
        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x
        dest_dy = dest_y - y
        proj = r1x * dest_dx + r1y * dest_dy
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi
        prev_beta = beta
        proj *= SCALE
        if proj > 0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * TRACK_DETAIL_STEP
        y += p1y * TRACK_DETAIL_STEP
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break

    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i == 0:
            return None
        pass_through_start = track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break
    track = track[i1 : i2 - 1]
    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)
    well_glued = math.sqrt(
        (first_perp_x * (track[0][2] - track[-1][2])) ** 2
        + (first_perp_y * (track[0][3] - track[-1][3])) ** 2
    )
    if well_glued > TRACK_DETAIL_STEP:
        return None

    track_xy = np.asarray([(x, y) for (_, _, x, y) in track], dtype=np.float64)
    track_poly = shp.Polygon(track_xy)
    if not track_poly.is_valid or track_poly.is_empty:
        return None
    track_in = track_poly.buffer(WIDTH)
    track_out = track_poly.buffer(-WIDTH)
    if track_in.is_empty or track_out.is_empty:
        return None
    return track_xy, np.array(track_in.exterior.coords), np.array(track_out.exterior.coords)


def _world_to_px(xy: np.ndarray, origin_xy: np.ndarray, resolution: float, height: int) -> np.ndarray:
    # ROS map: x right, y up; image row 0 is top
    px = (xy[:, 0] - origin_xy[0]) / resolution
    py = height - 1 - (xy[:, 1] - origin_xy[1]) / resolution
    return np.stack([px, py], axis=1)


def convert_track(
    track: np.ndarray,
    track_int: np.ndarray,
    track_ext: np.ndarray,
    out_dir: Path,
    name: str,
    resolution: float = 0.05,
    *,
    scale: float = 1.0,
    write_pgm: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if scale != 1.0:
        s = float(scale)
        track, track_int, track_ext = track * s, track_int * s, track_ext * s

    # Bounds with margin
    all_pts = np.vstack([track_int, track_ext, track])
    min_xy = all_pts.min(axis=0) - 5.0
    max_xy = all_pts.max(axis=0) + 5.0
    # Scale generator units → meters (generator space ~ same as f1tenth script * 0.05 later)
    # Match upstream: treat generator coords as pixels at 0.05 m after transform.
    # We rasterize in meters directly using resolution.
    width_m = float(max_xy[0] - min_xy[0])
    height_m = float(max_xy[1] - min_xy[1])
    w = max(64, int(math.ceil(width_m / resolution)))
    h = max(64, int(math.ceil(height_m / resolution)))
    origin = np.array([min_xy[0], min_xy[1]], dtype=np.float64)

    # Free = white (255), occupied = black (0)
    img = np.zeros((h, w), dtype=np.uint8)

    def poly_px(poly_xy: np.ndarray) -> np.ndarray:
        return np.round(_world_to_px(poly_xy, origin, resolution, h)).astype(np.int32)

    # Fill exterior free space ring: exterior wall outer, interior hole
    # Track corridor = between outer and inner boundaries.
    # Outer boundary (track_out is inner wall in upstream naming — buffer(-WIDTH) = inside)
    # Upstream: track_int = buffer(+WIDTH) exterior wall, track_ext = buffer(-WIDTH) interior wall
    outer = poly_px(track_int)
    inner = poly_px(track_ext)
    cv2.fillPoly(img, [outer], 255)
    cv2.fillPoly(img, [inner], 0)
    # Draw wall lines thicker for collision reliability
    cv2.polylines(img, [outer], isClosed=True, color=0, thickness=2)
    cv2.polylines(img, [inner], isClosed=True, color=0, thickness=2)

    png_path = out_dir / f"{name}.png"
    pgm_path = out_dir / f"{name}.pgm"
    yaml_path = out_dir / f"{name}.yaml"
    center_path = out_dir / "centerline.csv"
    start_path = out_dir / "start_pose.txt"

    cv2.imwrite(str(png_path), img)
    # PGM is the ROS map_server convention but costs ~400x the png on disk at
    # this raster size; the gym/UI read whatever `image:` points at.
    if write_pgm:
        cv2.imwrite(str(pgm_path), img)
    image_name = f"{name}.pgm" if write_pgm else f"{name}.png"

    yaml_path.write_text(
        "\n".join(
            [
                f"image: {image_name}",
                f"resolution: {resolution:.6f}",
                f"origin: [{origin[0]:.6f}, {origin[1]:.6f}, 0.000000]",
                "negate: 0",
                "occupied_thresh: 0.45",
                "free_thresh: 0.196",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Centerline in world meters
    with center_path.open("w", encoding="utf-8") as f:
        for x, y in track:
            f.write(f"{x:.6f}, {y:.6f}\n")

    # Start pose: first centerline point, heading toward second
    p0 = track[0]
    p1 = track[min(5, len(track) - 1)]
    theta = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    start_path.write_text(f"{p0[0]:.6f} {p0[1]:.6f} {theta:.6f}\n", encoding="utf-8")

    return yaml_path


def map_yaml_path(out_root: Path, name: str) -> Path:
    return Path(out_root) / name / f"{name}.yaml"


def _map_rng(seed: int, index: int, attempt: int) -> np.random.Generator:
    """Independent stream per (seed, index) so each map id regenerates identically."""
    return np.random.default_rng([int(seed), int(index), int(attempt)])


def generate_map(
    out_root: Path,
    index: int,
    *,
    seed: int = 123,
    prefix: str = "map",
    resolution: float = 0.05,
    scale: float = 1.0,
    write_pgm: bool = False,
    max_attempts: int = 40,
) -> dict | None:
    """Generate one map ``<prefix><index>``; returns its spec, or None if all attempts fail."""
    name = f"{prefix}{index}"
    for attempt in range(max(1, int(max_attempts))):
        result = create_track(_map_rng(seed, index, attempt))
        if result is None:
            continue
        track, track_int, track_ext = result
        try:
            yaml_path = convert_track(
                track,
                track_int,
                track_ext,
                Path(out_root) / name,
                name,
                resolution=resolution,
                scale=scale,
                write_pgm=write_pgm,
            )
        except Exception:
            continue
        return {
            "id": name,
            "index": int(index),
            "yaml": yaml_path,
            "gen_seed": int(seed),
            "gen_attempt": int(attempt),
            "gen_version": GEN_VERSION,
            "resolution": float(resolution),
            "scale": float(scale),
            "has_pgm": bool(write_pgm),
            "n_centerline": int(len(track)),
        }
    return None


def generate_map_specs(
    out_root: Path,
    num_maps: int = 3,
    seed: int = 123,
    prefix: str = "map",
    *,
    start_index: int = 0,
    skip_existing: bool = True,
    resolution: float = 0.05,
    scale: float = 1.0,
    write_pgm: bool = False,
) -> list[dict]:
    """Ensure maps ``<prefix><start_index..>`` exist on disk; cached ones are not redrawn.

    Specs of cached maps carry ``cached=True`` and no generation provenance (it
    lives in the map pack manifest instead).
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    specs: list[dict] = []
    for i in range(int(start_index), int(start_index) + int(num_maps)):
        name = f"{prefix}{i}"
        existing = map_yaml_path(out_root, name)
        if skip_existing and existing.is_file():
            specs.append({"id": name, "index": i, "yaml": existing, "cached": True})
            continue
        spec = generate_map(
            out_root,
            i,
            seed=seed,
            prefix=prefix,
            resolution=resolution,
            scale=scale,
            write_pgm=write_pgm,
            max_attempts=40,
        )
        if spec is not None:
            spec["cached"] = False
            specs.append(spec)
    return specs


def generate_maps(
    out_root: Path,
    num_maps: int = 3,
    seed: int = 123,
    prefix: str = "map",
    **kwargs,
) -> list[Path]:
    """Backwards-compatible wrapper: returns the ensured map yaml paths."""
    specs = generate_map_specs(out_root, num_maps=num_maps, seed=seed, prefix=prefix, **kwargs)
    return [s["yaml"] for s in specs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: generate F1TENTH race maps")
    parser.add_argument("--num_maps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "maps"),
    )
    parser.add_argument("--prefix", type=str, default="map")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Shrink/grow track geometry (1.0 = legacy size; smaller = tighter, cheaper raster)",
    )
    parser.add_argument("--pgm", action="store_true", help="Also write the ~40MB ROS .pgm twin")
    parser.add_argument("--force", action="store_true", help="Redraw maps that already exist")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip adopting the results into maps/map_pack.json",
    )
    args = parser.parse_args(argv)

    out_root = Path(args.out)
    specs = generate_map_specs(
        out_root,
        num_maps=args.num_maps,
        seed=args.seed,
        prefix=args.prefix,
        start_index=args.start_index,
        skip_existing=not args.force,
        resolution=args.resolution,
        scale=args.scale,
        write_pgm=args.pgm,
    )
    for s in specs:
        print(f"{'cached ' if s.get('cached') else 'wrote  '}{s['yaml']}")
    if not args.no_register and specs:
        try:
            from .map_pack import register_specs

            register_specs(specs, maps_root=out_root)
            print(f"registered in {out_root / 'map_pack.json'}")
        except Exception as exc:  # pack registration is advisory, never fatal here
            print(f"NOTE: map pack not updated ({exc})")
    if len(specs) < args.num_maps:
        print(f"Only produced {len(specs)}/{args.num_maps} maps after retries")
        return 1
    print(f"OK: {len(specs)} map sets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
