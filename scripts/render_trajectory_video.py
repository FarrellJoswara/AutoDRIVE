#!/usr/bin/env python3
"""Render a recorded trajectory clip into an MP4 or GIF.

Example:
  python scripts/render_trajectory_video.py runs/.../trajectory_clips/clip-000001.jsonl -o comparison.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_clip(path: Path):
    samples = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("kind") == "sample":
                samples.append(item)
    if not samples:
        raise ValueError(f"clip contains no samples: {path}")
    return samples


def _map_transform(yaml_path: Path, image: Image.Image, crop_origin=(0, 0), original_height=None):
    text = yaml_path.read_text(encoding="utf-8")
    resolution = float(re.search(r"^resolution:\s*([0-9.eE+-]+)", text, re.MULTILINE).group(1))
    origin = [float(x) for x in re.search(r"^origin:\s*\[([^]]+)\]", text, re.MULTILINE).group(1).split(",")]
    height = image.height if original_height is None else original_height
    return lambda pose: (
        int(round((float(pose[0]) - origin[0]) / resolution)) - crop_origin[0],
        int(round(height - crop_origin[1] - (float(pose[1]) - origin[1]) / resolution)),
    )


def render(path: Path, output: Path, fps: int, tail: int, map_image: Path | None = None, map_yaml: Path | None = None) -> None:
    samples = load_clip(path)
    env_ids = sorted({int(car["env_id"]) for row in samples for car in row.get("cars", [])})
    if not env_ids:
        raise ValueError("clip contains no cars")
    if (map_image is None) != (map_yaml is None):
        raise ValueError("--map-image and --map-yaml must be supplied together")
    base = Image.open(map_image).convert("RGB") if map_image else None
    original_map_height = base.height if base is not None else None
    crop_origin = (0, 0)
    if base is not None:
        # Occupancy maps commonly use 205 (unknown/background), while track
        # surface and walls are materially lighter/darker. Crop to that useful
        # region so the map view does not waste most of the video on margins.
        gray = base.convert("L")
        pixels = gray.load()
        coords = [
            (x, y)
            for y in range(gray.height)
            for x in range(gray.width)
            if abs(int(pixels[x, y]) - 205) > 8
        ]
        if coords:
            left = max(0, min(x for x, _ in coords) - 16)
            top = max(0, min(y for _, y in coords) - 16)
            right = min(base.width, max(x for x, _ in coords) + 17)
            bottom = min(base.height, max(y for _, y in coords) + 17)
            crop_origin = (left, top)
            base = base.crop((left, top, right, bottom))
    if base is not None:
        raw_transform = _map_transform(map_yaml, base, crop_origin, original_map_height)
        display_scale = max(1.0, 360.0 / base.height)
        if display_scale > 1.0:
            base = base.resize(
                (round(base.width * display_scale), round(base.height * display_scale)),
                Image.Resampling.NEAREST,
            )
        transform = lambda pose: tuple(round(value * display_scale) for value in raw_transform(pose))
    all_points = [car["pose"] for row in samples for car in row.get("cars", []) if car.get("pose")]
    if not all_points:
        raise ValueError("clip contains no car poses")
    if base is None:
        xs, ys = zip(*all_points)
        scale = 80.0
        pad = 40
        width = int((max(xs) - min(xs)) * scale) + 2 * pad
        height = int((max(ys) - min(ys)) * scale) + 2 * pad
        transform = lambda pose: (int((pose[0] - min(xs)) * scale) + pad, height - int((pose[1] - min(ys)) * scale) - pad)
        base = Image.new("RGB", (max(320, width), max(240, height)), "#20252b")
    colors = [(235, 70, 70), (70, 150, 255), (70, 210, 120), (245, 190, 55), (210, 100, 240)]
    panel_width = 190
    histories = {env: [] for env in env_ids}
    font = ImageFont.load_default()
    canvas_width = base.width + panel_width
    canvas_height = base.height
    # yuv420p (the broadly compatible MP4 pixel format) requires even sizes.
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trajectory-video-") as frames_dir:
        for index, sample in enumerate(samples):
            frame = Image.new("RGB", (canvas_width, canvas_height), "#171b21")
            frame.paste(base, (0, 0))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((8, 8, min(base.width - 8, 330), 30), fill=(0, 0, 0))
            draw.text((14, 14), f"MAP VIEW  |  {index + 1}/{len(samples)}", fill="white", font=font)
            draw.rectangle((base.width, 0, frame.width, frame.height), fill="#171b21")
            draw.text((base.width + 12, 14), "LAP TELEMETRY", fill="white", font=font)
            for car in sample.get("cars", []):
                env = int(car["env_id"])
                pose = car.get("pose")
                if not pose or len(pose) < 2:
                    continue
                point = transform(pose)
                histories[env].append(point)
                visible = histories[env][-tail:] if tail > 0 else histories[env]
                color = colors[env % len(colors)]
                if len(visible) > 1:
                    draw.line(visible, fill=color, width=2)
                radius = 5
                draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color, outline="white", width=1)
                yaw = car.get("yaw")
                if yaw is not None:
                    heading = (point[0] + int(14 * math.sin(float(yaw))), point[1] - int(14 * math.cos(float(yaw))))
                    draw.line((point, heading), fill="white", width=2)
                draw.text((point[0] + 7, point[1] - 7), f"car {env}", fill=color, font=font)
            panel_y = 38
            for env in env_ids:
                car = next((item for item in sample.get("cars", []) if int(item.get("env_id", -1)) == env), {})
                color = colors[env % len(colors)]
                draw.text((base.width + 12, panel_y), f"CAR {env}", fill=color, font=font)
                panel_y += 16
                lap_count = car.get("lap_count", 0)
                draw.text((base.width + 12, panel_y), f"Lap: {lap_count}", fill="white", font=font)
                panel_y += 14
                last = car.get("last_lap_time_s")
                best = car.get("best_lap_time_s")
                elapsed = car.get("lap_elapsed_s")
                speed = car.get("speed")
                def fmt(value):
                    return f"{float(value):.2f}s" if value is not None else "--"
                draw.text((base.width + 12, panel_y), f"Last: {fmt(last)}", fill="white", font=font)
                panel_y += 14
                draw.text((base.width + 12, panel_y), f"Best: {fmt(best)}", fill="white", font=font)
                panel_y += 14
                draw.text((base.width + 12, panel_y), f"Now:  {fmt(elapsed)}", fill="white", font=font)
                panel_y += 14
                draw.text((base.width + 12, panel_y), f"Speed: {fmt(speed)}", fill="white", font=font)
                panel_y += 14
                status = "COLLISION" if car.get("collision") else "OK"
                draw.text((base.width + 12, panel_y), status, fill="#ff5555" if status != "OK" else "#70d278", font=font)
                panel_y += 24
            frame.save(Path(frames_dir) / f"frame-{index:06d}.png")
        if output.suffix.lower() == ".gif":
            frames = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in sorted(Path(frames_dir).glob("*.png"))]
            frames[0].save(output, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                "-i", str(Path(frames_dir) / "frame-%06d.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(output),
            ], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--tail", type=int, default=0, help="keep only this many recent samples per car; 0 keeps the full trail")
    parser.add_argument("--map-image", type=Path, help="occupancy/map image used as the map-view underlay")
    parser.add_argument("--map-yaml", type=Path, help="map YAML containing resolution and origin")
    args = parser.parse_args()
    if args.fps < 1 or args.tail < 0:
        parser.error("--fps must be positive and --tail cannot be negative")
    render(args.clip, args.output, args.fps, args.tail, args.map_image, args.map_yaml)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
