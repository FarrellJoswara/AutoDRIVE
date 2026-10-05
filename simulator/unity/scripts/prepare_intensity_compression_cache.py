"""Temporarily cache the gzip payload for unchanged LiDAR intensity arrays."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "simulator" / "unity" / "AutoDRIVE"
LIDAR = PROJECT / "Assets" / "Scripts" / "LIDAR.cs"
SOCKET = PROJECT / "Assets" / "Scripts" / "Socket.cs"
BACKUP = ROOT / "simulator" / "_build" / "intensity-cache-source-backup"
FILES = (LIDAR, SOCKET)


def apply() -> None:
    if BACKUP.exists():
        raise RuntimeError(f"source backup already exists; restore it first: {BACKUP}")
    lidar = LIDAR.read_text(encoding="utf-8")
    socket = SOCKET.read_text(encoding="utf-8")
    if "CurrentCompressedIntensityArray" in lidar or "CurrentCompressedIntensityArray" in socket:
        raise RuntimeError("intensity compression cache is already applied")

    field = "\tprivate string[] IntensityArray; // Array storing range values of a scan\n"
    if lidar.count(field) != 1:
        raise RuntimeError("could not find the unique LiDAR intensity field")
    lidar = lidar.replace(field, field +
        "\tprivate string intensityArrayText;\n"
        "\tprivate string compressedIntensityArray;\n", 1)

    getter = "\tpublic string[] CurrentIntensityArray{get{return IntensityArray;}}\n"
    if lidar.count(getter) != 1:
        raise RuntimeError("could not find the unique LiDAR intensity getter")
    lidar = lidar.replace(getter, getter +
        "\tpublic string CurrentCompressedIntensityArray{get{return compressedIntensityArray;}}\n", 1)

    assignment = "\t\t\tIntensityArray[i] = Intensity.ToString(); // Update intensity\n"
    if lidar.count(assignment) != 1:
        raise RuntimeError("could not find the per-beam intensity assignment")
    lidar = lidar.replace(assignment, "", 1)
    visual = "\t\t// Visualize laser scan (if enabled) -- Rendered in editor and standalone mode\n"
    if lidar.count(visual) != 1:
        raise RuntimeError("could not find the post-raycast visualization anchor")
    update_cache = (
        "\t\tstring intensityText = Intensity.ToString();\n"
        "\t\tif (intensityArrayText != intensityText)\n"
        "\t\t{\n"
        "\t\t\tfor (int i = 0; i < MeasurementsPerScan; i++) IntensityArray[i] = intensityText;\n"
        "\t\t\tcompressedIntensityArray = DataCompressor.CompressArray(IntensityArray);\n"
        "\t\t\tintensityArrayText = intensityText;\n"
        "\t\t}\n\n"
    )
    lidar = lidar.replace(visual, update_cache + visual, 1)

    compress = "DataCompressor.CompressArray(LIDARUnits[i].CurrentIntensityArray)"
    if socket.count(compress) != 1:
        raise RuntimeError("could not find the unique intensity compression call")
    socket = socket.replace(compress, "LIDARUnits[i].CurrentCompressedIntensityArray", 1)

    BACKUP.mkdir(parents=True)
    for path in FILES:
        shutil.copy2(path, BACKUP / path.name)
    LIDAR.write_text(lidar, encoding="utf-8", newline="")
    SOCKET.write_text(socket, encoding="utf-8", newline="")
    print("Applied unchanged-intensity compression cache")


def restore() -> None:
    if not BACKUP.is_dir():
        raise RuntimeError(f"no source backup found: {BACKUP}")
    for path in FILES:
        saved = BACKUP / path.name
        if not saved.is_file():
            raise FileNotFoundError(saved)
        shutil.copy2(saved, path)
    shutil.rmtree(BACKUP)
    print("Restored original LiDAR and Socket sources")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "restore"))
    if parser.parse_args().action == "apply":
        apply()
    else:
        restore()


if __name__ == "__main__":
    main()
