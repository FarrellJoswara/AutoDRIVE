"""Temporarily remove repeated LiDAR scan allocations in ignored AutoDRIVE source."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "simulator" / "unity" / "AutoDRIVE"
BACKUP = ROOT / "simulator" / "_build" / "lidar-allocation-source-backup"
SOURCE = PROJECT / "Assets" / "Scripts" / "LIDAR.cs"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one source anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


def apply() -> None:
    if BACKUP.exists():
        raise RuntimeError(f"backup already exists; restore first: {BACKUP}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    BACKUP.mkdir(parents=True)
    shutil.copy2(SOURCE, BACKUP / SOURCE.name)
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "\tprivate string[] IntensityArray; // Array storing range values of a scan\n",
        "\tprivate string[] IntensityArray; // Array storing range values of a scan\n"
        "\tprivate Vector3[] laserRayDirections;\n"
        "\tprivate RaycastHit[] hits;\n",
    )
    text = replace_once(
        text,
        "\t\tIntensityArray = new string[MeasurementsPerScan]; // Array storing range values of a scan\n",
        "\t\tIntensityArray = new string[MeasurementsPerScan]; // Array storing range values of a scan\n"
        "\t\tlaserRayDirections = new Vector3[MeasurementsPerScan];\n"
        "\t\thits = new RaycastHit[MeasurementsPerScan];\n",
    )
    text = replace_once(
        text,
        "\t\tVector3[] laserRayDirections = new Vector3[MeasurementsPerScan];\n",
        "",
    )
    text = replace_once(
        text,
        "\t\tRaycastHit[] hits = new RaycastHit[MeasurementsPerScan];\n",
        "",
    )
    text = replace_once(
        text,
        "\t\t// Perform raycasts\n\t\tfor (int i = 0; i < MeasurementsPerScan; i++)\n",
        "\t\t// Reuse one formatted value while keeping its serialized text unchanged.\n"
        "\t\tstring intensityText = Intensity.ToString();\n"
        "\t\t// Perform raycasts\n\t\tfor (int i = 0; i < MeasurementsPerScan; i++)\n",
    )
    text = replace_once(
        text,
        "\t\t\tIntensityArray[i] = Intensity.ToString(); // Update intensity\n",
        "\t\t\tIntensityArray[i] = intensityText; // Update intensity\n",
    )
    SOURCE.write_text(text, encoding="utf-8", newline="")
    print("LiDAR allocation experiment applied to", SOURCE)


def restore() -> None:
    saved = BACKUP / SOURCE.name
    if not saved.is_file():
        raise FileNotFoundError(saved)
    shutil.copy2(saved, SOURCE)
    shutil.rmtree(BACKUP)
    print("Restored original AutoDRIVE LiDAR source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "restore"))
    args = parser.parse_args()
    if args.action == "apply":
        apply()
    else:
        restore()


if __name__ == "__main__":
    main()
