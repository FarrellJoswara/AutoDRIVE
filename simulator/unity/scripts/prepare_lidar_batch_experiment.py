"""Temporarily replace serial LiDAR raycasts with a staged Unity job batch."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "simulator" / "unity" / "AutoDRIVE"
SOURCE = PROJECT / "Assets" / "Scripts" / "LIDAR.cs"
BACKUP = ROOT / "simulator" / "_build" / "lidar-batch-source-backup.cs"


def apply() -> None:
    if BACKUP.exists():
        raise RuntimeError(f"source backup already exists; restore it first: {BACKUP}")
    text = SOURCE.read_text(encoding="utf-8")
    if "private NativeArray<RaycastCommand> batchCommands;" in text:
        raise RuntimeError("LiDAR batch experiment is already applied")

    old_start = "\tprivate string[] IntensityArray; // Array storing range values of a scan\n"
    new_start = old_start + (
        "\tprivate NativeArray<RaycastCommand> batchCommands;\n"
        "\tprivate NativeArray<RaycastHit> batchHits;\n"
        "\tprivate QueryParameters batchQueryParameters;\n"
    )
    if text.count(old_start) != 1:
        raise RuntimeError("could not find the unique LiDAR array field anchor")
    text = text.replace(old_start, new_start, 1)

    old_init = "\t\tIntensityArray = new string[MeasurementsPerScan]; // Array storing range values of a scan\n"
    new_init = old_init + (
        "\t\tbatchCommands = new NativeArray<RaycastCommand>(MeasurementsPerScan, Allocator.Persistent);\n"
        "\t\tbatchHits = new NativeArray<RaycastHit>(MeasurementsPerScan, Allocator.Persistent);\n"
        "\t\tbatchQueryParameters = new QueryParameters(layer_mask, false, QueryTriggerInteraction.UseGlobal, false);\n"
    )
    if text.count(old_init) != 1:
        raise RuntimeError("could not find the unique LiDAR initialization anchor")
    text = text.replace(old_init, new_init, 1)

    on_destroy_anchor = "\n\tvoid FixedUpdate()\n"
    on_destroy = (
        "\n\tprivate void OnDestroy()\n"
        "\t{\n"
        "\t\tif (batchCommands.IsCreated) batchCommands.Dispose();\n"
        "\t\tif (batchHits.IsCreated) batchHits.Dispose();\n"
        "\t}\n"
    )
    if text.count(on_destroy_anchor) != 1:
        raise RuntimeError("could not find the unique FixedUpdate anchor")
    text = text.replace(on_destroy_anchor, on_destroy + on_destroy_anchor, 1)

    old_raycast = """\t\t// Perform raycasts
		RaycastHit[] hits = new RaycastHit[MeasurementsPerScan];
		for (int i = 0; i < MeasurementsPerScan; i++)
		{
			if (Physics.Raycast(Head.transform.position, laserRayDirections[i], out hits[i], MaximumLinearRange, layer_mask))
			{
				float hitDistance = hits[i].distance;
				RangeArray[i] = (hitDistance > MinimumLinearRange) ? hitDistance.ToString() : "inf"; // Store range result
			}
			else
			{
				RangeArray[i] = "inf"; // No hit
			}

			IntensityArray[i] = Intensity.ToString(); // Update intensity
"""
    new_raycast = """\t\t// Schedule the same ordered rays as a batch, then wait before reading results.
		Vector3 rayOrigin = Head.transform.position;
		for (int i = 0; i < MeasurementsPerScan; i++)
		{
			batchCommands[i] = new RaycastCommand(rayOrigin, laserRayDirections[i], batchQueryParameters, MaximumLinearRange);
		}
		RaycastCommand.ScheduleBatch(batchCommands, batchHits, 32, default(JobHandle)).Complete();
		NativeArray<RaycastHit> hits = batchHits;
		for (int i = 0; i < MeasurementsPerScan; i++)
		{
			if (hits[i].collider != null)
			{
				float hitDistance = hits[i].distance;
				RangeArray[i] = (hitDistance > MinimumLinearRange) ? hitDistance.ToString() : "inf"; // Store range result
			}
			else
			{
				RangeArray[i] = "inf"; // No hit
			}

			IntensityArray[i] = Intensity.ToString(); // Update intensity
"""
    if text.count(old_raycast) != 1:
        raise RuntimeError("could not find the unique serial raycast loop")
    text = text.replace(old_raycast, new_raycast, 1)

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, BACKUP)
    SOURCE.write_text(text, encoding="utf-8", newline="")
    print("Applied isolated LiDAR batch experiment to", SOURCE)


def restore() -> None:
    if not BACKUP.is_file():
        raise RuntimeError(f"no source backup found: {BACKUP}")
    shutil.copy2(BACKUP, SOURCE)
    BACKUP.unlink()
    print("Restored original LiDAR source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "restore"))
    action = parser.parse_args().action
    if action == "apply":
        apply()
    else:
        restore()


if __name__ == "__main__":
    main()
