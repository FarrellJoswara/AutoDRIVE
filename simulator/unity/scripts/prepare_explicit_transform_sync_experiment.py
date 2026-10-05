"""Temporarily batch wheel-transform physics synchronization for an A/B build."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "simulator" / "unity" / "AutoDRIVE"
CONTROLLER = PROJECT / "Assets" / "Scripts" / "VehicleController.cs"
SETTINGS = PROJECT / "ProjectSettings" / "DynamicsManager.asset"
CONTROLLER_BACKUP = ROOT / "simulator" / "_build" / "explicit-sync-controller-backup.cs"
SETTINGS_BACKUP = ROOT / "simulator" / "_build" / "explicit-sync-settings-backup.asset"


def apply() -> None:
    if CONTROLLER_BACKUP.exists() or SETTINGS_BACKUP.exists():
        raise RuntimeError("an explicit-sync backup already exists; restore it first")
    controller = CONTROLLER.read_text(encoding="utf-8")
    anchor = "UpdateWheelPoses();"
    if controller.count(anchor) != 1:
        raise RuntimeError("could not find the unique wheel-pose update call")
    if "Physics.SyncTransforms();" in controller:
        raise RuntimeError("controller already contains an explicit transform sync")
    settings = SETTINGS.read_text(encoding="utf-8")
    enabled = "m_AutoSyncTransforms: 1"
    disabled = "m_AutoSyncTransforms: 0"
    if settings.count(enabled) != 1:
        raise RuntimeError("expected exactly one enabled Auto Sync Transforms setting")

    CONTROLLER_BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONTROLLER, CONTROLLER_BACKUP)
    shutil.copy2(SETTINGS, SETTINGS_BACKUP)
    patched = controller.replace(anchor,
        anchor + "\n            Physics.SyncTransforms();", 1)
    CONTROLLER.write_text(patched, encoding="utf-8", newline="")
    SETTINGS.write_text(settings.replace(enabled, disabled, 1), encoding="utf-8", newline="")
    print("Applied one explicit sync after the four wheel pose updates")


def restore() -> None:
    if not CONTROLLER_BACKUP.is_file() or not SETTINGS_BACKUP.is_file():
        raise RuntimeError("both explicit-sync source backups are required for restore")
    shutil.copy2(CONTROLLER_BACKUP, CONTROLLER)
    shutil.copy2(SETTINGS_BACKUP, SETTINGS)
    CONTROLLER_BACKUP.unlink()
    SETTINGS_BACKUP.unlink()
    print("Restored original controller and Auto Sync Transforms setting")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "restore"))
    if parser.parse_args().action == "apply":
        apply()
    else:
        restore()


if __name__ == "__main__":
    main()
