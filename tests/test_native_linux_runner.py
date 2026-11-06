import hashlib
import json
import runpy
from pathlib import Path

import pytest


def test_runner_competition_profile_accepts_pinned_competition_and_rejects_practice(tmp_path):
    runner = runpy.run_path("scripts/native-linux-runner.py")
    runtime = tmp_path / "competition-runtime"
    rootfs = runtime / "rootfs"
    simulator = rootfs / "home/autodrive_simulator/AutoDRIVE Simulator.x86_64"
    simulator.parent.mkdir(parents=True)
    simulator.write_bytes(b"competition player")
    simulator.chmod(0o755)
    data = simulator.parent / "Data"
    data.mkdir()
    (data / "level0").write_bytes(b"competition scene")
    (data / "globalgamemanagers").write_bytes(b"competition build")
    digest = "sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d"
    identity = {
        "repository": "autodriveecosystem/autodrive_roboracer_sim",
        "tag": "2026-iros-compete", "digest": digest,
        "manifest_verified": True, "architecture": "linux/amd64",
        "rootfs_tree_digest": "sha256:" + "a" * 64,
        "simulator_sha256": hashlib.sha256(simulator.read_bytes()).hexdigest(),
        "level0_sha256": hashlib.sha256((data / "level0").read_bytes()).hexdigest(),
        "globalgamemanagers_sha256": hashlib.sha256((data / "globalgamemanagers").read_bytes()).hexdigest(),
    }
    (runtime / "IMAGE-IDENTITY.json").write_text(json.dumps(identity))
    (rootfs / ".OCI-COMPLETE.json").write_text(json.dumps({"tag": identity["tag"], "digest": digest, "verified_layers": 30}))
    runner["require_identity"]("competition", runtime, simulator)

    identity.update(tag="2026-iros-practice", digest="sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76")
    (runtime / "IMAGE-IDENTITY.json").write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="pinned official linux/amd64 competition"):
        runner["require_identity"]("competition", runtime, simulator)
