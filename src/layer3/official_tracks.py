"""Choose a privileged route matching the official simulator variant."""
from pathlib import Path
import os
import hashlib
from pathlib import PureWindowsPath


def official_frontier_path(*, backend=None, image=None, profile=None):
    backend = backend if backend is not None else os.environ.get("AICAR_SIMULATOR_BACKEND", "")
    image = image if image is not None else os.environ.get("AICAR_OFFICIAL_SIMULATOR_IMAGE", "")
    profile = profile if profile is not None else os.environ.get("AICAR_OFFICIAL_TRACK_PROFILE", "")
    if not profile:
        profile = "competition" if backend == "windows_external" or "compete" in image else "practice"
    root = Path(__file__).resolve().parents[2] / "competition" / "iros2026"
    if profile == "competition":
        return root / "tracks" / "competition" / "official_centerline.csv"
    if profile == "practice":
        return root / "official_centerline.csv"
    raise ValueError(f"Unknown official track profile: {profile}")


# Scene hashes verified read-only from the official Windows releases. Unknown
# builds require an explicit geometry verification instead of a silent fallback.
WINDOWS_SCENES = {
    "competition": "a5d11e435857e2ead32ec0eb065dbbd011c41fa80d1bf8a187177014e014b674",
    "practice": "5edbd1bacc15647746e6e89d9bcf1bb9e202d62d47e76ae352497be1404b3f8d",
}


def verify_windows_scene(frontier_path, executable=None):
    if executable is None:
        executable = os.environ.get("AICAR_OFFICIAL_SIMULATOR_EXE", "").strip()
        if not executable:
            saved = Path(__file__).resolve().parents[2] / "logs/layer4/official_simulator_path.txt"
            executable = saved.read_text(encoding="utf-8-sig").strip()
    win = PureWindowsPath(str(executable))
    if os.name != "nt" and win.drive:
        executable = Path("/mnt") / win.drive[0].lower() / Path(*win.parts[1:])
    scene = Path(executable).parent / "Data" / "level0"
    digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    profile = "competition" if Path(frontier_path).parent.name == "competition" else "practice"
    if digest != WINDOWS_SCENES[profile]:
        raise ValueError(
            f"Official simulator scene does not match {profile} route: {scene} ({digest}). "
            "Verify the track geometry and select the matching AICAR_OFFICIAL_TRACK_PROFILE."
        )
    return {"scene_path": str(scene), "scene_sha256": digest, "track_profile": profile, "verified": True}
