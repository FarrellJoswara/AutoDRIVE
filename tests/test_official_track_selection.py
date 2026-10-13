import json
import numpy as np
import pytest
from src.layer3.official_tracks import official_frontier_path
from src.layer2.route_progress import RouteProgressTracker


def test_native_competition_and_docker_practice_select_different_routes():
    native = official_frontier_path(backend="windows_external", image="", profile="")
    practice = official_frontier_path(backend="docker", image="2026-iros-practice", profile="")
    compete = official_frontier_path(backend="docker", image="2026-iros-compete", profile="")
    assert native == compete and native != practice
    route = RouteProgressTracker.from_csv(native)
    assert route.closed
    assert 45 < route.length_m < 47
    # The competition straight continues where the mismatched Porto route
    # turns toward the wall. Verify against the scene-derived route.
    station = route.project_pose(0.8, -5.4, enforce_corridor=False)
    point, tangent = route.point_and_tangent_at(station)
    assert np.linalg.norm(point - [0.8, -5.4]) < 0.1
    assert tangent[1] < -0.99
    meta = json.loads(native.with_name("official_frontier.json").read_text())
    assert meta["verified_scene_active"]["SRL 2026 IROS Track"]
    assert not meta["verified_scene_active"]["Porto Track"]


def test_explicit_profile_and_invalid_profile():
    assert official_frontier_path(backend="windows_external",profile="practice").parent.name == "iros2026"
    with pytest.raises(ValueError,match="Unknown official track"):
        official_frontier_path(profile="typo")


def test_scene_guard_accepts_only_verified_matching_assets(tmp_path, monkeypatch):
    import hashlib
    from src.layer3 import official_tracks
    scene = tmp_path / "Data" / "level0"
    scene.parent.mkdir()
    scene.write_bytes(b"known competition test fixture")
    monkeypatch.setitem(official_tracks.WINDOWS_SCENES,"competition",hashlib.sha256(scene.read_bytes()).hexdigest())
    frontier = official_frontier_path(profile="competition")
    assert official_tracks.verify_windows_scene(frontier,tmp_path / "sim.exe")["verified"]
    scene.write_bytes(b"other scene")
    with pytest.raises(ValueError, match="does not match"):
        official_tracks.verify_windows_scene(frontier,tmp_path / "sim.exe")
