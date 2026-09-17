"""Headless smoke for the Watch overlay (race KPIs, banners, crash cues).

Runs without a display: OpenCV window calls are stubbed and frames are written
to ``rl/logs/watch_overlay/*.png`` so the overlay can be eyeballed after a run.

    python -m rl.test_watch_overlay          # run everything, print PASS/FAIL
    pytest "rl/test_watch_overlay.py"        # same checks as pytest cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

RL_ROOT = Path(__file__).resolve().parent
OUT_DIR = RL_ROOT / "logs" / "watch_overlay"
MAPS_ROOT = RL_ROOT / "maps"

_frames: list[np.ndarray] = []


def _stub_cv2_gui() -> None:
    """Make MapViewer usable with no display; keep imwrite/drawing real."""
    cv2.namedWindow = lambda *a, **k: None  # type: ignore[assignment]
    cv2.resizeWindow = lambda *a, **k: None  # type: ignore[assignment]
    cv2.destroyWindow = lambda *a, **k: None  # type: ignore[assignment]
    cv2.destroyAllWindows = lambda *a, **k: None  # type: ignore[assignment]
    cv2.getWindowProperty = lambda *a, **k: 1.0  # type: ignore[assignment]
    cv2.waitKey = lambda *a, **k: -1  # type: ignore[assignment]

    def _imshow(_window, frame):
        _frames.append(frame.copy())

    cv2.imshow = _imshow  # type: ignore[assignment]


_stub_cv2_gui()

from .racing_env import RacingEnv, resolve_map_yaml  # noqa: E402
from .viewer import MapViewer  # noqa: E402
from .watch_kpi import (  # noqa: E402
    FleetKpi,
    adjusted_or_dq,
    adjusted_time,
    beat_line,
    contracts_warning,
    map_ids_in_run_id,
    map_mismatch_warning,
    train_status_line,
)


def test_adjusted_time_matches_leaderboard_formula():
    assert adjusted_time(30.0, 2) == 50.0
    assert adjusted_time(None, 2) is None
    # No lap → timeout proxy, same rule the official protocol uses (protocol_timeout_s).
    from .eval_protocol import protocol_timeout_s

    assert adjusted_or_dq(None, 1) == float(protocol_timeout_s()) + 10.0


def test_fleet_kpi_tracks_laps_collisions_and_crash_tags():
    kpi = FleetKpi(3, label="PPO")
    kpi.start_episode(3)
    assert kpi.record({"collision": True, "crash_tag": "wall"}) == "wall"
    kpi.note_finished()
    kpi.record({"lap_time": 31.0})
    kpi.note_finished()
    assert kpi.record({"crash_tag": "stall", "stall": True}) == "stall"
    kpi.note_finished()

    assert kpi.ep.collisions == 1
    assert kpi.ep.lap_times == [31.0]
    assert kpi.ep.adjusted_time == 41.0
    assert kpi.ep.dq is False
    line = kpi.line()
    assert "adj 41.0" in line and "col 1" in line and "laps 1/3" in line

    kpi.end_episode()
    assert kpi.session.episodes == 1
    assert "wall 1" in kpi.session.crash_summary()
    assert "stall 1" in kpi.session.crash_summary()


def test_dq_shown_when_no_lap_completed():
    kpi = FleetKpi(2)
    kpi.start_episode(2)
    kpi.record({"collision": True, "crash_tag": "wall"})
    assert kpi.ep.dq is True
    assert kpi.ep.adjusted_time is None
    assert "DQ~" in kpi.line()


def test_session_window_rolls_so_scores_do_not_inflate():
    kpi = FleetKpi(1, window=2)
    for _ in range(3):
        kpi.start_episode(1)
        kpi.record({"collision": True, "crash_tag": "wall"})
        kpi.end_episode()
    assert kpi.session.episodes == 2, "window must drop the oldest episode"
    assert kpi.session.collisions == 2
    assert "last 2 ep" in kpi.session_line()


def test_crash_markers_merge_at_the_same_corner():
    from .watch import _push_marker

    markers: list[dict] = []
    _push_marker(markers, {"x": 1.0, "y": 1.0, "tag": "wall"})
    _push_marker(markers, {"x": 1.3, "y": 1.1, "tag": "wall"})  # same corner
    _push_marker(markers, {"x": 1.1, "y": 1.0, "tag": "stall"})  # other tag
    _push_marker(markers, {"x": 9.0, "y": 9.0, "tag": "wall"})  # far away
    _push_marker(markers, None)
    assert [(m["tag"], m["n"]) for m in markers] == [
        ("wall", 2),
        ("stall", 1),
        ("wall", 1),
    ]


def test_beat_line_points_at_the_faster_side():
    ppo, ftg = FleetKpi(1, "PPO"), FleetKpi(1, "FTG")
    ppo.start_episode(1)
    ftg.start_episode(1)
    ppo.record({"lap_time": 30.0})
    ftg.record({"lap_time": 35.0})
    assert "PPO ahead" in beat_line(ppo, ftg)
    ppo.record({"collision": True})  # +10 s penalty flips it
    assert "FTG ahead" in beat_line(ppo, ftg)


def test_kpi_lines_drop_ghost_rows_when_ghost_disabled():
    from .watch import _kpi_lines

    twins = FleetKpi(2, "PPO")
    twins.start_episode(2)
    lines = _kpi_lines(twins, None, {"timesteps": 100})
    assert len(lines) == 2 and lines[0].startswith("PPO")
    assert not any("FTG" in ln or "ghost" in ln for ln in lines)
    assert lines[-1].startswith("train")


def test_kpi_lines_compact_keeps_race_drops_session():
    """Compact overlay: live score + short train; drops session dump / dense train row."""
    from .watch import _kpi_lines

    twins = FleetKpi(2, "PPO")
    twins.start_episode(2)
    twins.record({"progress": 0.1, "collision": False})
    twins.end_episode()
    twins.start_episode(2)

    class _GhostStub:
        def __init__(self):
            self.kpi = FleetKpi(1, "FTG")
            self.kpi.start_episode(1)

        def line(self) -> str:
            return "FTG  ghost dump row"

    status = {
        "run_id": "overnight_soak_demo",
        "timesteps": 12000,
        "steps_per_sec": 200.0,
        "collisions_estimate": 3,
        "ep_rew_mean": 1.2,
        "phase": "learning",
        "crash_rate_estimate": 0.25,
    }
    full = _kpi_lines(twins, _GhostStub(), status, compact=False)
    compact = _kpi_lines(twins, _GhostStub(), status, compact=True)
    assert any(ln.startswith("PPO") for ln in compact)
    assert any("vs FTG" in ln or "ahead" in ln or "delta" in ln for ln in compact)
    assert len(compact) < len(full)
    assert not any("ghost dump" in ln for ln in compact)
    assert any(ln.startswith("train") and "ts 12000" in ln for ln in compact)
    assert any("overnight_soak_demo" in ln for ln in compact)
    assert any("learning" in ln for ln in compact)
    assert any("crash 25%" in ln for ln in compact)
    assert not any("rew" in ln for ln in compact)


def test_map_mismatch_and_contracts_banners():
    map_yaml = resolve_map_yaml("map0", MAPS_ROOT)
    same = {"maps": [str(map_yaml)]}
    assert map_mismatch_warning(map_yaml, same) is None

    other = {"maps": [str(map_yaml.parent.parent / "map7" / "map7.yaml")]}
    warn = map_mismatch_warning(map_yaml, other)
    assert warn and "MAP MISMATCH" in warn and "map7" in warn

    changed = {"maps": [str(map_yaml)], "map_hashes": {"map0": "deadbeefdeadbeef"}}
    warn = map_mismatch_warning(map_yaml, changed)
    assert warn and "MAP CHANGED" in warn

    # No config.json: fall back to map<N> tokens in the run id, suffixes ignored.
    assert map_ids_in_run_id("20260917_ppo_gym_map3_hard") == ["map3"]
    assert map_mismatch_warning(map_yaml, None, run_id="x_ppo_gym_map0_hard") is None
    warn = map_mismatch_warning(map_yaml, None, run_id="x_ppo_gym_map3_hard")
    assert warn and "map3" in warn
    assert map_mismatch_warning(map_yaml, None, run_id="x_ppo_gym_custom") is None

    assert contracts_warning({"contracts_version": "1.0.0"})
    assert contracts_warning({"contracts_version": "2.0.0"}) is None


def test_train_status_line_leads_with_race_counters():
    line = train_status_line(
        {
            "timesteps": 49152,
            "steps_per_sec": 812.4,
            "collisions_estimate": 34,
            "crash_rate_estimate": 0.42,
            "ep_rew_mean": 12.5,
        }
    )
    assert line is not None
    assert line.index("crash eps") < line.index("rew")
    assert train_status_line(None) is None


def test_validating_not_mislabeled_stale_and_keeps_honesty():
    """phase=validating must not become TRAIN STALE; lag-behind stays unofficial."""
    import time

    from .watch import _lag_sub_label, _status_stale_warning, _watching_label

    validating = {
        "phase": "validating",
        "timesteps": 545488,
        "unix_time": time.time() - 200.0,
        "msg": "racing validation map (map3) — timesteps paused (not stuck)",
        "run_id": "overnight_soak_20260917_082739",
    }
    assert _status_stale_warning(validating) is None
    label = _watching_label(validating)
    assert "validating" in label and "not hung" in label
    # Learning + same age still fires stale.
    learning = dict(validating, phase="learning")
    warn = _status_stale_warning(learning)
    assert warn and "TRAIN STALE" in warn
    sub = _lag_sub_label(None)
    assert "lag-behind" in sub and "unofficial" in sub


def _render_sample_frame() -> np.ndarray:
    """One synthetic frame with every overlay layer on at once."""
    env = RacingEnv(map_yaml=resolve_map_yaml("map0", MAPS_ROOT), seed=0)
    env.reset(seed=0)
    viewer = MapViewer(env, window="test")
    x, y = float(env._state[0]), float(env._state[1])
    agents = [
        {"x": x, "y": y, "theta": 0.3, "speed": 3.1, "scan": env._raw_scan},
        {"x": x + 1.0, "y": y + 0.6, "theta": 0.1, "speed": 0.0, "done": True},
        {
            "x": x + 2.0,
            "y": y - 0.6,
            "theta": -0.2,
            "speed": 2.4,
            "ghost": True,
            "label": "FTG ",
            "color": (235, 235, 235),
        },
    ]
    markers = [
        {"x": x + 1.2, "y": y + 1.4, "tag": "wall", "n": 3},
        {"x": x - 1.2, "y": y - 1.4, "tag": "stall", "n": 1},
        {"x": x + 2.6, "y": y + 1.0, "tag": "ttc", "n": 1},
    ]
    metrics = {
        "watch_label": "watching iter 12 | ts=49152",
        "sub_label": "lag-behind replay of latest_model.zip (8s old) - unofficial KPIs (not the training cars)",
        "warn": "MAP MISMATCH: watching 'map0' but run trained on map3",
        "kpi_lines": [
            "PPO  adj 41.3 | lap 31.3 | col 1 | laps 3/8",
            "FTG  last 2 ep: adj 38.9 | col 0 | crash none",
            "delta +2.4s vs FTG ghost -> FTG ahead",
            "train ts 49152 | 812 steps/s | crash eps 34 | rate 42%",
        ],
    }
    frame = viewer.render_frame(metrics, agents=agents, markers=markers)
    viewer.close()
    return frame


def test_overlay_frame_renders_all_layers():
    frame = _render_sample_frame()
    assert frame.ndim == 3 and frame.shape[2] == 3
    # Red banner band at the top and light text in the KPI block.
    band = frame[0:26, :]
    assert int(band[:, :, 2].max()) > 200, "warn banner should be red"
    kpi_band = frame[frame.shape[0] - 90 : frame.shape[0] - 20, :]
    assert int(kpi_band.max()) > 200, "KPI text should be drawn bottom-left"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "overlay_layers.png"), frame)


def test_standalone_watch_episode_runs_and_draws():
    """End-to-end: real env + FTG twins + ghost through rl.watch main()."""
    from . import watch

    _frames.clear()
    code = watch.main(
        [
            "--policy",
            "ftg",
            "--map",
            "map0",
            "--n-envs",
            "3",
            "--episodes",
            "1",
            "--every",
            "8",
            # Short budget for smoke; production Watch defaults to protocol (400 s).
            "--timeout-s",
            "20",
        ]
    )
    assert code == 0
    assert _frames, "viewer produced no frames"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "standalone_last.png"), _frames[-1])


def _find_v2_run() -> str | None:
    """Newest run with contracts-v2 config + saved weights (skip marker if none)."""
    import json

    best: tuple[float, str] | None = None
    for cfg_path in (RL_ROOT / "models").glob("*/config.json"):
        if not (cfg_path.parent / "latest_model.zip").is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(cfg.get("contracts_version")) != "2.0.0":
            continue
        mtime = cfg_path.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, cfg_path.parent.name)
    return best[1] if best else None


def test_follow_mode_runs_with_ghost_and_map_mismatch_banner():
    """End-to-end --follow: PPO twins + FTG ghost + wrong map on purpose."""
    from . import watch

    run_id = _find_v2_run()
    if run_id is None:
        print("  (skip: no contracts-v2 run with latest_model.zip on disk)")
        return

    _frames.clear()
    # map3 vs a map0-trained run: the banner must fire.
    code = watch.main(
        [
            "--follow",
            "--run_id",
            run_id,
            "--map",
            "map3",
            "--n-envs",
            "2",
            "--episodes",
            "1",
            "--every",
            "10",
            "--timeout-s",
            "20",
        ]
    )
    assert code == 0
    assert _frames, "follow mode produced no frames"
    frame = _frames[-1]
    assert int(frame[0:26, :, 2].max()) > 200, "map mismatch banner should be red"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "follow_last.png"), frame)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - smoke script wants the message
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed; frames in {OUT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
