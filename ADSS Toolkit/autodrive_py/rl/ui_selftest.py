"""Offline checks for the control UI (helpers + HTTP surface). No training, no browser.

Run from ADSS Toolkit/autodrive_py:
  python -m rl.ui_selftest

Refusal paths only: nothing here starts train_ppo or writes into rl/maps.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path

from . import control_ui as cui
from . import ui_ops
from .contracts import CONTRACTS_VERSION, N_LIDAR_DEFAULT, obs_dim

RL_DIR = Path(__file__).resolve().parent
_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def _fake_run(root: Path, run_id: str, *, contracts: str = CONTRACTS_VERSION) -> Path:
    """Minimal models/<run_id>/ with a plausible zip + config."""
    run_dir = root / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    blob = b"PK\x03\x04" + b"0" * 4096
    (run_dir / "best_model.zip").write_bytes(blob)
    (run_dir / "latest_model.zip").write_bytes(blob)
    (run_dir / "checkpoints" / "ppo_10000_steps.zip").write_bytes(blob)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "contracts_version": contracts,
                "obs_dim": obs_dim(N_LIDAR_DEFAULT),
                "n_lidar": N_LIDAR_DEFAULT,
                "map": str(RL_DIR / "maps" / "map1" / "map1.yaml"),
                "maps": [str(RL_DIR / "maps" / "map1" / "map1.yaml")],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_map_and_holdout_labels() -> None:
    print("map / holdout labels")
    maps_dir = RL_DIR / "maps"
    labels = {m["id"]: m for m in ui_ops.map_labels(maps_dir)}
    check("map_labels sees real maps", "map0" in labels, f"ids={sorted(labels)}")
    check("map2 tagged sealed", bool(labels.get("map2", {}).get("sealed")))
    check("map2 label shows HOLDOUT", "[HOLDOUT]" in str(labels.get("map2", {}).get("label")))
    check("map0 not sealed", not labels.get("map0", {}).get("sealed"))

    # Seals the map pack knows about but eval_protocol.yaml does not must still bind.
    sealed_ids = [m["id"] for m in labels.values() if m["sealed"]]
    pinned = [m["id"] for m in labels.values() if not m["sealed"] and not m["train_safe"]]
    check("every sealed map is labelled", all("[HOLDOUT]" in labels[i]["label"] for i in sealed_ids),
          f"sealed={sealed_ids}")
    check("validation pin labelled and not train-safe",
          all("[VALIDATION]" in labels[i]["label"] for i in pinned), f"pinned={pinned}")

    ok, msg = ui_ops.start_guard("map0", maps_dir=maps_dir)
    check("start_guard allows train map", ok and not msg)
    for mid in sealed_ids:
        ok, msg = ui_ops.start_guard(mid, maps_dir=maps_dir)
        check(f"start_guard refuses sealed {mid}", (not ok) and "Refuse Start" in msg, msg)
    for mid in pinned:
        ok, msg = ui_ops.start_guard(mid, maps_dir=maps_dir)
        check(f"start_guard refuses validation pin {mid}", (not ok) and "Refuse Start" in msg, msg)
    ok, msg = ui_ops.start_guard("map2", allow_holdout=True, maps_dir=maps_dir)
    check("start_guard allows sealed with override + warning", ok and "WARNING" in msg)


def test_map_generation() -> None:
    print("map generation")
    with tempfile.TemporaryDirectory() as tmp:
        maps = Path(tmp)
        r = ui_ops.generate_maps_op(maps, count=0, seed=1)
        check("refuses count=0", not r["ok"], r["msg"])
        r = ui_ops.generate_maps_op(maps, count=2, seed=1, prefix="map")
        check("refuses protocol prefix 'map'", not r["ok"], r["msg"])

        calls: dict = {}

        def fake_gen(out_root, num_maps=1, seed=0, prefix="map"):
            calls.update(out_root=out_root, num_maps=num_maps, seed=seed, prefix=prefix)
            written = []
            for i in range(num_maps):
                d = Path(out_root) / f"{prefix}{i}"
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{prefix}{i}.yaml"
                p.write_text("image: x.pgm\n", encoding="utf-8")
                written.append(p)
            return written

        r = ui_ops.generate_maps_op(maps, count=2, seed=42, generator=fake_gen)
        check("generates with unique prefix", r["ok"] and r["created"] == ["gen42_0", "gen42_1"], r["msg"])
        check("passes seed through to generator", calls.get("seed") == 42 and calls.get("prefix") == "gen42_")
        (maps / "gen7_0").mkdir()
        r = ui_ops.generate_maps_op(maps, count=1, seed=7, generator=fake_gen)
        check("injected generator refuses to overwrite", not r["ok"], r["msg"])

    # Real map-pack generation into a throwaway root: registers role + hashes, caches on repeat.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        r = ui_ops.generate_maps_op(root, count=1, seed=2024)
        mid = r["created"][0] if r["created"] else ""
        check("map pack generate writes a map set", r["ok"] and (root / mid / f"{mid}.yaml").is_file(), r["msg"])

        manifest = json.loads((root / "map_pack.json").read_text(encoding="utf-8"))
        entry = manifest["maps"].get(mid, {})
        check("generated map registered in manifest", entry.get("role") == "train_ok" and bool(entry.get("map_hash")),
              str(entry.get("role")))
        check("generated map is labelled train-safe",
              ui_ops.map_role(mid, root)["train_safe"], str(ui_ops.map_role(mid, root)))

        again = ui_ops.generate_maps_op(root, count=1, seed=2024)
        check("regenerating same seed reuses cached maps",
              (not again["ok"]) and mid in again.get("cached", []), again["msg"])


def test_models_ops() -> None:
    print("model load / delete / timeline")
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp)
        _fake_run(models, "run_good")
        _fake_run(models, "run_v1", contracts="1.0.0")

        rows = {r["run_id"]: r for r in ui_ops.list_run_models(models)}
        check("lists runs with artifacts", rows["run_good"]["has_best"] and rows["run_good"]["checkpoint"])

        timeline = ui_ops.model_timeline(models, "run_good")
        kinds = {e["kind"] for e in timeline}
        check("timeline has best/latest/checkpoint", kinds == {"best", "latest", "checkpoint"}, str(kinds))

        path, why = ui_ops.resolve_model_choice(models, "run_good", "best")
        check("resolve best_model", path is not None and path.name == "best_model.zip", why)
        path, why = ui_ops.resolve_model_choice(models, "run_good", "checkpoint")
        check("resolve last checkpoint", path is not None and "10000" in path.name, why)
        path, why = ui_ops.resolve_model_choice(models, "../escape", "best")
        check("resolve refuses traversal", path is None, why)

        good = models / "run_good" / "best_model.zip"
        ok, msg = ui_ops.precheck_model(good)
        check("precheck accepts current contracts", ok, msg)
        ok, msg = ui_ops.precheck_model(models / "run_v1" / "best_model.zip")
        check("precheck refuses v1 contracts", (not ok) and "Refuse load" in msg, msg)
        stub = models / "run_good" / "partial.zip"
        stub.write_bytes(b"x")
        ok, msg = ui_ops.precheck_model(stub)
        check("precheck refuses partial zip", not ok, msg)

        check("run_map_ids recovers train map", ui_ops.run_map_ids(models, "run_good") == ["map1"])
        check(
            "run_map_ids infers map from run_id when config missing",
            ui_ops.run_map_ids(models, "20260917_014513_ppo_gym_gen231_0_hard") == ["gen231_0"],
        )

        check(
            "delete refuses traversal",
            "Refuse delete" in ui_ops.delete_run_model(models, "../boom"),
        )
        check(
            "delete refuses the live run",
            "Refuse delete" in ui_ops.delete_run_model(models, "run_good", active_run_id="run_good"),
        )
        lock = models / "run_good" / "train.lock"
        lock.write_text(json.dumps({"pid": 999999999, "timestamp": "now"}), encoding="utf-8")
        owner = ui_ops.lock_owner(models, "run_good")
        check("lock_owner reports stale lock", bool(owner and owner["stale"]), str(owner))
        check("clear_stale_lock removes a dead lock", bool(ui_ops.clear_stale_lock(models, "run_good")))
        check("stale lock file gone", not lock.is_file())

        # Our own pid is genuinely alive: that lock must survive.
        lock.write_text(json.dumps({"pid": os.getpid(), "timestamp": "now"}), encoding="utf-8")
        owner = ui_ops.lock_owner(models, "run_good")
        check("lock_owner reports a live lock", bool(owner and owner["alive"]), str(owner))
        check("clear_stale_lock spares a live lock", ui_ops.clear_stale_lock(models, "run_good") is None)
        check("delete refuses a live lock", "Refuse delete" in ui_ops.delete_run_model(models, "run_good"))
        lock.unlink()

        check("delete removes an unlocked run", "Deleted" in ui_ops.delete_run_model(models, "run_good"))
        check("deleted dir is gone", not (models / "run_good").is_dir())


def test_continue_argv() -> None:
    print("continue-train argv")
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp)
        (models / "empty_run").mkdir()
        argv, why = ui_ops.continue_train_argv(
            run_id="empty_run", models_dir=models, timesteps=1000, n_envs=1
        )
        check("refuses run without checkpoint", argv == [] and "No complete checkpoint" in why, why)

        _fake_run(models, "run_good")
        argv, why = ui_ops.continue_train_argv(
            run_id="run_good", models_dir=models, timesteps=5000, n_envs=4, map_id="map1"
        )
        pairs = dict(zip(argv[::2], argv[1::2]))
        check("resumes same run_id", pairs.get("--resume") == "run_good" and pairs.get("--run_id") == "run_good")
        check("keeps the run's own map", pairs.get("--map") == "map1", str(argv))
        check("vec matches n_envs", pairs.get("--vec-env") == "subproc" and pairs.get("--n-envs") == "4")
        argv, _ = ui_ops.continue_train_argv(
            run_id="run_good", models_dir=models, timesteps=5000, n_envs=1
        )
        check("single env resumes on dummy vec", "dummy" in argv)

        argv, why = ui_ops.continue_train_argv(
            run_id="run_good",
            models_dir=models,
            timesteps=5000,
            n_envs=1,
            stop_on_budget=False,
            early_stop_patience=0,
        )
        check("continue refuses budget-off + patience 0", argv == [] and "patience is 0" in why, why)

        argv, why = ui_ops.continue_train_argv(
            run_id="run_good",
            models_dir=models,
            timesteps=5000,
            n_envs=1,
            stop_on_budget=False,
            early_stop_patience=3,
            early_stop_min_improve=1.0,
            race_eval_every=25_000,
        )
        check("continue unlimited passes flag", "--unlimited-timesteps" in argv, str(argv))
        check("continue unlimited sets early-stop", "--early-stop-patience" in argv and "3" in argv, str(argv))
        check("continue passes min-improve", "--early-stop-min-improve" in argv and "1" in argv, str(argv))
        check("continue passes race-eval-every", "--race-eval-every" in argv and "25000" in argv, str(argv))
        check("continue unlimited uses safety ceiling", str(ui_ops.UNLIMITED_TIMESTEPS_SAFETY) in argv, str(argv))

        cfg = models / "run_good" / "config.json"
        cfg.write_text(
            json.dumps({"collision_first": True, "speed_gate": True, "maps": ["map0"]}),
            encoding="utf-8",
        )
        argv, why = ui_ops.continue_train_argv(
            run_id="run_good", models_dir=models, timesteps=1000, n_envs=1
        )
        check("continue inherits collision-first from config", "--collision-first" in argv, why)
        check("continue inherits speed-gate from config", "--speed-gate" in argv, why)
        flags = ui_ops.run_curriculum_flags(models, "run_good")
        check("run_curriculum_flags reads config", flags["collision_first"] and flags["speed_gate"], str(flags))

        _fake_run(models, "run_cf")
        cfg = json.loads((models / "run_cf" / "config.json").read_text(encoding="utf-8"))
        cfg["collision_first"] = True
        cfg["speed_gate"] = True
        (models / "run_cf" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        flags = ui_ops.run_curriculum_flags(models, "run_cf")
        check("curriculum flags read collision_first", flags["collision_first"] and flags["speed_gate"], str(flags))
        argv, why = ui_ops.continue_train_argv(
            run_id="run_cf", models_dir=models, timesteps=1000, n_envs=1
        )
        check("continue restores --collision-first from config", "--collision-first" in argv, str(argv))
        check("continue restores --speed-gate from config", "--speed-gate" in argv, str(argv))
        check("continue msg mentions collision_first", "collision_first=True" in why, why)


def test_seal_verify() -> None:
    print("seal verify summary")
    report = ui_ops.seal_verify_summary(RL_DIR / "maps")
    check("seal_verify returns ok flag", "ok" in report and "msg" in report, str(report.get("msg")))
    check("seal_verify msg is operator-readable", "SEAL" in report["msg"] or "PACK" in report["msg"], report["msg"])
    # Real pack should have sealed holdouts; prefer SEALS OK when maps intact.
    if report.get("ok"):
        check("intact pack reports SEALS OK", "SEALS OK" in report["msg"], report["msg"])
    msg = cui._verify_seals()
    check("UI verify_seals sets msg", "SEAL" in msg or "PACK" in msg, msg)


def test_stop_budget() -> None:
    print("stop-on-budget resolve")
    on = ui_ops.resolve_stop_budget(stop_on_budget=True, timesteps=100_000, early_stop_patience=0)
    check("budget on keeps timesteps", on["ok"] and on["effective_timesteps"] == 100_000 and not on["unlimited"])
    check("budget on label", on["budget_label"] == "budget: 100,000", on["budget_label"])

    bad = ui_ops.resolve_stop_budget(stop_on_budget=False, timesteps=100_000, early_stop_patience=0)
    check("budget off + patience 0 refused", not bad["ok"] and "patience is 0" in (bad["error"] or ""))

    off = ui_ops.resolve_stop_budget(stop_on_budget=False, timesteps=100_000, early_stop_patience=3)
    check(
        "budget off uses safety ceiling",
        off["ok"]
        and off["unlimited"]
        and off["effective_timesteps"] == ui_ops.UNLIMITED_TIMESTEPS_SAFETY,
        str(off),
    )
    check("budget off label", off["budget_label"] == "budget: off (early-stop only)", off["budget_label"])

    # Mock busy so overnight train_ppo does not mask the budget refuse path.
    old_busy = cui._train_busy
    try:
        cui._train_busy = lambda force=True: None  # type: ignore[assignment]
        msg = cui._start_train("map0", 2048, 1, stop_on_budget=False, early_stop_patience=0)
        check("Start refuses budget-off + patience 0", "Refuse" in msg and "patience is 0" in msg, msg)
    finally:
        cui._train_busy = old_busy


def test_dual_writer_guards() -> None:
    """Start/Continue must refuse when a trainer (or lock) already owns the run."""
    print("dual-writer / busy refuse")

    class _AliveProc:
        pid = 424242

        def poll(self):
            return None

    old_proc, old_rid = cui._train_proc, cui._train_run_id
    kills: list[str] = []
    old_kill = cui._kill_train_tree
    try:
        cui._kill_train_tree = lambda: kills.append("kill") or 0  # type: ignore[assignment]
        cui._train_proc = _AliveProc()  # type: ignore[assignment]
        cui._train_run_id = "owned_run"
        busy = cui._train_busy(force=False)
        check("busy when UI owns a live train", bool(busy and "already running" in busy), str(busy))
        msg = cui._start_train("map0", 2048, 1)
        check("Start refuses while UI train alive", "already running" in msg, msg)
        check("refused Start left owned proc alone", cui._train_proc is not None and cui._train_proc.poll() is None)
        check("refused Start did not kill", kills == [], str(kills))
    finally:
        cui._train_proc = old_proc
        cui._train_run_id = old_rid
        cui._kill_train_tree = old_kill

    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp)
        _fake_run(models, "locked_run")
        lock = models / "locked_run" / "train.lock"
        lock.write_text(json.dumps({"pid": os.getpid(), "timestamp": "now"}), encoding="utf-8")
        old_models = cui.MODELS_DIR
        kills = []
        old_kill = cui._kill_train_tree
        old_busy = cui._train_busy
        old_ext = cui._find_external_train
        try:
            cui.MODELS_DIR = models
            cui._kill_train_tree = lambda: kills.append("kill") or 0  # type: ignore[assignment]
            # Isolate from any live overnight/external train_ppo on this machine.
            cui._find_external_train = lambda force=False: None  # type: ignore[assignment]
            cui._train_busy = lambda force=True: None  # type: ignore[assignment]
            msg = cui._continue_train("locked_run", 2048, 1)
            check("Continue refuses live train.lock", "Refuse Continue" in msg and "locked" in msg.lower(), msg)
            check("Continue on healthy lock did not kill", kills == [], str(kills))

            # Busy false-negative + live lock: Start must refuse via lock scan, never kill.
            cui._train_busy = old_busy
            cui._train_proc = None
            busy = cui._train_busy(force=True)
            check(
                "busy via live lock when PID scan empty",
                bool(busy and "lock held" in busy.lower()),
                str(busy),
            )
            kills.clear()
            msg = cui._start_train("map0", 2048, 1, stop_on_budget=True)
            check("Start refuses live lock without PID", "lock held" in msg.lower() or "already running" in msg.lower(), msg)
            check("Start live-lock refuse did not kill", kills == [], str(kills))
        finally:
            cui.MODELS_DIR = old_models
            cui._kill_train_tree = old_kill
            cui._train_busy = old_busy
            cui._find_external_train = old_ext
            cui._train_proc = old_proc
            cui._train_run_id = old_rid


def test_start_continue_never_kill_on_false_busy() -> None:
    """Start/Continue must never call _kill_train_tree even if busy looks idle."""
    print("Start/Continue no-kill (busy false-negative)")
    kills: list[str] = []
    spawns: list[str] = []
    old_kill = cui._kill_train_tree
    old_spawn = cui._spawn_train
    old_busy = cui._train_busy
    old_ext = cui._find_external_train
    old_models = cui.MODELS_DIR
    old_proc, old_rid = cui._train_proc, cui._train_run_id
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp)
        try:
            cui.MODELS_DIR = models  # empty — no live locks
            cui._train_proc = None
            cui._train_run_id = None
            cui._kill_train_tree = lambda: kills.append("kill") or 0  # type: ignore[assignment]
            cui._spawn_train = (  # type: ignore[assignment]
                lambda argv, run_id, n_envs=1, action="x", log_mode="w": (
                    spawns.append(run_id) or f"mock-spawn {run_id}"
                )
            )
            cui._train_busy = lambda force=True: None  # type: ignore[assignment]
            cui._find_external_train = lambda force=False: None  # type: ignore[assignment]

            msg = cui._start_train("map0", 2048, 1, stop_on_budget=True, early_stop_patience=0)
            check("Start false-busy did not kill", kills == [], f"kills={kills} msg={msg}")
            check("Start false-busy reached spawn (no kill path)", bool(spawns), msg)

            kills.clear()
            spawns.clear()
            _fake_run(models, "cont_run")
            # Continue with no live lock, busy false-negative — still no kill.
            msg = cui._continue_train("cont_run", 2048, 1, stop_on_budget=True, early_stop_patience=0)
            check("Continue false-busy did not kill", kills == [], f"kills={kills} msg={msg}")
        finally:
            cui._kill_train_tree = old_kill
            cui._spawn_train = old_spawn
            cui._train_busy = old_busy
            cui._find_external_train = old_ext
            cui.MODELS_DIR = old_models
            cui._train_proc = old_proc
            cui._train_run_id = old_rid


def test_heartbeat_cannot_clobber_early_stopped() -> None:
    """RaceBest heartbeat must join before early_stopped and never overwrite it."""
    print("validation heartbeat vs early_stopped")
    import time

    from .live_status import is_terminal_phase, read_live_status
    from .train_ppo import RaceBestModelCallback

    check("early_stopped is terminal", is_terminal_phase("early_stopped"))
    check("validating is not terminal", not is_terminal_phase("validating"))

    with tempfile.TemporaryDirectory() as tmp:
        status = Path(tmp) / "live_status.json"
        cb = RaceBestModelCallback(
            run_dir=Path(tmp),
            eval_maps=[],
            n_lidar=64,
            eval_episodes=1,
            eval_freq=1000,
            seed=0,
            status_path=status,
            status_run_id="hb_test",
        )
        # BaseCallback.num_timesteps may need a nudge before status writes.
        try:
            cb.num_timesteps = 50_000
        except Exception:
            pass

        cb._start_heartbeat("racing validation map (map3)")
        check("heartbeat thread started", cb._hb_thread is not None and cb._hb_thread.is_alive())

        # Production order: join heartbeat, then mark terminal, then write.
        cb._stop_heartbeat()
        check(
            "heartbeat joined (thread dead)",
            cb._hb_thread is None or not cb._hb_thread.is_alive(),
        )
        cb.early_stopped = True
        cb._write_status(phase="early_stopped", msg="early-stop: test")
        data = read_live_status(status)
        check("phase is early_stopped after join+write", bool(data and data.get("phase") == "early_stopped"), str(data))

        # Late pulse / mistaken validating write must no-op.
        cb._write_status(phase="validating", msg="late pulse must not win")
        data = read_live_status(status)
        check("validating cannot clobber early_stopped", bool(data and data.get("phase") == "early_stopped"), str(data))

        # Even if someone restarts heartbeat after terminal, pulses must no-op.
        cb._start_heartbeat("should not start or should no-op")
        check("no heartbeat after early_stopped", cb._hb_thread is None or not cb._hb_thread.is_alive())
        time.sleep(0.2)
        data = read_live_status(status)
        check("still early_stopped after blocked hb", bool(data and data.get("phase") == "early_stopped"), str(data))


def test_banner_and_coach() -> None:
    print("status banner / coach")
    b = ui_ops.classify_banner(train_alive=False, timesteps=None, prev_timesteps=None, status_age_s=None)
    check("idle when no trainer", b["state"] == "Idle", b["reason"])
    b = ui_ops.classify_banner(train_alive=True, timesteps=1000, prev_timesteps=900, status_age_s=2.0)
    check("learning while stepping", b["state"] == "Learning")
    b = ui_ops.classify_banner(train_alive=True, timesteps=1000, prev_timesteps=1000, status_age_s=120.0)
    check("stale when trail freezes", b["state"] == "Stale", b["reason"])
    b = ui_ops.classify_banner(
        train_alive=False, timesteps=1000, prev_timesteps=1000, status_age_s=1.0, exit_code=2
    )
    check("crashed on non-zero exit", b["state"] == "Crashed", b["reason"])
    b = ui_ops.classify_banner(
        train_alive=True, timesteps=10, prev_timesteps=5, status_age_s=1.0, stopping=True
    )
    check("stopping wins over learning", b["state"] == "Stopping")
    b = ui_ops.classify_banner(
        train_alive=True, timesteps=10, prev_timesteps=5, status_age_s=1.0, saving=True
    )
    check("saving while writing checkpoint", b["state"] == "Saving")
    b = ui_ops.classify_banner(
        train_alive=True, timesteps=10, prev_timesteps=5, status_age_s=1.0, saving=True, stopping=True
    )
    check("stopping wins over saving", b["state"] == "Stopping")
    b = ui_ops.classify_banner(
        train_alive=False,
        timesteps=None,
        prev_timesteps=None,
        status_age_s=0.5,
        exit_code=None,
        phase="starting",
    )
    check("starting phase not crashed", b["state"] == "Idle", b["reason"])
    b = ui_ops.classify_banner(
        train_alive=True,
        timesteps=1000,
        prev_timesteps=1000,
        status_age_s=120.0,
        phase="validating",
        msg="racing validation map...",
    )
    check("validating not stale", b["state"] == "Validating", b["reason"])
    check("validating shows msg", "racing validation" in b["reason"], b["reason"])
    b = ui_ops.classify_banner(
        train_alive=False,
        timesteps=70000,
        prev_timesteps=70000,
        status_age_s=2.0,
        exit_code=0,
        phase="early_stopped",
        msg="early-stop: no meaningful improvement",
    )
    check("early-stop banner", b["state"] == "EarlyStop", b["reason"])
    check("early-stop not crashed", "early-stop" in b["reason"].lower(), b["reason"])

    hints = ui_ops.coach_hints({"crash_rate_estimate": 0.9, "steps_per_sec": 12}, watch_opened=False, n_envs=16)
    check("coach flags crash rate", any("Crash rate high" in h for h in hints), str(hints))
    check("coach flags slow steps/sec", any("steps/sec low" in h for h in hints))
    hints = ui_ops.coach_hints({"crash_rate_estimate": 0.0}, watch_opened=True, n_envs=4)
    check("coach stays quiet when healthy", any("looking healthy" in h for h in hints), str(hints))


def _http(server_url: str, path: str, data: dict | None = None) -> dict:
    url = server_url + path
    if data is None:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def test_http_surface() -> None:
    print("HTTP surface (local server, refusal paths only)")
    server = cui.QuietThreadingHTTPServer(("127.0.0.1", 0), cui.Handler)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(base + "/", timeout=20) as r:
            html = r.read().decode("utf-8")
        check("index serves panel", r.status == 200 and "Start preview" in html and "Coach / health" in html)
        check("index has budget stop toggle", "stop_on_budget" in html and "Stop on timesteps budget" in html)
        check("index has min improvement control", 'id="min_improve"' in html and "Min improvement" in html)
        check("index has race-eval-every control", 'id="race_eval_every"' in html)
        check("index has collision-first checkbox", 'id="collision_first"' in html and "Collision-first" in html)
        check("index has speed-gate checkbox", 'id="speed_gate"' in html and "Speed-gate" in html)
        check(
            "index has fast-probe checkbox",
            'id="fast_probe"' in html and "NON-OFFICIAL" in html and ("never promotes" in html.lower() or "does not promote" in html.lower()),
        )
        check("index has verify seals button", "verify_seals" in html and "Verify pack seals" in html)
        check(
            "index has Auto-train experimental stub",
            'id="auto_train_btn"' in html
            and "EXPERIMENTAL" in html
            and "disabled" in html[html.find("auto_train_btn") : html.find("auto_train_btn") + 120],
        )
        check(
            "index Auto-train warn mentions overnight/CLI",
            'id="auto_train_warn"' in html
            and "auto_train" in html
            and ("overnight" in html.lower() or "chaos" in html.lower()),
        )
        check("index has soft-stop Continue hint", "Soft-stop" in html or "complete" in html.lower())
        check(
            "index build stamp bumped",
            'id="ui_build"' in html and ("w4-auto-train" in html or "w3-ux-soft" in html or "20260917" in html),
            html[html.find("ui build") : html.find("ui build") + 90] if "ui build" in html else "",
        )
        check(
            "index has glossary drawer",
            'id="glossary"' in html
            and "timesteps" in html
            and "adjusted_time" in html
            and "holdout" in html.lower()
            and "Validating" in html,
        )

        st = _http(base, "/api/status")
        for key in ("banner", "coach", "presets", "steps_per_sec", "crash_rate_estimate", "vec_env_active"):
            check(f"status exposes {key}", key in st)
        check("status banner has a state", bool(st["banner"].get("state")), str(st["banner"]))
        check("status maps carry seal labels", any(m.get("sealed") for m in st["maps"]),
              f"labels={[m.get('label') for m in st['maps']]}")
        check("presets include debug/quick/overnight", set(st["presets"]) == {"debug", "quick", "overnight"})

        pv = _http(base, "/api/preview?map=map0&timesteps=50000&n_envs=4")
        check("preview reports vec + run path", pv["vec_env"] == "subproc" and pv["timesteps"] == 50000, str(pv))
        check("preview budget on by default", pv.get("budget_label") == "budget: 50,000" and pv.get("stop_on_budget"), str(pv))
        check("preview clean on train map", not pv["holdout_warn"] and pv["map_role"] == "train_ok")
        pv = _http(base, "/api/preview?map=map0&timesteps=50000&n_envs=4&stop_on_budget=0&early_stop=3&min_improve=1&race_eval_every=20000")
        check(
            "preview budget off label",
            pv.get("budget_label") == "budget: off (early-stop only)" and pv.get("unlimited"),
            str(pv),
        )
        check("preview reports min_improve", pv.get("early_stop_min_improve") == 1.0, str(pv))
        check("preview reports race_eval_every", pv.get("race_eval_every") == 20000, str(pv))
        pv = _http(base, "/api/preview?map=map0&timesteps=50000&n_envs=1&stop_on_budget=0&early_stop=0")
        check("preview flags budget+patience refuse", bool(pv.get("budget_error")), str(pv.get("budget_error")))
        pv = _http(base, "/api/preview?map=map2&timesteps=1000&n_envs=1")
        check("preview flags holdout map", pv["holdout_warn"] and pv["vec_env"] == "dummy")
        pv = _http(base, "/api/preview?map=map3&timesteps=1000&n_envs=1")
        check("preview flags validation pin", pv["holdout_warn"] and "VALIDATION" in str(pv.get("map_label", "")).upper(),
              str(pv.get("map_label")))
        check("status maps expose train_safe", all("train_safe" in m for m in st["maps"]))

        models = _http(base, "/api/models")
        check("models endpoint returns runs", isinstance(models.get("runs"), list), str(type(models.get("runs"))))
        check("models endpoint reports race candidate slot", "race_candidate" in models)

        pv = _http(base, "/api/preview?map=map0&timesteps=50000&n_envs=4&collision_first=1")
        check("preview reports collision_first", pv.get("collision_first") is True, str(pv))
        pv = _http(base, "/api/preview?map=map0&timesteps=50000&n_envs=4&speed_gate=1&fast_probe=1")
        check("preview reports speed_gate", pv.get("speed_gate") is True, str(pv))
        check("preview reports fast_probe", pv.get("fast_probe") is True, str(pv))

        r = _http(base, "/api/action", {"op": "verify_seals"})
        check("verify_seals op returns seal msg", "SEAL" in r["msg"] or "PACK" in r["msg"], r["msg"])

        # Isolate HTTP Start refuse paths from any parallel overnight / A/B train.
        old_busy = cui._train_busy
        old_ext = cui._find_external_train
        old_proc = cui._train_proc
        old_rid = cui._train_run_id
        try:
            cui._train_busy = lambda force=True: None  # type: ignore[assignment]
            cui._find_external_train = lambda force=False: None  # type: ignore[assignment]
            cui._train_proc = None
            cui._train_run_id = None
            r = _http(base, "/api/action", {"op": "start", "map": "map2", "timesteps": "2048", "n_envs": "1"})
            check("Start refuses sealed holdout", "Refuse Start" in r["msg"], r["msg"])
            check(
                "refused Start spawned nothing",
                cui._train_proc is None,
                f"proc={cui._train_proc} msg={r.get('msg')}",
            )

            r = _http(
                base,
                "/api/action",
                {
                    "op": "start",
                    "map": "map0",
                    "timesteps": "2048",
                    "n_envs": "1",
                    "stop_on_budget": "0",
                    "early_stop": "0",
                },
            )
            check("Start refuses budget-off + patience 0", "patience is 0" in r["msg"], r["msg"])
            check(
                "budget refuse spawned nothing",
                cui._train_proc is None,
                f"proc={cui._train_proc} msg={r.get('msg')}",
            )

            r = _http(base, "/api/action", {"op": "start", "map": "map3", "timesteps": "2048", "n_envs": "1"})
            check("Start refuses validation pin", "Refuse Start" in r["msg"], r["msg"])
        finally:
            cui._train_busy = old_busy
            cui._find_external_train = old_ext
            cui._train_proc = old_proc
            cui._train_run_id = old_rid

        class _AliveProc:
            pid = 777001

            def poll(self):
                return None

        old_proc, old_rid = cui._train_proc, cui._train_run_id
        try:
            cui._train_proc = _AliveProc()  # type: ignore[assignment]
            cui._train_run_id = "http_owned"
            r = _http(base, "/api/action", {"op": "start", "map": "map0", "timesteps": "2048", "n_envs": "1"})
            check("HTTP Start refuses dual writer", "already running" in r["msg"], r["msg"])
            check("dual-writer Start did not clear owned proc", cui._train_proc is not None)
        finally:
            cui._train_proc = old_proc
            cui._train_run_id = old_rid

        old_busy2 = cui._train_busy
        old_ext2 = cui._find_external_train
        old_kill2 = cui._kill_train_tree
        try:
            cui._train_busy = lambda force=True: None  # type: ignore[assignment]
            cui._find_external_train = lambda force=False: None  # type: ignore[assignment]
            cui._kill_train_tree = lambda: 0  # type: ignore[assignment]

            r = _http(base, "/api/action", {"op": "continue", "run_id": "__no_such_run__", "timesteps": "2048"})
            check(
                "Continue refuses unknown run",
                "Continue refused" in r["msg"] or "no run" in r["msg"].lower() or "No complete" in r["msg"],
                r["msg"],
            )

            r = _http(base, "/api/action", {"op": "delete_model", "run_id": "../evil"})
            check("Delete refuses traversal", "Refuse delete" in r["msg"], r["msg"])

            r = _http(base, "/api/action", {"op": "load_model", "run_id": "__no_such_run__", "which": "best"})
            check("Load refuses unknown run", "Load refused" in r["msg"], r["msg"])

            r = _http(base, "/api/action", {"op": "gen_maps", "count": "0", "seed": "1"})
            check("Generate refuses bad count", "Generate refused" in r["msg"], r["msg"])

            r = _http(base, "/api/action", {"op": "stop"})
            check(
                "Stop is honest when idle",
                "no trainer was running" in r["msg"].lower() or "nothing to kill" in r["msg"].lower(),
                r["msg"],
            )
        finally:
            cui._train_busy = old_busy2
            cui._find_external_train = old_ext2
            cui._kill_train_tree = old_kill2

        r = _http(base, "/api/action", {"op": "bogus"})
        check("unknown op reported", "unknown op" in r["msg"], r["msg"])
    finally:
        server.shutdown()
        server.server_close()


def test_meaningful_improvement() -> None:
    """Early-stop epsilon vs race_score_key (finishers / DNFs) + patience state machine."""
    print("early-stop min improvement + patience tick")
    from .eval_protocol import race_score_key
    from .train_ppo import (
        early_stop_patience_tick,
        is_meaningful_race_improvement,
        race_eval_patience_update,
    )

    fin = lambda adj, cols=0: {
        "mean_lap_time": float(adj) - 10.0 * cols,
        "adjusted_time": float(adj),
        "mean_progress_frac": 1.0,
        "total_collisions": cols,
    }
    dnf = lambda prog, adj=400.0, cols=0: {
        "mean_lap_time": None,
        "adjusted_time": float(adj),
        "mean_progress_frac": float(prog),
        "total_collisions": cols,
    }

    check("first eval always meaningful", is_meaningful_race_improvement(fin(100), None, min_improve=0.5))
    check(
        "tiny finisher gain not meaningful at 0.5",
        not is_meaningful_race_improvement(fin(99.8), fin(100), min_improve=0.5),
    )
    check(
        "0.5s finisher gain is meaningful",
        is_meaningful_race_improvement(fin(99.5), fin(100), min_improve=0.5),
    )
    check(
        "min_improve 0 = any key improvement",
        is_meaningful_race_improvement(fin(99.9), fin(100), min_improve=0.0),
    )
    check(
        "DNF->finish always meaningful",
        is_meaningful_race_improvement(fin(250), dnf(0.4), min_improve=0.5),
    )
    check(
        "DNF tiny progress is meaningful (aligns with promotion)",
        is_meaningful_race_improvement(dnf(0.405), dnf(0.40), min_improve=0.5),
    )
    check(
        "DNF +1% progress meaningful",
        is_meaningful_race_improvement(dnf(0.41), dnf(0.40), min_improve=0.5),
    )
    # BUG-3 fix: far-then-crash beats idle-timeout-at-2% under progress-first DNF key.
    idle = dnf(0.02, adj=60.0, cols=0)
    far = dnf(0.40, adj=70.0, cols=1)
    check(
        "far-then-crash ranks above idle timeout",
        race_score_key(far) < race_score_key(idle),
        f"far={race_score_key(far)} idle={race_score_key(idle)}",
    )
    n, stop = race_eval_patience_update(improved=False, no_improve=2, patience=3)
    check("patience trips on 3rd miss", n == 3 and stop)
    n, stop = race_eval_patience_update(improved=True, no_improve=2, patience=3)
    check("meaningful reset clears counter", n == 0 and not stop)

    # State machine: baseline / warmup / grace / miss / stop / unscorable.
    n, stop, ev = early_stop_patience_tick(
        improved=True, no_improve=0, patience=3, eval_count=1, timesteps=10_000,
        warmup_evals=2, min_timesteps=100_000, is_baseline=True,
    )
    check("baseline never stops", n == 0 and not stop and ev == "baseline")
    n, stop, ev = early_stop_patience_tick(
        improved=False, no_improve=0, patience=3, eval_count=2, timesteps=20_000,
        warmup_evals=2, min_timesteps=100_000, is_baseline=False,
    )
    check("warmup miss does not tick", n == 0 and not stop and ev == "warmup")
    n, stop, ev = early_stop_patience_tick(
        improved=False, no_improve=0, patience=3, eval_count=3, timesteps=50_000,
        warmup_evals=2, min_timesteps=100_000, is_baseline=False,
    )
    check("pre-min-timesteps miss does not tick", n == 0 and not stop and ev == "grace_timesteps")
    n, stop, ev = early_stop_patience_tick(
        improved=False, no_improve=0, patience=3, eval_count=4, timesteps=120_000,
        warmup_evals=2, min_timesteps=100_000, is_baseline=False,
    )
    check("post-warmup miss increments", n == 1 and not stop and ev == "miss")
    n, stop, ev = early_stop_patience_tick(
        improved=False, no_improve=2, patience=3, eval_count=6, timesteps=200_000,
        warmup_evals=2, min_timesteps=100_000, is_baseline=False,
    )
    check("patience trips after grace", n == 3 and stop and ev == "stop")
    n, stop, ev = early_stop_patience_tick(
        improved=False, no_improve=2, patience=3, eval_count=6, timesteps=200_000,
        warmup_evals=0, min_timesteps=0, scorable=False,
    )
    check("unscorable never early-stops", n == 2 and not stop and ev == "unscorable")

    ov = ui_ops.PRESETS["overnight"]
    check("overnight preset budget off", ov.get("stop_on_budget") is False)
    check("overnight patience 5", ov.get("early_stop_patience") == 5)
    check("overnight eval every 50k", ov.get("race_eval_every") == 50_000)
    check("overnight min_improve 0.5", ov.get("early_stop_min_improve") == 0.5)
    check("overnight select_timeout >= 180", float(ov.get("select_timeout") or 0) >= 180)
    check("overnight min_timesteps >= 50k", int(ov.get("early_stop_min_timesteps") or 0) >= 50_000)
    check("mid-train timeout >= 180", ui_ops.MID_TRAIN_SELECT_TIMEOUT_S >= 180)
    check("auto eval floor >= 50k", ui_ops.AUTO_RACE_EVAL_EVERY_FLOOR >= 50_000)


def main() -> int:
    print(f"control UI selftest - contracts {CONTRACTS_VERSION}, obs_dim {obs_dim(N_LIDAR_DEFAULT)}\n")
    for fn in (
        test_map_and_holdout_labels,
        test_map_generation,
        test_models_ops,
        test_continue_argv,
        test_seal_verify,
        test_stop_budget,
        test_dual_writer_guards,
        test_start_continue_never_kill_on_false_busy,
        test_heartbeat_cannot_clobber_early_stopped,
        test_banner_and_coach,
        test_meaningful_improvement,
        test_http_surface,
    ):
        fn()
        print()
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)}: {', '.join(_FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
