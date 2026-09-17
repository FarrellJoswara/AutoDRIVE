"""P0 bugfix smokes: atomic save never destroys prior zip; exclusive train.lock."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.live_status import write_live_status, read_live_status, LiveStatusCallback  # noqa: E402
from rl.metrics_io import (  # noqa: E402
    acquire_run_lock,
    atomic_save_sb3,
    atomic_write_json,
    find_last_complete_checkpoint,
    release_run_lock,
)
from rl.map_pack import CorruptManifest, HoldoutViolation, assert_train_safe, load_pack  # noqa: E402
from rl.racing_env import assert_resolved_map_id, resolve_map_yaml  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


class _FakeModel:
    """Minimal SB3-like save that writes a valid zip (size >1KiB)."""

    def __init__(self, payload: bytes = b"GOOD_V1________"):
        self.payload = payload

    def save(self, path: str) -> None:
        import zipfile

        p = Path(path)
        # Mirror SB3: path without .zip → path.zip
        out = p if p.suffix == ".zip" else Path(str(p) + ".zip")
        with zipfile.ZipFile(out, "w") as zf:
            zf.writestr("data.bin", self.payload + (b"\0" * 2048))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bugfix_p0_"))

    # --- atomic_save: never unlink final when bak-move fails --------------------
    dest = tmp / "best_model"
    final = atomic_save_sb3(_FakeModel(b"GOOD_V1________"), dest)
    before = final.read_bytes()
    check("seed best_model exists", final.is_file() and len(before) > 8, f"size={len(before)}")

    real_replace = Path.replace

    def boom_replace(self: Path, target):  # noqa: ANN001
        if self == final and Path(target).suffix.endswith(".bak"):
            raise OSError(5, "simulated WinError 5 on move-aside")
        return real_replace(self, target)

    with mock.patch.object(Path, "replace", boom_replace):
        try:
            atomic_save_sb3(_FakeModel(b"NEW_SHOULD_FAIL"), dest)
            # replace-in-place may succeed in the mock path if only bak-move fails
            # and produced.replace(final) is allowed — that is OK (final updated).
            still = final.is_file()
            check(
                "locked move-aside does not delete final",
                still and final.stat().st_size > 0,
                f"exists={still} size={final.stat().st_size if still else 0}",
            )
        except OSError:
            check(
                "locked move-aside does not delete final",
                final.is_file() and final.read_bytes() == before,
                "raised OSError but prior zip intact",
            )

    # Explicit: the old unlink-on-failure path must be gone — simulate both
    # bak-move and in-place replace failing; prior bytes must remain.
    def boom_all_replace(self: Path, target):  # noqa: ANN001
        raise OSError(32, "simulated sharing violation")

    final.write_bytes(before)
    with mock.patch.object(Path, "replace", boom_all_replace):
        raised = False
        try:
            atomic_save_sb3(_FakeModel(b"NEVER"), dest)
        except OSError:
            raised = True
    check(
        "double-fail replace keeps prior zip bytes",
        raised and final.is_file() and final.read_bytes() == before,
        f"raised={raised} match={final.read_bytes() == before}",
    )

    # bak recovery for Continue
    bak = final.with_suffix(final.suffix + ".bak")
    bak.write_bytes(b"PK\x05\x06" + b"\x00" * 2000)
    orphan = tmp / "orphan_run"
    orphan.mkdir()
    (orphan / "best_model.zip.bak").write_bytes(b"PK\x05\x06" + b"\x00" * 2000)
    found = find_last_complete_checkpoint(orphan)
    check(
        "find_last_complete_checkpoint recovers .bak",
        found is not None and found.name.endswith(".bak"),
        str(found),
    )

    # --- exclusive train.lock -------------------------------------------------
    run_a = tmp / "run_lock_a"
    run_a.mkdir()
    lock = acquire_run_lock(run_a, pid=os.getpid())
    check("acquire_run_lock creates train.lock", lock.is_file(), str(lock))
    refused = False
    try:
        acquire_run_lock(run_a, pid=os.getpid() + 99999)
    except RuntimeError as exc:
        refused = "dual-writer" in str(exc).lower() or "Refuse" in str(exc)
    check("second acquire while alive refuses", refused)

    # Stale lock (dead pid) can be cleared
    run_b = tmp / "run_lock_b"
    run_b.mkdir()
    stale = run_b / "train.lock"
    stale.write_text(json.dumps({"pid": 1, "timestamp": "x"}), encoding="utf-8")
    # pid 1 may or may not be alive on Windows; force dead via mock
    with mock.patch("rl.metrics_io._pid_alive", return_value=False):
        lock_b = acquire_run_lock(run_b, pid=os.getpid())
    check("stale lock cleared then acquired", lock_b.is_file())
    release_run_lock(run_b)
    release_run_lock(run_a)

    # O_EXCL: two creates cannot both succeed
    run_c = tmp / "run_lock_c"
    run_c.mkdir()
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd1 = os.open(str(run_c / "train.lock"), flags, 0o644)
    os.close(fd1)
    second_excl = False
    try:
        os.open(str(run_c / "train.lock"), flags, 0o644)
    except FileExistsError:
        second_excl = True
    check("O_EXCL second create raises FileExistsError", second_excl)

    # --- B0.1: strict map resolve refuses silent wrong-map fallback -----------
    # Soft fallback only matches **/map*.yaml (or demo) — use map0 so soft works.
    maps_tmp = tmp / "maps_b01"
    maps_tmp.mkdir()
    real = maps_tmp / "map0"
    real.mkdir()
    (real / "map0.yaml").write_text("image: map0.png\nresolution: 0.05\n", encoding="utf-8")
    soft = resolve_map_yaml("does_not_exist_xyz", maps_tmp, strict=False)
    check(
        "B0.1 soft resolve still falls back",
        soft.stem == "map0",
        str(soft),
    )
    strict_refused = False
    try:
        resolve_map_yaml("does_not_exist_xyz", maps_tmp, strict=True)
    except FileNotFoundError as exc:
        strict_refused = "strict" in str(exc).lower() or "refuse" in str(exc).lower()
    check("B0.1 strict missing id raises FileNotFoundError", strict_refused)

    mismatch_refused = False
    try:
        assert_resolved_map_id("berlin", soft)
    except FileNotFoundError as exc:
        mismatch_refused = "mismatch" in str(exc).lower()
    check("B0.1 assert_resolved_map_id catches stem mismatch", mismatch_refused)

    ok_id = resolve_map_yaml("map0", maps_tmp, strict=True)
    try:
        assert_resolved_map_id("map0", ok_id)
        id_ok = True
    except FileNotFoundError:
        id_ok = False
    check("B0.1 exact id resolves + assert passes", id_ok and ok_id.stem == "map0")

    # assert_train_safe: missing yaml + not-in-pack (empty manifest, no adopt)
    empty_pack = {"pack_version": 1, "maps": {}, "validation_map": None}
    missing_gate = False
    try:
        assert_train_safe(["ghost_map_zzz"], maps_root=maps_tmp, pack=empty_pack)
    except HoldoutViolation as exc:
        missing_gate = "missing" in str(exc).lower() or "unresolved" in str(exc).lower()
    check("B0.1 assert_train_safe refuses missing map yaml", missing_gate)

    not_safe_gate = False
    try:
        # yaml exists on disk but pack has no train_ok entry
        assert_train_safe(["map0"], maps_root=maps_tmp, pack=empty_pack)
    except HoldoutViolation as exc:
        not_safe_gate = "train_safe" in str(exc).lower() or "unregistered" in str(exc).lower()
    check("B0.1 assert_train_safe refuses unregistered map", not_safe_gate)

    safe_pack = {
        "pack_version": 1,
        "validation_map": None,
        "maps": {
            "map0": {
                "id": "map0",
                "role": "train_ok",
                "sealed": False,
                "exists": True,
            }
        },
    }
    try:
        assert_train_safe(["map0"], maps_root=maps_tmp, pack=safe_pack)
        train_ok_gate = True
    except HoldoutViolation:
        train_ok_gate = False
    check("B0.1 assert_train_safe allows train_ok with yaml", train_ok_gate)

    # --- live_status does not clobber early_stopped ---------------------------
    status = tmp / "live_status.json"
    write_live_status(
        status,
        {
            "run_id": "t",
            "phase": "early_stopped",
            "msg": "early-stop: done",
            "timesteps": 1000,
        },
    )
    # Construct callback and force a learning write attempt
    cb = LiveStatusCallback(
        status_path=status,
        run_id="t",
        every_steps=1,
        every_rollouts=1,
        n_envs=1,
        vec_env_active="dummy",
        vec_env_fallback=True,
    )
    # Minimal fake for _write without full SB3 train loop
    class _M:
        ep_info_buffer = []
        _n_updates = 0
        num_timesteps = 2000

    cb.model = _M()  # type: ignore[attr-defined]
    cb.num_timesteps = 2000  # type: ignore[attr-defined]
    cb._write(force_save_latest=False)  # type: ignore[attr-defined]
    after = read_live_status(status) or {}
    check(
        "LiveStatus refuses to clobber early_stopped phase",
        after.get("phase") == "early_stopped",
        f"phase={after.get('phase')}",
    )

    # Fresh learning write must surface Subproc→Dummy honesty flag (B1.1 / P1-7)
    status2 = tmp / "live_status_fallback.json"
    cb2 = LiveStatusCallback(
        status_path=status2,
        run_id="t2",
        every_steps=1,
        every_rollouts=1,
        n_envs=8,
        vec_env_active="dummy",
        vec_env_fallback=True,
    )
    cb2.model = _M()  # type: ignore[attr-defined]
    cb2.num_timesteps = 100  # type: ignore[attr-defined]
    cb2._write(force_save_latest=False)  # type: ignore[attr-defined]
    fb = read_live_status(status2) or {}
    check(
        "live_status records vec_env_fallback=true",
        fb.get("vec_env_fallback") is True and fb.get("vec_env_active") == "dummy",
        f"fallback={fb.get('vec_env_fallback')} active={fb.get('vec_env_active')}",
    )

    # P1-2: best_model_meta must be written atomically (tmp → replace), not torn write_text
    meta_path = tmp / "best_model_meta.json"
    atomic_write_json(
        meta_path,
        {"selected_by": "adjusted_time", "dnf": False, "timesteps": 1000},
    )
    meta1 = json.loads(meta_path.read_text(encoding="utf-8"))
    atomic_write_json(
        meta_path,
        {"selected_by": "progress_frac (DNF, no scored lap)", "dnf": True, "timesteps": 2000},
    )
    meta2 = json.loads(meta_path.read_text(encoding="utf-8"))
    check(
        "P1-2 atomic_write_json best_model_meta roundtrip",
        meta1.get("dnf") is False and meta2.get("dnf") is True and meta2.get("timesteps") == 2000,
        f"meta1_dnf={meta1.get('dnf')} meta2={meta2}",
    )
    train_src = Path(__file__).with_name("train_ppo.py").read_text(encoding="utf-8")
    uses_atomic_meta = (
        "atomic_write_json(self.run_dir / \"best_model_meta.json\"" in train_src
        or "atomic_write_json(self.run_dir / 'best_model_meta.json'" in train_src
    )
    still_torn = "best_model_meta.json\").write_text" in train_src or "best_model_meta.json').write_text" in train_src
    check(
        "P1-2 RaceBest uses atomic_write_json for meta",
        uses_atomic_meta and not still_torn,
        f"atomic={uses_atomic_meta} torn_write_text={still_torn}",
    )

    # P1-3: corrupt map_pack.json must refuse (never silent empty + adopt as train_ok)
    pack_root = tmp / "maps_corrupt"
    pack_root.mkdir()
    (pack_root / "map_pack.json").write_text("{not-json", encoding="utf-8")
    corrupt_raised = False
    try:
        load_pack(pack_root, adopt=False)
    except CorruptManifest as exc:
        corrupt_raised = "corrupt" in str(exc).lower() or "json" in str(exc).lower()
    check("P1-3 load_pack refuses corrupt JSON manifest", corrupt_raised)

    bad_shape = tmp / "maps_bad_shape"
    bad_shape.mkdir()
    (bad_shape / "map_pack.json").write_text('{"pack_version": 1, "maps": []}', encoding="utf-8")
    shape_raised = False
    try:
        load_pack(bad_shape, adopt=False)
    except CorruptManifest as exc:
        shape_raised = "shape" in str(exc).lower() or "maps" in str(exc).lower()
    check("P1-3 load_pack refuses bad-shape manifest", shape_raised)

    # Checkpoint finder skips incomplete/partial zip payloads
    import io
    import zipfile

    ckpt_run = tmp / "ckpt_run"
    (ckpt_run / "checkpoints").mkdir(parents=True)
    partial = ckpt_run / "checkpoints" / "ppo_1_steps.zip"
    partial.write_bytes(b"not-a-real-zip-file-just-noise-xxxx")
    good = ckpt_run / "checkpoints" / "ppo_2_steps.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("policy.pth", b"G" * 2048)
    good.write_bytes(buf.getvalue())
    found_ckpt = find_last_complete_checkpoint(ckpt_run)
    check(
        "find_last_complete_checkpoint skips bad zip, keeps good",
        found_ckpt is not None and found_ckpt.resolve() == good.resolve(),
        str(found_ckpt),
    )

    # P1-1: AtomicCheckpointCallback wired in train_ppo
    uses_atomic_ckpt = (
        "class AtomicCheckpointCallback" in train_src
        and "AtomicCheckpointCallback(" in train_src
        and "atomic_save_sb3(self.model" in train_src
    )
    check("P1-1 AtomicCheckpointCallback wired", uses_atomic_ckpt)

    # P1-5: Monitor info_keywords constant
    from rl.train_ppo import MONITOR_INFO_KEYWORDS

    check(
        "P1-5 MONITOR_INFO_KEYWORDS has collision+progress",
        "collision" in MONITOR_INFO_KEYWORDS and "progress_frac" in MONITOR_INFO_KEYWORDS,
        str(MONITOR_INFO_KEYWORDS),
    )
    check(
        "P1-5 _make_env uses MONITOR_INFO_KEYWORDS",
        "info_keywords=MONITOR_INFO_KEYWORDS" in train_src,
    )

    print()
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
