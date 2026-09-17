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
    find_last_complete_checkpoint,
    release_run_lock,
)

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


class _FakeModel:
    """Minimal SB3-like save that writes an empty-ish zip stem."""

    def __init__(self, payload: bytes = b"PK\x05\x06" + b"\x00" * 18):
        self.payload = payload

    def save(self, path: str) -> None:
        p = Path(path)
        # Mirror SB3: path without .zip → path.zip
        out = p if p.suffix == ".zip" else Path(str(p) + ".zip")
        out.write_bytes(self.payload + os.urandom(32))


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

    print()
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
