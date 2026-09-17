"""Self-contained map pack regression smoke.

Builds a throwaway pack in a temp dir (small scale, fast) and checks the
guarantees siblings rely on: cached generation, role queries, the train gate,
seal tamper detection, and per-index determinism. Touches no real maps.

    python -m rl.test_map_pack          # run everything, print PASS/FAIL
    pytest "rl/test_map_pack.py"        # same checks as one pytest case
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from . import map_pack as mp

PREFIX = "smk"
SCALE = 0.25  # small raster keeps the smoke ~seconds


class _Checks:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def ok(self, label: str, cond: bool, extra: str = "") -> None:
        print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
        if not cond:
            self.fails.append(label)

    def violation(self, label: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except (mp.HoldoutViolation, mp.SealBroken) as exc:
            self.ok(label, True, str(exc)[:70])
            return
        self.ok(label, False, "no exception raised")


def run_checks(root: Path) -> list[str]:
    """Exercise the pack under ``root``; returns the list of failed labels."""
    c = _Checks()
    gen = dict(prefix=PREFIX, maps_root=root, scale=SCALE, seed=4242)
    train_id, val_id, hold_id = f"{PREFIX}0", f"{PREFIX}1", f"{PREFIX}2"

    res = mp.ensure_maps(1, role=mp.ROLE_TRAIN_OK, **gen)
    c.ok("generate creates train map", res["created"] == [train_id], str(res["created"]))

    res = mp.ensure_maps(2, **gen)
    c.ok(
        "generate is cached (only new id drawn)",
        res["created"] == [val_id] and res["cached"] == [train_id],
        f"created={res['created']} cached={res['cached']}",
    )

    res = mp.ensure_maps(3, seal=True, seal_reason="smoke holdout", **gen)
    c.ok("generate --seal creates sealed holdout", res["created"] == [hold_id])

    mp.pin_validation_map(val_id, maps_root=root)
    c.ok("validation pin recorded", mp.validation_map(root) == val_id)

    c.ok(
        "train_safe = train map only",
        mp.train_safe_maps(root) == [train_id],
        str(mp.train_safe_maps(root)),
    )
    c.ok("holdout list", mp.holdout_maps(root) == [hold_id], str(mp.holdout_maps(root)))
    c.ok("is_holdout(sealed)", mp.is_holdout(hold_id, root))
    c.ok("is_holdout(train) false", not mp.is_holdout(train_id, root))
    c.ok("validation is not train safe", not mp.is_train_safe(val_id, root))

    mp.assert_train_safe([train_id], maps_root=root)
    c.ok("gate allows train map", True)
    c.violation("gate refuses sealed holdout", mp.assert_train_safe, [hold_id], maps_root=root)
    c.violation("gate refuses validation pin", mp.assert_train_safe, [val_id], maps_root=root)
    mp.assert_train_safe([hold_id], maps_root=root, allow_holdout=True)
    c.ok("allow_holdout override", True)

    c.violation("force regen refuses sealed map", mp.ensure_maps, 3, force=True, **gen)
    c.violation("unseal requires force", mp.unseal_maps, [hold_id], maps_root=root)

    fp = mp.pack_fingerprint(root)
    c.ok(
        "fingerprint pins holdout hash",
        bool(fp["holdouts"].get(hold_id)) and fp["validation_map"] == val_id,
        json.dumps(fp["holdouts"]),
    )

    report = mp.verify_pack(root)
    c.ok(
        "all maps verify clean",
        all(r["ok"] for r in report["maps"]),
        "; ".join(i for r in report["maps"] for i in r["issues"]),
    )
    mp.assert_seals_intact(root)
    c.ok("seals intact before tampering", True)

    # per-index determinism: same (seed, index) redraws byte-identical geometry
    from .trackgen import generate_map

    scratch = root / "_regen"
    spec = generate_map(scratch, 0, seed=4242, prefix=PREFIX, scale=SCALE)
    c.ok(
        "map regenerates byte-identical from (seed, index)",
        mp.map_content_hash(spec["yaml"]) == mp.map_content_hash(mp.map_yaml_for(train_id, root)),
    )
    shutil.rmtree(scratch, ignore_errors=True)

    # tampering with a sealed map's centerline must break its seal
    cl = root / hold_id / "centerline.csv"
    cl.write_text(cl.read_text(encoding="utf-8") + "0.0, 0.0\n", encoding="utf-8")
    entry = next(r for r in mp.verify_pack(root)["maps"] if r["id"] == hold_id)
    c.ok(
        "tampered centerline breaks the seal",
        any("SEAL BROKEN" in i for i in entry["issues"]),
        "; ".join(entry["issues"])[:90],
    )
    c.violation("assert_seals_intact raises on tamper", mp.assert_seals_intact, root)

    return c.fails


def test_map_pack(tmp_path: Path | None = None) -> None:
    """Pytest entry point (also used by ``main``)."""
    own_tmp = tmp_path is None
    base = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp(prefix="map_pack_smoke_"))
    try:
        fails = run_checks(base / "maps")
    finally:
        if own_tmp:
            shutil.rmtree(base, ignore_errors=True)
    assert not fails, f"map pack checks failed: {fails}"


def main(argv: list[str] | None = None) -> int:
    try:
        test_map_pack()
    except AssertionError as exc:
        print(f"\n{exc}")
        return 1
    print("\nmap pack smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
