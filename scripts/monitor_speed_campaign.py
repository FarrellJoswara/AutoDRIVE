#!/usr/bin/env python3
"""Read-only watchdog for a persisted speed campaign.

It never submits training or evaluation work. Status is atomically written to
<campaign>/monitor_status.json and newly observed alerts to monitor_alerts.jsonl.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def get_json(url: str, timeout: float = 5) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def age_seconds(path: Path, now: float) -> float | None:
    try:
        return max(0.0, now - path.stat().st_mtime)
    except OSError:
        return None


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ValueError, TypeError, ProcessLookupError, PermissionError):
        return False


def lock_owner_pid(path: Path) -> int | None:
    """Find the PID holding this file's flock on Linux, without changing it."""
    try:
        stat = path.stat()
        device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
        for line in Path("/proc/locks").read_text().splitlines():
            fields = line.split()
            # Linux flock rows: id FLOCK ADVISORY WRITE pid dev:inode range
            if len(fields) >= 6 and fields[1] == "FLOCK" and fields[5] == device:
                try:
                    return int(fields[4])
                except ValueError:
                    continue
    except (OSError, ValueError):
        pass
    return None


def audit_progress(outdir: Path) -> dict[str, Any]:
    path = outdir / "ppo_optimizer_audit.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    latest = rows[-1] if rows else {}
    return {"step": latest.get("num_timesteps"), "audit_count": len(rows),
            "audit_age_s": age_seconds(path, time.time()) if path.exists() else None,
            "audit_finite": all(latest.get(k) is True for k in
                ("metrics_finite", "gradients_finite", "parameters_finite", "optimizer_state_finite")) if latest else None}


def historical_production_cycle(state: dict[str, Any], key: str) -> bool:
    if not key.startswith("production:"):
        return False
    cycle = (state.get("production") or {}).get("cycle")
    if not isinstance(cycle, int):
        return False
    try:
        return int(key.split(":", 1)[1]) < cycle
    except ValueError:
        return False


def inspect(campaign_dir: Path, hub_url: str, *, now: float | None = None,
            runner_grace_s: float = 90, audit_stall_s: float = 900,
            telemetry_stall_s: float = 600, checkpoint_stall_s: float = 1800,
            eval_stall_s: float = 900) -> dict[str, Any]:
    now = time.time() if now is None else now
    state_path = campaign_dir / "state.json"
    state = load(state_path, {})
    alerts: list[str] = []
    status: dict[str, Any] = {"checked_utc": utc(), "campaign_dir": str(campaign_dir),
        "state_path": str(state_path), "phase": state.get("phase"), "hub": {"healthy": False},
        "runner": {}, "runs": {}, "evaluation": {}, "alerts": alerts}

    try:
        health = get_json(hub_url.rstrip("/") + "/health")
        status["hub"] = {"healthy": health.get("status") == "ok", "status": health.get("status"), "pid": health.get("hub_pid")}
    except Exception as exc:
        status["hub"] = {"healthy": False, "error": f"{type(exc).__name__}: {exc}"}
    if not status["hub"]["healthy"]:
        alerts.append("hub_unavailable")

    recorded_pid = load(campaign_dir / "runner.pid", None)
    pid = recorded_pid
    pid_ok = pid_alive(pid)
    # The runner's flock is authoritative when its PID file is stale.
    lock = campaign_dir / "runner.lock"
    lock_held = False
    try:
        # This only acquires/releases a lock; it does not write file contents.
        # LOCK_EX requires a writable descriptor on the campaign filesystem.
        fd = os.open(lock, os.O_RDWR | os.O_NONBLOCK)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_held = True
        finally:
            if not lock_held:
                os.close(fd)
    except OSError:
        pass
    if lock_held and not pid_ok:
        owner_pid = lock_owner_pid(lock)
        if owner_pid is not None:
            pid = owner_pid
            pid_ok = pid_alive(pid)
    runner_live = pid_ok or lock_held
    status["runner"] = {"pid": pid, "recorded_pid": recorded_pid,
                        "pid_alive": pid_ok, "lock_held": lock_held, "live": runner_live,
                        "pid_file_age_s": age_seconds(campaign_dir / "runner.pid", now)}
    phase = state.get("phase")
    if phase and not runner_live:
        alerts.append("runner_stopped")

    rows: dict[str, Any] = {}
    try:
        response = get_json(hub_url.rstrip("/") + "/train/runs")
        rows = {str(x.get("run_id")): x for x in response.get("runs", []) if isinstance(x, dict)}
    except Exception as exc:
        alerts.append("run_inventory_unavailable")
        status["run_inventory_error"] = f"{type(exc).__name__}: {exc}"

    for key, intent in state.get("runs", {}).items():
        if not isinstance(intent, dict):
            continue
        rid = intent.get("run_id")
        row = rows.get(str(rid), {}) if rid else {}
        run_state = row.get("state", "missing")
        output = Path(row.get("output_dir") or intent.get("output_dir") or "")
        audit = audit_progress(output) if str(output) != "." else {"step": None, "audit_count": 0, "audit_age_s": None, "audit_finite": None}
        telemetry_path = output / "telemetry.jsonl"
        checkpoint_dir = output / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.zip")) if checkpoint_dir.is_dir() else []
        cp_age = min((age_seconds(x, now) for x in checkpoints), default=None)
        newest_checkpoint_mtime = max((x.stat().st_mtime for x in checkpoints), default=None)
        telemetry_age = age_seconds(telemetry_path, now)
        fleet = row.get("latest_fleet") or row.get("latest_telemetry") or {}
        telemetry_step = fleet.get("step") if isinstance(fleet, dict) else None
        try:
            telemetry_step = int(telemetry_step) if telemetry_step is not None else None
        except (TypeError, ValueError):
            telemetry_step = None
        active = run_state in {"queued", "running", "stopping"}
        item = {"run_id": rid, "state": run_state, "display_name": intent.get("display_name"),
            "active": active, "progress": audit, "telemetry_age_s": telemetry_age,
            "telemetry_mtime": telemetry_path.stat().st_mtime if telemetry_path.exists() else None,
            "telemetry_step": telemetry_step,
            "telemetry_ts": fleet.get("ts") if isinstance(fleet, dict) else None,
            "checkpoint_count": len(checkpoints), "newest_checkpoint_age_s": cp_age,
            "newest_checkpoint_mtime": newest_checkpoint_mtime,
            "retry_count": intent.get("retry_count", 0), "retry_exhausted": bool(intent.get("retry_exhausted"))}
        status["runs"][key] = item
        if active:
            if audit.get("audit_age_s") is not None and audit["audit_age_s"] > audit_stall_s:
                alerts.append(f"audit_stalled:{key}")
            if telemetry_age is not None and telemetry_age > telemetry_stall_s:
                alerts.append(f"telemetry_stalled:{key}")
            if checkpoints and cp_age is not None and cp_age > checkpoint_stall_s:
                alerts.append(f"checkpoint_stalled:{key}")
            if audit.get("audit_finite") is False:
                alerts.append(f"optimizer_audit_failed:{key}")
        historical = historical_production_cycle(state, key)
        if run_state == "failed" and not historical:
            alerts.append(f"run_failed:{key}")
        elif run_state == "missing" and rid and not historical:
            alerts.append(f"run_missing:{key}")
        if intent.get("retry_exhausted") and not historical:
            alerts.append(f"retry_exhausted:{key}")

    try:
        ev = get_json(hub_url.rstrip("/") + "/api/replay/status")
        running = ev.get("state") in {"queued", "running", "stopping"}
        eval_item = {"state": ev.get("state"), "evaluation_id": ev.get("evaluation_id"),
            "validation_mode": ev.get("validation_mode"), "attempts_completed": ev.get("attempts_completed"),
            "attempts_required": ev.get("attempts_required"), "model": ev.get("model"),
            "started_at": ev.get("started_at"), "active": running}
        status["evaluation"] = eval_item
        if running:
            # Attempt count is evaluated across monitor samples and changes trigger
            # progress. The timestamp age is a conservative fallback signal.
            start = ev.get("started_at")
            if start:
                try:
                    started = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
                    if now - started > eval_stall_s and int(ev.get("attempts_completed", 0) or 0) == 0:
                        alerts.append("evaluation_attempt_stalled")
                except (ValueError, TypeError):
                    pass
            if ev.get("validation_mode") != "porto_practice_diagnostic":
                alerts.append("evaluation_validation_mode_unexpected")
            if any(x.get("active") for x in status["runs"].values()):
                alerts.append("evaluation_overlaps_training")
    except Exception as exc:
        status["evaluation"] = {"error": f"{type(exc).__name__}: {exc}"}
        alerts.append("evaluation_status_unavailable")

    if phase and phase.startswith("halted"):
        alerts.append("campaign_halted")
    status["alerts"] = sorted(set(alerts))
    return status


def write_status(campaign_dir: Path, status: dict[str, Any]) -> None:
    status_path = campaign_dir / "monitor_status.json"
    previous = load(status_path, {})
    now = time.time()
    progress = previous.get("progress_watch", {})
    tracked: dict[str, Any] = {}
    limits = {"audit": 900, "telemetry": 600, "checkpoint": 1800, "evaluation": 900}
    for key, item in status.get("runs", {}).items():
        if not item.get("active"):
            continue
        p = item.get("progress", {})
        signatures = {
            "audit": [p.get("step"), p.get("audit_count")],
            # Hub fleet step is the producer's progress counter. Some native
            # runs do not write telemetry.jsonl, so mtime alone can stay null
            # while live telemetry advances. Keep mtime as a fallback and as
            # evidence of refreshed telemetry when the step is unchanged.
            "telemetry": [item.get("telemetry_step"), item.get("telemetry_mtime")],
            "checkpoint": [item.get("checkpoint_count"), item.get("newest_checkpoint_mtime")],
        }
        for kind, signature in signatures.items():
            name = f"{kind}:{key}"
            old = progress.get(name, {})
            if kind == "telemetry":
                old_signature = old.get("signature")
                old_step = old_signature[0] if isinstance(old_signature, list) and old_signature else None
                old_mtime = old_signature[1] if isinstance(old_signature, list) and len(old_signature) > 1 else old_signature
                step, mtime = signature
                changed = (
                    step is not None and (old_step is None or step > old_step)
                ) or (
                    mtime is not None and (old_mtime is None or mtime > old_mtime)
                )
            else:
                changed = old.get("signature") != signature
            since = now if changed else old.get("unchanged_since", now)
            tracked[name] = {"signature": signature, "unchanged_since": since}
            if not changed and now - since >= limits[kind]:
                status["alerts"].append(f"{kind}_progress_stalled:{key}")
    evaluation = status.get("evaluation", {})
    if evaluation.get("active"):
        name = "evaluation"
        signature = [evaluation.get("evaluation_id"), evaluation.get("attempts_completed")]
        old = progress.get(name, {})
        changed = old.get("signature") != signature
        since = now if changed else old.get("unchanged_since", now)
        tracked[name] = {"signature": signature, "unchanged_since": since}
        if not changed and now - since >= limits["evaluation"]:
            status["alerts"].append("evaluation_progress_stalled")
    status["alerts"] = sorted(set(status.get("alerts", [])))
    status["progress_watch"] = tracked
    atomic_json(status_path, status)
    old = set(previous.get("alerts", []))
    added = sorted(set(status.get("alerts", [])) - old)
    if added:
        event = {"utc": status["checked_utc"], "new_alerts": added, "all_alerts": status["alerts"],
                 "phase": status.get("phase"), "runner": status.get("runner"), "hub": status.get("hub")}
        with (campaign_dir / "monitor_alerts.jsonl").open("a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=Path("logs/rl/speed_campaign_20261110"))
    parser.add_argument("--hub-url", default="http://127.0.0.1:8090")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least one second")
    args.campaign_dir = args.campaign_dir.resolve()
    while True:
        try:
            status = inspect(args.campaign_dir, args.hub_url)
            write_status(args.campaign_dir, status)
            print(json.dumps({"checked_utc": status["checked_utc"], "phase": status["phase"],
                              "runner_live": status["runner"].get("live"), "alerts": status["alerts"]}), flush=True)
        except Exception as exc:
            # The monitor loop itself must remain visible and keep trying.
            print(f"monitor iteration failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
