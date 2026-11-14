#!/usr/bin/env python3
"""Restart the persisted speed campaign runner after unexpected exits.

This supervisor only relaunches ``run_speed_campaign.py``. It never creates
training jobs itself and relies on the runner's state reconciliation and
``runner.lock`` for idempotence.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PHASES = {
    "wait_adaptation", "baseline_eval", "screen", "screen_evaluations",
    "confirmation", "confirmation_evaluations", "decision",
    "production_precheck", "production",
}
TERMINAL_PHASE_PREFIX = "halted"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def restart_block_reason(state: dict[str, Any] | None) -> str | None:
    if state is None:
        return "state_unreadable"
    phase = state.get("phase")
    if phase not in ACTIVE_PHASES:
        return f"phase_not_active:{phase}"
    if str(phase).startswith(TERMINAL_PHASE_PREFIX):
        return f"terminal_phase:{phase}"
    production = state.get("production") or {}
    active_production_cycle = production.get("cycle")
    for key, intent in (state.get("runs") or {}).items():
        if not isinstance(intent, dict):
            continue
        # A completed prior production cycle remains in the state as immutable
        # failure history. Its exhausted retry budget must not prevent a
        # separately numbered, explicitly authorized continuation cycle.
        if key.startswith("production:") and isinstance(active_production_cycle, int):
            try:
                if int(key.split(":", 1)[1]) < active_production_cycle:
                    continue
            except ValueError:
                pass
        if intent.get("retry_exhausted"):
            return f"retry_exhausted:{key}"
        # A failed current run is restartable only when the runner persisted its
        # explicitly bounded retry transition before exiting.
        if intent.get("state") == "failed" and not intent.get("retry_pending"):
            return f"terminal_failed_run:{key}"
        if intent.get("run_id") and intent.get("terminal_state") == "failed" and not intent.get("retry_pending"):
            return f"terminal_failed_run:{key}"
    return None


def hub_healthy(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            body = json.loads(response.read())
        return isinstance(body, dict) and body.get("status") == "ok"
    except Exception:
        return False


def run_inventory(url: str, timeout: float = 10.0) -> dict[str, dict[str, Any]] | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/train/runs", timeout=timeout) as response:
            body = json.loads(response.read())
        return {str(row.get("run_id")): row for row in body.get("runs", [])
                if isinstance(row, dict) and row.get("run_id")}
    except Exception:
        return None


def terminal_failure_reason(state: dict[str, Any], inventory: dict[str, dict[str, Any]]) -> str | None:
    production = state.get("production") or {}
    active_production_cycle = production.get("cycle")
    for key, intent in (state.get("runs") or {}).items():
        if not isinstance(intent, dict) or intent.get("retry_pending"):
            continue
        if key.startswith("production:") and isinstance(active_production_cycle, int):
            try:
                if int(key.split(":", 1)[1]) < active_production_cycle:
                    continue
            except ValueError:
                pass
        rid = intent.get("run_id")
        row = inventory.get(str(rid)) if rid else None
        if row and row.get("state") == "failed":
            return f"terminal_failed_run:{key}:{rid}"
    return None


def runner_lock_held(path: Path) -> bool:
    try:
        # LOCK_EX needs a writable descriptor on this filesystem; opening
        # read-only made a free lock look conservatively occupied forever.
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    except OSError:
        # Conservatively avoid launching if lock status cannot be established.
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def append_log(path: Path, event: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"utc": utc(), "event": event, **details}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def fresh_monitor_stall_signals(path: Path, *, now: float | None = None,
                                max_age_s: float = 120.0) -> set[str]:
    """Return monitor-confirmed stall alerts only while its snapshot is fresh."""
    try:
        status = json.loads(path.read_text())
        checked = datetime.fromisoformat(status["checked_utc"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(now, timezone.utc)) - checked
        if age.total_seconds() < -30 or age.total_seconds() > max_age_s:
            return set()
        alerts = status.get("alerts", [])
        return {str(alert) for alert in alerts if "stall" in str(alert).lower()}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return set()


def monitor_child(child: Any, monitor_path: Path, log_path: Path, interval: float,
                  sleep: Callable[[float], None]) -> int | None:
    """Poll child liveness and log monitor-confirmed stalls without intervening."""
    active_signals: set[str] = set()
    while True:
        code = child.poll()
        if code is not None:
            return code
        signals = fresh_monitor_stall_signals(monitor_path, max_age_s=max(120.0, interval * 4))
        for signal in sorted(signals - active_signals):
            append_log(log_path, "trainer_stall_alert", signal=signal,
                       action="none; runner and trainer left untouched")
        for signal in sorted(active_signals - signals):
            append_log(log_path, "trainer_stall_cleared", signal=signal)
        active_signals = signals
        sleep(interval)


def runner_command(args: argparse.Namespace) -> list[str]:
    command = [args.python, "-u", str(ROOT / "scripts/run_speed_campaign.py"),
            "--adaptation-run-id", args.adaptation_run_id,
            "--base-config", str(args.base_config.resolve()),
            "--output", str(args.campaign_dir.resolve()),
            "--hub-url", args.hub_url, "--poll-seconds", str(args.runner_poll_seconds),
            "--execute"]
    if getattr(args, "continue_after_failed_production_eval", False):
        command.append("--continue-after-failed-production-eval")
    return command


def run_supervisor(args: argparse.Namespace, *, popen: Callable[..., Any] = subprocess.Popen,
                   sleep: Callable[[float], None] = time.sleep,
                   healthy: Callable[[str], bool] = hub_healthy,
                   inventory_reader: Callable[[str], dict[str, dict[str, Any]] | None] = run_inventory) -> int:
    campaign = args.campaign_dir.resolve()
    state_path = campaign / "state.json"
    lock_path = campaign / "runner.lock"
    log_path = args.log.resolve() if args.log else campaign / "supervisor.log"
    supervisor_lock_path = campaign / "supervisor.lock"
    campaign.mkdir(parents=True, exist_ok=True)
    lock_stream = supervisor_lock_path.open("a")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        append_log(log_path, "already_running")
        return 0

    command = runner_command(args)
    delay = args.backoff_initial
    while True:
        state = load_state(state_path)
        reason = restart_block_reason(state)
        if reason:
            append_log(log_path, "supervision_stopped", reason=reason)
            return 0 if reason != "state_unreadable" else 2

        if runner_lock_held(lock_path):
            append_log(log_path, "runner_already_locked")
            sleep(args.interval)
            continue

        while not healthy(args.hub_url):
            append_log(log_path, "hub_unhealthy_waiting", hub_url=args.hub_url)
            sleep(args.interval)
            reason = restart_block_reason(load_state(state_path))
            if reason:
                append_log(log_path, "supervision_stopped", reason=reason)
                return 0

        inventory = inventory_reader(args.hub_url)
        if inventory is None:
            append_log(log_path, "run_inventory_unavailable", hub_url=args.hub_url)
            sleep(args.interval)
            continue
        failed_reason = terminal_failure_reason(state, inventory)
        if failed_reason:
            append_log(log_path, "supervision_stopped", reason=failed_reason)
            return 0

        append_log(log_path, "runner_starting", command=command, backoff_s=delay)
        try:
            child = popen(command, cwd=str(ROOT), start_new_session=True)
            code = monitor_child(child, campaign / "monitor_status.json", log_path,
                                 args.interval, sleep)
        except Exception as exc:
            code = None
            append_log(log_path, "runner_launch_error", error=f"{type(exc).__name__}: {exc}")
        append_log(log_path, "runner_exited", returncode=code)

        reason = restart_block_reason(load_state(state_path))
        if reason:
            append_log(log_path, "supervision_stopped", reason=reason)
            return 0
        sleep(delay)
        delay = min(args.backoff_max, max(args.backoff_initial, delay * 2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=Path("logs/rl/speed_campaign_20261110"))
    parser.add_argument("--adaptation-run-id", required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--hub-url", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--python",
        default=os.environ.get("AICAR_NATIVE_PYTHON", sys.executable),
        help="Python executable used for the campaign runner (defaults to AICAR_NATIVE_PYTHON)",
    )
    parser.add_argument("--runner-poll-seconds", type=float, default=5)
    parser.add_argument("--continue-after-failed-production-eval", action="store_true",
                        help="pass the explicit failed-evaluation continuation override to the durable runner")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--backoff-initial", type=float, default=5)
    parser.add_argument("--backoff-max", type=float, default=300)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)
    if args.interval < 1 or args.runner_poll_seconds < 1:
        parser.error("intervals must be at least one second")
    if args.backoff_initial < 1 or args.backoff_max < args.backoff_initial:
        parser.error("backoff maximum must be at least the initial delay")
    return args


def main(argv: list[str] | None = None) -> int:
    return run_supervisor(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
