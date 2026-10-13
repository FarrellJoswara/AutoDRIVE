"""Process-independent monitor for the persisted official PPO campaign clock.

Run with ``python -m src.layer4.hub.campaign_supervisor`` from the repository
root. The monitor never creates training/evaluation work. Admission remains in
the Mission Control API; this process requests bounded normal stop operations.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.layer4.hub.campaign_deadline import (
    append_campaign_event,
    campaign_clock_status,
    default_state_path,
    initialize_campaign,
    load_campaign_state,
    record_campaign_status,
)


DEFAULT_HUB_URL = "http://127.0.0.1:8090"
HARD_STOP_LEAD_SECONDS = 45.0
REQUEST_TIMEOUT_SECONDS = 3.0
STOP_RETRY_SECONDS = 5.0
TRAINING_FREEZE_GRACE_SECONDS = 60.0
HEARTBEAT_LOG_SECONDS = 60.0
RELAY_NEUTRAL_SETTLE_SECONDS = 0.6
PROCESS_TERM_SETTLE_SECONDS = 0.8
_SIGCONT = getattr(signal, "SIGCONT", 18)
_SIGKILL = getattr(signal, "SIGKILL", 9)


def _bounded_call(function, timeout: float):
    """Run a potentially slow HTTP stop in a daemon thread with a hard wait bound."""
    if timeout <= 0:
        raise TimeoutError("campaign deadline leaves no time for another blocking request")
    result: list[Any] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(function())
        except BaseException as exc:
            failure.append(exc)

    worker = threading.Thread(target=run, name="campaign-stop-request", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"campaign HTTP request exceeded {timeout:.3f}s bound")
    if failure:
        raise failure[0]
    return result[0]


def _retry_due(state: dict[str, Any], kind: str, run_id: str, now_mono_ns: int) -> bool:
    for event in reversed(state.get("events", [])):
        if not isinstance(event, dict) or event.get("event") != "supervisor_action":
            continue
        if event.get("run_id") != run_id or event.get("kind") not in {
            f"{kind}_stop_requested", f"{kind}_stop_failed",
        }:
            continue
        previous_mono = event.get("monotonic_ns")
        if isinstance(previous_mono, int) and now_mono_ns >= previous_mono:
            return (now_mono_ns - previous_mono) / 1_000_000_000 >= STOP_RETRY_SECONDS
        return True
    return True


def _seconds_since_last_stop(state: dict[str, Any], kind: str, run_id: str, now_mono_ns: int) -> Optional[float]:
    for event in reversed(state.get("events", [])):
        if not isinstance(event, dict) or event.get("event") != "supervisor_action":
            continue
        if event.get("run_id") != run_id or event.get("kind") not in {
            f"{kind}_stop_requested", f"{kind}_stop_failed",
        }:
            continue
        previous_mono = event.get("monotonic_ns")
        if isinstance(previous_mono, int) and now_mono_ns >= previous_mono:
            return (now_mono_ns - previous_mono) / 1_000_000_000
        return None
    return None


def _seconds_since_first_stop_request(
    state: dict[str, Any], kind: str, run_id: str, now_mono_ns: int,
    *, reason: Optional[str] = None,
) -> Optional[float]:
    """Measure grace from the first durable stop attempt for one run."""
    for event in state.get("events", []):
        if not isinstance(event, dict) or event.get("event") != "supervisor_action":
            continue
        if event.get("run_id") != run_id or event.get("kind") not in {
            f"{kind}_stop_requested", f"{kind}_stop_failed",
        }:
            continue
        if reason is not None and event.get("reason") != reason:
            continue
        first_mono = event.get("monotonic_ns")
        if isinstance(first_mono, int) and now_mono_ns >= first_mono:
            return (now_mono_ns - first_mono) / 1_000_000_000
        return None
    return None


def _seconds_since_event(state: dict[str, Any], event_name: str, now_mono_ns: int) -> Optional[float]:
    for event in reversed(state.get("events", [])):
        if isinstance(event, dict) and event.get("event") == event_name:
            previous_mono = event.get("monotonic_ns")
            if isinstance(previous_mono, int) and now_mono_ns >= previous_mono:
                return (now_mono_ns - previous_mono) / 1_000_000_000
            return None
    return None


def _proc_identity(pid: int, proc_root: str | Path = "/proc") -> Optional[dict[str, Any]]:
    """Read Linux process identity fields without relying on command-name searches."""
    proc = Path(proc_root) / str(pid)
    try:
        raw_stat = (proc / "stat").read_text(encoding="ascii")
        close = raw_stat.rfind(")")
        if close < 0:
            return None
        fields = raw_stat[close + 2:].split()  # starts with field 3, state
        command = (proc / "cmdline").read_bytes().split(b"\0")
        argv = [item.decode("utf-8", errors="replace") for item in command if item]
        return {
            "pid": pid,
            "state": fields[0],
            "parent_pid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "starttime_ticks": int(fields[19]),
            "argv": argv,
        }
    except (OSError, ValueError, IndexError):
        return None


def _relay_neutral_report(record: dict[str, Any]) -> dict[str, Any]:
    log_path = record.get("log")
    if not isinstance(log_path, str) or not log_path:
        return {"verified": False, "reason": "relay_log_path_missing"}
    path = Path(log_path)
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"verified": False, "reason": f"relay_log_unreadable:{exc}", "log_path": log_path}
    releases = [row for row in rows if isinstance(row, dict)
                and row.get("event") == "shutdown_neutral_release_complete"]
    if not releases:
        return {"verified": False, "reason": "neutral_release_completion_not_logged", "log_path": log_path}
    result = releases[-1]
    attempted = result.get("attempted")
    sent = result.get("sent")
    failed = result.get("failed")
    verified = (isinstance(attempted, int) and attempted > 0 and isinstance(sent, int)
                and sent == attempted and isinstance(failed, int) and failed == 0)
    return {"verified": verified, "attempted": attempted, "sent": sent, "failed": failed,
            "transport_acknowledged": False, "log_path": log_path}


def _manifest_records_for_run(run_id: str, *, repository_root: Optional[str | Path] = None) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    root = Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[3]
    run_dir = root / "logs" / "rl" / f"official_run_{run_id}"
    manifest_path = run_dir / "native_group_api.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = manifest.get("Labels", {})
    if not isinstance(labels, dict) or labels.get("aicar.run_id") != run_id:
        raise ValueError("native manifest run identity does not match requested run")
    if labels.get("aicar.output_dir", "").rstrip("/").casefold() != str(run_dir).replace("\\", "/").rstrip("/").casefold():
        # Windows-hosted artifacts record their corresponding /mnt/c path.
        expected_wsl = (
            "/mnt/" + str(run_dir.drive[0]).casefold() + str(run_dir)[2:].replace("\\", "/")
            if run_dir.drive else str(run_dir).replace("\\", "/")
        )
        if labels.get("aicar.output_dir", "").rstrip("/").casefold() != expected_wsl.casefold():
            raise ValueError("native manifest output directory does not match requested run")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("native manifest records are missing")
    return manifest_path, manifest, records


def force_stop_native_training_run(
    run_id: str,
    *,
    repository_root: Optional[str | Path] = None,
    signal_group=None,
    sleep=time.sleep,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Identity-checked POSIX fallback after the manager's graceful stop grace.

    This function is only called by the deadline supervisor. It refuses to
    operate from Windows, and never targets hub PIDs, image names, or global
    process patterns. SIGTERM to a recorded relay first gives it the bounded
    neutral-throttle/steering/reset release path before any process kill.
    """
    default_signal = signal_group is None
    if default_signal:
        signal_group = getattr(os, "killpg", None)
    if signal_group is None or (default_signal and os.name != "posix") or not Path(proc_root).is_dir():
        return {"status": "not_supported", "reason": "identity-checked /proc cleanup requires Linux/WSL"}
    try:
        manifest_path, manifest, records = _manifest_records_for_run(
            run_id, repository_root=repository_root
        )
    except Exception as exc:
        return {"status": "refused", "reason": f"manifest_invalid:{type(exc).__name__}:{exc}"}

    hub_pid = manifest.get("hub_pid")
    backend_id = manifest.get("backend_instance_id")
    labels = manifest.get("Labels", {})
    if not isinstance(hub_pid, int) or not backend_id or labels.get("aicar.kind") != "training":
        return {"status": "refused", "reason": "manifest lacks expected training hub/backend identity"}

    verified: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            return {"status": "refused", "reason": "manifest contains malformed process record"}
        role = record.get("role")
        if role == "hub" or not isinstance(role, str) or not role.startswith(("trainer", "relay-", "devkit-")):
            return {"status": "refused", "reason": f"manifest contains non-owned process role:{role}"}
        if record.get("handle_origin") != "created_by_popen" or record.get("process_handle_class") != "subprocess.Popen":
            return {"status": "refused", "reason": f"process ownership not proven for role:{role}"}
        pid = record.get("pid")
        pgid = record.get("process_group_id")
        expected_start = record.get("process_starttime_ticks")
        expected_parent = record.get("launch_parent_pid")
        expected_argv = record.get("launch_command")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (pid, pgid, expected_start, expected_parent)):
            return {"status": "refused", "reason": f"invalid process identity fields for role:{role}"}
        if pid <= 1 or pgid <= 1 or pid == hub_pid or expected_parent != hub_pid:
            return {"status": "refused", "reason": f"unsafe PID/PGID/parent for role:{role}"}
        if not isinstance(expected_argv, list) or not expected_argv or not all(isinstance(x, str) for x in expected_argv):
            return {"status": "refused", "reason": f"launch command missing for role:{role}"}
        current = _proc_identity(pid, proc_root)
        if current is None:
            verified.append({"role": role, "pid": pid, "process_group_id": pgid,
                             "status": "already_exited", "starttime_ticks": expected_start})
            continue
        if (current["starttime_ticks"] != expected_start or current["process_group_id"] != pgid
                or current["parent_pid"] != expected_parent or current["argv"] != expected_argv):
            return {"status": "refused", "reason": f"live process identity changed for role:{role}",
                    "observed": {key: current[key] for key in ("pid", "parent_pid", "process_group_id", "starttime_ticks", "argv")},
                    "expected": {"pid": pid, "parent_pid": expected_parent, "process_group_id": pgid,
                                 "starttime_ticks": expected_start, "argv": expected_argv}}
        verified.append({"role": role, "pid": pid, "process_group_id": pgid,
                         "starttime_ticks": expected_start, "parent_pid": expected_parent,
                         "launch_command": expected_argv, "status": "identity_verified"})

    live = [row for row in verified if row["status"] == "identity_verified"]
    signals: list[dict[str, Any]] = []
    relay_rows = [row for row in live if str(row["role"]).startswith("relay-")]
    other_rows = [row for row in live if row not in relay_rows]
    for row in relay_rows:
        try:
            signal_group(row["process_group_id"], _SIGCONT)
            signal_group(row["process_group_id"], signal.SIGTERM)
        except OSError as exc:
            return {"status": "incomplete", "reason": f"relay_signal_failed:{exc}",
                    "run_id": run_id, "manifest_path": str(manifest_path),
                    "neutral_release_sent": False, "transport_acknowledged": False,
                    "verified_processes": verified}
        signals.append({"role": row["role"], "signal": "SIGTERM", "neutral_release_attempted": True,
                        "transport_acknowledged": False})
    if relay_rows:
        sleep(RELAY_NEUTRAL_SETTLE_SECONDS)
    neutral_reports = []
    for relay in relay_rows:
        source = next(record for record in records if record.get("pid") == relay["pid"])
        neutral_reports.append({"role": relay["role"], **_relay_neutral_report(source)})
    neutral_verified = not relay_rows or all(row.get("verified") for row in neutral_reports)
    for row in other_rows:
        try:
            signal_group(row["process_group_id"], _SIGCONT)
            signal_group(row["process_group_id"], signal.SIGTERM)
        except OSError as exc:
            return {"status": "incomplete", "reason": f"process_signal_failed:{row['role']}:{exc}",
                    "run_id": run_id, "manifest_path": str(manifest_path),
                    "neutral_release_sent": bool(relay_rows), "transport_acknowledged": False,
                    "verified_processes": verified, "signals": signals}
        signals.append({"role": row["role"], "signal": "SIGTERM", "neutral_release_attempted": False})
    if other_rows:
        sleep(PROCESS_TERM_SETTLE_SECONDS)

    killed: list[str] = []
    survivors: list[dict[str, Any]] = []
    for row in live:
        current = _proc_identity(row["pid"], proc_root)
        if current is None:
            continue
        if (current["starttime_ticks"] != row["starttime_ticks"]
                or current["process_group_id"] != row["process_group_id"]
                or current["parent_pid"] != row["parent_pid"]
                or current["argv"] != row["launch_command"]):
            survivors.append({"role": row["role"], "pid": row["pid"], "status": "identity_changed_before_kill"})
            continue
        try:
            signal_group(row["process_group_id"], _SIGKILL)
        except OSError as exc:
            survivors.append({"role": row["role"], "pid": row["pid"], "status": f"sigkill_failed:{exc}"})
            continue
        killed.append(row["role"])
    remaining = []
    for _ in range(20 if killed else 1):
        remaining = [
            {"role": row["role"], "pid": row["pid"]}
            for row in live if _proc_identity(row["pid"], proc_root) is not None
        ]
        if not remaining:
            break
        sleep(0.05)
    terminal_context = manifest_path.parent / "native_terminal_context.json"
    terminal_context_sha256 = None
    if terminal_context.exists():
        terminal_context_sha256 = hashlib.sha256(terminal_context.read_bytes()).hexdigest()
    return {
        "status": "force_stopped" if not survivors else "incomplete",
        "run_id": run_id,
        "backend_instance_id": backend_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "terminal_context_path": str(terminal_context) if terminal_context.exists() else None,
        "terminal_context_sha256": terminal_context_sha256,
        "verified_processes": verified,
        "signals": signals,
        "sigkill_roles": killed,
        "survivors": survivors + remaining,
        "neutral_release_sent": bool(relay_rows) and neutral_verified,
        "neutral_release_reports": neutral_reports,
        "transport_acknowledged": False,
        "simulator_cleanup": "Windows simulator cleanup must use run-scoped identity-checked companion; never use a global process-name kill",
    }


def force_stop_windows_simulators(
    run_id: str,
    *,
    repository_root: Optional[str | Path] = None,
    hard_deadline_reached: bool = False,
    runner=subprocess.run,
    powershell: Optional[str] = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Call the run-scoped Windows simulator identity checker from WSL/Windows.

    There is deliberately no process-name or PID-list fallback here. The
    PowerShell companion must validate the run's identity manifest before it
    terminates any recorded simulator process.
    """
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-fA-F]{12}", run_id):
        return {"status": "refused", "reason": "run_id must be a 12-character hex id"}
    root = Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[3]
    script = root / "scripts" / "terminate_verified_official_simulators.ps1"
    if not script.is_file():
        return {"status": "not_ready", "reason": "verified Windows simulator helper is missing",
                "script": str(script)}

    script_arg = str(script)
    if os.name != "nt":
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", script_arg)
        if match:
            script_arg = f"{match.group(1).upper()}:\\" + match.group(2).replace("/", "\\")
        elif Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists() or os.environ.get("WSL_DISTRO_NAME"):
            # The injected runner is a test seam; it cannot invoke PowerShell
            # or touch a process. Keep the real WSL path restriction on the
            # production subprocess runner while allowing tests to validate
            # the run-ID and result contract from an ordinary temp directory.
            if runner is subprocess.run:
                return {"status": "refused", "reason": "repository is not on a mounted Windows path",
                        "script": str(script)}
        else:
            if runner is subprocess.run:
                return {"status": "not_supported", "reason": "Windows simulator helper requires Windows or WSL"}
    command = [
        powershell or "powershell.exe", "-NoProfile", "-NonInteractive",
        "-ExecutionPolicy", "Bypass", "-File", script_arg,
        "-RunId", run_id, "-ApiStopAttempted", "-GraceExpired",
    ]
    if hard_deadline_reached:
        command.append("-HardDeadlineReached")
    try:
        completed = runner(command, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return {"status": "failed", "reason": f"helper_invocation_failed:{type(exc).__name__}:{exc}",
                "run_id": run_id}
    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    try:
        payload = json.loads(stdout.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"status": "failed", "reason": "helper_output_not_json", "run_id": run_id,
                "exit_code": getattr(completed, "returncode", None),
                "stdout": stdout[-2000:], "stderr": stderr[-2000:]}
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        return {"status": "refused", "reason": "helper_output_run_id_mismatch", "run_id": run_id,
                "payload": payload}
    required_arrays = (
        "targets", "terminated_pids", "already_exited_pids", "rejected_pids", "errors",
    )
    if (payload.get("result") not in {"success", "validated_only"}
            or any(not isinstance(payload.get(key), list) for key in required_arrays)
            or not isinstance(payload.get("recorded_at_utc"), str)
            or not payload["recorded_at_utc"].strip()
            or not isinstance(payload.get("api_state"), str)
            or not payload["api_state"].strip()):
        return {"status": "failed", "reason": "helper_output_schema_invalid", "run_id": run_id,
                "payload": payload, "exit_code": getattr(completed, "returncode", None),
                "stderr": stderr[-2000:]}
    if (getattr(completed, "returncode", 1) != 0 or payload.get("errors")
            or payload.get("rejected_pids")):
        return {"status": "incomplete", "run_id": run_id, "payload": payload,
                "stderr": stderr[-2000:]}
    return {"status": "verified", "run_id": run_id, "payload": payload}


def _get_json(url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object response from {url}")
    return payload


def _post_json(url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object response from {url}")
    return payload


def _is_active(row: dict[str, Any]) -> bool:
    return str(row.get("state", "")).casefold() in {"starting", "running", "stopping"}


def _active_work(hub_url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    active_training: list[dict[str, Any]] = []
    active_evaluation: Optional[dict[str, Any]] = None
    runs_payload = _get_json(f"{hub_url.rstrip('/')}/train/runs", timeout=timeout)
    runs = runs_payload.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("/train/runs response has no run list")
    for row in runs:
        if isinstance(row, dict) and _is_active(row) and row.get("run_id"):
            active_training.append({"run_id": str(row["run_id"]), "state": row.get("state")})

    replay = _get_json(f"{hub_url.rstrip('/')}/api/replay/status", timeout=timeout)
    if _is_active(replay) and replay.get("run_id"):
        active_evaluation = {
            "run_id": str(replay["run_id"]),
            "state": replay.get("state"),
            "pid": replay.get("pid"),
        }
    return {"training": active_training, "evaluation": active_evaluation}


def supervise_once(
    state_path: str | Path,
    *,
    hub_url: str = DEFAULT_HUB_URL,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
    boot_id: Optional[str] = None,
    api: Any = None,
) -> dict[str, Any]:
    """One bounded monitor pass; ``api`` can inject test transport methods."""
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    if state is None:
        raise FileNotFoundError(f"campaign state not found: {path}")
    now = (utc_now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    clock = campaign_clock_status(
        state, utc_now=now, monotonic_ns=monotonic_ns, boot_id=boot_id
    )
    deadline_reached = clock["phase"] == "expired"
    if state.get("status") != "active":
        return {"status": state.get("status"), "clock": clock, "active": None, "actions": []}

    monotonic_current = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    remaining_value = clock.get("remaining_seconds")
    remaining_before_query = float(remaining_value) if remaining_value is not None else None
    if remaining_before_query is not None and remaining_before_query <= 1.0 and not deadline_reached:
        return {"status": "active", "clock": clock, "active": None, "actions": [],
                "reason": "hard deadline is within one second; skipping blocking status requests"}
    api_bound = (
        1.0 if deadline_reached else REQUEST_TIMEOUT_SECONDS if remaining_before_query is None
        else min(REQUEST_TIMEOUT_SECONDS, max(0.05, (remaining_before_query - 0.5) / 3.0))
    )
    try:
        active = api.active_work(hub_url) if api else _bounded_call(
            lambda: _active_work(hub_url, timeout=api_bound),
            timeout=(api_bound if deadline_reached or remaining_before_query is None else
                     min(REQUEST_TIMEOUT_SECONDS, remaining_before_query - 0.5)),
        )
    except Exception as exc:
        outage_age = _seconds_since_event(state, "supervisor_api_unavailable", monotonic_current)
        if outage_age is None or outage_age >= HEARTBEAT_LOG_SECONDS:
            append_campaign_event(path, {
                "event": "supervisor_api_unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }, utc_now=now, monotonic_ns=monotonic_ns)
        if not deadline_reached:
            return {"status": "active", "clock": clock, "active": None, "actions": [],
                    "error": f"{type(exc).__name__}: {exc}"}
        # At expiry, a failed API query cannot make previously observed work
        # disappear. Retry the exact run IDs from the last durable observation.
        observed = next((event.get("active") for event in reversed(state.get("events", []))
                         if isinstance(event, dict)
                         and event.get("event") == "supervisor_active_work_observed"
                         and isinstance(event.get("active"), dict)), None)
        if observed is None:
            return {"status": "active", "clock": clock, "active": None, "actions": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "reason": "hard deadline reached but no durable active-work identity is available"}
        active = {
            "training": [{"run_id": run_id, "state": "previously_observed"}
                         for run_id in observed.get("training", [])
                         if isinstance(run_id, str)],
            "evaluation": ({"run_id": observed["evaluation"], "state": "previously_observed"}
                           if isinstance(observed.get("evaluation"), str) else None),
        }

    active_snapshot = {
        "training": sorted(str(row.get("run_id")) for row in active.get("training", [])
                            if isinstance(row, dict) and row.get("run_id")),
        "evaluation": (str(active["evaluation"].get("run_id"))
                       if isinstance(active.get("evaluation"), dict)
                       and active["evaluation"].get("run_id") else None),
    }
    previous_snapshot = next((event.get("active") for event in reversed(state.get("events", []))
                              if isinstance(event, dict)
                              and event.get("event") == "supervisor_active_work_observed"), None)
    if active_snapshot != previous_snapshot:
        append_campaign_event(path, {"event": "supervisor_active_work_observed", "active": active_snapshot},
                              utc_now=now, monotonic_ns=monotonic_ns)
        state = load_campaign_state(path) or state

    freeze_due = clock["phase"] in {"final_validation", "final_validation_requested"}
    hard_stop_due = deadline_reached or (
        remaining_before_query is not None and remaining_before_query <= HARD_STOP_LEAD_SECONDS
    )
    freeze_reason = (
        "hard_deadline_cleanup" if hard_stop_due else
        "readiness_validation_requested"
        if clock["phase"] == "final_validation_requested" else "training_freeze"
    )
    actions: list[dict[str, Any]] = []
    if freeze_due or hard_stop_due:
        for row in active.get("training", []):
            run_id = str(row.get("run_id", ""))
            if not run_id:
                continue
            if deadline_reached:
                # Make one final normal stop attempt, then enforce the global
                # cutoff against only this recorded run identity.
                try:
                    request_timeout = min(1.0, max(0.05, api_bound))
                    result = api.stop_training(hub_url, run_id) if api else _bounded_call(
                        lambda: _post_json(
                            f"{hub_url.rstrip('/')}/train/runs/{run_id}/stop", timeout=request_timeout
                        ), timeout=request_timeout,
                    )
                    stop_action = {"kind": "training_stop_requested", "run_id": run_id,
                                   "reason": "hard_deadline_cleanup", "state": result.get("state")}
                except Exception as exc:
                    stop_action = {"kind": "training_stop_failed", "run_id": run_id,
                                   "reason": "hard_deadline_cleanup",
                                   "error": f"{type(exc).__name__}: {exc}"}
                actions.append(stop_action)
                append_campaign_event(path, {"event": "supervisor_action", **stop_action},
                                      utc_now=now, monotonic_ns=monotonic_ns)
                if api and hasattr(api, "escalate_training"):
                    cleanup = api.escalate_training(hub_url, run_id)
                else:
                    cleanup = {
                        "native_processes": force_stop_native_training_run(run_id),
                        "windows_simulators": force_stop_windows_simulators(
                            run_id, hard_deadline_reached=True,
                        ),
                    }
                    cleanup["status"] = (
                        "force_stopped"
                        if cleanup["windows_simulators"].get("status") == "verified"
                        and cleanup["native_processes"].get("status") in {"force_stopped", "not_supported"}
                        else "incomplete"
                    )
                action = {"kind": "training_escalation", "run_id": run_id,
                          "reason": "hard_deadline_cleanup", "cleanup": cleanup}
                actions.append(action)
                append_campaign_event(path, {"event": "supervisor_action", **action},
                                      utc_now=now, monotonic_ns=monotonic_ns)
                continue
            elapsed_since_stop = _seconds_since_first_stop_request(
                state, "training", run_id, monotonic_current,
                reason=None if hard_stop_due else freeze_reason,
            )
            # A freeze at +9h reserves three hours for validation, so allow a
            # cooperative trainer time to finish its safe-boundary save and
            # neutral cleanup before escalating. The final 45-second hard-stop
            # window retains its short retry cadence to honor the global cutoff.
            escalation_grace = STOP_RETRY_SECONDS if hard_stop_due else TRAINING_FREEZE_GRACE_SECONDS
            if elapsed_since_stop is not None and elapsed_since_stop >= escalation_grace:
                latest_escalation = next((event for event in reversed(state.get("events", []))
                                          if isinstance(event, dict)
                                          and event.get("event") == "supervisor_action"
                                          and event.get("kind") == "training_escalation"
                                          and event.get("run_id") == run_id), None)
                escalation_mono = latest_escalation.get("monotonic_ns") if latest_escalation else None
                if not isinstance(escalation_mono, int) or monotonic_current - escalation_mono >= STOP_RETRY_SECONDS * 1_000_000_000:
                    if api and hasattr(api, "escalate_training"):
                        cleanup = api.escalate_training(hub_url, run_id)
                    else:
                        native_cleanup = force_stop_native_training_run(run_id)
                        windows_cleanup = force_stop_windows_simulators(
                            run_id,
                            hard_deadline_reached=clock["phase"] == "expired",
                        )
                        cleanup = {
                            "status": "force_stopped"
                            if (windows_cleanup.get("status") == "verified"
                                and native_cleanup.get("status") in {"force_stopped", "not_supported"})
                            else "incomplete",
                            "native_processes": native_cleanup,
                            "windows_simulators": windows_cleanup,
                        }
                    action = {"kind": "training_escalation", "run_id": run_id,
                              "reason": freeze_reason,
                              "cleanup": cleanup}
                    actions.append(action)
                    append_campaign_event(path, {"event": "supervisor_action", **action},
                                          utc_now=now, monotonic_ns=monotonic_ns)
                    continue
            if not _retry_due(state, "training", run_id, monotonic_current):
                continue
            try:
                request_timeout = (
                    REQUEST_TIMEOUT_SECONDS if remaining_before_query is None else
                    min(REQUEST_TIMEOUT_SECONDS, max(0.05, remaining_before_query - 1.0))
                )
                result = api.stop_training(hub_url, run_id) if api else _bounded_call(
                    lambda: _post_json(
                        f"{hub_url.rstrip('/')}/train/runs/{run_id}/stop", timeout=request_timeout
                    ), timeout=request_timeout,
                )
                action = {"kind": "training_stop_requested", "run_id": run_id,
                          "reason": freeze_reason,
                          "state": result.get("state")}
            except Exception as exc:
                action = {"kind": "training_stop_failed", "run_id": run_id,
                          "reason": freeze_reason,
                          "error": f"{type(exc).__name__}: {exc}"}
            actions.append(action)
            append_campaign_event(path, {"event": "supervisor_action", **action},
                                  utc_now=now, monotonic_ns=monotonic_ns)

    if hard_stop_due and active.get("evaluation"):
        evaluation = active["evaluation"]
        evaluation_id = str(evaluation.get("run_id", "evaluation"))
        try:
            should_retry = _retry_due(state, "evaluation", evaluation_id, monotonic_current)
            if not should_retry and not deadline_reached:
                return {"status": "active", "clock": clock, "active": active, "actions": actions}
            if should_retry:
                if api:
                    result = api.stop_evaluation(hub_url)
                else:
                    request_timeout = min(
                        1.0 if deadline_reached else 25.0,
                        max(0.05, float(clock.get("remaining_seconds", 0.0)) - 1.0),
                    )
                    result = _bounded_call(
                        lambda: _post_json(f"{hub_url.rstrip('/')}/api/replay/stop", timeout=request_timeout),
                        timeout=request_timeout,
                    )
                action = {"kind": "evaluation_stop_requested", "run_id": evaluation.get("run_id"),
                          "reason": "hard_deadline_cleanup", "state": result.get("state")}
            else:
                action = {"kind": "evaluation_stop_grace_already_attempted",
                          "run_id": evaluation.get("run_id"), "reason": "hard_deadline_cleanup"}
        except Exception as exc:
            action = {"kind": "evaluation_stop_failed", "run_id": evaluation.get("run_id"),
                      "reason": "hard_deadline_cleanup", "error": f"{type(exc).__name__}: {exc}"}
        actions.append(action)
        append_campaign_event(path, {"event": "supervisor_action", **action},
                              utc_now=now, monotonic_ns=monotonic_ns)

        if deadline_reached:
            cleanup = force_stop_windows_simulators(
                evaluation.get("native_run_id") or evaluation_id,
                hard_deadline_reached=True,
            )
            action = {"kind": "evaluation_escalation", "run_id": evaluation_id,
                      "reason": "hard_deadline_cleanup", "cleanup": cleanup}
            actions.append(action)
            append_campaign_event(path, {"event": "supervisor_action", **action},
                                  utc_now=now, monotonic_ns=monotonic_ns)

    if deadline_reached and not active.get("training") and not active.get("evaluation"):
        state = record_campaign_status(
            path, "expired", reason=str(clock.get("reason") or "hard_deadline"),
            utc_now=now, monotonic_ns=monotonic_ns,
        )
        return {"status": "expired", "clock": clock, "active": active, "actions": actions}

    return {"status": "active", "clock": clock, "active": active, "actions": actions}


def _append_monitor_log(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _exclusive_monitor_lock(lock_path: Path):
    """Use an OS-released lock so a restarted supervisor can safely take over."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another campaign supervisor holds the monitor lock") from exc
            locked = True
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another campaign supervisor holds the monitor lock") from exc
            locked = True
        yield
    finally:
        try:
            if locked and os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            elif locked:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def monitor(
    state_path: str | Path,
    *,
    hub_url: str = DEFAULT_HUB_URL,
    log_path: Optional[str | Path] = None,
    poll_interval_seconds: float = 1.0,
    once: bool = False,
) -> int:
    if not 0.1 <= poll_interval_seconds <= 5.0:
        raise ValueError("poll interval must be between 0.1 and 5 seconds")
    state_file = Path(state_path).resolve()
    log_file = Path(log_path).resolve() if log_path else state_file.with_name("supervisor.jsonl")
    lock_file = state_file.with_name(f".{state_file.stem}.monitor.lock")
    with _exclusive_monitor_lock(lock_file):
        return _monitor_loop(state_file, hub_url=hub_url, log_file=log_file,
                             poll_interval_seconds=poll_interval_seconds, once=once)


def _monitor_loop(
    state_file: Path,
    *,
    hub_url: str,
    log_file: Path,
    poll_interval_seconds: float,
    once: bool,
) -> int:
    last_signature: Optional[str] = None
    last_log_mono = 0
    while True:
        try:
            result = supervise_once(state_file, hub_url=hub_url)
        except Exception as exc:
            result = {"status": "fail_closed", "error": f"{type(exc).__name__}: {exc}"}
            _append_monitor_log(log_file, {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), **result,
            })
            return 2
        clock = result.get("clock", {})
        signature = json.dumps({
            "status": result.get("status"),
            "phase": clock.get("phase") if isinstance(clock, dict) else None,
            "active": result.get("active"),
            "actions": result.get("actions"),
            "error": result.get("error"),
            "reason": result.get("reason"),
        }, sort_keys=True, default=str)
        now_mono = time.monotonic_ns()
        if (signature != last_signature or (now_mono - last_log_mono) / 1_000_000_000 >= HEARTBEAT_LOG_SECONDS
                or result["status"] != "active"):
            _append_monitor_log(log_file, {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "campaign_state_path": str(state_file), **result,
            })
            last_signature = signature
            last_log_mono = now_mono
        if once or result["status"] != "active":
            return 0
        remaining_value = result.get("clock", {}).get("remaining_seconds")
        if remaining_value is None:
            delay = poll_interval_seconds
        else:
            remaining = float(remaining_value)
            delay = min(poll_interval_seconds, max(0.0, remaining - 0.05))
        if delay > 0:
            time.sleep(delay)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status", "monitor"))
    parser.add_argument("--state-path", default=str(default_state_path()))
    parser.add_argument("--gate-evidence", help="JSON gate report; required for init")
    parser.add_argument("--campaign-id", help="stable campaign id used by init")
    parser.add_argument("--hub-url", default=os.environ.get("AICAR_HUB_PUBLIC_URL", DEFAULT_HUB_URL))
    parser.add_argument("--log-path")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            if not args.gate_evidence:
                parser.error("init requires --gate-evidence")
            state = initialize_campaign(
                args.state_path, args.gate_evidence, campaign_id=args.campaign_id
            )
            print(json.dumps({"status": state["status"], "campaign_id": state["campaign_id"],
                              "started_at_utc": state["started_at_utc"],
                              "training_freeze_at_utc": state["training_freeze_at_utc"],
                              "hard_deadline_at_utc": state["hard_deadline_at_utc"],
                              "state_path": str(Path(args.state_path).resolve())}, indent=2))
            return 0
        if args.command == "status":
            state = load_campaign_state(args.state_path)
            if state is None:
                print(json.dumps({"status": "not_started", "state_path": str(Path(args.state_path).resolve())}))
                return 0
            clock = campaign_clock_status(state)
            print(json.dumps({"status": state["status"], "campaign_id": state.get("campaign_id"),
                              **clock, "state_path": str(Path(args.state_path).resolve())}, indent=2))
            return 0
        return monitor(args.state_path, hub_url=args.hub_url, log_path=args.log_path,
                       poll_interval_seconds=args.poll_interval, once=args.once)
    except Exception as exc:
        print(f"campaign supervisor error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
