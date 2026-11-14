#!/usr/bin/env python3
"""Wait for a qualified PPO pivot handoff, then arm its production campaign."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIVOT = ROOT / "logs/rl/ppo_pivot_20261111"
BASE_CONFIG = ROOT / "logs/rl/reward_sweep_966510/base_config.json"
PRODUCTION = ROOT / "logs/rl/ppo_pivot_production_20261111"
TMUX_LAUNCH = "ppo-pivot-production-launcher"
TMUX_SUPERVISOR = "ppo-pivot-production"
TMUX_MONITOR = "ppo-pivot-production-monitor"
HUB = "http://127.0.0.1:8090"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def get_json(url: str, path: str) -> Any:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=10) as response:
        return json.loads(response.read())


def record(status: str, **extra: Any) -> None:
    save_json(PRODUCTION / "launcher_status.json", {"updated_utc": utc(), "status": status, **extra})


def wait_for_handoff(interval: float) -> dict[str, Any] | None:
    handoff_path = PIVOT / "handoff.json"
    while True:
        state_path = PIVOT / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("phase") == "complete" and state.get("decision", {}).get("handoff_eligible") is False:
            if all((state.get("trials") or {}).get(n, {}).get("evaluation") for n in ("trial-1", "trial-2")):
                record("no_eligible_candidate_parent_control_pending", pivot_decision=state.get("decision"))
                return {"_no_eligible_candidate": True}
        if state.get("phase") == "complete" and handoff_path.is_file():
            return json.loads(handoff_path.read_text())
        if state.get("phase", "").startswith("halted"):
            record("pivot_halted_without_handoff", pivot_phase=state.get("phase"))
            return None
        time.sleep(interval)


def run_parent_control_diagnostic(hub_url: str, poll_seconds: float) -> None:
    """Run at most one fresh Porto 3x3 on the pivot's immutable common parent."""
    from scripts import run_speed_campaign as speed

    pivot_state = speed.read_json(PIVOT / "state.json")
    if pivot_state.get("phase") != "complete" or pivot_state.get("decision", {}).get("handoff_eligible") is not False:
        raise RuntimeError("parent control is permitted only after both pivot trials complete without a winner")
    trials = pivot_state.get("trials") or {}
    if any((trials.get(n) or {}).get("evaluation", {}).get("clean_gate")
           and speed.finite((trials.get(n) or {}).get("evaluation", {}).get("median_flying_lap_s"))
           and (trials.get(n) or {}).get("evaluation", {}).get("median_flying_lap_s") <= 24.768
           for n in ("trial-1", "trial-2")):
        raise RuntimeError("a pivot candidate qualifies; refusing parent-control fallback")
    parent = Path(pivot_state["parent_checkpoint"]).resolve()
    parent_hash = pivot_state["parent_sha256"]
    if not parent.is_file() or sha256(parent) != parent_hash:
        raise RuntimeError("recorded pivot parent is missing or its hash changed")
    if speed.step_of(parent) != 1195886 or speed.verified_checkpoint(parent.parent.parent, 1195886) != parent:
        raise RuntimeError("recorded 1,195,886-step parent failed checkpoint audit")
    diagnostic_dir = PRODUCTION / "parent_control_diagnostic"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    intent_path = diagnostic_dir / "intent.json"
    result_path = diagnostic_dir / "result.json"
    binding = {"schema_version": 1, "purpose": "single fresh parent control after both pivot trials fail",
              "parent_checkpoint": str(parent), "parent_sha256": parent_hash, "validation_mode": "porto_practice_diagnostic",
              "attempts": 3, "laps_per_attempt": 3, "created_utc": utc()}
    if result_path.exists():
        record("parent_control_already_recorded", result=str(result_path))
        return
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        if any(intent.get(k) != v for k, v in binding.items() if k != "created_utc"):
            raise RuntimeError("existing parent-control intent is bound to different evidence")
    else:
        hub_idle(hub_url)
        save_json(intent_path, binding)
        intent = binding
    api = speed.API(hub_url)
    evaluation_id = intent.get("evaluation_id")
    if not evaluation_id:
        def fresh_matches() -> list[dict[str, Any]]:
            history = api.get("/api/evaluations/history")
            current = api.get("/api/replay/status")
            cutoff = datetime.fromisoformat(intent["created_utc"])
            observed = []
            for candidate in [current, *history]:
                if not isinstance(candidate, dict):
                    continue
                stamp = candidate.get("started_at") or candidate.get("created_at") or candidate.get("started_utc")
                try:
                    observed_at = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    continue
                if (observed_at < cutoff or candidate.get("validation_mode") != "porto_practice_diagnostic"
                        or not candidate.get("model_path", candidate.get("model"))
                        or speed.model_path(candidate.get("model_path", candidate.get("model"))) != parent
                        or (candidate.get("model_sha256") or (candidate.get("result") or {}).get("model_sha256")) != parent_hash):
                    continue
                observed.append(candidate)
            # Current status and history may describe the same evaluation.
            unique = {str(r.get("evaluation_id")): r for r in observed if r.get("evaluation_id")}
            return list(unique.values())

        observed = fresh_matches()
        if len(observed) > 1:
            raise RuntimeError("multiple fresh matching parent diagnostics exist; refusing ambiguity")
        if observed:
            evaluation_id = observed[0].get("evaluation_id")
            intent["evaluation_id"] = evaluation_id
            intent["reconciled"] = True
            save_json(intent_path, intent)
        elif intent.get("post_started"):
            # A prior process may have lost the POST response. Reconcile only; never submit twice.
            while not observed:
                time.sleep(poll_seconds)
                observed = fresh_matches()
            if len(observed) != 1:
                raise RuntimeError("ambiguous parent-control POST reconciliation")
            evaluation_id = observed[0].get("evaluation_id")
            intent["evaluation_id"] = evaluation_id
            intent["reconciled"] = True
            save_json(intent_path, intent)
        else:
            hub_idle(hub_url)
            intent["post_started"] = True
            intent["post_started_utc"] = utc()
            save_json(intent_path, intent)
            model_id = parent.relative_to((ROOT / "logs/rl").resolve()).as_posix()
            response = api.post("/api/replay/start", {"model_id": model_id, "map_id": "none", "device": "cpu",
                                                      "trace_steps": 500, "validation_mode": "porto_practice_diagnostic"})
            evaluation_id = response.get("evaluation_id") if isinstance(response, dict) else None
            if not evaluation_id:
                # Leave intent unresolved. A restart will reconcile history and will never blindly repost.
                raise RuntimeError("parent-control evaluation POST returned no ID; intent retained for reconciliation")
            intent["evaluation_id"] = evaluation_id
            save_json(intent_path, intent)
    while True:
        history = api.get("/api/evaluations/history")
        row = next((r for r in history if isinstance(r, dict) and r.get("evaluation_id") == evaluation_id
                    and r.get("state") in speed.TERMINAL_EVALS), None)
        if row is None:
            current = api.get("/api/replay/status")
            if current.get("evaluation_id") == evaluation_id and current.get("state") in speed.TERMINAL_EVALS:
                row = current
        if row is not None:
            break
        time.sleep(poll_seconds)
    actual = row.get("model_path", row.get("model"))
    actual_hash = row.get("model_sha256") or (row.get("result") or {}).get("model_sha256")
    if (row.get("validation_mode") != "porto_practice_diagnostic" or not actual
            or speed.model_path(actual) != parent or actual_hash != parent_hash):
        raise RuntimeError("parent-control result failed model identity/hash audit")
    summary = speed.summarize(row)
    trial_summaries = {n: (trials.get(n) or {}).get("evaluation") for n in ("trial-1", "trial-2")}
    result = {"completed_utc": utc(), "intent": intent, "evaluation_id": evaluation_id,
              "parent_checkpoint": str(parent), "parent_sha256": parent_hash,
              "checkpoint_lineage_verified": True, "evaluation_identity_verified": True,
              "validation_mode": row.get("validation_mode"), "parent_control": summary,
              "pivot_trials": trial_summaries,
              "interpretation": "parent_clean_baseline_available; compare candidate failures against parent timing" if summary.get("clean_gate")
                               else "parent_also_failed_clean_gate; simulator/readiness conditions may explain trial failures",
              "raw_evaluation": row}
    save_json(result_path, result)
    record("parent_control_diagnostic_complete", result=str(result_path), clean_gate=summary.get("clean_gate"),
           median_flying_lap_s=summary.get("median_flying_lap_s"))


def audit_handoff(handoff: dict[str, Any], hub_url: str) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    from scripts import run_speed_campaign as speed

    if handoff.get("schema_version") != 1 or handoff.get("source") != "run_ppo_pivot":
        raise RuntimeError("unexpected handoff schema or source")
    if handoff.get("launch_authorized") is not False:
        raise RuntimeError("handoff authorization marker does not match pivot contract")
    if handoff.get("supervisor") != "scripts/supervise_speed_campaign.py":
        raise RuntimeError("handoff names an unexpected supervisor")
    selected = handoff.get("selected_trial")
    pivot_state = speed.read_json(PIVOT / "state.json")
    decision = pivot_state.get("decision") or {}
    if selected not in {"trial-1", "trial-2"} or decision.get("selected_trial") != selected or not decision.get("handoff_eligible"):
        raise RuntimeError("handoff selection does not match the completed pivot decision")
    trial = (pivot_state.get("trials") or {}).get(selected)
    if not isinstance(trial, dict) or not trial.get("completed"):
        raise RuntimeError("selected trial lacks completed persisted training evidence")
    checkpoint = Path(handoff["checkpoint"]).resolve()
    if checkpoint != Path(trial.get("checkpoint", "")).resolve():
        raise RuntimeError("handoff checkpoint differs from the selected trial checkpoint")
    expected_sha = trial.get("checkpoint_sha256")
    if not expected_sha or sha256(checkpoint) != expected_sha or handoff.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError("selected checkpoint hash does not match pivot and handoff records")
    config_hash = hashlib.sha256(json.dumps(trial.get("config"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if trial.get("config_sha256") != config_hash:
        raise RuntimeError("persisted selected trial config hash is invalid")
    if speed.verified_checkpoint(checkpoint.parent.parent, speed.step_of(checkpoint)) != checkpoint:
        raise RuntimeError("selected checkpoint failed optimizer audit or PPO lineage verification")
    if trial.get("parent_checkpoint") != pivot_state.get("parent_checkpoint") or trial.get("parent_sha256") != pivot_state.get("parent_sha256"):
        raise RuntimeError("trial parent is not the pivot's common parent")
    parent = Path(pivot_state["parent_checkpoint"]).resolve()
    if sha256(parent) != pivot_state["parent_sha256"]:
        raise RuntimeError("common parent checkpoint hash changed")
    parent_verified = speed.verified_checkpoint(parent.parent.parent, speed.step_of(parent))
    if parent_verified != parent:
        raise RuntimeError("common parent failed optimizer audit or PPO lineage verification")
    run_inventory = get_json(hub_url, "/train/runs").get("runs", [])
    matching = [r for r in run_inventory if r.get("run_id") == trial.get("run_id")]
    if len(matching) != 1 or matching[0].get("state") != "completed":
        raise RuntimeError("selected trial is not uniquely completed in hub inventory")
    actual = speed.comparable_config(matching[0].get("config") or {})
    expected = speed.comparable_config(trial.get("config") or {})
    if any(actual.get(k) != v for k, v in expected.items()):
        raise RuntimeError("selected trial hub config differs from persisted trial config")
    # Reconfirm the handoff's measured gate from raw attempt records.
    validation = handoff.get("validation") or {}
    summary = speed.summarize({"state": validation.get("state"), "validation_mode": validation.get("validation_mode"),
                               "evaluation_id": validation.get("evaluation_id"), "result": {"readiness_passed": validation.get("readiness_passed"),
                               "runs": validation.get("attempts_detail", [])}})
    pace = handoff.get("selection_pace_s")
    if (not summary["clean_gate"] or validation.get("clean_gate") is not True or not speed.finite(pace)
            or pace > 24.768 or pace != validation.get("median_flying_lap_s") or pace != decision.get("selection_pace_s", pace)):
        raise RuntimeError("handoff does not independently satisfy the clean 3x3 <=24.768s gate")
    if validation.get("validation_mode") != "porto_practice_diagnostic":
        raise RuntimeError("winner evaluation is not Porto practice diagnostic mode")
    evaluation_id = validation.get("evaluation_id")
    history = get_json(hub_url, "/api/evaluations/history")
    eval_rows = [r for r in history if isinstance(r, dict) and r.get("evaluation_id") == evaluation_id]
    if len(eval_rows) != 1:
        raise RuntimeError("winner evaluation is not uniquely present in hub history")
    evaluation = eval_rows[0]
    actual_model = evaluation.get("model_path", evaluation.get("model"))
    actual_model_sha = evaluation.get("model_sha256") or (evaluation.get("result") or {}).get("model_sha256")
    if (evaluation.get("state") not in speed.TERMINAL_EVALS
            or evaluation.get("validation_mode") != "porto_practice_diagnostic"
            or not actual_model or speed.model_path(actual_model) != checkpoint
            or actual_model_sha != expected_sha):
        raise RuntimeError("hub evaluation identity, terminal state, or checkpoint hash audit failed")
    config = handoff.get("training_config")
    if not isinstance(config, dict) or speed.comparable_config(config).get("resume") != str(checkpoint):
        raise RuntimeError("handoff training config does not resume from the selected checkpoint")
    allowed_changes = {"resume", "n_envs", "seed", "total_timesteps", "checkpoint_every",
                       "automatic_evaluation_enabled", "cooperative_pause_enabled", "stop_on_plateau",
                       "max_duration_hours"}
    normalized_config = speed.comparable_config(config)
    normalized_trial = speed.comparable_config(trial["config"])
    mismatch = [k for k, v in normalized_trial.items()
                if k not in allowed_changes and normalized_config.get(k) != v]
    extra = [k for k in normalized_config if k not in normalized_trial and k not in allowed_changes]
    if mismatch or extra:
        raise RuntimeError(f"handoff config differs from selected trial beyond allowed continuation fields: {mismatch + extra}")
    if config.get("total_timesteps") != 0 or config.get("n_envs") != 6:
        raise RuntimeError("handoff training config is not an unlimited six-environment continuation")
    # The reward parameters in the selected trial config become its production override.
    reward = {k: config[k] for k in ("frontier_pace_bonus_strength", "frontier_pace_target_mps", "time_cost_per_simulated_second") if k in config}
    if len(reward) != 3:
        raise RuntimeError("selected trial handoff omits production reward overrides")
    return pivot_state, trial, checkpoint, json.dumps(reward, sort_keys=True)


def hub_idle(hub_url: str) -> None:
    inventory = get_json(hub_url, "/train/runs").get("runs", [])
    active_training = [r.get("run_id") for r in inventory if r.get("state") in {"queued", "starting", "running", "stopping"}]
    replay = get_json(hub_url, "/api/replay/status")
    if active_training or replay.get("state") in {"queued", "running", "stopping"}:
        raise RuntimeError(f"hub is not idle (training={active_training}, evaluation={replay.get('state')})")


def seed_and_launch(handoff: dict[str, Any], pivot_state: dict[str, Any], trial: dict[str, Any], checkpoint: Path,
                    hub_url: str) -> None:
    from scripts import run_speed_campaign as speed

    PRODUCTION.mkdir(parents=True, exist_ok=True)
    lock_path = PRODUCTION / "launcher.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = PRODUCTION / "state.json"
        if state_path.exists():
            state = speed.read_json(state_path)
            if state.get("pivot_handoff_sha256") != sha256(PIVOT / "handoff.json"):
                raise RuntimeError("production state already exists with a different handoff")
        else:
            hub_idle(hub_url)
            base = speed.read_json(BASE_CONFIG)
            effective = dict(handoff["training_config"])
            effective.update({"resume": str(checkpoint), "n_envs": 6, "seed": 1, "total_timesteps": 0,
                              "checkpoint_every": 12288, "automatic_evaluation_enabled": False,
                              "cooperative_pause_enabled": False, "stop_on_plateau": False, "max_duration_hours": 0,
                              "centerline_shaping_strength": 0, "collision_penalty_magnitude": 250,
                              "failed_episode_penalty": 250})
            name = handoff["selected_trial"]
            reward = {k: effective[k] for k in ("frontier_pace_bonus_strength", "frontier_pace_target_mps", "time_cost_per_simulated_second") if k in effective}
            candidates = {c["name"]: {"reward": c} for c in speed.CANDIDATES}
            candidate_name = f"ppo-pivot-{name}"
            candidates[candidate_name] = {"reward": {"name": candidate_name, **reward}, "checkpoints": {"screen": str(checkpoint)},
                                          "screen_eval": handoff["validation"]}
            run_root = speed.resolve_outdir((pivot_state["trials"][name].get("output_dir") or
                next((r.get("output_dir") for r in get_json(hub_url, "/train/runs").get("runs", [])
                      if r.get("run_id") == trial.get("run_id")), "")))
            run_record = {"checkpoint": str(checkpoint), "output_dir": str(run_root), "run_id": trial.get("run_id")}
            state = {"version": 1, "created_utc": utc(), "phase": "production",
                     "adaptation_run_id": trial.get("run_id"), "base_config": str(BASE_CONFIG.resolve()),
                     "base_config_sha256": sha256(BASE_CONFIG), "effective_base_config": effective,
                     "adapted_checkpoint": str(Path(pivot_state["parent_checkpoint"]).resolve()),
                     "adapted_sha256": pivot_state["parent_sha256"], "adaptation_config": trial["config"],
                     "runs": {f"screen:{candidate_name}": run_record}, "evaluations": {}, "candidates": candidates,
                     "confirmation": {}, "decision": {"selected": candidate_name, "selected_checkpoint": str(checkpoint),
                         "improvement_confirmed": False, "production_eligible": True, "source": "qualified PPO pivot",
                         "production_validation_status": "fresh_clean_3x3_pivot_gate"},
                     "production": {"resume": str(checkpoint), "cycle": 0},
                     "production_training_evidence": {"clean_ten_lap_episode": False,
                         "source": "fresh audited Porto practice 3x3 pivot evaluation",
                         "clean_3x3_gate": True, "validation_mode": "porto_practice_diagnostic",
                         "evaluation_id": handoff["validation"].get("evaluation_id"),
                         "median_flying_lap_s": handoff["selection_pace_s"],
                         "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)},
                     "production_eval_interval_transitions": 49152,
                     "continue_after_failed_production_eval": True, "alerts": [],
                     "pivot_handoff_sha256": sha256(PIVOT / "handoff.json"),
                     "pivot_handoff": str(PIVOT / "handoff.json"), "pivot_validation": handoff["validation"]}
            save_json(state_path, state)
            save_json(PRODUCTION / "effective_base_config.json", effective)
            save_json(PRODUCTION / "report.json", {"created_utc": state["created_utc"], "phase": state["phase"],
                      "adapted_checkpoint": state["adapted_checkpoint"], "adapted_sha256": state["adapted_sha256"],
                      "effective_base_config": effective, "decision": state["decision"],
                      "production": state["production"], "production_eval_interval_transitions": 49152,
                      "continue_after_failed_production_eval": True,
                      "production_training_evidence": state["production_training_evidence"],
                      "pivot_handoff": str(PIVOT / "handoff.json"),
                      "pivot_handoff_sha256": state["pivot_handoff_sha256"]})
            record("production_state_seeded", selected_trial=name, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        # Recheck idle immediately before creating production workers.
        hub_idle(hub_url)
        sessions = subprocess.check_output(["tmux", "list-sessions", "-F", "#S"], text=True).splitlines()
        if TMUX_SUPERVISOR in sessions or TMUX_MONITOR in sessions:
            record("duplicate_worker_session_detected", sessions=sessions)
            return
        root = str(ROOT)
        supervisor_cmd = (f"cd {root} && exec bash scripts/supervise_speed_campaign_native.sh "
                          f"--campaign-dir logs/rl/ppo_pivot_production_20261111 "
                          f"--adaptation-run-id {trial['run_id']} "
                          f"--base-config logs/rl/reward_sweep_966510/base_config.json "
                          f"--hub-url {hub_url} --continue-after-failed-production-eval "
                          f"--log logs/rl/ppo_pivot_production_20261111/supervisor.log")
        monitor_cmd = (f"cd {root} && exec python3 -u scripts/monitor_speed_campaign.py "
                       f"--campaign-dir logs/rl/ppo_pivot_production_20261111 --hub-url {hub_url} --interval 30")
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SUPERVISOR, "bash", "-lc", supervisor_cmd], check=True)
        try:
            subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_MONITOR, "bash", "-lc", monitor_cmd], check=True)
        except Exception:
            subprocess.run(["tmux", "kill-session", "-t", TMUX_SUPERVISOR], check=False)
            raise
        record("production_supervisor_and_monitor_started", supervisor_session=TMUX_SUPERVISOR,
               monitor_session=TMUX_MONITOR, campaign_dir=str(PRODUCTION))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-url", default=HUB)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")
    PRODUCTION.mkdir(parents=True, exist_ok=True)
    try:
        record("waiting_for_pivot_handoff", pivot_dir=str(PIVOT), handoff=str(PIVOT / "handoff.json"))
        handoff = wait_for_handoff(args.poll_seconds)
        if handoff is None:
            return 0
        if handoff.get("_no_eligible_candidate"):
            run_parent_control_diagnostic(args.hub_url, args.poll_seconds)
            return 0
        pivot_state, trial, checkpoint, _ = audit_handoff(handoff, args.hub_url)
        seed_and_launch(handoff, pivot_state, trial, checkpoint, args.hub_url)
        return 0
    except Exception as exc:
        record("error", error=f"{type(exc).__name__}: {exc}")
        print(f"launch_ppo_pivot_winner: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
