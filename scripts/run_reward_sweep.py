#!/usr/bin/env python3
"""Durable, fail-closed Porto reward sweep orchestrator.

This script is intentionally an orchestrator only; it never executes training
unless --execute is supplied. State and reports live under --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import fcntl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / "logs/rl/official_run_11e94f5da0ce/checkpoints/post_optimizer_update_00000015_steps_000000966510.zip"
PARENT_SHA256 = "8527be92e5e08b2d9edd323b47c0846e87cac6574f620f7681b540b5a7c215b5"
BASELINE_ID_DEFAULT = "2b978ace17ff4964"
CANDIDATES = [
    {"name": f"pace-{pace:g}_time-{cost:g}", "frontier_pace_bonus_strength": pace,
     "time_cost_per_simulated_second": cost, "centerline_shaping_strength": 0.0}
    for pace in (0.0, 0.5, 1.0) for cost in (0.0, 0.02)
]
TERMINAL_RUNS = {"completed", "failed", "stopped"}
TERMINAL_EVALS = {"complete", "completed", "failed", "incomplete", "stopped"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except (AttributeError, OSError):
        pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class API:
    def __init__(self, base: str, timeout: float = 240):
        self.base, self.timeout = base.rstrip("/"), timeout

    def request(self, path: str, payload: dict | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"hub request {path} failed: {exc}") from exc
        return result


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def comparable_config(config: dict) -> dict:
    result = dict(config)
    value = result.get("resume")
    if isinstance(value, str):
        for prefix in ("/runs/", "/app/logs/"):
            if value.startswith(prefix):
                value = str(ROOT / "logs" / value.removeprefix(prefix)); break
        result["resume"] = str(Path(value).resolve())
    return result


def set_base_config(config: dict, checkpoint: Path) -> dict:
    result = dict(config)
    result.update({"resume": str(checkpoint.resolve()), "n_envs": 2,
        "n_steps": 2048, "learning_rate": 1e-5, "checkpoint_every": 4096,
        "seed": 0, "automatic_evaluation_enabled": False,
        "automatic_evaluation_mode": "competitive",
        "cooperative_pause_enabled": False, "total_timesteps": 0,
        "stop_on_plateau": False, "max_duration_hours": 0,
        "collision_penalty_magnitude": 250.0, "failed_episode_penalty": 250.0,
        "centerline_shaping_strength": 0.0,
        "observation_profile": "official_sensors_camera"})
    if result.get("time_cost_per_simulated_second") is None:
        result["time_cost_per_simulated_second"] = 0.0
    return result


def finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def probe_summary(row: dict) -> dict:
    result = row.get("latest_result") or row.get("result") or {}
    runs = result.get("runs") or row.get("attempts") or []
    lap_values: list[float] = []
    standing: list[float] = []
    flying: list[float] = []
    collisions = 0
    all_clean = True
    for attempt in runs:
        if not isinstance(attempt, dict): all_clean = False; continue
        laps = attempt.get("race_lap_times_s") or []
        if not isinstance(laps, list): laps = []
        clean_laps = [float(x) for x in laps if finite_number(x) and float(x) > 0]
        lap_values.extend(clean_laps)
        if clean_laps:
            standing.append(clean_laps[0])
            flying.extend(clean_laps[1:])
        count = attempt.get("collision_count")
        if count is None:
            count = attempt.get("race_collisions", attempt.get("raw_collision_count_at_finish"))
        if not finite_number(count): all_clean = False
        else: collisions += int(count)
        all_clean &= attempt.get("stop_reason") in {"race_complete", "evaluation_complete"}
        all_clean &= len(clean_laps) == 3
    ready = result.get("readiness_passed") is True
    nine = len(lap_values) == 9 and len(runs) == 3
    return {"evaluation_id": row.get("evaluation_id"), "state": row.get("state"),
        "validation_mode": row.get("validation_mode"), "readiness_passed": ready,
        "attempts": len(runs), "scored_laps": len(lap_values), "collisions": collisions,
        "clean_gate": bool(ready and nine and all_clean and collisions == 0),
        "median_all_laps_s": sorted(lap_values)[len(lap_values)//2] if len(lap_values)==9 else None,
        "median_standing_lap_s": sorted(standing)[len(standing)//2] if len(standing)==3 else None,
        "median_flying_lap_s": statistics.median(flying) if len(flying)==6 else None,
        "attempts_detail": runs}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=PARENT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hub-url", default="http://127.0.0.1:8090")
    p.add_argument("--baseline-evaluation-id", default=BASELINE_ID_DEFAULT)
    p.add_argument("--execute", action="store_true", help="allow hub mutations")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve()
    base_path = args.base_config.resolve()
    out = args.output.resolve()
    try:
        if not checkpoint.is_file(): raise ValueError(f"checkpoint missing: {checkpoint}")
        digest = sha256(checkpoint)
        if checkpoint == PARENT.resolve() and digest != PARENT_SHA256:
            raise ValueError("fixed parent checkpoint SHA-256 mismatch")
        base_raw = load_json(base_path)
        base = set_base_config(base_raw, checkpoint)
        if base_raw.get("collision_penalty_magnitude") != 250 or base_raw.get("centerline_shaping_strength") != 0:
            raise ValueError("base config must declare collision penalty 250 and centerline shaping 0")
        if not args.execute:
            print(json.dumps({"mutation": False, "parent": str(checkpoint), "sha256": digest,
                "candidate_count": 6, "base_config": base, "baseline_evaluation_id": args.baseline_evaluation_id}, indent=2))
            return 0
        if not args.baseline_evaluation_id:
            raise ValueError("--execute requires --baseline-evaluation-id for the already-launched Porto baseline")
        if args.poll_seconds < 1: raise ValueError("poll interval must be >=1s")
        out.mkdir(parents=True, exist_ok=True)
        lock_file = (out / "runner.lock").open("a")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / "state.json"
        report_path = out / "report.json"
        if state_path.exists():
            state = load_json(state_path)
            if state.get("parent_sha256") != digest or state.get("baseline_evaluation_id") != args.baseline_evaluation_id:
                raise ValueError("existing state is bound to a different parent or baseline evaluation")
        else:
            state = {"version": 1, "created_utc": utc(), "updated_utc": utc(),
                "phase": "baseline", "parent_checkpoint": str(checkpoint), "parent_sha256": digest,
                "baseline_evaluation_id": args.baseline_evaluation_id,
                "base_config": base, "candidates": {c["name"]: {"reward": c} for c in CANDIDATES},
                "run_intents": {}, "evaluations": {}, "decision": None}
            atomic_json(state_path, state)
        api = API(args.hub_url)

        def save():
            state["updated_utc"] = utc(); atomic_json(state_path, state)
            atomic_json(report_path, {"updated_utc": state["updated_utc"], "phase": state["phase"],
                "decision": state.get("decision"), "baseline": state.get("baseline"),
                "candidates": state.get("candidates"), "confirmation": state.get("confirmation"),
                "limitation": state.get("limitation")})

        def wait_eval(eid: str) -> dict:
            while True:
                history = api.request("/api/evaluations/history")
                if not isinstance(history, list): raise RuntimeError("invalid evaluation history response")
                row = next((x for x in history if isinstance(x, dict) and x.get("evaluation_id") == eid), None)
                if row and row.get("state") in TERMINAL_EVALS: return row
                status = api.request("/api/replay/status")
                if not isinstance(status, dict): raise RuntimeError("invalid replay status response")
                log_resource(out, "evaluation", eid, {"run_id": status.get("run_id"),
                    "state": status.get("state"), "fleet": status.get("last_fleet")})
                if status.get("evaluation_id") == eid and status.get("state") in TERMINAL_EVALS: return status
                time.sleep(args.poll_seconds)

        baseline_row = wait_eval(args.baseline_evaluation_id)
        baseline_path = baseline_row.get("model_path", baseline_row.get("model"))
        baseline_digest = baseline_row.get("model_sha256") or (baseline_row.get("result") or {}).get("model_sha256")
        if not baseline_path or Path(baseline_path).resolve() != checkpoint or baseline_digest != digest:
            raise RuntimeError("baseline evaluation does not authenticate the common parent checkpoint")
        if state["phase"] == "baseline":
            summary = probe_summary(baseline_row)
            if baseline_row.get("validation_mode") != "porto_practice_diagnostic" or not summary["clean_gate"]:
                state["baseline"] = summary; state["phase"] = "halted_baseline_gate"; save()
                raise RuntimeError("baseline Porto practice diagnostic failed its 3x3 clean readiness gate; halted")
            state["baseline"] = summary; state["phase"] = "screen_segment_1"; save()

        def run_group(stage: str, candidates: list[str], transitions: int, resume_for: dict[str, str]):
            # Persist each intent before POST. On restart, exact-name reconciliation
            # adopts one matching run; ambiguity or lost POST outcome fails closed.
            for name in candidates:
                key = f"{stage}:{name}"
                intent = state["run_intents"].get(key)
                if intent is None:
                    config = dict(base)
                    config.update({k:v for k,v in state["candidates"][name]["reward"].items() if k != "name"})
                    config.update({"resume": resume_for[name], "total_timesteps": transitions,
                        "seed": 1 if stage == "confirm" else 0, "n_envs": 2, "automatic_evaluation_enabled": False,
                        "cooperative_pause_enabled": False, "stop_on_plateau": False,
                        "max_duration_hours": 0, "checkpoint_every": 4096})
                    display = f"reward-sweep-{out.name}-{stage}-{name}"
                    intent = {"display_name": display, "config": config, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), "post_started": False}
                    state["run_intents"][key] = intent; save()
                    runs = api.request("/train/runs")
                    if not isinstance(runs, dict) or not isinstance(runs.get("runs"), list): raise RuntimeError("invalid training runs response")
                    matches = [r for r in runs["runs"] if r.get("display_name") == display]
                    if matches: raise RuntimeError(f"unexpected pre-existing run name {display}; inspect before continuing")
                    intent["post_started"] = True; save()
                    created = api.request("/train/runs", {"config": config, "display_name": display})
                    rid = created.get("run_id") if isinstance(created, dict) else None
                    if not isinstance(rid, str) or not rid: raise RuntimeError("run creation returned no run_id")
                    intent["run_id"] = rid; save()
                elif not intent.get("run_id"):
                    runs = api.request("/train/runs")
                    matches = [r for r in runs.get("runs", []) if r.get("display_name") == intent["display_name"]]
                    if len(matches) != 1: raise RuntimeError(f"ambiguous/lost create outcome for {intent['display_name']}; found {len(matches)}")
                    actual = matches[0].get("config") or {}
                    expected = comparable_config(intent["config"])
                    actual = comparable_config(actual)
                    if any(actual.get(k) != v for k,v in expected.items()):
                        raise RuntimeError("reconciled run config mismatch")
                    intent["run_id"] = matches[0]["run_id"]; save()
            active = {state["run_intents"][f"{stage}:{n}"]["run_id"]: n for n in candidates}
            while active:
                for rid, name in list(active.items()):
                    row = api.request(f"/train/runs/{rid}")
                    expected_config = comparable_config(state["run_intents"][f"{stage}:{name}"]["config"])
                    actual_config = comparable_config(row.get("config") or {})
                    mismatches = [k for k,v in expected_config.items() if actual_config.get(k) != v]
                    if mismatches:
                        raise RuntimeError(f"run {rid}: effective config mismatch: {mismatches}")
                    if row.get("state") in TERMINAL_RUNS:
                        if row.get("state") != "completed": raise RuntimeError(f"run {rid} ended {row.get('state')}: {row.get('error')}")
                        state["candidates"][name].setdefault("segments", {})[stage] = row
                        cp = newest_checkpoint(row.get("output_dir"), checkpoint)
                        expected = checkpoint_steps(Path(resume_for[name])) + transitions
                        if checkpoint_steps(cp) != expected:
                            raise RuntimeError(f"run {rid}: expected checkpoint step {expected}, got {cp}")
                        audits = [json.loads(x) for x in (cp.parent.parent / "ppo_optimizer_audit.jsonl").read_text().splitlines() if x.strip()]
                        if len(audits) != transitions // 4096:
                            raise RuntimeError(f"run {rid}: optimizer update count mismatch")
                        state["candidates"][name].setdefault("checkpoints", {})[stage] = str(cp)
                        del active[rid]; save()
                    else:
                        log_resource(out, stage, name, row)
                time.sleep(args.poll_seconds)

        def run_batch(stage, candidates, transitions, resume_for):
            for offset in range(0, len(candidates), 3):
                run_group(stage, candidates[offset:offset+3], transitions, resume_for)

        def evaluate(stage: str, names: list[str], destination: str):
            records = state.setdefault(destination, {})
            for name in names:
                record = records.get(name)
                cp = Path(state["candidates"][name]["checkpoints"][stage]).resolve()
                if record and record.get("result"):
                    continue
                if record and not record.get("evaluation_id"):
                    # Reconcile an accepted request whose HTTP reply was lost.
                    status = api.request("/api/replay/status")
                    history = api.request("/api/evaluations/history")
                    matches = [x for x in [status, *history] if x.get("model_path", x.get("model")) == str(cp)
                               and x.get("validation_mode") == "porto_practice_diagnostic"]
                    ids = {x["evaluation_id"] for x in matches}
                    if len(ids) != 1:
                        raise RuntimeError(f"cannot safely reconcile evaluation intent for {name}")
                    record["evaluation_id"] = ids.pop(); save()
                if not record:
                    if api.request("/api/replay/status").get("state") in {"queued", "running", "stopping"}:
                        raise RuntimeError("another evaluation is active; refusing to overlap")
                    records[name] = {"intent": True, "checkpoint": str(cp)}; save()
                    response = api.request("/api/replay/start", {
                        "model_id": cp.relative_to(ROOT / "logs/rl").as_posix(),
                        "validation_mode": "porto_practice_diagnostic", "trace_steps": 500})
                    record = records[name]
                    record["evaluation_id"] = response["evaluation_id"]; save()
                record["result"] = probe_summary(wait_eval(record["evaluation_id"])); save()

        if state["phase"] == "screen_segment_1":
            run_batch("screen1", [c["name"] for c in CANDIDATES], 24576,
                {c["name"]: str(checkpoint) for c in CANDIDATES})
            state["phase"] = "midpoint_evaluations"; save()
        if state["phase"] == "midpoint_evaluations":
            evaluate("screen1", [c["name"] for c in CANDIDATES], "midpoint")
            state["phase"] = "screen_segment_2"; save()
        if state["phase"] == "screen_segment_2":
            run_batch("screen2", [c["name"] for c in CANDIDATES], 28672,
                {c["name"]: state["candidates"][c["name"]]["checkpoints"]["screen1"] for c in CANDIDATES})
            state["phase"] = "screen_evaluations"; save()
        if state["phase"] == "screen_evaluations":
            evaluate("screen2", [c["name"] for c in CANDIDATES], "evaluations")
            for c in CANDIDATES:
                name = c["name"]
                ev = state["evaluations"].get(name)
                if not ev:
                    cp = Path(state["candidates"][name]["checkpoints"]["screen2"]).resolve()
                    model_id = cp.relative_to((ROOT / "logs/rl").resolve()).as_posix()
                    intent_key = f"screen:{name}"
                    state["evaluations"][name] = {"intent": True, "checkpoint": str(cp)}; save()
                    if api.request("/api/replay/status").get("state") in {"queued", "running", "stopping"}:
                        raise RuntimeError("replay queue/status is busy; state retained; rerun to continue after it drains")
                    response = api.request("/api/replay/start", {"model_id": model_id, "map_id": "none", "device": "cpu", "trace_steps": 500, "validation_mode": "porto_practice_diagnostic"})
                    eid = response.get("evaluation_id") if isinstance(response, dict) else None
                    if not isinstance(eid, str): raise RuntimeError("probe start returned no evaluation_id")
                    state["evaluations"][name] = {"evaluation_id": eid, "checkpoint": str(cp)}; save(); ev = state["evaluations"][name]
                row = wait_eval(ev["evaluation_id"])
                summary = probe_summary(row)
                state["evaluations"][name]["result"] = summary; save()
            state["phase"] = "confirmation_selection"; save()

        if state["phase"] == "confirmation_selection":
            eligible = []
            for c in CANDIDATES:
                name = c["name"]; result = state["evaluations"][name]["result"]
                if result["clean_gate"] and finite_number(result["median_all_laps_s"]): eligible.append((result["median_all_laps_s"], name))
            eligible.sort()
            state["confirmation_candidates"] = [name for _, name in eligible[:2]]
            control = CANDIDATES[0]["name"]
            if eligible and control not in state["confirmation_candidates"]:
                state["confirmation_candidates"].append(control)
            state["phase"] = "confirmation_training" if eligible else "decision"
            if not eligible:
                state["decision"] = {"status": "no_clean_candidate", "selected_checkpoint": str(checkpoint), "reason": "No candidate passed three-attempt, nine-lap clean diagnostic."}
            save()

        if state["phase"] == "confirmation_training":
            names = state["confirmation_candidates"]
            # Independent continuation from common parent, seed 1.
            run_batch("confirm", names, 106496, {name: str(checkpoint) for name in names})
            state["phase"] = "confirmation_evaluations"; save()

        if state["phase"] == "confirmation_evaluations":
            evaluate("confirm", state["confirmation_candidates"], "confirmation")
            state.setdefault("confirmation", {})
            for name in state["confirmation_candidates"]:
                intent = state["confirmation"].get(name)
                if not intent:
                    cp = Path(state["candidates"][name]["checkpoints"]["confirm"]).resolve()
                    model_id = cp.relative_to((ROOT / "logs/rl").resolve()).as_posix()
                    if api.request("/api/replay/status").get("state") in {"queued", "running", "stopping"}:
                        raise RuntimeError("evaluation queue is busy; durable phase retained")
                    state["confirmation"][name] = {"intent": True, "checkpoint": str(cp)}; save()
                    response = api.request("/api/replay/start", {"model_id": model_id, "map_id": "none", "device": "cpu", "trace_steps": 500, "validation_mode": "porto_practice_diagnostic"})
                    eid = response.get("evaluation_id") if isinstance(response, dict) else None
                    if not isinstance(eid, str): raise RuntimeError("confirmation probe returned no evaluation_id")
                    state["confirmation"][name]["evaluation_id"] = eid; save(); intent = state["confirmation"][name]
                row = wait_eval(intent["evaluation_id"])
                intent["result"] = probe_summary(row); save()
            state["phase"] = "decision"; save()

        if state["phase"] == "decision":
            baseline = state["baseline"]
            choices = []
            for name in state.get("confirmation_candidates", []):
                r = state.get("confirmation", {}).get(name, {}).get("result", {})
                if r.get("clean_gate") and finite_number(r.get("median_all_laps_s")):
                    choices.append((float(r["median_all_laps_s"]), name))
            choices.sort()
            base_med = baseline.get("median_all_laps_s")
            winner = choices[0] if choices else None
            control_name = CANDIDATES[0]["name"]
            control = state.get("confirmation", {}).get(control_name, {})
            control_result = control.get("result", {})
            control_med = control_result.get("median_all_laps_s") if control_result.get("clean_gate") else None
            proven = bool(winner and finite_number(base_med) and winner[0] < float(base_med) * 0.98
                          and (winner[1] == control_name or (finite_number(control_med) and winner[0] < float(control_med) * 0.98)))
            selected_name = winner[1] if proven else control_name
            selected_record = state.get("confirmation", {}).get(selected_name, {})
            if not selected_record.get("result", {}).get("clean_gate"):
                selected_record = state.get("evaluations", {}).get(selected_name, {})
            if not selected_record.get("result", {}).get("clean_gate"):
                state["phase"] = "halted_no_safe_selection"; save()
                raise RuntimeError("no validated control or confirmed winner; refusing automatic continuation")
            selected_cp = selected_record["checkpoint"]
            state["decision"] = {"status": "selected", "winner": selected_name,
                "selected_checkpoint": selected_cp, "baseline_median_all_laps_s": base_med,
                "candidate_median_all_laps_s": winner[0] if winner else None,
                "improvement_fraction": (1.0 - winner[0] / float(base_med)) if proven else None,
                "gate_passed": proven, "reward": {k:v for k,v in state["candidates"][selected_name]["reward"].items() if k != "name"},
                "note": "Confirmed improvement selected, otherwise validated zero-pace control retained. Unlimited six-environment continuation authorized."}
            state["production"] = {"cycle": 0, "resume": selected_cp}
            state["phase"] = "production"; save()
        if state["phase"] == "production":
            production = state["production"]
            name = state["decision"]["winner"]
            while True:
                cycle = production["cycle"]
                stage = f"production-{cycle}"
                if not production.get("run_id"):
                    display = f"reward-sweep-{out.name}-{stage}-{name}"
                    config = {**base, **state["decision"]["reward"], "resume": production["resume"],
                              "n_envs": 6, "total_timesteps": 0, "checkpoint_every": 12288}
                    if production.get("launch_intent"):
                        matches = [x for x in api.request("/train/runs")["runs"] if x["display_name"] == display]
                        if len(matches) != 1:
                            raise RuntimeError("cannot reconcile production launch intent")
                        row = matches[0]
                    else:
                        production["launch_intent"] = {"display": display, "config": config}; save()
                        row = api.request("/train/runs", {"config": config, "display_name": display})
                    production["run_id"] = row["run_id"]
                    production["evaluation_due_step"] = checkpoint_steps(Path(production["resume"])) + 98304
                    save()
                row = api.request(f"/train/runs/{production['run_id']}")
                if row.get("state") in TERMINAL_RUNS and not production.get("evaluation_requested"):
                    raise RuntimeError(f"unlimited trainer unexpectedly ended: {row.get('state')} {row.get('error')}")
                log_resource(out, stage, name, row)
                root = Path(row["output_dir"])
                audit = root / "ppo_optimizer_audit.jsonl"
                audits = [json.loads(x) for x in audit.read_text().splitlines() if x.strip()] if audit.exists() else []
                if audits and audits[-1]["num_timesteps"] >= production["evaluation_due_step"]:
                    production["evaluation_requested"] = True; save()
                if production.get("evaluation_requested"):
                    if row["state"] not in TERMINAL_RUNS:
                        api.request(f"/train/runs/{production['run_id']}/stop", {})
                        time.sleep(args.poll_seconds); continue
                    if row["state"] == "failed":
                        raise RuntimeError("production trainer failed before periodic evaluation")
                    cp = newest_checkpoint(root, checkpoint)
                    state["candidates"][name].setdefault("checkpoints", {})[stage] = str(cp); save()
                    evaluate(stage, [name], f"production_evaluation_{cycle}")
                    result = state[f"production_evaluation_{cycle}"][name]["result"]
                    # Retain the last validated model if the new checkpoint fails.
                    if result["clean_gate"]:
                        production["resume"] = str(cp)
                    production["cycle"] += 1
                    production.pop("run_id", None); production.pop("launch_intent", None)
                    production.pop("evaluation_requested", None); save()
                time.sleep(args.poll_seconds)
        print(json.dumps({"phase": state["phase"], "state_file": str(state_path), "report": str(report_path), "decision": state.get("decision")}, indent=2))
        return 0
    except Exception as exc:
        print(f"run_reward_sweep: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def checkpoint_steps(path: Path) -> int:
    match = re.search(r"steps_(\d+)\.zip$", path.name)
    if not match: raise RuntimeError(f"checkpoint lacks immutable step identity: {path}")
    return int(match.group(1))


def newest_checkpoint(output_dir: Any, parent: Path) -> Path:
    if not output_dir: raise RuntimeError("training run has no output_dir")
    root = Path(str(output_dir))
    if not root.exists() and str(output_dir).startswith("/app/logs/rl/"):
        root = ROOT / "logs/rl" / str(output_dir).removeprefix("/app/logs/rl/")
    if not root.exists() and str(output_dir).startswith("/runs/rl/"):
        root = ROOT / "logs/rl" / str(output_dir).removeprefix("/runs/rl/")
    audits = [json.loads(x) for x in (root / "ppo_optimizer_audit.jsonl").read_text().splitlines() if x.strip()]
    audited = {r["num_timesteps"] for r in audits if all(r.get(k) is True for k in
               ("metrics_finite", "gradients_finite", "parameters_finite", "optimizer_state_finite"))}
    if not audits or any(not all(r.get(k) is True for k in ("metrics_finite", "gradients_finite", "parameters_finite", "optimizer_state_finite")) for r in audits):
        raise RuntimeError(f"missing or nonfinite optimizer audit: {root}")
    pattern = re.compile(r"steps_(\d+)\.zip$")
    found = []
    for path in (root / "checkpoints").glob("*.zip"):
        match = pattern.search(path.name)
        if match and path.name.startswith("post_optimizer_update_") and int(match.group(1)) in audited:
            found.append((int(match.group(1)), path))
    if not found: raise RuntimeError(f"no immutable post-optimizer checkpoint under {root}")
    selected = max(found, key=lambda x: x[0])[1].resolve()
    from src.layer3.bc_artifact import verify_ppo_lineage_checkpoint
    if not verify_ppo_lineage_checkpoint(selected):
        raise RuntimeError(f"checkpoint lineage verification failed: {selected}")
    return selected


def log_resource(out: Path, stage: str, name: str, run: dict) -> None:
    """Append one fsynced host/API telemetry snapshot for each polling sample."""
    sample: dict[str, Any] = {"utc": utc(), "stage": stage, "candidate": name,
        "run_id": run.get("run_id"), "state": run.get("state"),
        "latest_metrics": run.get("latest_metrics"), "latest_telemetry": run.get("latest_telemetry")}
    fleet = run.get("fleet")
    if not fleet and run.get("output_dir"):
        try: fleet = load_json(Path(run["output_dir"]) / "watch_latest.json")
        except (OSError, ValueError): fleet = None
    if isinstance(fleet, dict):
        keys = ("env_id", "pose", "yaw", "speed", "speed_source", "v_long", "throttle_command", "steering_command",
                "collision", "collision_count", "lap_count", "lap_times_s", "reward_components", "current_route_speed_mps",
                "route_projection_valid", "neutral_fallback_count", "command_expiry_diagnostic")
        sample["fleet"] = {"step": fleet.get("step"), "ts": fleet.get("ts"),
                           "cars": [{k:c.get(k) for k in keys} for c in fleet.get("cars", [])]}
    try:
        sample["loadavg"] = list(os.getloadavg())
    except (AttributeError, OSError): pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines()[:4]:
            k, v = line.split(":", 1); sample.setdefault("meminfo_kb", {})[k] = int(v.strip().split()[0])
    except (OSError, ValueError, IndexError): pass
    try:
        sample["disk_free_bytes"] = shutil.disk_usage(out).free
    except OSError: pass
    try:
        sample["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], timeout=5, text=True).strip()
    except (OSError, subprocess.SubprocessError): sample["gpu"] = None
    path = out / "telemetry.jsonl"
    from src.layer3.reward_experiment_metrics import _json_value
    sample = _json_value(sample)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
