#!/usr/bin/env python3
"""Durable speed-reward campaign orchestrator. Mutations require --execute."""
from __future__ import annotations
import argparse, ast, fcntl, hashlib, json, math, os, re, statistics, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TERMINAL_RUNS = {"completed", "failed", "stopped"}
TERMINAL_EVALS = {"complete", "completed", "failed", "incomplete", "stopped"}
AUDIT_FLAGS = ("metrics_finite", "gradients_finite", "parameters_finite", "optimizer_state_finite")
CANDIDATES = [
    {"name":"control", "frontier_pace_bonus_strength":0.5, "frontier_pace_target_mps":6.0, "time_cost_per_simulated_second":0.0},
    {"name":"target15-strength05-time0", "frontier_pace_bonus_strength":0.5, "frontier_pace_target_mps":1.5, "time_cost_per_simulated_second":0.0},
    {"name":"target15-strength1-time0", "frontier_pace_bonus_strength":1.0, "frontier_pace_target_mps":1.5, "time_cost_per_simulated_second":0.0},
    {"name":"target15-strength1-time01", "frontier_pace_bonus_strength":1.0, "frontier_pace_target_mps":1.5, "time_cost_per_simulated_second":0.1},
]


def utc(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    with tmp.open("w") as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_DIRECTORY); os.fsync(fd); os.close(fd)
def read_json(path):
    v=json.loads(Path(path).read_text());
    if not isinstance(v,dict): raise RuntimeError(f"expected JSON object at {path}")
    return v
class API:
    def __init__(self,url): self.url=url.rstrip("/")
    def get(self,path): return self.call(path)
    def post(self,path,payload): return self.call(path,payload)
    def call(self,path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(self.url+path,data=data,headers={"Content-Type":"application/json"} if data else {},method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req,timeout=240) as r: return json.loads(r.read())
        except (urllib.error.URLError,TimeoutError,json.JSONDecodeError) as e: raise RuntimeError(f"hub request {path}: {e}") from e

def resolve_outdir(value):
    p=Path(str(value))
    if p.exists(): return p.resolve()
    for prefix in ("/app/logs/rl/","/runs/rl/"):
        if str(value).startswith(prefix): return (ROOT/"logs/rl"/str(value).removeprefix(prefix)).resolve()
    return p.resolve()
def comparable_config(config):
    result=dict(config)
    resume=result.get("resume")
    if isinstance(resume,str):
        for prefix in ("/runs/rl/","/app/logs/rl/"):
            if resume.startswith(prefix):
                resume=str((ROOT/"logs/rl"/resume.removeprefix(prefix)).resolve()); break
        result["resume"]=str(Path(resume).resolve())
    return result
def official_config_fields():
    source=ROOT/"src/layer4/official_settings.py"
    tree=ast.parse(source.read_text(encoding="utf-8"))
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="OfficialTrainSettings")
    return {n.target.id for n in model.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
def model_path(path):
    value=str(path)
    for prefix in ("/runs/rl/","/app/logs/rl/"):
        if value.startswith(prefix): return (ROOT/"logs/rl"/value.removeprefix(prefix)).resolve()
    return Path(value).resolve()
def step_of(path):
    m=re.search(r"steps_(\d+)\.zip$",Path(path).name)
    if not m: raise RuntimeError(f"checkpoint has no immutable step count: {path}")
    return int(m.group(1))
def verified_checkpoint(output_dir, expected_steps=None):
    root=resolve_outdir(output_dir); audit_path=root/"ppo_optimizer_audit.jsonl"
    if not audit_path.is_file(): raise RuntimeError(f"missing optimizer audit: {audit_path}")
    rows=[]
    for line in audit_path.read_text().splitlines():
        if line.strip(): rows.append(json.loads(line))
    if not rows or any(not all(x.get(k) is True for k in AUDIT_FLAGS) for x in rows): raise RuntimeError(f"optimizer audits missing/failed: {root}")
    audited={int(x["num_timesteps"]) for x in rows}
    found=[]
    for p in (root/"checkpoints").glob("post_optimizer_update_*_steps_*.zip"):
        n=step_of(p)
        if n in audited and (expected_steps is None or n==expected_steps): found.append((n,p.resolve()))
    if not found: raise RuntimeError(f"no audited checkpoint at expected step {expected_steps}: {root}")
    cp=max(found,key=lambda x:x[0])[1]
    from src.layer3.bc_artifact import verify_ppo_lineage_checkpoint
    if not verify_ppo_lineage_checkpoint(cp): raise RuntimeError(f"PPO lineage verification failed: {cp}")
    return cp

def is_zero_update_sensor_startup_failure(run_row):
    """Allow one retry only when startup timed out before any PPO update."""
    if run_row.get("state") != "failed" or run_row.get("error") != "official API or simulator container exited (trainer exit code 1)":
        return False
    root=resolve_outdir(run_row.get("output_dir", ""))
    log=root/"hub_train.log"
    progress=root/"tensorboard"/"progress.csv"
    audit=root/"ppo_optimizer_audit.jsonl"
    if not log.is_file() or audit.exists(): return False
    if progress.is_file() and progress.stat().st_size != 0: return False
    text=log.read_text(errors="replace")
    return "Timed out after 180.0s waiting for a fresh official ROS sensor frame" in text and "Official PPO configuration verified before rollout." in text

def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))
def summarize(row):
    result=row.get("latest_result") or row.get("result") or {}; runs=result.get("runs") or row.get("attempts") or []
    laps=[]; flying=[]; clean=True; collisions=0
    for a in runs:
        if not isinstance(a,dict): clean=False; continue
        v=[float(x) for x in a.get("race_lap_times_s",[]) if finite(x) and x>0]; laps.extend(v); flying.extend(v[1:])
        counters=[a.get(k) for k in ("collision_count","race_collisions","raw_collision_count_at_finish")]
        if any(not finite(c) or c!=0 for c in counters): clean=False
        collisions+=sum(int(c) for c in counters if finite(c))
        clean &= (len(a.get("race_lap_times_s",[]))==3 and len(v)==3
                  and a.get("stop_reason") in {"race_complete","evaluation_complete"})
    ready=result.get("readiness_passed") is True
    gate=ready and len(runs)==3 and len(laps)==9 and clean and collisions==0
    return {"evaluation_id":row.get("evaluation_id"),"state":row.get("state"),"validation_mode":row.get("validation_mode"),"readiness_passed":ready,"attempts":len(runs),"scored_laps":len(laps),"collisions":collisions,"clean_gate":bool(gate),"median_flying_lap_s":statistics.median(flying) if len(flying)==6 else None,"median_all_laps_s":statistics.median(laps) if len(laps)==9 else None,"attempts_detail":runs}

def production_gate_action(clean_gate, override):
    """Keep the default gate; explicit persisted override permits exploration."""
    return "continue" if clean_gate or override else "halt"
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adaptation-run-id",required=True); p.add_argument("--base-config",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--hub-url",default="http://127.0.0.1:8090"); p.add_argument("--execute",action="store_true"); p.add_argument("--poll-seconds",type=float,default=5)
    p.add_argument("--continue-after-failed-production-eval",action="store_true",
                    help="persist an explicit override to alert on failed production Porto evaluations and continue from audited checkpoints")
    a=p.parse_args(); base_path=a.base_config.resolve(); out=a.output.resolve()
    try:
        if a.poll_seconds<1: raise ValueError("--poll-seconds must be at least 1")
        raw=read_json(base_path)
        unknown=set(raw)-official_config_fields()
        if unknown: raise ValueError(f"base config contains unsupported official settings: {sorted(unknown)}")
        if raw.get("collision_penalty_magnitude")!=250 or raw.get("centerline_shaping_strength")!=0: raise ValueError("base config must specify penalty 250 and centerline shaping 0")
        if not a.execute:
            print(json.dumps({"mutation":False,"adaptation_run_id":a.adaptation_run_id,"base_config":str(base_path),"output":str(out),"candidate_names":[x["name"] for x in CANDIDATES]},indent=2)); return 0
        out.mkdir(parents=True,exist_ok=True); lock=(out/"runner.lock").open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=out/"state.json"; report=out/"report.json"
        if state_path.exists():
            s=read_json(state_path)
            if s.get("adaptation_run_id")!=a.adaptation_run_id or s.get("base_config_sha256")!=digest(base_path): raise RuntimeError("state is bound to another adaptation/config")
        else:
            s={"version":1,"created_utc":utc(),"phase":"wait_adaptation","adaptation_run_id":a.adaptation_run_id,"base_config":str(base_path),"base_config_sha256":digest(base_path),"runs":{},"evaluations":{},"candidates":{c["name"]:{"reward":c} for c in CANDIDATES},"decision":None,"continue_after_failed_production_eval":False,"alerts":[]}
        if a.continue_after_failed_production_eval:
            s["continue_after_failed_production_eval"] = True
            s.setdefault("alerts",[])
            save_json(state_path,s)
        api=API(a.hub_url)
        def persist():
            s["updated_utc"]=utc(); save_json(state_path,s); save_json(report,{k:s.get(k) for k in ("updated_utc","phase","adapted_checkpoint","baseline","candidates","confirmation","decision","production","continue_after_failed_production_eval","alerts")})
        def wait_run(rid):
            while True:
                row=api.get(f"/train/runs/{rid}")
                if row.get("run_id")!=rid: raise RuntimeError("adaptation/run identity mismatch")
                if row.get("state") in TERMINAL_RUNS: return row
                time.sleep(a.poll_seconds)
        if s["phase"]=="wait_adaptation":
            row=wait_run(a.adaptation_run_id)
            if row.get("state")!="completed": s["phase"]="halted_adaptation"; s["adaptation_error"]={"state":row.get("state"),"error":row.get("error")}; persist(); raise RuntimeError("adaptation did not complete; campaign halted")
            cfg=row.get("config") or {}
            if cfg.get("n_envs")!=6 or cfg.get("total_timesteps")!=122880 or cfg.get("centerline_shaping_strength")!=0: raise RuntimeError("adaptation config does not match authorized 6-env/122880/zero-centerline adaptation")
            if step_of(cfg.get("resume", ""))!=966510: raise RuntimeError("adaptation did not resume from the authorized 966510-step parent")
            configured_parent=cfg.get("resume")
            declared_parent=raw.get("resume")
            if not declared_parent or model_path(configured_parent)!=model_path(declared_parent) or digest(model_path(configured_parent))!=digest(model_path(declared_parent)):
                raise RuntimeError("adaptation source checkpoint does not match base-config resume lineage")
            cp=verified_checkpoint(row.get("output_dir"),966510+122880)
            s["adapted_checkpoint"]=str(cp); s["adapted_sha256"]=digest(cp); s["adaptation_config"]=cfg
            base={**raw,"resume":str(cp),"n_envs":2,"n_steps":2048,"learning_rate":1e-5,"checkpoint_every":4096,"seed":0,"automatic_evaluation_enabled":False,"automatic_evaluation_mode":"competitive","cooperative_pause_enabled":False,"total_timesteps":0,"stop_on_plateau":False,"max_duration_hours":0,"collision_penalty_magnitude":250,"failed_episode_penalty":250,"centerline_shaping_strength":0,"observation_profile":"official_sensors_camera"}
            s["effective_base_config"]=base; s["phase"]="baseline_eval"; persist()
        parent=Path(s["adapted_checkpoint"]).resolve()
        if digest(parent)!=s["adapted_sha256"]: raise RuntimeError("adapted parent checkpoint changed")
        def wait_eval(eid):
            while True:
                hist=api.get("/api/evaluations/history")
                row=next((x for x in hist if isinstance(x,dict) and x.get("evaluation_id")==eid),None)
                if row and row.get("state") in TERMINAL_EVALS: return row
                status=api.get("/api/replay/status")
                if status.get("evaluation_id")==eid and status.get("state") in TERMINAL_EVALS: return status
                time.sleep(a.poll_seconds)
        def evaluate(key,cp):
            rec=s["evaluations"].get(key)
            if rec and rec.get("result"): return rec["result"]
            cp=Path(cp).resolve()
            if not rec:
                while True:
                    status=api.get("/api/replay/status")
                    if status.get("state") not in {"queued","running","stopping"}: break
                    time.sleep(a.poll_seconds)
                rec={"intent":True,"checkpoint":str(cp),"checkpoint_sha256":digest(cp)}; s["evaluations"][key]=rec; persist()
                resp=api.post("/api/replay/start",{"model_id":cp.relative_to((ROOT/"logs/rl").resolve()).as_posix(),"map_id":"none","device":"cpu","trace_steps":500,"validation_mode":"porto_practice_diagnostic"})
                rec["evaluation_id"]=resp["evaluation_id"]; persist()
            elif not rec.get("evaluation_id"):
                status=api.get("/api/replay/status"); hist=api.get("/api/evaluations/history")
                matches=[x for x in [status,*hist] if isinstance(x,dict) and x.get("model_path",x.get("model")) and model_path(x.get("model_path",x.get("model")))==cp and x.get("validation_mode")=="porto_practice_diagnostic"]
                ids={x.get("evaluation_id") for x in matches if x.get("evaluation_id")}
                if len(ids)!=1: raise RuntimeError(f"ambiguous/lost evaluation POST outcome for {key}")
                rec["evaluation_id"]=ids.pop(); persist()
            row=wait_eval(rec["evaluation_id"])
            if row.get("validation_mode")!="porto_practice_diagnostic": raise RuntimeError("wrong validation mode")
            actual=row.get("model_path",row.get("model")); md=row.get("model_sha256") or (row.get("result") or {}).get("model_sha256")
            if not actual or model_path(actual)!=cp: raise RuntimeError("evaluation model path missing or mismatched")
            if not md or md!=digest(cp): raise RuntimeError("evaluation checkpoint hash missing or mismatched")
            rec["result"]=summarize(row); persist(); return rec["result"]
        if s["phase"]=="baseline_eval":
            b=evaluate("adapted-parent",parent); s["baseline"]=b
            if not b["clean_gate"]: s["phase"]="halted_baseline_gate"; persist(); raise RuntimeError("adapted checkpoint failed Porto 3x3 clean gate")
            s["phase"]="screen"; persist()
        def post_run(key,stage,name,parent_cp,steps,seed):
            intent=s["runs"].get(key)
            cfg=dict(s["effective_base_config"])
            cfg.update({k:v for k,v in s["candidates"][name]["reward"].items() if k!="name"})
            cfg.update({"resume":str(Path(parent_cp).resolve()),"n_envs":2,"seed":seed,"total_timesteps":steps,"checkpoint_every":4096,"automatic_evaluation_enabled":False,"cooperative_pause_enabled":False,"stop_on_plateau":False,"max_duration_hours":0,"centerline_shaping_strength":0,"collision_penalty_magnitude":250,"failed_episode_penalty":250})
            config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            display=f"speed-campaign-{out.name}-{stage}-{name}"
            if not intent:
                intent={"display_name":display,"config":cfg,"config_sha256":config_hash,"parent_checkpoint":str(Path(parent_cp).resolve()),"parent_sha256":digest(parent_cp),"expected_steps":step_of(parent_cp)+steps,"post_started":False}
                s["runs"][key]=intent; persist()
                runs=api.get("/train/runs").get("runs",[])
                matches=[r for r in runs if r.get("display_name")==display]
                if matches: raise RuntimeError(f"unexpected existing run named {display}; refusing duplicate")
                intent["post_started"]=True; persist()
                created=api.post("/train/runs",{"config":cfg,"display_name":display})
                intent["run_id"]=created.get("run_id") if isinstance(created,dict) else None; persist()
                if isinstance(created,dict): append_snapshot(out,stage,name,created)
                if not intent["run_id"]: raise RuntimeError("training POST returned no run_id")
            elif intent.get("config_sha256")!=config_hash: raise RuntimeError(f"persisted run intent mismatch for {key}")
            elif intent.get("retry_pending"):
                if intent.get("run_id") or intent.get("retry_count")!=1:
                    raise RuntimeError(f"invalid bounded retry state for {key}")
                if intent.get("parent_checkpoint")!=str(Path(parent_cp).resolve()) or intent.get("parent_sha256")!=digest(parent_cp):
                    raise RuntimeError(f"retry parent changed for {key}")
                rows=api.get("/train/runs").get("runs",[])
                matches=[r for r in rows if r.get("display_name")==intent["display_name"]]
                if matches: raise RuntimeError(f"retry display name already exists for {key}; refusing duplicate")
                intent["post_started"]=True; persist()
                created=api.post("/train/runs",{"config":cfg,"display_name":intent["display_name"]})
                intent["run_id"]=created.get("run_id") if isinstance(created,dict) else None
                intent["retry_pending"]=False; persist()
                if isinstance(created,dict): append_snapshot(out,stage,name,created)
                if not intent["run_id"]: raise RuntimeError("retry training POST returned no run_id")
            elif not intent.get("run_id"):
                rows=api.get("/train/runs").get("runs",[]); matches=[r for r in rows if r.get("display_name")==display]
                if len(matches)!=1: raise RuntimeError(f"lost create outcome ambiguous for {display}: {len(matches)} matches")
                actual=comparable_config(matches[0].get("config") or {}); expected=comparable_config(cfg)
                for k,v in expected.items():
                    if actual.get(k)!=v: raise RuntimeError(f"reconciled {display} config mismatch at {k}")
                intent["run_id"]=matches[0]["run_id"]; persist()
            return intent
        def train_group(stage,names,steps,parent_by_name,seed):
            intents={n:post_run(f"{stage}:{n}",stage,n,parent_by_name[n],steps,seed) for n in names}
            active={x["run_id"]:n for n,x in intents.items()}
            while active:
                for rid,name in list(active.items()):
                    intent=intents[name]; row=api.get(f"/train/runs/{rid}")
                    if row.get("run_id")!=rid: raise RuntimeError("trainer identity mismatch")
                    actual=comparable_config(row.get("config") or {}); expected=comparable_config(intent["config"])
                    mismatches=[k for k,v in expected.items() if actual.get(k)!=v]
                    if mismatches: raise RuntimeError(f"effective run config mismatch {rid}: {mismatches}")
                    append_snapshot(out,stage,name,row)
                    if row.get("state") in TERMINAL_RUNS:
                        if row.get("state")!="completed":
                            if (intent.get("retry_count",0)==0 and is_zero_update_sensor_startup_failure(row)
                                and intent.get("parent_checkpoint") and intent.get("parent_sha256")==digest(intent["parent_checkpoint"])):
                                failed={"run_id":rid,"display_name":intent["display_name"],"state":row.get("state"),"error":row.get("error"),"output_dir":row.get("output_dir"),"failed_utc":utc()}
                                intent.setdefault("failed_attempts",[]).append(failed)
                                intent["retry_count"]=1; intent["retry_pending"]=True
                                intent["display_name"]=intent["display_name"]+"-retry1"
                                intent["run_id"]=None; intent["post_started"]=False
                                persist(); del active[rid]
                                retry=post_run(f"{stage}:{name}",stage,name,parent_by_name[name],steps,seed)
                                active[retry["run_id"]]=name
                                continue
                            raise RuntimeError(f"run {rid} ended {row.get('state')}: {row.get('error')}")
                        cp=verified_checkpoint(row.get("output_dir"),intent["expected_steps"])
                        intent["output_dir"]=str(resolve_outdir(row.get("output_dir"))); intent["checkpoint"]=str(cp); intent["checkpoint_sha256"]=digest(cp); intent["completed"]=True
                        s["candidates"][name].setdefault("checkpoints",{})[stage]=str(cp); persist(); del active[rid]
                if active: time.sleep(a.poll_seconds)
        def train_batch(stage,names,steps,parent_by_name,seed):
            for i in range(0,len(names),3): train_group(stage,names[i:i+3],steps,parent_by_name,seed)
        if s["phase"]=="screen":
            names=[c["name"] for c in CANDIDATES]
            for offset in range(0,len(names),3):
                batch=names[offset:offset+3]
                train_group("screen",batch,53248,{n:str(parent) for n in batch},0)
                for n in batch: s["candidates"][n]["screen_eval"]=evaluate("screen:"+n,s["candidates"][n]["checkpoints"]["screen"])
                s["screen_batches_evaluated"]=offset//3+1; persist()
            s["phase"]="screen_evaluations"; persist()
        if s["phase"]=="screen_evaluations":
            names=[c["name"] for c in CANDIDATES]
            for n in names: s["candidates"][n]["screen_eval"]=evaluate("screen:"+n,s["candidates"][n]["checkpoints"]["screen"])
            control=s["candidates"]["control"]["screen_eval"]
            eligible=[]
            for n in names:
                ev=s["candidates"][n]["screen_eval"]
                if ev["clean_gate"] and finite(ev.get("median_flying_lap_s")): eligible.append((ev["median_flying_lap_s"],n))
            eligible.sort(); best_noncontrol=next((n for _,n in eligible if n!="control"),None)
            selected=["control"]
            if best_noncontrol: selected.append(best_noncontrol)
            s["confirmation_candidates"]=selected
            s["phase"]="confirmation" if eligible else "decision"
            s["screen_selection"]={"eligible":eligible,"control_clean":control["clean_gate"]}; persist()
        if s["phase"]=="confirmation":
            names=s["confirmation_candidates"]
            train_batch("confirm",names,106496,{n:str(parent) for n in names},1)
            s["phase"]="confirmation_evaluations"; persist()
        if s["phase"]=="confirmation_evaluations":
            s["confirmation"]={}
            for n in s["confirmation_candidates"]: s["confirmation"][n]=evaluate("confirm:"+n,s["candidates"][n]["checkpoints"]["confirm"])
            s["phase"]="decision"; persist()
        if s["phase"]=="decision":
            control_confirm=s.get("confirmation",{}).get("control",{})
            control_screen=s["candidates"]["control"].get("screen_eval",{})
            control=control_confirm if control_confirm.get("clean_gate") else control_screen
            clean=[(v.get("median_flying_lap_s"),n) for n,v in s.get("confirmation",{}).items() if v.get("clean_gate") and finite(v.get("median_flying_lap_s"))]
            clean.sort(); best=clean[0] if clean else None
            # Select a confirmed challenger only when its median flying lap improves on the confirmed control.
            proven=bool(best and best[1]!="control" and control_confirm.get("clean_gate") and best[0] < control_confirm["median_flying_lap_s"])
            chosen=best[1] if proven else "control"
            chosen_ev=s.get("confirmation",{}).get(chosen,{})
            chosen_cp=s["candidates"][chosen]["checkpoints"].get("confirm") if chosen_ev.get("clean_gate") else s["candidates"][chosen]["checkpoints"].get("screen")
            if not control.get("clean_gate") and not (proven and chosen_ev.get("clean_gate")):
                s["phase"]="halted_no_clean_control"; persist(); raise RuntimeError("no clean control or confirmed improvement; production halted")
            if chosen=="control" and not control.get("clean_gate"): s["phase"]="halted_no_clean_control"; persist(); raise RuntimeError("no clean control; production halted")
            s["decision"]={"selected":chosen,"selected_checkpoint":chosen_cp,"improvement_confirmed":proven,"control_median_flying_lap_s":control.get("median_flying_lap_s"),"selected_median_flying_lap_s":(chosen_ev if chosen_ev.get("clean_gate") else control).get("median_flying_lap_s"),"production_eligible":False}
            s["production"]={"resume":chosen_cp,"cycle":0}; s["phase"]="production_precheck"; persist()
        if s["phase"]=="production_precheck":
            name=s["decision"]["selected"]
            chosen_cp=Path(s["decision"]["selected_checkpoint"]).resolve()
            run=next((r for stage in ("confirm", "screen")
                      if (r:=s["runs"].get(f"{stage}:{name}"))
                      and Path(r.get("checkpoint", "")).resolve()==chosen_cp), None)
            evidence=training_evidence(Path(run["output_dir"]) if run else None)
            s["production_training_evidence"]=evidence
            if not evidence["clean_ten_lap_episode"]:
                s["phase"]="halted_no_training_evidence"; persist(); raise RuntimeError("selected candidate lacks a clean ten-lap race_metrics training episode")
            s["decision"]["production_eligible"]=True; s["phase"]="production"; persist()
        if s["phase"]=="production":
            # Evaluate on an audited rollout boundary, then resume the campaign.
            evaluation_interval = s.get("production_eval_interval_transitions", 98304)
            if type(evaluation_interval) is not int or evaluation_interval <= 0 or evaluation_interval % 12288:
                raise ValueError("production evaluation interval must be a positive multiple of 12288")
            prod=s["production"]; name=s["decision"]["selected"]
            while True:
                cycle=prod["cycle"]; key=f"production:{cycle}"; intent=s["runs"].get(key)
                if not intent:
                    cfg=dict(s["effective_base_config"]); cfg.update({k:v for k,v in s["candidates"][name]["reward"].items() if k!="name"})
                    cfg.update({"resume":str(Path(prod["resume"]).resolve()),"n_envs":6,"seed":1,"total_timesteps":0,"checkpoint_every":12288,"automatic_evaluation_enabled":False,"cooperative_pause_enabled":False,"centerline_shaping_strength":0,"collision_penalty_magnitude":250,"failed_episode_penalty":250})
                    display=f"speed-campaign-{out.name}-production-{cycle}-{name}"
                    intent={"display_name":display,"config":cfg,"parent_checkpoint":prod["resume"],"parent_sha256":digest(prod["resume"]),"evaluation_due_step":step_of(prod["resume"])+evaluation_interval,"post_started":False}; s["runs"][key]=intent; persist()
                    matches=[r for r in api.get("/train/runs").get("runs",[]) if r.get("display_name")==display]
                    if matches: raise RuntimeError(f"unexpected production run name already exists: {display}")
                    intent["post_started"]=True; persist(); created=api.post("/train/runs",{"config":cfg,"display_name":display}); intent["run_id"]=created.get("run_id"); persist()
                    if not intent["run_id"]: raise RuntimeError("production POST returned no run ID")
                elif intent.get("retry_pending"):
                    # A single startup-only retry keeps the original failed run in
                    # failed_attempts and reuses the exact production config/parent.
                    if intent.get("retry_count") != 1 or intent.get("run_id"):
                        raise RuntimeError("invalid production retry state")
                    if intent.get("parent_checkpoint") != prod["resume"] or intent.get("parent_sha256") != digest(prod["resume"]):
                        raise RuntimeError("production retry parent changed")
                    matches=[r for r in api.get("/train/runs").get("runs",[]) if r.get("display_name")==intent["display_name"]]
                    if matches:
                        if len(matches)!=1: raise RuntimeError("ambiguous production retry name; refusing duplicate")
                        actual=comparable_config(matches[0].get("config") or {})
                        requested=comparable_config(intent["config"])
                        if any(actual.get(k)!=v for k,v in requested.items()):
                            raise RuntimeError("existing production retry config mismatch")
                        intent["run_id"]=matches[0]["run_id"]
                    else:
                        intent["post_started"]=True; persist()
                        created=api.post("/train/runs",{"config":intent["config"],"display_name":intent["display_name"]})
                        intent["run_id"]=created.get("run_id")
                    intent["retry_pending"]=False; persist()
                    if not intent["run_id"]: raise RuntimeError("production retry POST returned no run ID")
                rid=intent.get("run_id")
                if not rid:
                    matches=[r for r in api.get("/train/runs").get("runs",[]) if r.get("display_name")==intent["display_name"]]
                    if len(matches)!=1: raise RuntimeError("cannot reconcile production create outcome")
                    observed=comparable_config(matches[0].get("config") or {}); requested=comparable_config(intent["config"])
                    if any(observed.get(k)!=v for k,v in requested.items()): raise RuntimeError("reconciled production config mismatch")
                    intent["run_id"]=matches[0]["run_id"]; rid=intent["run_id"]; persist()
                row=api.get(f"/train/runs/{rid}"); append_snapshot(out,f"production-{cycle}",name,row)
                effective=comparable_config(row.get("config") or {}); requested=comparable_config(intent["config"])
                mismatch=[k for k,v in requested.items() if effective.get(k)!=v]
                if mismatch: raise RuntimeError(f"production effective config mismatch: {mismatch}")
                root=resolve_outdir(row.get("output_dir")); cp=None
                if row.get("state") in TERMINAL_RUNS and not intent.get("stop_requested"):
                    if (row.get("state")=="failed" and intent.get("retry_count",0)==0
                        and is_zero_update_sensor_startup_failure(row)
                        and intent.get("parent_checkpoint")==prod["resume"]
                        and intent.get("parent_sha256")==digest(prod["resume"])):
                        intent.setdefault("failed_attempts",[]).append({"run_id":rid,"display_name":intent["display_name"],"state":row.get("state"),"error":row.get("error"),"output_dir":row.get("output_dir"),"failed_utc":utc()})
                        intent["retry_count"]=1; intent["retry_pending"]=True
                        intent["display_name"]+="-retry1"; intent["run_id"]=None; intent["post_started"]=False
                        persist(); continue
                    if row.get("state")=="failed" and intent.get("retry_count",0)>=1:
                        failures=intent.setdefault("failed_attempts",[])
                        if not any(x.get("run_id")==rid for x in failures):
                            failures.append({"run_id":rid,"display_name":intent["display_name"],"state":row.get("state"),"error":row.get("error"),"output_dir":row.get("output_dir"),"failed_utc":utc()})
                        intent["retry_exhausted"]=True
                        s["phase"]="halted_production_startup"; persist()
                        raise RuntimeError("one-time production startup retry failed; campaign halted")
                    raise RuntimeError(f"unlimited production trainer unexpectedly ended: {row.get('state')}")
                audits=root/"ppo_optimizer_audit.jsonl"
                if audits.is_file():
                    audit_text=audits.read_text()
                    complete_lines=audit_text.splitlines() if audit_text.endswith("\n") else audit_text.splitlines()[:-1]
                    records=[json.loads(x) for x in complete_lines if x.strip()]
                    if records and records[-1].get("num_timesteps",0)>=intent["evaluation_due_step"]: intent["stop_requested"]=True; persist()
                if intent.get("stop_requested"):
                    # stop_run is idempotent. After a lost response or runner restart,
                    # a still-running trainer needs the stop request sent again.
                    if row.get("state") not in {"stopping",*TERMINAL_RUNS}:
                        intent["stop_sent"]=True; persist()
                        api.post(f"/train/runs/{rid}/stop",{})
                    time.sleep(a.poll_seconds); row=api.get(f"/train/runs/{rid}")
                    if row.get("state") not in TERMINAL_RUNS: continue
                    if row.get("state")!="stopped": raise RuntimeError(f"production stop ended {row.get('state')}")
                    cp=verified_checkpoint(row.get("output_dir"))
                    if step_of(cp)<intent["evaluation_due_step"]: raise RuntimeError("production stop checkpoint below evaluation boundary")
                    ev=evaluate(f"production:{cycle}",cp); prod["last_evaluation"]=ev
                    if ev["clean_gate"]:
                        prod["resume"]=str(cp)
                    else:
                        prod["last_failed_checkpoint"]=str(cp)
                        alert={"utc":utc(),"kind":"failed_production_porto_evaluation","cycle":cycle,
                               "evaluation_id":ev.get("evaluation_id"),"checkpoint":str(cp),
                               "checkpoint_sha256":digest(cp),"clean_gate":False,
                               "action":"continuing from latest immutable audited checkpoint by persisted user override",
                               "validated_improvement":False}
                        s.setdefault("alerts",[]).append(alert)
                        if production_gate_action(False,s.get("continue_after_failed_production_eval",False)) == "halt":
                            s["phase"]="halted_production_gate"
                            s["production_halt_reason"]={"cycle":cycle,"evaluation_id":ev.get("evaluation_id"),"checkpoint":str(cp),"reason":"Porto clean gate failed; refusing another production continuation"}
                            persist()
                            raise RuntimeError("production Porto gate failed; campaign halted")
                        # Failed evaluation remains in the evaluation map, alert log, and cycle run record.
                        # Continue only from the exact audited immutable checkpoint just evaluated.
                        prod["resume"]=str(cp)
                        s.setdefault("decision",{})["validated_improvement"]=False
                        s["decision"]["production_validation_status"]="unvalidated_exploratory_continuation"
                    prod["cycle"]+=1; persist()
                time.sleep(a.poll_seconds)
        print(json.dumps({"phase":s["phase"],"state_file":str(state_path),"report":str(report),"decision":s.get("decision")},indent=2)); return 0
    except Exception as e:
        print(f"run_speed_campaign: {type(e).__name__}: {e}",file=sys.stderr); return 2

def append_snapshot(out,stage,name,row):
    resources={"load_average_1m":os.getloadavg()[0]}
    try:
        memory={line.split(":",1)[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemTotal:","MemAvailable:"))}
        resources.update({"memory_total_kib":memory.get("MemTotal"),"memory_available_kib":memory.get("MemAvailable")})
    except (OSError, ValueError): pass
    item={"utc":utc(),"stage":stage,"candidate":name,"run_id":row.get("run_id"),"state":row.get("state"),"latest_metrics":row.get("latest_metrics"),"latest_telemetry":row.get("latest_telemetry"),"host_resources":resources}
    def safe(value):
        if isinstance(value,float) and not math.isfinite(value): return None
        if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [safe(v) for v in value]
        if value is None or isinstance(value,(str,int,float,bool)): return value
        return str(value)
    with (out/"telemetry.jsonl").open("a") as f: f.write(json.dumps(safe(item),allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
def training_evidence(root):
    if not root: return {"clean_ten_lap_episode":False,"reason":"missing trainer output"}
    path=root/"race_metrics.jsonl"; qualified=[]; malformed=0
    if not path.is_file(): return {"clean_ten_lap_episode":False,"reason":"missing race_metrics.jsonl"}
    for line in path.read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: malformed+=1; continue
        laps=row.get("race_lap_times_s")
        if (row.get("event_type")=="episode" and row.get("clean_10_lap_episode") is True
            and finite(row.get("race_laps_completed")) and row["race_laps_completed"]>=10
            and row.get("race_lap_timing_valid") is True and isinstance(laps,list)
            and len([x for x in laps if finite(x) and x>0])>=10
            and row.get("race_collisions")==0 and row.get("raw_collision_count")==0):
            qualified.append(row)
    return {"clean_ten_lap_episode":bool(qualified),"qualifying_records":len(qualified),"malformed_lines":malformed,"path":str(path)}
if __name__=="__main__": raise SystemExit(main())
