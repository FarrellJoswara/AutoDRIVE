#!/usr/bin/env python3
"""Durable bounded two-stage PPO pivot; hub mutations require --execute."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_speed_campaign as speed
TRIALS=("trial-1","trial-2"); STEPS=24576; BASELINE=24.768

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--parent-checkpoint',type=Path); p.add_argument('--trial-1-config',type=Path); p.add_argument('--trial-2-config',type=Path)
 p.add_argument('--output',type=Path); p.add_argument('--hub-url',default='http://127.0.0.1:8090'); p.add_argument('--fresh-parent-diagnostic',action='store_true'); p.add_argument('--execute',action='store_true'); p.add_argument('--poll-seconds',type=float,default=5); p.add_argument('--print-schema',action='store_true')
 a=p.parse_args(); fields=list(speed.official_config_fields())
 if a.print_schema:
  print(json.dumps({'schema_version':1,'trial_config':{'type':'object','ordered_fields':fields,'required':['total_timesteps'],'constraints':{'total_timesteps':STEPS,'n_envs':6,'resume':'overridden by orchestrator'}},'selection':{'porto_validation_mode':'porto_practice_diagnostic','attempts':3,'laps_per_attempt':3,'clean_gate':'readiness_passed and exactly 9 positive finite lap times and zero collisions','pace_metric':'median_flying_lap_s','baseline_s':BASELINE,'eligible':'clean_gate and median_flying_lap_s <= baseline_s'},'handoff':'<output>/handoff.json; supervisor launch remains separate'},indent=2)); return 0
 try:
  if not all((a.parent_checkpoint,a.trial_1_config,a.trial_2_config,a.output)): raise ValueError('parent, both trial configs, and output are required')
  if a.poll_seconds<1: raise ValueError('poll interval must be >=1 second')
  parent=a.parent_checkpoint.resolve()
  if not parent.is_file(): raise ValueError(f'missing parent checkpoint: {parent}')
  parent_sha=speed.digest(parent); cfgs={n:speed.read_json(v.resolve()) for n,v in zip(TRIALS,(a.trial_1_config,a.trial_2_config))}
  for n,c in cfgs.items():
   unknown=set(c)-set(fields)
   if unknown: raise ValueError(f'{n}: unsupported fields {sorted(unknown)}')
   if c.get('total_timesteps')!=STEPS: raise ValueError(f'{n}: total_timesteps must equal {STEPS}')
  out=a.output.resolve()
  if not a.execute:
   print(json.dumps({'mutation':False,'parent_checkpoint':str(parent),'parent_sha256':parent_sha,'trial_configs':{n:{**c,'n_envs':6,'total_timesteps':STEPS} for n,c in cfgs.items()},'output':str(out)},indent=2)); return 0
  if speed.verified_checkpoint(parent.parent.parent, speed.step_of(parent)) != parent:
   raise RuntimeError('parent does not match the audited checkpoint')
  out.mkdir(parents=True,exist_ok=True); lock=(out/'runner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  sp=out/'state.json'; rp=out/'report.json'; binding={'parent_sha256':parent_sha,'config_sha256':{n:hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest() for n,c in cfgs.items()},'fresh_parent_diagnostic':a.fresh_parent_diagnostic}
  if sp.exists():
   s=speed.read_json(sp)
   if any(s.get(k)!=v for k,v in binding.items()): raise RuntimeError('persisted state input binding mismatch')
  else: s={**binding,'version':1,'phase':'parent_eval' if a.fresh_parent_diagnostic else 'trial-1','parent_checkpoint':str(parent),'trials':{},'evaluations':{},'created_utc':speed.utc()}
  def save():
   s['runs']=s['trials']; s['updated_utc']=speed.utc(); speed.save_json(sp,s); speed.save_json(rp,{k:s.get(k) for k in ('updated_utc','phase','parent_checkpoint','parent_sha256','parent_diagnostic','trials','decision')})
  api=speed.API(a.hub_url)
  def config_matches(row,cfg):
   actual=speed.comparable_config(row.get('config') or {})
   return all(actual.get(k)==v for k,v in speed.comparable_config(cfg).items())
  def evaluate(key,cp):
   cp=Path(cp).resolve(); rec=s['evaluations'].get(key)
   if rec and rec.get('result'): return rec['result']
   if not rec:
    while api.get('/api/replay/status').get('state') in {'queued','running','stopping'}: time.sleep(a.poll_seconds)
    rec={'intent':True,'checkpoint':str(cp),'checkpoint_sha256':speed.digest(cp)}; s['evaluations'][key]=rec; save()
    mid=cp.relative_to((ROOT/'logs/rl').resolve()).as_posix(); res=api.post('/api/replay/start',{'model_id':mid,'map_id':'none','device':'cpu','trace_steps':500,'validation_mode':'porto_practice_diagnostic'}); rec['evaluation_id']=res.get('evaluation_id') if isinstance(res,dict) else None; save()
    if not rec['evaluation_id']: raise RuntimeError('evaluation POST returned no ID')
   elif not rec.get('evaluation_id'):
    hist=api.get('/api/evaluations/history'); rows=[api.get('/api/replay/status'),*hist]; ids={r.get('evaluation_id') for r in rows if isinstance(r,dict) and r.get('validation_mode')=='porto_practice_diagnostic' and r.get('model_path',r.get('model')) and speed.model_path(r.get('model_path',r.get('model')))==cp and r.get('evaluation_id')}
    if len(ids)!=1: raise RuntimeError(f'ambiguous/lost eval POST for {key}')
    rec['evaluation_id']=ids.pop(); save()
   eid=rec['evaluation_id']
   while True:
    hist=api.get('/api/evaluations/history'); row=next((r for r in hist if isinstance(r,dict) and r.get('evaluation_id')==eid and r.get('state') in speed.TERMINAL_EVALS),None)
    if row is None:
     row=api.get('/api/replay/status')
     if row.get('evaluation_id')!=eid or row.get('state') not in speed.TERMINAL_EVALS: time.sleep(a.poll_seconds); continue
    break
   actual=row.get('model_path',row.get('model')); sha=row.get('model_sha256') or (row.get('result') or {}).get('model_sha256')
   if row.get('validation_mode')!='porto_practice_diagnostic' or not actual or speed.model_path(actual)!=cp or sha!=speed.digest(cp): raise RuntimeError('evaluation checkpoint identity/hash audit failed')
   rec['result']=speed.summarize(row); save(); return rec['result']
  if s['phase']=='parent_eval': s['parent_diagnostic']=evaluate('parent',parent); s['phase']='trial-1'; save()
  for i,n in enumerate(TRIALS):
   if s['phase'] not in (n,n+'-eval','decision','complete'): continue
   if s['phase']==n:
    pp=parent; cfg={**cfgs[n],'resume':str(pp),'n_envs':6,'total_timesteps':STEPS,'automatic_evaluation_enabled':False,'cooperative_pause_enabled':False,'stop_on_plateau':False,'max_duration_hours':0}; display=f'ppo-pivot-{out.name}-{n}'; ch=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest(); it=s['trials'].get(n); expected=speed.step_of(pp)+STEPS
    if not it:
     it={'display_name':display,'config':cfg,'config_sha256':ch,'parent_checkpoint':str(pp),'parent_sha256':speed.digest(pp),'expected_steps':expected,'post_started':False}; s['trials'][n]=it; save()
     existing=[r for r in api.get('/train/runs').get('runs',[]) if r.get('display_name')==display]
     if existing: raise RuntimeError(f'unexpected existing run {display}; refusing duplicate')
     it['post_started']=True; save(); created=api.post('/train/runs',{'config':cfg,'display_name':display}); it['run_id']=created.get('run_id') if isinstance(created,dict) else None; save()
     if not it['run_id']: raise RuntimeError('training POST returned no run ID')
    elif it.get('config_sha256')!=ch: raise RuntimeError(f'persisted {n} config mismatch')
    elif not it.get('run_id'):
     matches=[r for r in api.get('/train/runs').get('runs',[]) if r.get('display_name')==display]
     if len(matches)!=1 or not config_matches(matches[0],cfg): raise RuntimeError(f'ambiguous/lost training POST for {n}')
     it['run_id']=matches[0]['run_id']; save()
    while True:
     row=api.get(f"/train/runs/{it['run_id']}")
     if row.get('run_id')!=it['run_id']: raise RuntimeError('training identity mismatch')
     if row.get('state') in speed.TERMINAL_RUNS: break
     speed.append_snapshot(out,n,n,row); time.sleep(a.poll_seconds)
    if row.get('state')!='completed': raise RuntimeError(f"{n} ended {row.get('state')}: {row.get('error')}")
    if not config_matches(row,cfg): raise RuntimeError('effective config mismatch')
    cp=speed.verified_checkpoint(row.get('output_dir'),expected); it.update({'checkpoint':str(cp),'checkpoint_sha256':speed.digest(cp),'completed':True}); s['phase']=n+'-eval'; save()
   if s['phase']==n+'-eval':
    s['trials'][n]['evaluation']=evaluate(n,s['trials'][n]['checkpoint']); s['phase']=TRIALS[i+1] if i==0 else 'decision'; save()
  if s['phase']=='decision':
   qualified=[]
   for n,t in s['trials'].items():
    ev=t.get('evaluation',{}); pace=ev.get('median_flying_lap_s')
    if ev.get('clean_gate') and speed.finite(pace) and pace<=BASELINE: qualified.append((pace,n))
   qualified.sort(); winner=qualified[0][1] if qualified else None
   selected_cp=s['trials'][winner]['checkpoint'] if winner else (str(parent) if s.get('parent_diagnostic',{}).get('clean_gate') else None)
   s['decision']={'selected_trial':winner,'selected_checkpoint':selected_cp,'baseline_s':BASELINE,'pace_metric':'median_flying_lap_s','selection_safe':selected_cp is not None,'promotion':False,'handoff_eligible':winner is not None,'reason':'clean Porto 3x3 and median flying lap <= 24.768s' if winner else 'no trial met clean Porto 3x3 and baseline pace threshold'}
   if winner:
    cp=Path(selected_cp).resolve(); base={**cfgs[winner],'resume':str(cp),'n_envs':6,'total_timesteps':0,'automatic_evaluation_enabled':False,'cooperative_pause_enabled':False,'stop_on_plateau':False,'max_duration_hours':0}
    speed.save_json(out/'handoff.json',{'schema_version':1,'status':'ready_for_supervisor_review','source':'run_ppo_pivot','selected_trial':winner,'checkpoint':str(cp),'checkpoint_sha256':speed.digest(cp),'validation':s['trials'][winner]['evaluation'],'baseline_s':BASELINE,'selection_pace_s':s['trials'][winner]['evaluation']['median_flying_lap_s'],'training_config':base,'supervisor':'scripts/supervise_speed_campaign.py','launch_authorized':False})
   s['phase']='complete'; save()
  print(json.dumps({'phase':s['phase'],'report':str(rp),'handoff':str(out/'handoff.json') if (out/'handoff.json').exists() else None,'decision':s.get('decision')},indent=2)); return 0
 except Exception as e:
  print(f'run_ppo_pivot: {e}',file=sys.stderr); return 2
if __name__=='__main__': raise SystemExit(main())
