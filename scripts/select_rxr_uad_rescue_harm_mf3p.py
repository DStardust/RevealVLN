#!/usr/bin/env python3
"""Select the causal first-crossing MF3P rule on consumed ranks."""

from __future__ import annotations

import json, math, statistics, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from revealnav_mf3 import (MF3B_SCOPE,OnlineUADFeatureDataset,PairwiseSwitchUtility,
    collate_online_uad,median_mad_lower_confidence,top2_rescue_harm_logit,top2_switch_targets)
from scripts.train_rxr_uad_correction_mf3e import atomic_json,sha256_file
from scripts.train_rxr_uad_rescue_harm_mf3p import DATA,DESIGN,HIDDEN_DIMS,OUT as TRAIN,POSITIVE_WEIGHT,SEEDS

MAD_WEIGHTS=(0.,.5,1.,1.5,2.);THRESHOLDS=(0.,.25,.5,.75,1.,1.5,2.);PERSISTENCE=(1,2,3)
OUT=ROOT/"artifacts/evaluation/mf3p_rescue_harm_development_v1"

def load_models(hidden,device):
  models=[];evidence=[]
  for seed in SEEDS:
    p=TRAIN/f"hidden_{hidden}/seed_{seed}/rescue_harm_mf3p.pt";x=torch.load(p,map_location="cpu",weights_only=True)
    if not(x.get("schema_version")=="revealnav-mf3p-checkpoint/1" and x.get("hidden_dim")==hidden and x.get("seed")==seed and abs(float(x.get("rescue_positive_weight"))-POSITIVE_WEIGHT)<1e-12):raise RuntimeError("MF3P checkpoint drift")
    m=PairwiseSwitchUtility(768,1536,hidden);m.load_state_dict(x["model_state_dict"],strict=True);models.append(m.to(device).eval());evidence.append({"seed":seed,"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
  return tuple(models),evidence

def collect(models,split,device,data=DATA):
  loader=DataLoader(OnlineUADFeatureDataset(data,split),batch_size=1,shuffle=False,collate_fn=collate_online_uad);episodes=[]
  with torch.no_grad():
    for cpu in loader:
      b={k:v.to(device) for k,v in cpu.items()};values=[]
      for m in models:
        o=m(b["history_embeddings"],b["candidate_embeddings"],b["candidate_mask"],b["instruction_embedding"],b["native_scores"],b["native_index"]);values.append(top2_rescue_harm_logit(o,b)[0])
      labels,runner,valid=top2_switch_targets(b);sequence=[]
      for step in range(valid.shape[1]):
        if not bool(valid[0,step]):sequence.append(None);continue
        native=int(b["native_index"][0,step]);sequence.append({"outcome":"RESCUE" if int(labels[0,step])==1 else "HARM" if int(labels[0,step])==2 else "NEITHER","member_logits":[float(v[0,step]) for v in values],"native_margin":float(b["native_scores"][0,step,native]-b["native_scores"][0,step,runner[0,step]])})
      episodes.append(sequence)
  return episodes

def robust(row,weight):return float(median_mad_lower_confidence(torch.tensor(row["member_logits"]),mad_weight=weight))

def first_crossing(sequence,weight,threshold,persistence):
  run=[]
  for row in sequence:
    if row is None or robust(row,weight)<=threshold:run=[];continue
    run.append(row)
    if len(run)>=persistence:return row
  return None

def summarize(episodes,weight,threshold,persistence):
  selected=[x for seq in episodes if (x:=first_crossing(seq,weight,threshold,persistence)) is not None];r=sum(x["outcome"]=="RESCUE" for x in selected);h=sum(x["outcome"]=="HARM" for x in selected)
  return {"interventions":len(selected),"rescues":r,"harms":h,"neither":len(selected)-r-h,"net_rescues":r-h}

def uncertainty_summary(episodes,margin_max):
  selected=[]
  for sequence in episodes:
    row=next((x for x in sequence if x is not None and x["native_margin"]<=margin_max),None)
    if row is not None:selected.append(row)
  r=sum(x["outcome"]=="RESCUE" for x in selected);h=sum(x["outcome"]=="HARM" for x in selected)
  return {"interventions":len(selected),"rescues":r,"harms":h,"neither":len(selected)-r-h,"net_rescues":r-h}

def wilson(r,h):
  n=r+h
  if not n:return 0.
  z=1.96;p=r/n;return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def main():
  arch=[]
  for hidden in HIDDEN_DIMS:
    nll=[json.loads((TRAIN/f"hidden_{hidden}/seed_{seed}/RESULT.json").read_text())["calibration"]["balanced_bce"] for seed in SEEDS];arch.append({"hidden_dim":hidden,"member_calibration_bce":nll,"median_calibration_bce":statistics.median(nll)})
  selected_arch=min(arch,key=lambda x:(x["median_calibration_bce"],x["hidden_dim"]));device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");models,checkpoints=load_models(int(selected_arch["hidden_dim"]),device)
  strata={"calibration_ranks_18_20":collect(models,"calibration",device),"development_ranks_21_23":collect(models,"diagnostic",device)};candidates=[]
  for weight in MAD_WEIGHTS:
    for threshold in THRESHOLDS:
      for persistence in PERSISTENCE:
        s={k:summarize(v,weight,threshold,persistence) for k,v in strata.items()};pooled={key:sum(x[key] for x in s.values()) for key in ("interventions","rescues","harms","neither","net_rescues")};counts=[x["interventions"] for x in s.values()];ratio=min(counts)/max(counts) if max(counts) else 0.;lower=wilson(pooled["rescues"],pooled["harms"]);qualifies=all(x["interventions"]>=10 and x["net_rescues"]>0 for x in s.values()) and pooled["interventions"]>=25 and ratio>=.5 and lower>.5
        candidates.append({"mad_weight":weight,"logit_threshold":threshold,"persistence_steps":persistence,"strata":s,"pooled":pooled,"stratum_intervention_ratio":ratio,"rescue_precision_wilson95_lower":lower,"qualifies":qualifies})
  q=[x for x in candidates if x["qualifies"]];rule=max(q,key=lambda x:(min(v["net_rescues"] for v in x["strata"].values()),x["rescue_precision_wilson95_lower"],x["pooled"]["net_rescues"],-x["pooled"]["harms"],x["logit_threshold"],x["persistence_steps"],x["mad_weight"])) if q else None
  uncertainty=None
  if rule:
    pooled_episodes=[sequence for values in strata.values() for sequence in values];margins=sorted({row["native_margin"] for sequence in pooled_episodes for row in sequence if row is not None});target=rule["pooled"]["interventions"]
    options=[(abs(uncertainty_summary(pooled_episodes,m)["interventions"]-target),uncertainty_summary(pooled_episodes,m)["interventions"]>target,m,uncertainty_summary(pooled_episodes,m)) for m in margins]
    _,_,margin_max,summary=min(options)
    uncertainty={"native_margin_max":margin_max,"development":summary,"target_intervention_budget":target,"online_rule":"first eligible margin crossing, once per episode"}
  atomic_json(OUT/"MF3P_DEVELOPMENT_SELECTION.json",{"schema_version":"revealnav-mf3p-development-selection/1","status":"DEVELOPMENT_PASS" if rule else "DEVELOPMENT_FAIL","architecture_candidates":arch,"selected_architecture":selected_arch,"rule_candidates":candidates,"selected_rule":rule,"uncertainty_rule":uncertainty,"eligible_episodes_by_stratum":{k:len(v) for k,v in strata.items()},"checkpoints":checkpoints,"ranks24_29_payload_read":False,"data_sha256":sha256_file(DATA),"design_sha256":sha256_file(DESIGN),**MF3B_SCOPE});return 0 if rule else 2
if __name__=="__main__":raise SystemExit(main())
