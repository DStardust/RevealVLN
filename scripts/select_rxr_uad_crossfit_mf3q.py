#!/usr/bin/env python3
"""Select MF3Q from episode-disjoint out-of-fold predictions."""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from revealnav_mf3 import MF3B_SCOPE,OnlineUADFeatureDataset,PairwiseSwitchUtility,collate_online_uad,top2_rescue_harm_logit,top2_switch_targets
from scripts.select_rxr_uad_rescue_harm_mf3p import MAD_WEIGHTS,PERSISTENCE,THRESHOLDS,first_crossing,summarize,uncertainty_summary,wilson
from scripts.train_rxr_uad_correction_mf3e import atomic_json,sha256_file
from scripts.train_rxr_uad_crossfit_mf3q import DESIGN,FOLDS,OUT as TRAIN,POSITIVE_WEIGHT,SEEDS,manifest_path
OUT=ROOT/"artifacts/evaluation/mf3q_crossfit_development_v1"
def load_models(fold,device):
  models=[];evidence=[]
  for seed in SEEDS:
    p=TRAIN/f"fold_{fold}/seed_{seed}/crossfit_mf3q.pt";x=torch.load(p,map_location="cpu",weights_only=True)
    if not(x.get("schema_version")=="revealnav-mf3q-checkpoint/1" and x.get("fold")==fold and x.get("seed")==seed and x.get("optimizer_steps")==200 and abs(float(x.get("rescue_positive_weight"))-POSITIVE_WEIGHT)<1e-12):raise RuntimeError("MF3Q checkpoint drift")
    m=PairwiseSwitchUtility(768,1536,64);m.load_state_dict(x["model_state_dict"],strict=True);models.append(m.to(device).eval());evidence.append({"fold":fold,"seed":seed,"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
  return tuple(models),evidence
def collect(models,split,device,data):
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
def main():
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");strata={};checkpoints=[]
  for fold in FOLDS:
    models,evidence=load_models(fold,device);checkpoints.extend(evidence);strata[f"oof_fold_{fold}"]=collect(models,"calibration",device,manifest_path(fold))
  _,final_checkpoints=load_models("final",device);candidates=[]
  for weight in MAD_WEIGHTS:
    for threshold in THRESHOLDS:
      for persistence in PERSISTENCE:
        s={k:summarize(v,weight,threshold,persistence) for k,v in strata.items()};pooled={key:sum(x[key] for x in s.values()) for key in ("interventions","rescues","harms","neither","net_rescues")};counts=[x["interventions"] for x in s.values()];ratio=min(counts)/max(counts) if max(counts) else 0.;lower=wilson(pooled["rescues"],pooled["harms"]);qualifies=all(x["interventions"]>=10 and x["net_rescues"]>0 for x in s.values()) and pooled["interventions"]>=45 and ratio>=.4 and lower>.5
        candidates.append({"mad_weight":weight,"logit_threshold":threshold,"persistence_steps":persistence,"strata":s,"pooled":pooled,"fold_intervention_ratio":ratio,"rescue_precision_wilson95_lower":lower,"qualifies":qualifies})
  q=[x for x in candidates if x["qualifies"]];rule=max(q,key=lambda x:(min(v["net_rescues"] for v in x["strata"].values()),x["rescue_precision_wilson95_lower"],x["pooled"]["net_rescues"],-x["pooled"]["harms"],x["logit_threshold"],x["persistence_steps"],x["mad_weight"])) if q else None;uncertainty=None
  if rule:
    pooled_episodes=[seq for values in strata.values() for seq in values];margins=sorted({row["native_margin"] for seq in pooled_episodes for row in seq if row is not None});target=rule["pooled"]["interventions"];options=[(abs(uncertainty_summary(pooled_episodes,m)["interventions"]-target),uncertainty_summary(pooled_episodes,m)["interventions"]>target,m,uncertainty_summary(pooled_episodes,m)) for m in margins];_,_,margin,summary=min(options);uncertainty={"native_margin_max":margin,"development":summary,"target_intervention_budget":target,"online_rule":"first eligible margin crossing, once per episode"}
  atomic_json(OUT/"MF3Q_DEVELOPMENT_SELECTION.json",{"schema_version":"revealnav-mf3q-development-selection/1","status":"DEVELOPMENT_PASS" if rule else "DEVELOPMENT_FAIL","selected_architecture":{"hidden_dim":64,"optimizer_steps":200},"rule_candidates":candidates,"selected_rule":rule,"uncertainty_rule":uncertainty,"oof_episodes_by_fold":{k:len(v) for k,v in strata.items()},"crossfit_checkpoints":checkpoints,"final_checkpoints":final_checkpoints,"ranks24_29_payload_read":False,"design_sha256":sha256_file(DESIGN),**MF3B_SCOPE});return 0 if rule else 2
if __name__=="__main__":raise SystemExit(main())
