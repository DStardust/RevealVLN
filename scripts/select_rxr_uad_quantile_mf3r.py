#!/usr/bin/env python3
"""Select MF3R using unlabeled training-score quantile calibration."""
from __future__ import annotations
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from revealnav_mf3 import MF3B_SCOPE,median_mad_lower_confidence
from scripts.select_rxr_uad_crossfit_mf3q import collect,load_models
from scripts.select_rxr_uad_rescue_harm_mf3p import PERSISTENCE,summarize,uncertainty_summary,wilson
from scripts.train_rxr_uad_correction_mf3e import atomic_json,sha256_file
from scripts.train_rxr_uad_crossfit_mf3q import FOLDS,manifest_path
DESIGN=ROOT/"artifacts/design/METHOD_FREEZE_3R_QUANTILE_CALIBRATED_UAD.md";MF3Q=ROOT/"artifacts/evaluation/mf3q_crossfit_development_v1/MF3Q_DEVELOPMENT_SELECTION.json";OUT=ROOT/"artifacts/evaluation/mf3r_quantile_development_v1"
MAD_WEIGHTS=(0.,.5,1.,1.5,2.);QUANTILES=(.95,.975,.985,.99,.995)
def score(row,weight):return float(median_mad_lower_confidence(torch.tensor(row["member_logits"]),mad_weight=weight))
def threshold(episodes,weight,quantile):
  values=torch.tensor([score(row,weight) for sequence in episodes for row in sequence if row is not None])
  if values.numel()<1:raise RuntimeError("empty MF3R training-score distribution")
  return float(torch.quantile(values,quantile))
def main():
  prior=json.loads(MF3Q.read_text())
  if prior.get("status")!="DEVELOPMENT_FAIL" or prior.get("ranks24_29_payload_read",False):raise RuntimeError("MF3R prior boundary drift")
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");folds={};checkpoints=[]
  for fold in FOLDS:
    models,evidence=load_models(fold,device);checkpoints.extend(evidence);folds[fold]={"train":collect(models,"fit",device,manifest_path(fold)),"oof":collect(models,"calibration",device,manifest_path(fold))}
  final_models,final_checkpoints=load_models("final",device);final_train=collect(final_models,"fit",device,manifest_path("final"));candidates=[]
  for weight in MAD_WEIGHTS:
    for quantile in QUANTILES:
      fold_thresholds={fold:threshold(value["train"],weight,quantile) for fold,value in folds.items()}
      for persistence in PERSISTENCE:
        strata={f"oof_fold_{fold}":summarize(value["oof"],weight,fold_thresholds[fold],persistence) for fold,value in folds.items()};pooled={key:sum(x[key] for x in strata.values()) for key in ("interventions","rescues","harms","neither","net_rescues")};counts=[x["interventions"] for x in strata.values()];ratio=min(counts)/max(counts) if max(counts) else 0.;lower=wilson(pooled["rescues"],pooled["harms"]);qualifies=all(x["interventions"]>=10 and x["net_rescues"]>0 for x in strata.values()) and pooled["interventions"]>=45 and ratio>=.4 and lower>.5
        candidates.append({"mad_weight":weight,"training_score_quantile":quantile,"persistence_steps":persistence,"fold_thresholds":fold_thresholds,"strata":strata,"pooled":pooled,"fold_intervention_ratio":ratio,"rescue_precision_wilson95_lower":lower,"qualifies":qualifies})
  q=[x for x in candidates if x["qualifies"]];rule=max(q,key=lambda x:(min(v["net_rescues"] for v in x["strata"].values()),x["rescue_precision_wilson95_lower"],x["pooled"]["net_rescues"],-x["pooled"]["harms"],x["training_score_quantile"],x["persistence_steps"],x["mad_weight"])) if q else None;uncertainty=None
  if rule:
    rule["final_training_threshold"]=threshold(final_train,float(rule["mad_weight"]),float(rule["training_score_quantile"]));pooled_episodes=[seq for value in folds.values() for seq in value["oof"]];margins=sorted({row["native_margin"] for seq in pooled_episodes for row in seq if row is not None});target=rule["pooled"]["interventions"];options=[(abs(uncertainty_summary(pooled_episodes,m)["interventions"]-target),uncertainty_summary(pooled_episodes,m)["interventions"]>target,m,uncertainty_summary(pooled_episodes,m)) for m in margins];_,_,margin,summary=min(options);uncertainty={"native_margin_max":margin,"development":summary,"target_intervention_budget":target,"online_rule":"first eligible margin crossing, once per episode"}
  atomic_json(OUT/"MF3R_DEVELOPMENT_SELECTION.json",{"schema_version":"revealnav-mf3r-development-selection/1","status":"DEVELOPMENT_PASS" if rule else "DEVELOPMENT_FAIL","selected_architecture":{"hidden_dim":64,"optimizer_steps":200},"rule_candidates":candidates,"selected_rule":rule,"uncertainty_rule":uncertainty,"oof_episodes_by_fold":{f"oof_fold_{f}":len(v["oof"]) for f,v in folds.items()},"crossfit_checkpoints":checkpoints,"final_checkpoints":final_checkpoints,"ranks24_29_payload_read":False,"design_sha256":sha256_file(DESIGN),**MF3B_SCOPE});return 0 if rule else 2
if __name__=="__main__":raise SystemExit(main())
