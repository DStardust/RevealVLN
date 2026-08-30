#!/usr/bin/env python3
"""Select MF3S on OOF folds plus consumed ranks 24--29."""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from revealnav_mf3 import MF3B_SCOPE,median_mad_lower_confidence
from scripts.select_rxr_uad_crossfit_mf3q import collect,load_models
from scripts.select_rxr_uad_rescue_harm_mf3p import PERSISTENCE,wilson
from scripts.train_rxr_uad_correction_mf3e import atomic_json,sha256_file
from scripts.train_rxr_uad_crossfit_mf3q import FOLDS,manifest_path
DESIGN=ROOT/"artifacts/design/METHOD_FREEZE_3S_POLICY_RISK_ADJUSTED_UAD.md";MF3R_GATE=ROOT/"artifacts/evaluation/mf3r_quantile_shadow_gate_v1/MF3R_SHADOW_GATE.json";CONSUMED=ROOT/"artifacts/phase1/mf3n_top2_utility_rank29/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json";OUT=ROOT/"artifacts/evaluation/mf3s_policy_risk_development_v1"
MAD_WEIGHTS=(0.,.5,1.,1.5,2.);BETAS=(.25,.5,1.,2.,4.,8.);QUANTILES=(.95,.975,.985,.99,.995)
def _score_terms(row):
  """Cache invariant scalar terms used by the frozen MF3S grid.

  The previous implementation rebuilt a Torch tensor for every row at every
  grid point.  These three scalars depend only on the stored online record, so
  caching them changes neither the score nor the selected operating point.
  """
  cached=row.get("_mf3s_score_terms")
  if cached is None:
    values=torch.tensor(row["member_logits"])
    median=values.median()
    mad=(values-median).abs().median()
    cached=(median,mad,math.log1p(max(0.,float(row["native_margin"]))))
    row["_mf3s_score_terms"]=cached
  return cached

def hybrid(row,weight,beta):
  median,mad,log_margin=_score_terms(row)
  # Keep the original float32 Torch arithmetic for the robust term so this is
  # bit-for-bit equivalent to ``median_mad_lower_confidence``.
  return float(median-float(weight)*mad)-beta*log_margin
def threshold(episodes,weight,beta,quantile):return float(torch.quantile(torch.tensor([hybrid(r,weight,beta) for s in episodes for r in s if r is not None]),quantile))
def summarize(episodes,weight,beta,cutoff,persistence):
  ys=[]
  for seq in episodes:
    run=0
    for row in seq:
      if row is None or hybrid(row,weight,beta)<=cutoff:run=0;continue
      run+=1
      if run>=persistence:ys.append(row["outcome"]);break
  r=ys.count("RESCUE");h=ys.count("HARM");return {"interventions":len(ys),"rescues":r,"harms":h,"neither":len(ys)-r-h,"net_rescues":r-h}
def exact_control(episodes,budget):
  margins=sorted({r["native_margin"] for s in episodes for r in s if r is not None});best=None
  for cap in margins:
    ys=[]
    for seq in episodes:
      for row in seq:
        if row is not None and row["native_margin"]<=cap:ys.append(row["outcome"]);break
    key=(abs(len(ys)-budget),len(ys)>budget)
    if best is None or key<best[0]:
      r=ys.count("RESCUE");h=ys.count("HARM");best=(key,{"native_margin_max":cap,"interventions":len(ys),"rescues":r,"harms":h,"neither":len(ys)-r-h,"net_rescues":r-h})
  return best[1]
def main():
  prior=json.loads(MF3R_GATE.read_text())
  if prior.get("status")!="SHADOW_GATE_FAIL" or prior.get("ranks24_29_payload_read") is not True:raise RuntimeError("MF3S prior boundary drift")
  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");trains={};strata={};checkpoints=[]
  for fold in FOLDS:
    models,evidence=load_models(fold,device);checkpoints.extend(evidence);trains[fold]=collect(models,"fit",device,manifest_path(fold));strata[fold]=collect(models,"calibration",device,manifest_path(fold))
  final_models,final_checkpoints=load_models("final",device);trains[3]=collect(final_models,"fit",device,manifest_path("final"));strata[3]=collect(final_models,"shadow",device,CONSUMED);candidates=[];controls={}
  for weight in MAD_WEIGHTS:
    for beta in BETAS:
      for quantile in QUANTILES:
        cuts={i:threshold(trains[i],weight,beta,quantile) for i in range(4)}
        for persistence in PERSISTENCE:
          summaries={f"oof_fold_{i}" if i<3 else "consumed_ranks_24_29":summarize(strata[i],weight,beta,cuts[i],persistence) for i in range(4)};pooled={key:sum(x[key] for x in summaries.values()) for key in ("interventions","rescues","harms","neither","net_rescues")};lower=wilson(pooled["rescues"],pooled["harms"]);consumed=summaries["consumed_ranks_24_29"];budget=consumed["interventions"]
          if budget not in controls:controls[budget]=exact_control(strata[3],budget)
          control=controls[budget];qualifies=all(x["interventions"]>=10 and x["net_rescues"]>0 for x in summaries.values()) and lower>.5 and consumed["net_rescues"]>control["net_rescues"]
          candidates.append({"mad_weight":weight,"policy_risk_beta":beta,"training_score_quantile":quantile,"persistence_steps":persistence,"stratum_thresholds":cuts,"strata":summaries,"pooled":pooled,"rescue_precision_wilson95_lower":lower,"consumed_exact_budget_uncertainty":control,"qualifies":qualifies})
  q=[x for x in candidates if x["qualifies"]];rule=max(q,key=lambda x:(x["rescue_precision_wilson95_lower"],min(v["net_rescues"] for v in x["strata"].values()),x["pooled"]["net_rescues"],-x["pooled"]["harms"],x["training_score_quantile"],x["policy_risk_beta"],x["mad_weight"])) if q else None
  uncertainty=None
  if rule:
    rule["final_training_threshold"]=threshold(trains[3],float(rule["mad_weight"]),float(rule["policy_risk_beta"]),float(rule["training_score_quantile"]));uncertainty=rule["consumed_exact_budget_uncertainty"]|{"online_rule":"first eligible native-margin crossing, once per episode","source":"consumed ranks 24-29 exact learned budget"}
  atomic_json(OUT/"MF3S_DEVELOPMENT_SELECTION.json",{"schema_version":"revealnav-mf3s-development-selection/1","status":"DEVELOPMENT_PASS" if rule else "DEVELOPMENT_FAIL","selected_architecture":{"hidden_dim":64,"optimizer_steps":200},"rule_candidates":candidates,"selected_rule":rule,"uncertainty_rule":uncertainty,"stratum_episodes":{"oof_fold_0":len(strata[0]),"oof_fold_1":len(strata[1]),"oof_fold_2":len(strata[2]),"consumed_ranks_24_29":len(strata[3])},"crossfit_checkpoints":checkpoints,"final_checkpoints":final_checkpoints,"ranks30_35_payload_read":False,"design_sha256":sha256_file(DESIGN),**MF3B_SCOPE});return 0 if rule else 2
if __name__=="__main__":raise SystemExit(main())
