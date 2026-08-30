#!/usr/bin/env python3
"""Train the sealed MF3P rescue-versus-harm ensemble."""

from __future__ import annotations

import argparse, copy, json, os, random, statistics, sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from revealnav_mf3 import (MF3B_SCOPE,OnlineUADFeatureDataset,PairwiseSwitchUtility,
    collate_online_uad,top2_rescue_harm_logit,top2_rescue_harm_loss,top2_switch_targets)
from scripts.train_rxr_uad_correction_mf3e import atomic_json,move,sha256_file
from scripts.train_rxr_uad_top2_utility_mf3n import validate_data

SEEDS=(20260826,20260827,20260828); HIDDEN_DIMS=(64,128)
DATA=ROOT/"artifacts/phase1/mf3n_top2_utility_rank23/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json"
DESIGN=ROOT/"artifacts/design/METHOD_FREEZE_3P_ONLINE_RESCUE_HARM.md"
MF3O_SELECTION=ROOT/"artifacts/evaluation/mf3o_cost_sensitive_top2_development_v1/MF3O_DEVELOPMENT_SELECTION.json"
OUT=ROOT/"artifacts/training/mf3p_rescue_harm_v1"
POSITIVE_WEIGHT=7700/1437; EPOCHS=24; PATIENCE=5

def protocol():
    validate_data(); prior=json.loads(MF3O_SELECTION.read_text())
    if prior.get("status")!="DEVELOPMENT_FAIL" or prior.get("ranks24_29_payload_read",False): raise RuntimeError("MF3P prior boundary drift")
    return {"schema_version":"revealnav-mf3p-training-protocol/1","status":"SEALED_BEFORE_TRAINING",
      "architectures":[{"hidden_dim":h,"seeds":list(SEEDS)} for h in HIDDEN_DIMS],
      "model":"PairwiseSwitchUtility frozen-runner-up rescue-harm logit",
      "fit_consequential_counts":{"RESCUE":1437,"HARM":7700},"rescue_positive_weight":POSITIVE_WEIGHT,
      "loss":"episode-balanced consequential binary BCE","epochs":EPOCHS,"patience":PATIENCE,
      "epoch_selection":"minimum ranks18-20 consequential BCE",
      "optimizer":{"name":"AdamW","lr":3e-4,"weight_decay":1e-4},
      "data_sha256":sha256_file(DATA),"design_sha256":sha256_file(DESIGN),"mf3o_selection_sha256":sha256_file(MF3O_SELECTION),**MF3B_SCOPE}

def eval_model(model,loader,device):
    model.eval();losses=[];count=correct=0
    with torch.no_grad():
      for cpu in loader:
        b=move(cpu,device);o=model(b["history_embeddings"],b["candidate_embeddings"],b["candidate_mask"],b["instruction_embedding"],b["native_scores"],b["native_index"])
        losses.append(float(top2_rescue_harm_loss(o,b,rescue_positive_weight=POSITIVE_WEIGHT)))
        logit,_,valid=top2_rescue_harm_logit(o,b);labels,_,_=top2_switch_targets(b); consequential=valid&((labels==1)|(labels==2));count+=int(consequential.sum());correct+=int(((logit>0)==(labels==1))[consequential].sum())
    if not losses or not count:raise RuntimeError("MF3P evaluation has no consequential labels")
    return {"episodes":len(losses),"consequential_steps":count,"balanced_bce":statistics.mean(losses),"balanced_accuracy_at_zero":correct/count}

def seal():
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/"MF3P_TRAINING_PROTOCOL.json";v=protocol()
    if p.exists() and json.loads(p.read_text())!=v:raise RuntimeError("MF3P protocol drift")
    if not p.exists():atomic_json(p,v)
    return 0

def train(hidden,seed,device):
    if hidden not in HIDDEN_DIMS or seed not in SEEDS:raise ValueError("unsealed MF3P run")
    pp=OUT/"MF3P_TRAINING_PROTOCOL.json"
    if json.loads(pp.read_text())!=protocol():raise RuntimeError("MF3P protocol drift")
    run=OUT/f"hidden_{hidden}/seed_{seed}"
    if run.exists():raise RuntimeError(f"refusing to overwrite {run}")
    run.mkdir(parents=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
    fit=DataLoader(OnlineUADFeatureDataset(DATA,"fit"),batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=collate_online_uad)
    cal=DataLoader(OnlineUADFeatureDataset(DATA,"calibration"),batch_size=1,shuffle=False,collate_fn=collate_online_uad)
    model=PairwiseSwitchUtility(768,1536,hidden).to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    best=None;state=None;stale=0;history=[]
    for epoch in range(1,EPOCHS+1):
      model.train();train_losses=[]
      for cpu in fit:
        b=move(cpu,device);opt.zero_grad(set_to_none=True);o=model(b["history_embeddings"],b["candidate_embeddings"],b["candidate_mask"],b["instruction_embedding"],b["native_scores"],b["native_index"]);loss=top2_rescue_harm_loss(o,b,rescue_positive_weight=POSITIVE_WEIGHT)
        if not torch.isfinite(loss):raise RuntimeError("non-finite MF3P loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();train_losses.append(float(loss.detach()))
      metrics=eval_model(model,cal,device);key=(-metrics["balanced_bce"],metrics["balanced_accuracy_at_zero"]);history.append({"epoch":epoch,"train_balanced_bce":statistics.mean(train_losses),**metrics})
      if best is None or key>best:best=key;state=copy.deepcopy(model.state_dict());stale=0
      else:
        stale+=1
        if stale>=PATIENCE:break
    model.load_state_dict(state,strict=True);final=eval_model(model,cal,device);ck=run/"rescue_harm_mf3p.pt"
    torch.save({"schema_version":"revealnav-mf3p-checkpoint/1","hidden_dim":hidden,"seed":seed,"candidate_feature_dim":1536,"rescue_positive_weight":POSITIVE_WEIGHT,"model_state_dict":model.state_dict(),"protocol_sha256":sha256_file(pp),**MF3B_SCOPE},ck)
    atomic_json(run/"RESULT.json",{"status":"TRAINING_COMPLETE","hidden_dim":hidden,"seed":seed,"calibration":final,"history":history,"checkpoint":{"path":str(ck.relative_to(ROOT)),"bytes":ck.stat().st_size,"sha256":sha256_file(ck)},**MF3B_SCOPE});return 0

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seal",action="store_true");ap.add_argument("--hidden",type=int);ap.add_argument("--seed",type=int);ap.add_argument("--device",default="cuda");a=ap.parse_args()
    if a.seal:return seal()
    if a.hidden is None or a.seed is None:raise SystemExit("training requires --hidden and --seed")
    return train(a.hidden,a.seed,torch.device(a.device))
if __name__=="__main__":raise SystemExit(main())
