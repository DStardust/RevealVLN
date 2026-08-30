#!/usr/bin/env python3
"""Train MF3O with the sealed selective-utility class costs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE, OnlineUADFeatureDataset, PairwiseSwitchUtility,
    collate_online_uad, top2_cost_sensitive_utility_loss,
    top2_expected_switch_utility, top2_switch_targets,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, move, sha256_file  # noqa: E402
from scripts.train_rxr_uad_top2_utility_mf3n import build_data_view, validate_data  # noqa: E402

SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIMS = (32, 64)
DATA = ROOT / "artifacts/phase1/mf3n_top2_utility_rank23/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json"
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3O_COST_SENSITIVE_TOP2.md"
MF3N_SELECTION = ROOT / "artifacts/evaluation/mf3n_top2_utility_development_v1/MF3N_DEVELOPMENT_SELECTION.json"
OUT = ROOT / "artifacts/training/mf3o_cost_sensitive_top2_v1"
EPOCHS, PATIENCE = 24, 5
CLASS_WEIGHTS = (1.0, 2.0, 0.5)


def protocol() -> dict:
    validate_data()
    return {
        "schema_version": "revealnav-mf3o-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "architectures": [{"hidden_dim": h, "seeds": list(SEEDS)} for h in HIDDEN_DIMS],
        "model": "PairwiseSwitchUtility frozen-runner-up projection",
        "outcomes": ["NEITHER", "RESCUE", "HARM"],
        "class_weights": {"NEITHER": 1.0, "RESCUE": 2.0, "HARM": 0.5},
        "score": "P(RESCUE)-P(HARM)",
        "loss": "episode-balanced cost-sensitive three-class CE",
        "epochs": EPOCHS, "patience": PATIENCE,
        "epoch_selection": "minimum ranks18-20 cost-sensitive top-2 NLL",
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "data_sha256": sha256_file(DATA), "design_sha256": sha256_file(DESIGN),
        "mf3n_selection_sha256": sha256_file(MF3N_SELECTION), **MF3B_SCOPE,
    }


def eval_model(model, loader, device) -> dict:
    model.eval(); losses=[]; steps=correct=rescue=harm=neither=0
    with torch.no_grad():
        for cpu in loader:
            b=move(cpu,device); o=model(b["history_embeddings"],b["candidate_embeddings"],b["candidate_mask"],b["instruction_embedding"],b["native_scores"],b["native_index"])
            losses.append(float(top2_cost_sensitive_utility_loss(o,b)))
            score,runner,valid=top2_expected_switch_utility(o,b); labels,_,_=top2_switch_targets(b)
            z=o.outcome_logits.gather(2,runner[...,None,None].expand(*runner.shape,1,3)).squeeze(2)
            steps+=int(valid.sum()); correct+=int((z.argmax(-1)[valid]==labels[valid]).sum())
            selected=valid&(score>0); rescue+=int((selected&(labels==1)).sum()); harm+=int((selected&(labels==2)).sum()); neither+=int((selected&(labels==0)).sum())
    if not losses: raise RuntimeError("MF3O evaluation has no supervision")
    return {"episodes":len(losses),"steps":steps,"top2_nll":statistics.mean(losses),"top2_accuracy":correct/steps,"zero_threshold_rescues":rescue,"zero_threshold_harms":harm,"zero_threshold_neither":neither}


def seal() -> int:
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/"MF3O_TRAINING_PROTOCOL.json"; value=protocol()
    if p.exists() and json.loads(p.read_text())!=value: raise RuntimeError("MF3O protocol drift")
    if not p.exists(): atomic_json(p,value)
    return 0


def train(hidden:int, seed:int, device:torch.device) -> int:
    if hidden not in HIDDEN_DIMS or seed not in SEEDS: raise ValueError("unsealed MF3O run")
    pp=OUT/"MF3O_TRAINING_PROTOCOL.json"
    if json.loads(pp.read_text())!=protocol(): raise RuntimeError("MF3O protocol drift")
    run=OUT/f"hidden_{hidden}/seed_{seed}"; run.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    fit=DataLoader(OnlineUADFeatureDataset(DATA,"fit"),batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=collate_online_uad)
    cal=DataLoader(OnlineUADFeatureDataset(DATA,"calibration"),batch_size=1,shuffle=False,collate_fn=collate_online_uad)
    model=PairwiseSwitchUtility(768,1536,hidden).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    best=None; best_state=None; stale=0; history=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); train_losses=[]
        for cpu in fit:
            b=move(cpu,device); opt.zero_grad(set_to_none=True); o=model(b["history_embeddings"],b["candidate_embeddings"],b["candidate_mask"],b["instruction_embedding"],b["native_scores"],b["native_index"]); loss=top2_cost_sensitive_utility_loss(o,b)
            if not torch.isfinite(loss): raise RuntimeError("non-finite MF3O loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); train_losses.append(float(loss.detach()))
        metrics=eval_model(model,cal,device); key=(-metrics["top2_nll"],metrics["top2_accuracy"]); history.append({"epoch":epoch,"train_top2_nll":statistics.mean(train_losses),**metrics})
        if best is None or key>best: best=key; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=PATIENCE: break
    model.load_state_dict(best_state,strict=True); final=eval_model(model,cal,device); ck=run/"cost_sensitive_mf3o.pt"
    torch.save({"schema_version":"revealnav-mf3o-checkpoint/1","hidden_dim":hidden,"seed":seed,"candidate_feature_dim":1536,"class_weights":CLASS_WEIGHTS,"model_state_dict":model.state_dict(),"protocol_sha256":sha256_file(pp),**MF3B_SCOPE},ck)
    atomic_json(run/"RESULT.json",{"status":"TRAINING_COMPLETE","hidden_dim":hidden,"seed":seed,"calibration":final,"history":history,"checkpoint":{"path":str(ck.relative_to(ROOT)),"bytes":ck.stat().st_size,"sha256":sha256_file(ck)},**MF3B_SCOPE}); return 0


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--seal",action="store_true"); ap.add_argument("--hidden",type=int); ap.add_argument("--seed",type=int); ap.add_argument("--device",default="cuda"); a=ap.parse_args()
    if a.seal:return seal()
    if a.hidden is None or a.seed is None: raise SystemExit("training requires --hidden and --seed")
    return train(a.hidden,a.seed,torch.device(a.device))

if __name__=="__main__": raise SystemExit(main())
