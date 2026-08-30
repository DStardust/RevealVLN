#!/usr/bin/env python3
"""Create deterministic MF3Q folds and train fixed-step models."""
from __future__ import annotations
import argparse,hashlib,json,os,random,statistics,sys
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from revealnav_mf3 import MF3B_SCOPE,OnlineUADFeatureDataset,PairwiseSwitchUtility,collate_online_uad,top2_rescue_harm_loss
from scripts.train_rxr_uad_correction_mf3e import atomic_json,move,sha256_file
SOURCE=ROOT/"artifacts/phase1/mf3m_robust_top2_rank23/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json"
DESIGN=ROOT/"artifacts/design/METHOD_FREEZE_3Q_CROSSFIT_RESCUE_HARM.md"
MF3P_SELECTION=ROOT/"artifacts/evaluation/mf3p_rescue_harm_development_v1/MF3P_DEVELOPMENT_SELECTION.json"
OUT=ROOT/"artifacts/training/mf3q_crossfit_v1";SEEDS=(20260826,20260827,20260828);FOLDS=(0,1,2);STEPS=200;POSITIVE_WEIGHT=7700/1437

def fold_of(eid):return int(hashlib.sha256(f"mf3q-fold:{eid}".encode()).hexdigest(),16)%3
def manifest_path(fold):
  name="mf3q_final_rank23" if fold=="final" else f"mf3q_crossfit_fold{fold}"
  return ROOT/f"artifacts/phase1/{name}/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json"
def build_views():
  source=json.loads(SOURCE.read_text())
  if source.get("status")!="PASS" or len(source.get("records",[]))!=1303:raise RuntimeError("MF3Q source drift")
  outputs={}
  for fold in (*FOLDS,"final"):
    records=[]
    for original in source["records"]:
      row=dict(original);row["source_split"]=row["split"];row["crossfit_fold"]=fold_of(str(row["episode_id"]));row["split"]="fit" if fold=="final" or row["crossfit_fold"]!=fold else "calibration";records.append(row)
    counts={s:sum(r["split"]==s for r in records) for s in ("fit","calibration")}
    value={"schema_version":"revealnav-mf3b-online-manifest/1","status":"PASS","counts":counts,"failures":[],"records":records,"source_manifest":{"path":str(SOURCE.relative_to(ROOT)),"sha256":sha256_file(SOURCE)},"crossfit":{"fold":fold,"formula":"sha256(mf3q-fold:<episode_id>) mod 3","payload_copied":False},"public_unseen_authorized":False};outputs[fold]=value
  return outputs
def protocol():
  prior=json.loads(MF3P_SELECTION.read_text())
  if prior.get("status")!="DEVELOPMENT_FAIL" or prior.get("ranks24_29_payload_read",False):raise RuntimeError("MF3Q prior boundary drift")
  return {"schema_version":"revealnav-mf3q-training-protocol/1","status":"SEALED_BEFORE_TRAINING","folds":list(FOLDS),"final_model":True,"hidden_dim":64,"seeds":list(SEEDS),"optimizer_steps":STEPS,"rescue_positive_weight":POSITIVE_WEIGHT,"optimizer":{"name":"AdamW","lr":3e-4,"weight_decay":1e-4},"design_sha256":sha256_file(DESIGN),"source_sha256":sha256_file(SOURCE),"mf3p_selection_sha256":sha256_file(MF3P_SELECTION),**MF3B_SCOPE}
def seal():
  for fold,value in build_views().items():
    p=manifest_path(fold);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and json.loads(p.read_text())!=value:raise RuntimeError("MF3Q view drift")
    if not p.exists():atomic_json(p,value)
  OUT.mkdir(parents=True,exist_ok=True);p=OUT/"MF3Q_TRAINING_PROTOCOL.json";v=protocol()
  if p.exists() and json.loads(p.read_text())!=v:raise RuntimeError("MF3Q protocol drift")
  if not p.exists():atomic_json(p,v)
  return 0
def train(fold,seed,device):
  if fold not in (*FOLDS,"final") or seed not in SEEDS:raise ValueError("unsealed MF3Q run")
  pp=OUT/"MF3Q_TRAINING_PROTOCOL.json"
  if json.loads(pp.read_text())!=protocol():raise RuntimeError("MF3Q protocol drift")
  run=OUT/f"fold_{fold}/seed_{seed}"
  if run.exists():raise RuntimeError(f"refusing to overwrite {run}")
  run.mkdir(parents=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
  loader=DataLoader(OnlineUADFeatureDataset(manifest_path(fold),"fit"),batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=collate_online_uad);iterator=iter(loader);model=PairwiseSwitchUtility(768,1536,64).to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);losses=[]
  for step in range(STEPS):
    try:cpu=next(iterator)
    except StopIteration:iterator=iter(loader);cpu=next(iterator)
    b=move(cpu,device);opt.zero_grad(set_to_none=True);o=model(b["history_embeddings"],b["candidate_embeddings"],b["candidate_mask"],b["instruction_embedding"],b["native_scores"],b["native_index"]);loss=top2_rescue_harm_loss(o,b,rescue_positive_weight=POSITIVE_WEIGHT)
    if not torch.isfinite(loss):raise RuntimeError("non-finite MF3Q loss")
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  ck=run/"crossfit_mf3q.pt";torch.save({"schema_version":"revealnav-mf3q-checkpoint/1","fold":fold,"seed":seed,"hidden_dim":64,"optimizer_steps":STEPS,"rescue_positive_weight":POSITIVE_WEIGHT,"model_state_dict":model.state_dict(),"protocol_sha256":sha256_file(pp),**MF3B_SCOPE},ck);atomic_json(run/"RESULT.json",{"status":"TRAINING_COMPLETE","fold":fold,"seed":seed,"optimizer_steps":STEPS,"train_balanced_bce_mean":statistics.mean(losses),"checkpoint":{"path":str(ck.relative_to(ROOT)),"bytes":ck.stat().st_size,"sha256":sha256_file(ck)},**MF3B_SCOPE});return 0
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--seal",action="store_true");ap.add_argument("--fold");ap.add_argument("--seed",type=int);ap.add_argument("--device",default="cuda");a=ap.parse_args()
  if a.seal:return seal()
  fold="final" if a.fold=="final" else int(a.fold) if a.fold is not None else None
  if fold is None or a.seed is None:raise SystemExit("training requires fold and seed")
  return train(fold,a.seed,torch.device(a.device))
if __name__=="__main__":raise SystemExit(main())
