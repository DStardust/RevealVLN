#!/usr/bin/env python3
"""Open MF3P ranks 24--29 once after a sealed development pass."""
from __future__ import annotations
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from revealnav_mf3 import MF3B_SCOPE
from scripts.select_rxr_uad_rescue_harm_mf3p import OUT as DEV,collect,first_crossing,load_models,summarize,uncertainty_summary
from scripts.train_rxr_uad_correction_mf3e import atomic_json,sha256_file
DATA=ROOT/"artifacts/phase1/mf3n_top2_utility_rank29/dataset_v1/MF3B_ONLINE_DATA_MANIFEST.json"
SELECTION=DEV/"MF3P_DEVELOPMENT_SELECTION.json";OUT=ROOT/"artifacts/evaluation/mf3p_rescue_harm_shadow_gate_v1"
def main():
  selection=json.loads(SELECTION.read_text())
  if not(selection.get("status")=="DEVELOPMENT_PASS" and selection.get("ranks24_29_payload_read") is False):raise RuntimeError("MF3P development does not authorize shadow")
  manifest=json.loads(DATA.read_text())
  if not(manifest.get("status")=="PASS" and manifest.get("counts")=={"fit":967,"calibration":168,"diagnostic":168,"shadow":336}):raise RuntimeError("MF3P fresh manifest drift")
  architecture=selection["selected_architecture"];rule=selection["selected_rule"];device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");models,checkpoints=load_models(int(architecture["hidden_dim"]),device);episodes=collect(models,"shadow",device,DATA)
  shadow=summarize(episodes,float(rule["mad_weight"]),float(rule["logit_threshold"]),int(rule["persistence_steps"]));uncertainty=uncertainty_summary(episodes,float(selection["uncertainty_rule"]["native_margin_max"]));gates={"fresh_shadow_has_fifteen_interventions":shadow["interventions"]>=15,"fresh_shadow_net_rescue_positive":shadow["net_rescues"]>0,"fresh_shadow_beats_uncertainty":shadow["net_rescues"]>uncertainty["net_rescues"] or (shadow["net_rescues"]==uncertainty["net_rescues"] and shadow["harms"]<uncertainty["harms"])};passed=all(gates.values())
  atomic_json(OUT/"MF3P_SHADOW_GATE.json",{"schema_version":"revealnav-mf3p-shadow-gate/1","status":"SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL","selected_architecture":architecture,"selected_rule":rule,"uncertainty_rule":selection["uncertainty_rule"],"shadow":shadow,"uncertainty_matched_shadow":uncertainty,"gates":gates,"task_metric_run_authorized":passed,"ranks24_29_payload_read":True,"fresh_data":{"path":str(DATA.relative_to(ROOT)),"bytes":DATA.stat().st_size,"sha256":sha256_file(DATA),"shadow_episodes":336},"checkpoints":checkpoints,**MF3B_SCOPE});return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
