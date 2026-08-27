#!/usr/bin/env python3
"""Extract frozen features for one automatic scale-v1 lane."""

import rxr_multibranch_feature_v2_lane as lane


ROOT = lane.ROOT
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1/automatic"
V2 = BASE / "multibranch"
lane.BASE = BASE
lane.V2 = V2
lane.INDEX = V2 / "RXR_SCALE_TRAINING_INDEX.json"
lane.CAUSAL = V2 / "RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json"
lane.TX_RUNS = V2 / "tx_runs/round1"


if __name__ == "__main__":
    raise SystemExit(lane.main())

