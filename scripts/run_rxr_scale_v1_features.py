#!/usr/bin/env python3
"""Extract the automatic scale-v1 frozen feature manifest."""

import run_rxr_multibranch_feature_v2 as features


ROOT = features.ROOT
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1/automatic"
V2 = BASE / "multibranch"
features.V2 = V2
features.INDEX = V2 / "RXR_SCALE_TRAINING_INDEX.json"
features.TX = V2 / "RXR_SCALE_TX_GATE.json"
features.WORKER = ROOT / "scripts/rxr_scale_v1_feature_lane.py"
features.FEATURES = V2 / "frozen_features"
features.OUT = V2 / "RXR_SCALE_FEATURE_MANIFEST.json"
features.GATE = V2 / "RXR_SCALE_FEATURE_GATE.json"
features.REQUIRED_SPLITS = ("train", "development")
features.HUMAN_AUDIT_STATUS = "NOT_PERFORMED_AUTOMATIC_SCALE"
features.REMAINING_BLOCKER = None
features.FEATURE_GATE_PASS_STATUS = "FEATURE_GATE_PASS_AUTOMATIC_SCALE_READY"
features.TRAINING_AUTHORIZED_AFTER_FEATURE_GATE = True
features.RECORD_EXTRA = {
    "label_source": "automatic_scale_pseudolabel",
    "quality_role": "train_or_development_only",
}
features.METADATA_EXTRA = {
    "label_source": "automatic_scale_pseudolabel",
    "evaluation_use_authorized": False,
    "gold_payload_read": False,
}


if __name__ == "__main__":
    raise SystemExit(features.main())
