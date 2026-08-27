#!/usr/bin/env python3
"""Extract the train-only secondary frozen-feature manifest."""

import run_rxr_multibranch_feature_v2 as features


ROOT = features.ROOT
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
V2 = BASE / "multibranch"
features.V2 = V2
features.INDEX = V2 / "RXR_SECONDARY_TRAINING_INDEX.json"
features.TX = V2 / "RXR_SECONDARY_TX_GATE.json"
features.WORKER = ROOT / "scripts/rxr_secondary_feature_lane.py"
features.FEATURES = V2 / "frozen_features"
features.OUT = V2 / "RXR_SECONDARY_FEATURE_MANIFEST.json"
features.GATE = V2 / "RXR_SECONDARY_FEATURE_GATE.json"
features.REQUIRED_SPLITS = ("train",)
features.HUMAN_AUDIT_STATUS = "NOT_PERFORMED_AUTOMATIC_TRAIN_ONLY"
features.REMAINING_BLOCKER = None
features.FEATURE_GATE_PASS_STATUS = "FEATURE_GATE_PASS_AUTOMATIC_TRAIN_READY"
features.TRAINING_AUTHORIZED_AFTER_FEATURE_GATE = True
features.RECORD_EXTRA = {
    "label_source": "automatic_secondary_pseudolabel",
    "quality_role": "train_only",
}
features.METADATA_EXTRA = {
    "label_source": "automatic_secondary_pseudolabel",
    "evaluation_use_authorized": False,
    "requires_primary_only_vs_augmented_ablation": True,
}


if __name__ == "__main__":
    raise SystemExit(features.main())
