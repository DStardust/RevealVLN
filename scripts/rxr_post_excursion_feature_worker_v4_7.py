#!/usr/bin/env python3
"""Full-generation entry point for the accepted V4.6.1 feature worker."""

from pathlib import Path

import rxr_post_excursion_feature_worker_v4_6 as worker


worker.PROTOCOL = Path("/mnt/daiyang/vla").resolve() / (
    "artifacts/phase1/rxr_train_expansion/post_excursion_v4_7/"
    "RXR_POST_EXCURSION_FULL_PROTOCOL_V4_7.json"
)


if __name__ == "__main__":
    raise SystemExit(worker.main())
