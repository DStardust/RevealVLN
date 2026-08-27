# TopoReveal / RevealNav research prototype

This repository contains the first, dependency-free prototype for the
recoverable-commitment component of RevealNav.  It deliberately does not
implement perception, SLAM, or a VLN backbone.  Those systems should provide
calibrated branch and reveal estimates through the typed interfaces in
`toporeveal/`.

The current prototype covers:

- language-conditioned option-checkpoint admission;
- a topological memory of recoverable branch choices;
- risk-aware branch ranking and sequential commitment;
- public-benchmark annotation invariants for U/A/D reveal events;
- a high-recall metadata screener for official RxR-CE/R2R-CE annotations.

Run the tests with:

```bash
.envs/phase0-tools/bin/python -m unittest discover -s tests -v
```

The current Phase 0 tool interpreter is project-local at
`.envs/phase0-tools`; its managed CPython 3.10 installation and package search
path remain under `/mnt/daiyang/vla`.

Phase 0 screening is restricted to official train/val-seen VLN-CE metadata.
For example:

```bash
.envs/phase0-tools/bin/python scripts/screen_vlnce_instructions.py \
  data/phase0/raw/rxr_vlnce_v0/train/train_guide.json.gz \
  --language en-US --language en-IN \
  --sample-count 50 --sample-seed 20260822 \
  --output artifacts/phase0/rxr_train_screening_seed20260822.json
```

Dataset and split are derived from the hash-bound project manifest, not caller
arguments. `val_unseen` is absent from that manifest.

The frozen research scope, novelty boundary, public-data plan, metrics, and
stop/go criteria are documented in [FROZEN_SPEC.md](FROZEN_SPEC.md).
The active self-contained feasibility run is tracked in
[PHASE0_PROTOCOL.md](PHASE0_PROTOCOL.md).
