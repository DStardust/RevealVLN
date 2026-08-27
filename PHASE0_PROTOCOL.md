# RevealNav Phase 0 Execution Protocol

**Started:** 2026-08-22  
**Frozen method:** Method-Freeze-2  
**Current decision:** IN PROGRESS / NO-GO until every recorded blocker is cleared

## 1. Project boundary

Every runtime input must be independently acquired and stored under
`/mnt/daiyang/vla`. Data, environments, checkpoints, code, caches and symlinks
from other project or user directories are forbidden. System executables may
bootstrap project-local tools, but the final run must not import packages or
read assets outside the project.

Phase 0 may inspect only `train` and `val_seen`. It must not extract, parse,
screen, run or use `val_unseen`, test-set episodes, their statistics or
baseline outputs. If an authoritative upstream asset is available only as one
indivisible archive containing excluded splits, the complete archive may be
retained as a sealed provenance object. Only its central-directory metadata
and whole-archive integrity may be audited; excluded members must never enter
the extracted runtime tree. This packaging exception does not permit reading
episode payloads or computing any excluded-split statistic.

## 2. Evidence lanes

### A. Public metadata and language feasibility

- Canonical provenance: `data/phase0/manifest.json`; human-readable record:
  `data/phase0/MANIFEST.md`
- Machine-readable screening outputs: `artifacts/phase0/*_screening*.json`
- Frozen 50-trajectory manual-review queue: the `samples` array in
  `artifacts/phase0/rxr_train_screening_seed20260822.json`, sampled uniformly
  without replacement from 6,219 unique candidate trajectories;
- Earlier trigger-balanced language precheck:
  `artifacts/phase0/rxr_train_language_pilot.csv` (diagnostic only)
- Independent language-only comparison:
  `artifacts/phase0/LANGUAGE_PRECHECK.md` (diagnostic only; not a human gate)

The lexical screener is deliberately high recall. A screened instruction is
not a Reveal Event. It becomes a valid event only after a reviewer identifies
a genuine branch-dependent constraint and the simulator pipeline produces
strict prefixes, evolving candidates and one reproducible expiry.

Current train counts:

| Dataset | Screened instructions | Unique trajectories | Scenes |
|---|---:|---:|---:|
| RxR-CE-en | 11,487 | 6,219 | 59 |
| R2R-CE | 6,848 | 3,339 | 61 |

### B. Event protocol

`toporeveal.benchmark.RevealEvent` is the single event-level contract. Each
event records dataset/scene/split/episode/event IDs, sensor protocol, candidate
frontend, return controller, hash-chained strict observation prefixes,
candidate IDs, fixed decisive constraints, candidate separation, reveal
interval, resolvability and counterfactual action costs.

U/A/D is never independently typed by an annotator or model label:

- U: target branch is absent from the declared candidate set;
- A: target is present, but separation or decisive evidence is incomplete;
- D: target is present, separable and all decisive evidence is closed.

The event validator requires three consecutive D prefixes. `T_X` is not an
annotator-supplied index: it is derived as the last prefix with a replayable,
SHA-256-bound safe controller witness, followed by an observed prefix with no
safe option. Collection validation rejects scene/split leakage and
counterfactual groups crossing episode boundaries.

### C. Frozen baseline reproduction

- Static audit: `artifacts/upstream/ETP_R1_AUDIT.md`
- Sealed official extra archive:
  `artifacts/upstream/ETP-R1-extra-files-86cacf29.zip`, revision
  `86cacf29b761f736db9948c009f4e4341f9a2c35`, 15,238,627,709 bytes,
  SHA-256
  `f3de48e9184eeff380b4fdc83769131a358e6b7e0ea37e057f4d431cfb027fa0`;
  member audit: `artifacts/upstream/ETP_R1_EXTRA_ARCHIVE_INVENTORY.txt`
- Project-local checkout: `third_party/ETP-R1`
- Pinned commit: `a94b5c8fe20d1631e9e150c430a925543eb1cba9`

The production compatibility path is fixed to Python 3.10, official PyTorch
2.11.0 CUDA 12.8 wheels, Habitat-Sim/Lab 0.1.7 task semantics, and the pinned
ETP-R1 source/checkpoints. Compatibility changes are limited to removed NumPy
aliases, optional TensorFlow imports, trusted checkpoint loading with explicit
`weights_only`, `torchrun` argument handling, project-local caches and strict
state-dict key validation. Moving to Habitat 0.3.x semantics is not an
acceptable substitute for reproducing the frozen baseline.

The smoke-test order is:

1. build the environment entirely under `.envs/etpr1`;
2. verify CUDA execution on one RTX 5090;
3. initialize RGB/depth sensors on one authorized train scene;
4. strictly load joint-pretraining and final R2R/RxR checkpoints;
5. run one train episode, then ten train/val-seen episodes;
6. run RxR with languages explicitly restricted to `en-US,en-IN`;
7. record configuration, seed, GPU, environment count and logs.

No baseline counts as reproduced when `missing_keys` or `unexpected_keys` is
non-empty, or when it reads a cache/path outside the project.

## 3. Frozen go/no-go calculation

The current evidence snapshot is `artifacts/phase0/evidence_current.json` and is
evaluated with:

```bash
.envs/phase0-tools/bin/python scripts/evaluate_phase0.py \
  artifacts/phase0/evidence_current.json
```

The evaluator returns exit 1 for a valid NO-GO and exit 2 for malformed or
unverifiable input. Every positive runtime claim must cite project-local
SHA-256 evidence. Event-count projection is not an input field: it is derived
from the manually reviewed rate using its 95% Wilson lower bound and at most
one projected event per unique candidate trajectory.

GO requires all of the following:

- self-contained project runtime;
- authorized project-local 90-scene MP3D;
- verified official R2R-CE/RxR-CE metadata;
- runnable Habitat, frozen waypoint frontend and ETP-R1 checkpoint;
- 50 reviewed candidates and at least 25% valid-event rate;
- at least 300 estimated valid Reveal Events;
- every accepted event has one unique reproducible expiry.

## 4. Storage policy

The user-created `.disk_reserve/reserve_10G_*.bin` files reserve shared `/mnt`
capacity. They are not data and must remain untouched until an actual download
or build needs space. Release only explicit files, immediately before use, and
record the released byte count. Do not release reserve capacity merely for
convenience or speculative future work.
