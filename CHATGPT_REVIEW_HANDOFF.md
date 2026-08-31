# RevealVLN — MF3ZO temporal oracle-gap review handoff

Snapshot date: 2026-08-31

Repository: `DStardust/RevealVLN`

Purpose: request an independent audit and the next scientifically legitimate
data/label protocol. This snapshot must not be interpreted as a request for
another single-decision gate or for tuning on a public split.

## Request to the reviewer

Please inspect the implementation, tests, sealed protocols, and compact result
artifacts at this repository revision. Then answer:

1. Is MF3ZO's fail-closed Probe-A outcome correctly classified as a
   supervision/data-readiness failure rather than a numerical disproof of
   temporal or oracle-state relevance?
2. What is the smallest new, versioned data/sensor-observation protocol that
   can produce trustworthy per-prefix embeddings and independently verified
   U/A/D, Reveal, and Expiry labels without consuming a public split or the old
   confirmation scenes?
3. Which fields can be generated automatically from frozen rollout evidence,
   which require human review, and what agreement/adjudication standard is
   needed before a formal identifiability audit?
4. Should a new pilot reuse the already-consumed development scenes under a
   newly sealed observation/label protocol, or must it reserve a new scene
   cohort? Please separate statistical validity from engineering convenience.
5. What one-shot PASS/FAIL criteria should be frozen before collecting those
   labels? Do not propose model, threshold, regularization, loss, or feature
   searches on the current 39-scene universe.

## Frozen historical conclusion

MF3ZM-CAR's implementation, nested whole-scene protocol, fast/reference
equivalence, and provenance were audited. Its scientific experiment failed:
five outer folds had no feasible inner candidate, and semantic, 28D,
policy-only, hard/soft/no-risk, scene/no-scene, pooled, RxR-only, and R2R-only
variants did not restore feasibility on 1,540 exact events.

Accordingly:

```text
FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED = true
```

CAR v2, RCSP v2, another single-decision gate, threshold/WD/loss/architecture
sweeps, and longer-training rescue attempts are out of scope. The prior
CAR/RCSP/DSR results must not be rerun or overwritten.

## MF3ZN/TUAD status

`mf3zn_tuad_v1` and its data protocol `mf3zn_teal_v1` implement the intended
next information structure:

```text
strict causal temporal state
  -> U/A/D and Reveal/Expiry identifiability
  -> native-inclusive exact counterfactual action selection
```

The formal protocol is:

```text
artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json
SHA-256 b502629d898879c65031a92b91496fd39d640e7c0f09097bd8bce8ebd9118772
status SEALED_BEFORE_IDENTIFIABILITY_RESULTS
```

Formal collection, TUAD training, checkpoint generation, and every public
split remain unauthorized. Existing historical traces have 7,572 prefix rows
from 1,448 trace files, but they lack complete per-prefix 768D embeddings and
complete trustworthy U/A/D, Reveal, and Expiry supervision.

A separately labeled exploratory no-review scalar-history ridge pretest failed
on both R2R and RxR. That result only rejects that fixed scalar-history probe;
it does not establish that all temporal evidence or full TUAD is ineffective.

## MF3ZO one-shot pilot

MF3ZO is an observability/oracle-gap audit, not a deployment algorithm or new
gate. Its method and protocol were sealed before results. The protocol is:

```text
artifacts/training/mf3zo_temporal_oracle_gap_v1/
  MF3ZO_TEMPORAL_ORACLE_GAP_PROTOCOL.json
SHA-256 74f2b0e737f9d5a89cfd1ad63ae9bfc93b4245ddcfa6179d9e2dd6d1471c989f
status SEALED_BEFORE_TEMPORAL_ORACLE_GAP_RESULTS
```

The deterministic, outcome-blind pilot contains 150 events across all 39 raw
MP3D development scenes:

| domain | events | scenes |
|---|---:|---:|
| R2R | 75 | 37 |
| RxR | 75 | 38 |

It reconstructs 681 strictly causal prefix rows (length 1–19, mean 4.54).
All 150 decision-time snapshots have checkpoint/native/runner embeddings, but
only 1/150 events has complete per-prefix checkpoint and executable-candidate
embedding coverage. Most importantly, 0/150 has a complete independently
verified oracle record. Consumed-confirmation overlap is empty, and all public
split access flags are false.

The pilot physically separates `CausalTemporalRecord` from
`TemporalOracleLabel`; inference tensor builders reject outcome, oracle,
future, geometry, and pose fields. Missing oracle fields remain explicitly
unavailable rather than being filled with a heuristic surrogate.

## First scientific stop

Probe A could not be estimated honestly:

```text
status       TEMPORAL_ORACLE_RELEVANCE_FAIL
executed     false
failure_kind REQUIRED_ORACLE_SUPERVISION_UNAVAILABLE
complete verified oracle labels 0 / 150
target payload read false
surrogate labels substituted false
```

The one-shot stop rule therefore triggered before Probe B or C. No numerical
claim about oracle-state relevance was produced. No TEAL collection, full TUAD
training, checkpoint, or public evaluation occurred.

The precise supported conclusion is:

> The fixed MF3ZO pilot stopped at Probe A because required independently
> verified U/A/D and Reveal/Expiry supervision was unavailable. This is a
> support/data-readiness failure, not evidence that oracle state or all
> temporal evidence is ineffective.

## Evidence in this Git snapshot

Review these files together:

- `METHOD_REVISION_3ZN_TUAD.md`
- `METHOD_REVISION_3ZO_TEMPORAL_ORACLE_GAP.md`
- `artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json`
- `artifacts/training/mf3zn_tuad_exploratory_no_review_v1/MF3ZN_NO_REVIEW_TEMPORAL_PRETEST.json`
- `artifacts/training/mf3zo_temporal_oracle_gap_v1/MF3ZO_TEMPORAL_ORACLE_GAP_PROTOCOL.json`
- `artifacts/training/mf3zo_temporal_oracle_gap_v1/MF3ZO_PILOT_DATA_AUDIT.json`
- `artifacts/training/mf3zo_temporal_oracle_gap_v1/MF3ZO_PROBE_A_ORACLE_RELEVANCE.json`
- `artifacts/training/mf3zo_temporal_oracle_gap_v1/MF3ZO_FINAL_RESULT.json`
- `revealnav_mf3/mf3zo_*.py`
- `scripts/run_mf3zo_temporal_oracle_gap.py`
- `tests/test_mf3zo_*.py`

The full regression suite reports 359/359 passing tests.

## Deliberately omitted from Git

Raw Matterport/R2R/RxR payloads, images, checkpoints, virtual environments,
caches, credentials, large logs, and the 150 derived embedding arrays remain
local. Their hashes and coverage metadata are retained in the compact audit
artifacts. This omission avoids redistributing dataset-derived arrays and does
not change the reported fail-closed result.
