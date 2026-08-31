# RevealVLN — MF3ZL-RCSP v1r1 review handoff

Snapshot date: 2026-08-31
Repository: `DStardust/RevealVLN`
Purpose: give an independent ChatGPT/reviewer enough current, machine-readable
evidence to assess the method and recommend one scientifically valid next step.

## Request to the reviewer

Please review the implementation and the linked compact artifacts, with the
latest result treated as a genuine train-development failure. Do not assume
that a positive result is required. In particular, answer:

1. Is the failure more consistent with an implementation/protocol defect, an
   objective/representation problem, or insufficient support coverage?
2. What is the smallest *versioned* algorithm revision worth attempting next,
   without tuning on consumed confirmation data or any public split?
3. Which controls and diagnostics are required before authorizing an unseen
   evaluation? Separate correctness checks from publishable evidence.
4. Does the proposed RCSP formulation make a defensible contribution over a
   frozen-policy reranker, margin baseline, selective prediction gate, or
   residual policy?
5. Should the frozen-proposal gate line be continued, or should the proposal
   mechanism/causal observation state be revised instead?

Please inspect the source rather than inferring success from names. Do not
recommend threshold/feature changes based on the old confirmation outcomes.

## Frozen boundaries

`FROZEN_SPEC.md` and the earlier accepted revisions remain unchanged. The
ETP-R1 policy, MF3V proposal ranker, MF3ZG proposal hierarchy, checkpoints,
utility definition, exact one-switch pairing rule, and public-split prohibition
are frozen. No revision in this snapshot fine-tunes the backbone, changes the
proposal hierarchy, constructs non-exact pairs, or reads `val_seen`,
`val_unseen`, or `test` for selection.

The old 52-episode confirmation is consumed evidence. It is retained only as
a retrospective failure and is not reused for fitting, feature design,
threshold selection, or a claim of fresh validation.

## Current data and audit status

The v1r1 dense replay was outcome-blind and restricted to already-consumed
development scenes. Its native shadow pass completed 2,703/2,703 episodes;
290 targeted treatments completed 290/290 with no runtime failures. The
versioned independent audit corrected a record-field naming mistake in the
first audit without changing labels or rollouts:

| source | unique exact events | scenes |
|---|---:|---:|
| RxR combined development | 997 | 38 |
| R2R combined development | 543 | 37 |
| joint canonical set | 1,540 | 39 |

The corrected audit reports zero identity conflicts and
`rcsp_training_authorized=true`. Public split access remains false. The R2R
v1r1 variant contribution is 290 events (151 positive, 19 catastrophic);
these counts are descriptive, not an outcome-dependent stopping rule.

## Latest algorithm results

`MF3ZL-RCSP v1r1 train` uses the zero-relative-delta correctness revision of
RCSP, nested whole-MP3D-scene cross-fitting (5 outer / 4 inner folds),
domain-scene-episode-event weighting, common random numbers, fixed model and
primal-dual constants, and only the pre-sealed weight-decay grid
`{1e-4, 1e-3, 1e-2}`. It uses 1,540 rows and 39 scenes; no public split was
read.

The result is:

```text
status                    TRAIN_DEVELOPMENT_FAIL
mainline                  NESTED_RCSP_FAIL
first failure             outer_fold_0:no_feasible_inner_candidate
checkpoint_created        false
confirmation_authorized   false
public_unseen_authorized  false
```

The best-looking first outer-fold trial (weight decay `0.01`) is still not
scientifically feasible: R2R total utility is positive (`2.1418`), but RxR
leave-one-selected-scene minimum is negative (`-0.2203`), RxR catastrophic
rate is above the relevant simple baselines, and the candidate does not pass
the pre-registered cross-domain/baseline criteria. The other weight decays
have clearly negative RxR utility or fail the same criteria. Controls were
intentionally skipped because the mainline did not produce a complete outer
OOF result; this is recorded, not hidden.

For transparency, the first outer-fold evidence (the run stops fail-closed at
this fold) is:

| weight decay | R2R total utility | RxR total utility | R2R catastrophic rate | RxR catastrophic rate |
|---:|---:|---:|---:|---:|
| `1e-4` | `+1.048716` | `-6.042747` | `4.58%` | `11.11%` |
| `1e-3` | `+1.861237` | `-1.253639` | `5.22%` | `10.08%` |
| `1e-2` | `+2.141801` | `+2.086910` | `5.16%` | `10.11%` |

These are aggregate event-level totals within one outer-fold evaluation, not
benchmark SR/SPL/nDTW claims. The `1e-2` row still fails its RxR
leave-one-selected-scene criterion (`-0.220339`) and the pre-registered
matched-baseline/risk checks, so it cannot be selected or exported as a model.

The isolated RxR-only zero-delta probe (997 rows) and a fixed 2,400-step
long-training diagnostic both independently failed at
`outer_fold_0:no_feasible_inner_candidate`. More optimization steps alone
did not resolve the issue. These are diagnostics, not public benchmark
results.

## Evidence files in this Git snapshot

The following small JSON files are intentionally included so a reviewer can
reproduce the reported decisions from the repository:

- `artifacts/training/mf3zl_rcsp_v1r1_audit_fix_v2/MF3ZL_V1R1_DATA_SUPPORT_AUDIT_CORRECTED.json`
- `artifacts/training/mf3zl_rcsp_v1r1_audit_fix_v2/MF3ZL_V1R1_AUDIT_FIX_V2_PROTOCOL.json`
- `artifacts/training/mf3zl_rcsp_v1r1_train/MF3ZL_RCSP_V1R1_TRAIN_PROTOCOL.json`
- `artifacts/training/mf3zl_rcsp_v1r1_train/MF3ZL_RCSP_V1R1_TRAIN_DEVELOPMENT_RESULT.json`
- `artifacts/training/mf3zl_rcsp_rxr_probe_v1_1/MF3ZL_RXR_PROBE_PROTOCOL.json`
- `artifacts/training/mf3zl_rcsp_rxr_probe_v1_1/MF3ZL_RXR_PROBE_RESULT.json`
- `artifacts/training/mf3zl_rcsp_rxr_longtrain/MF3ZL_RXR_LONGTRAIN_PROTOCOL.json`
- `artifacts/training/mf3zl_rcsp_rxr_longtrain/MF3ZL_RXR_LONGTRAIN_RESULT.json`

The corresponding versioned method notes and source are tracked in
`METHOD_REVISION_3ZL_RCSP*.md`, `revealnav_mf3/rcsp*.py`, and the associated
`scripts/` and `tests/` files. Artifact paths are project-relative; their
source protocols contain the expected provenance and access flags.

## Interpretation that must not be overclaimed

The data-support gate passes, so the current negative result is not evidence
that the proposal pool is empty. It does show that this RCSP objective and
representation do not yet satisfy the pre-registered scene-level safety and
utility criteria. R2R and RxR behavior differs; a joint positive aggregate
would not be allowed to conceal a domain regression. No checkpoint was
authorized and no claim of SOTA, public benchmark improvement, reproducibility
on unseen data, or acceptance probability follows from this snapshot.

Any next experiment must be a new, explicitly versioned revision. It may use
the sealed development artifacts for diagnostics, but it must not retune on
the consumed confirmation cohort or on a public split. A future public
evaluation can be authorized only by a separate, independently written
authorization artifact after all pre-registered development gates pass.

## Deliberately omitted from Git

Raw Matterport/R2R/RxR payloads, visual features, checkpoints, virtual
environments, caches, per-episode traces, large logs, reserve files, and all
credentials/API keys remain local to the self-contained workspace. This keeps
the review snapshot small and prevents accidental data or secret disclosure.
