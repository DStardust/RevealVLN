# MF3ZK-NP v9 train-development result

Run date: 2026-08-30

Verdict: **TRAIN_DEVELOPMENT_FAIL**.  This is a retained negative result, not
a confirmation or public benchmark result.  No old confirmation outcome was
used for selection, and no `val_seen`, `val_unseen`, `test`, or
`test_challenge` payload was read.

## Correctness revision applied

- L2 and return/harm thresholds are selected by nested whole-MP3D-scene
  cross-fitting: five outer folds and four inner folds.
- Outer development authorization uses the rule selected inside that outer
  fold.  The final modal-L2/median-threshold rule is not applied back to outer
  targets for the status decision.
- Core and expansion share one return/harm estimator; tier is not a model
  feature.  The frozen MF3ZG hierarchy still determines proposal routing.
- Fifty byte-identical RxR episode/step counterfactuals occurred in both
  historical core and expansion sources.  They were verified identical and
  counted once.  The cohort therefore contains 249 unique rows rather than
  299 source rows: 147 RxR and 102 R2R, over 39 MP3D scenes.
- The pooled bootstrap samples each MP3D scene once across benchmarks and
  then applies benchmark-balanced observation weights.
- The safety constraint compares catastrophic rate, not raw catastrophic
  count, so lower coverage cannot create a mechanical risk pass.
- The controller now fails closed on duplicate or non-round-tripping action
  identities before feature construction and execution.

## Result

| Arm | Fit | Scientific control | Authorized | Utility total | Minimum leave-one-selected-scene total |
| --- | --- | --- | ---: | ---: | ---: |
| Joint | FAIL | FAIL | not evaluated | not evaluated | not evaluated |
| RxR-only | PASS | FAIL | 82 / 147 | -0.460063 | -1.031240 |
| R2R-only | PASS | FAIL | 57 / 102 | 0.147039 | -0.372747 |

The joint arm failed closed because outer fold 1 had no feasible inner rule.
Neither single-domain control satisfied the predeclared robustness criteria.
RxR utility was negative; R2R utility was positive but its leave-one-scene
total was negative and its catastrophic rate (5.26%) exceeded the ungated
candidate rate (4.90%).
At the same R2R intervention budget, the nested gate also underperformed the
simple low-native-margin diagnostic (utility 0.147039 versus 0.697947).

The evidence therefore does not authorize a deployable gate, a fresh
confirmation run, or public evaluation.  The next action is an explicit
algorithm decision; threshold relaxation on this result is prohibited.

## Local evidence identity

- Protocol:
  `artifacts/training/mf3zk_nested_pooled_v9/MF3ZK_NESTED_POOLED_PROTOCOL.json`
  (`sha256 36847f560e1b33d3884d201fa703463183253112a1970d6a1466989e24675b95`)
- Result:
  `artifacts/training/mf3zk_nested_pooled_v9/MF3ZK_NESTED_POOLED_TRAINING_RESULT.json`
  (`sha256 c43fd88009efd9ab3c904fa15cddde45dd96f6e699bd4005288605de1324356b`)

The generated artifacts, datasets, model files, environments, and credentials
remain project-local and are intentionally excluded from GitHub.
