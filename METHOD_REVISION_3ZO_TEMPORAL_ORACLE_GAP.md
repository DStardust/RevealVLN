# MF3ZO Temporal Observability & Oracle-Gap Pilot v1

**Identifier:** `mf3zo_temporal_oracle_gap_v1`  
**Scope:** train-development scientific audit only; this is not a deployment
controller, gate, checkpoint, or public-benchmark experiment.

## 1. Frozen boundary

The completed MF3ZM-CAR, MF3ZL-RCSP, and MF3ZK-DSR experiments remain
immutable historical evidence.  Their frozen single-decision learned-gate
family is stopped.  MF3ZO neither reruns those experiments nor searches a new
single-decision decision surface.

MF3ZO asks one staged question:

> Does strictly causal temporal observation contain scene-generalizable
> information about U/A/D and Reveal/Expiry, and does that state explain exact
> intervention value beyond the current snapshot?

The pilot uses only development scenes and excludes every known consumed
confirmation scene.  Access to `val_seen`, `val_unseen`, `test`, and
`test_challenge` is prohibited.

## 2. Physical information separation

`CausalTemporalRecord` contains only signals available at prefixes
`j <= decision_step`.  It may explicitly mark historically absent checkpoint
or candidate embeddings with masks.  Missing embeddings are never imputed.

`TemporalOracleLabel` is stored in a separate file.  Its fields are:

- target-in-set sequence;
- candidate-separation sequence;
- decisive-evidence-closure sequence;
- Reveal interval;
- Expiry step;
- resolvability.

An unavailable field is stored as `null`, named in `unavailable_fields`, and
accompanied by provenance.  A heuristic, route label, future outcome, or
surrogate classifier may not be relabeled as oracle truth.

The causal tensor builder rejects outcome, future, geometry, pose, and oracle
keys.  Mutation of future observations, exact return, or an oracle label must
not change the inference tensor.

## 3. Frozen U/A/D semantics

For each prefix:

- **U:** the target branch is absent from the declared candidate set;
- **A:** the target is present but separation or evidence closure is false;
- **D:** presence, separation, and evidence closure have all remained true for
  exactly the predeclared stability requirement of `K=3` causal prefixes.

The state is deterministically derived.  No free U/A/D classifier may redefine
these labels.

## 4. Pilot population

The fixed capacity is 150 exact events: 75 R2R and 75 RxR.  Within each domain,
events are allocated round-robin across raw MP3D scenes and ordered by the
sealed SHA-256 salt.  Selection can use only event identity and causal-source
availability.  It cannot use exact return, catastrophe, a historical model
error, or any public-split outcome.

Historical causal records are reconstructed without new treatment rollout.
The audit reports terminal and full-prefix embedding coverage separately.

## 5. Fixed probes

All probes use five whole-raw-scene folds shared across R2R and RxR.  A scene
never crosses folds.  Continuous features are standardized using fit-fold
statistics only.  Ridge regularization is fixed at L2=1.  There is no model,
feature, threshold, optimizer, seed, or regularization search.  Final bootstrap
confidence intervals use 10,000 raw-scene-cluster replicates and fixed seeds.

### Probe A: oracle relevance

The current-only input is the ten frozen policy scalars and six decision-time
semantic cosines.  The oracle augmentation is fixed as final U/A/D one-hot,
both endpoints of the Reveal interval represented as `endpoint - t`, Expiry
slack `T_X - t`, and resolvability.  Both models are fixed ridge regressors for
exact runner-minus-native utility.  The primary per-domain statistic is:

`DeltaHuber = Huber(current) - Huber(oracle_augmented)`.

Both R2R and RxR require an observed improvement above zero and a raw-scene
bootstrap 95% lower bound above zero.  Missing complete verified oracle labels
is a fail-closed `TEMPORAL_ORACLE_RELEVANCE_FAIL`, not an imputation license.

### Probe B: temporal observability

This probe runs only after Probe A passes.  Snapshot and full strictly causal
history probes predict target presence, separation, evidence closure, Reveal
hazard, and Expiry hazard.  U/A/D is derived from the first three outputs using
the frozen rule.  Primary metrics, fixed before results, are:

1. U/A/D macro-F1 improvement;
2. Reveal NLL improvement;
3. Expiry NLL improvement.

Every metric must have positive observed improvement and positive scene-
cluster bootstrap lower bound in each domain.  Missing reliable supervision or
full causal inputs fails closed as `TEMPORAL_CAUSAL_OBSERVABILITY_FAIL`.

### Probe C: learned-state relevance

This probe runs only after A and B pass.  Its one temporal encoder is trained
only on U/A/D and Reveal/Expiry supervision, then frozen.  A fixed ridge
action-value probe compares current snapshot against current plus frozen state.
The runner is selected iff its OOF prediction is greater than zero.  No
decision threshold is searched.

Both domains require positive DeltaHuber with positive scene-bootstrap lower
bound, positive selected utility, positive minimum leave-one-scene utility,
nonnegative utility and nonzero selection in every fold/domain, and a
catastrophic rate no higher than the strongest fold/domain-budget-matched
deterministic baseline.

## 6. One-shot stop rule

The probes execute A, then B, then C.  The first failure permanently stops this
pilot version.  Later probe results are not fabricated.  Failure forbids
feature redefinition, model substitution, threshold/regularization tuning, or
an MF3ZO-v2 search on the same pilot scenes.  Further work requires an
explicitly new data/sensor protocol or separately authorized formal human
label audit.

Even if all three probes pass, MF3ZO only reports
`TEMPORAL_ORACLE_GAP_PILOT_PASS`.  It does not authorize TEAL treatment
collection, TUAD training, checkpoint generation, or any public split.

