# MF3ZU — RxR Evidence-Memory Feasibility v1

## 1. Revision purpose

Identifier:

`mf3zu_rxr_evidence_memory_feasibility_v1`

MF3ZU is a new, explicitly RxR-only diagnostic revision. It does not alter,
reinterpret, or rescue the immutable MF3ZT result. MF3ZT remains failed because
its pre-registered two-domain target-support prerequisite was not met in R2R.

MF3ZU answers one narrower feasibility question:

> On RxR train-development decisions with an existing exact current-candidate
> teacher, does explicit instruction-conditioned semantic evidence memory improve
> held-scene candidate ranking over frozen ETP-R1 current-state information?

R2R is outside this revision. An MF3ZU result cannot be reported as the
two-domain MF3ZT result.

## 2. Scope and prohibitions

MF3ZU is a candidate-ranking feasibility study only. It authorizes train-only causal observation
replay needed to materialize evidence for the fixed decision population. It does
not authorize a complete navigation evaluation, SR/SPL optimization, policy
gradient, RL, ETP-R1 fine-tuning, a deployment checkpoint, or access to
`val_seen`, `val_unseen`, `test`, or `test_challenge`.

The following historical artifacts stay byte-immutable:

* `METHOD_REVISION_3ZT_EVIDENCE_MEMORY_DECISION_PROBE.md`
* `MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_PROTOCOL.json`
* `MF3ZT_DECISION_TARGET_SUPPORT_AUDIT.json`
* `MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_RESULT.json`

No MF3ZT field is rewritten to omit R2R.

## 3. Fixed source and outcome-blind population

The source is the existing MF3B RxR train-guide corpus:

* 156 episodes;
* 59 raw MP3D scenes;
* 1,920 frozen ETP feature rows;
* exact current-candidate teacher labels already materialized; target existence
  and index may be read only by the exact-support predicate, while target values
  never enter the sanitized population, Qwen payload, evidence generation, or
  memory-required classification.

The exact-support population rule is fixed before evidence extraction:

```text
include every source row whose frozen candidate_mask contains at least 2 actions
and whose pre-existing exact target index names an active candidate feature slot
```

No sampling is permitted. The scientific population is exactly 1,428 legal
rankable decisions from 154 episodes and all 59 scenes. The source has 1,815
rows with at least two candidates; target existence and its in-mask feature
slot are used only to establish exact candidate-ranking support. Selection does
not inspect whether frozen ETP ranks that target correctly and never uses
reward, utility, success, route outcome, or future observation.

The builder writes two immutable artifacts. The sanitized decision population
contains no target field. A separate exact-target artifact contains only event
identity, target feature-slot index, coordinate-system declaration, and frozen
source provenance. Its hash is sealed in the population manifest. Annotation,
evidence construction, and memory-required classification may open only the
sanitized population; they must not inventory or open the exact-target artifact.
The trainer may open exact targets only after the evidence-memory manifest has
the frozen status required by the protocol.

Feature-row position is not assumed to equal the physical navigation step. The
builder pairs the ascending eligible-feature ordinal with the hash-chained
shadow-decision ordinal, verifies candidate cardinality, resolves the shadow
physical step to exactly one native-trace row, and stores both coordinates.
Candidate feature slots, graph indices, and action IDs remain distinct coordinate
systems until replay produces a byte-exact binding audit.

## 4. Scene folds

Five folds are assigned by raw MP3D scene. Scenes are ordered by a fixed salted
SHA-256 key and assigned round-robin to folds 0 through 4. Every decision from a
scene remains in one fold. The old MF3B fit/calibration/shadow labels do not
define MF3ZU folds.

## 5. Evidence generation boundary

Only causal observations at or before the decision step are allowed. Train-only
native observation replay is authorized solely to reconstruct those prefixes.
The fixed extractor is:

```text
model: qwen3.8-max
temperature: 0
thinking: false
max_tokens: 8000
```

There is no model, prompt, temperature, or ensemble search. Evidence and the
outcome-blind `MEMORY_REQUIRED` / `MEMORY_NOT_REQUIRED` classification must be
materialized and hashed before training-time target access.

At the user's direction, manual review is skipped for this scientific attempt.
Every generated annotation must therefore carry `human_verified=false` and
`gold=false`; neither the population nor the result may be described as human
validated.

The fixed ontology is:

```text
LANDMARK_SEEN
LANDMARK_PASSED
RELATION_SATISFIED
ORDINAL_COUNT
DIRECTIONAL_CONTEXT
```

Confidence is one of `OBSERVED`, `AMBIGUOUS`, or `ABSENT`. Retrieval uses exact
active instruction-atom association, compatible evidence type, deterministic
recency ordering, and `K_MEM = 8`. Retrieval is not trainable. Complete history
RGB or generic history embeddings are not memory-arm inputs.

`MEMORY_REQUIRED` classification must not read the teacher/correct candidate,
success, reward, utility, or a future frame.

## 6. Fixed arms and model

The only arms are:

1. `ETP_CURRENT`
2. `ETP_PLUS_EVIDENCE_MEMORY`
3. `ETP_PLUS_SHUFFLED_MEMORY`

ETP-R1, candidate generation, the visual backbone, and the topology encoder are
frozen. Arm A is the original frozen ETP masked `native_scores` and is not
trained. Arms B and C use the same fixed, small candidate-score residual
reranker. The true-memory arm receives retrieved semantic records; the shuffled
arm receives a train-fold-safe, different-event permutation that preserves
memory count and feature distribution. A held-fold decision can never donate
memory to training.

Each evidence record first has a fixed candidate-independent 77-D
representation: ontology one-hot (5), historical confidence one-hot (3),
current-status one-hot (3), `log1p(age)` (1), reciprocal recency (1), and signed
SHA token hash (64). For each candidate, one candidate-binding coordinate is
appended, producing the fixed 78-D candidate-conditioned evidence feature. At
most eight such features are mean-pooled per candidate. Candidate 768-D features
and candidate-conditioned evidence 78-D features are each projected to 64-D.
Their concatenation with the elementwise interaction enters
`Linear(192,64)-GELU-Linear(64,1)`, and the residual is added directly to the
frozen ETP base score.

The ranking loss is candidate-set cross entropy. Arms B and C share common
initialization and batch order and use AdamW (`lr=1e-3`, `weight_decay=1e-4`),
batch size 64, exactly 40 epochs, and seed 20260901. There is no early stopping
or best-checkpoint selection. There is no architecture sweep, threshold search,
weight-decay grid, seed selection, or multi-seed rescue.

## 7. Evaluation

Evaluation is five-fold raw-scene-disjoint OOF on the fixed 1,428 exact-support
rows. Exact-target values are opened by the trainer only after evidence freeze.
Standardization is fit on each training fold only.

Metrics are `Acc@1`, MRR, mean target rank, and correct-vs-competitor pairwise
ranking accuracy, reported for:

* `ALL`;
* `MEMORY_REQUIRED`;
* `MEMORY_NOT_REQUIRED`.

Raw-scene clustered bootstrap is fixed to 10,000 replicates with seed 20260901.

## 8. RxR-only PASS rule

Support first requires at least 50 `MEMORY_REQUIRED` rankable decisions from at
least 10 raw scenes. Otherwise the result is
`MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL`.

For the RxR `MEMORY_REQUIRED` subgroup, all of the following are required:

* `Acc(B) - Acc(A) > 0`;
* its scene-bootstrap 95% lower bound is greater than zero;
* `MRR(B) - MRR(A) > 0`;
* `Acc(B) - Acc(C) > 0`;
* its scene-bootstrap 95% lower bound is greater than zero.

For `MEMORY_NOT_REQUIRED`, `Acc(B) - Acc(A) >= -0.01`. For `ALL`,
`Acc(B) - Acc(A) >= 0`.

Passing yields `MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PASS`. Any failed gate
yields `MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_FAIL` and stops this fixed
RxR-only feasibility study. A PASS still leaves full navigation, public evaluation, and
deployment checkpoint generation false and requires review before another
revision.

## 9. Interpretation

A FAIL supports only this statement:

> In this fixed RxR train-development population, ontology, extractor, retrieval
> rule, and frozen ETP-R1 representation, explicit semantic evidence memory did
> not reliably improve memory-required candidate decisions.

A PASS supports only this statement:

> Explicit semantic evidence memory added held-scene information for RxR
> memory-required candidate decisions beyond frozen ETP-R1 current-state input.

Neither result is a complete-navigation or two-domain claim.
