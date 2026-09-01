# MF3ZT — Evidence-Memory Decision Probe v1

Revision: `mf3zt_evidence_memory_decision_probe_v1`

This is a versioned, train-development-only falsification test. It does not
modify the frozen MF3ZQ/MF3ZR revisions or reopen the stopped Reveal, Expiry,
U/A/D, option-preservation, or single-decision gate families.

## Scientific question

MF3ZT asks exactly one question:

> Does explicit, instruction-conditioned semantic evidence memory improve
> held-scene ETP candidate decisions, especially when relevant evidence was
> observed causally in the past but is absent from the current observation?

It is a candidate-ranking decision probe, not a navigation experiment. It
does not run full rollouts, optimize SR/SPL, train with RL, fine-tune ETP-R1,
or evaluate any public split.

## Mandatory target-support gate

Before a decision population can be built, both R2R and RxR must have a
pre-existing, project-local, auditable candidate-ranking target aligned
exactly to the frozen candidate set at the same episode, prefix, and decision.
The accepted target priority is:

1. exact train-native action/candidate supervision;
2. exact same-episode, same-prefix branch supervision;
3. an existing frozen causal decision target.

The following are not candidate targets: the frozen policy's own native
action, route-level reward or utility, SR/SPL/nDTW/SDTW, a target reconstructed
from route truth, nearest-candidate matching, cross-episode pairing, CAR rescue
labels, or any public-split label. A target derived after this revision begins
is not "pre-existing" support.

If either domain lacks legal support, the fixed result is
`MF3ZT_DECISION_TARGET_SUPPORT_FAIL`. The revision must stop before population
materialization, memory-required classification, evidence extraction,
training, cross-validation, bootstrap, checkpoint creation, or navigation.
No empty population is written and no downstream component is implemented as
a placeholder.

## Probe design, conditional on target support

The unit is one navigation decision containing dataset, raw MP3D scene,
episode, decision step, instruction, causal observation history, current
observation, causal topology history, current frozen candidate set, frozen ETP
candidate scores/features, and the legal candidate-ranking target.

All data remain train-development only. Five folds are raw-scene-disjoint;
shared R2R/RxR MP3D scenes stay in one fold. Normalization is fit on the
training scenes only. Event-level random splitting is prohibited.

Each decision is classified outcome-blind, before target evaluation, as:

- `MEMORY_REQUIRED`: an instruction-relevant fact appeared in causal history,
  is absent or insufficient in the current observation, and is semantically
  required for the current candidate decision;
- `MEMORY_NOT_REQUIRED`: the current observation is sufficient or the
  decision does not require historical semantic state.

The classifier may use instruction semantics, causal visual history, the
current observation, current candidate appearance/geometry, and instruction
constraint structure. It may not use the correct candidate, success, reward,
utility, or future frames. Each domain must contain at least 50
`MEMORY_REQUIRED` decisions spanning at least 10 raw scenes. Otherwise the
fixed result is `MF3ZT_MEMORY_REQUIRED_SUPPORT_FAIL`.

## Frozen evidence memory

The v1 ontology is fixed to exactly:

- `LANDMARK_SEEN`
- `LANDMARK_PASSED`
- `RELATION_SATISFIED`
- `ORDINAL_COUNT`
- `DIRECTIONAL_CONTEXT`

An `EvidenceRecord` contains an immutable evidence ID, event ID, causal source
step and topology node, instruction atom ID, ontology type, semantic value,
confidence class, and source-observation SHA-256. Confidence is one of
`OBSERVED`, `AMBIGUOUS`, or `ABSENT`. All records satisfy
`source_step <= decision_step`; future observations, candidates, outcomes,
utility, route truth, pose, and public data are forbidden.

Evidence is materialized and frozen before training. A single fixed extractor
may be used; model, prompt, temperature, and ensemble searches are prohibited.
Retrieval is deterministic: active atom order, exact atom association,
compatible evidence type, `OBSERVED` confidence, then descending source step
and evidence ID. The retrieval budget is fixed at `K_MEM = 8` and is not tuned.
Complete historical RGB is not re-encoded.

## Fixed arms and learner

The only arms are:

1. `ETP_CURRENT`
2. `ETP_PLUS_EVIDENCE_MEMORY`
3. `ETP_PLUS_SHUFFLED_MEMORY`

ETP-R1, its visual backbone, topology encoder, and candidate generator are
frozen. Only a small residual candidate reranker is trainable. The memory arm
uses mean-pooled evidence followed by a learned projection and the
candidate-specific interaction `[h_b, m, h_b * m]`. A fixed two-layer MLP
outputs a residual added directly to the frozen ETP score; there is no tunable
mixture coefficient, architecture grid, threshold search, or multi-seed
rescue. The loss is candidate-set cross entropy, or a fixed legal
multi-positive ranking loss only when equivalence is already present in the
target source.

Shuffled memory preserves count and feature distribution but permutes evidence
to a different event within the training fold. Held-fold targets never supply
memory to another example, and test-fold evidence is not imported across
events.

## Metrics and fixed gates

The metrics are Acc@1, MRR, mean target rank, and pairwise ranking accuracy,
reported separately for `ALL`, `MEMORY_REQUIRED`, and
`MEMORY_NOT_REQUIRED`, in R2R and RxR.

Raw MP3D scene-clustered bootstrap uses 10,000 replicates and seed `20260901`.
For both domains independently, PASS requires:

- on `MEMORY_REQUIRED`, Acc(B) - Acc(A) > 0 with lower95 > 0, and
  MRR(B) - MRR(A) > 0;
- on `MEMORY_REQUIRED`, Acc(B) - Acc(C) > 0 with lower95 > 0;
- on `MEMORY_NOT_REQUIRED`, Acc(B) - Acc(A) >= -0.01;
- on `ALL`, Acc(B) - Acc(A) >= 0.

Failure of support, either domain's memory-required gain, evidence specificity,
the overall non-degradation gate, or the negative-control gate yields
`MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_FAIL`. No ontology, retrieval budget,
reranker, loss, population, subgroup definition, domain-specific rescue, seed,
or difficult scene may then be changed inside v1.

PASS yields `MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_PASS`, while
`full_navigation_run`, `checkpoint_for_deployment`, and `public_evaluation`
remain false pending a separately versioned review.

## Correctness and execution boundary

The protocol is sealed as
`SEALED_BEFORE_MF3ZT_RESULTS`. It records source commit and source hashes,
population/target/evidence hashes when materialized, domain and scene counts,
ontology, `K_MEM`, frozen ETP checkpoint hashes, learner/loss, folds, bootstrap,
fixed gates, and closed public/navigation/checkpoint flags.

Required invariants include: no future observation or candidate; no public
split; no outcome-adaptive selection; outcome-blind memory-required labels;
frozen ETP/backbone/topology/candidate generator; pre-training evidence freeze;
raw-scene and shared-scene integrity; train-fold-only normalization;
`K_MEM == 8`; causal evidence sources; train-fold-safe shuffle; immutable
results; no navigation or deployment checkpoint; unchanged historical failed
revisions; and full regression PASS.

The execution order is fail-closed. Read-only repository inspection first
determines the candidate source inventory. The implementation and complete
conditional design are then tested and the protocol is sealed before the
formal target-support audit result is materialized. A target failure stops
immediately. Only a pass may build and classify the fixed population, audit
subgroup support, freeze memory, implement the fixed learner, run scene-OOF
evaluation and bootstrap, apply the gates, write an immutable result, regress,
commit, and push. Thus even an early support failure is governed by a
pre-result seal.

## Interpretation

A target-support failure says only that the sealed audited source inventory
cannot furnish the legal R2R/RxR candidate-ranking labels required to run this
probe. It is not evidence for or against semantic evidence memory.

A completed probe failure says only that the fixed ontology and frozen ETP-R1
representation did not stably improve the specified held-scene decisions. A
PASS says only that explicit memory supplied measurable held-scene decision
information beyond the frozen current representation; it does not establish a
full-VLN improvement.
