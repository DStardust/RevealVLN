# MF3ZU v1r2 — RxR evidence-annotation contract recovery

## 1. Purpose

Identifier:

`mf3zu_rxr_evidence_annotation_contract_recovery_v1r2`

This revision is a bounded, outcome-blind recovery of the 252 evidence
responses rejected by the sealed MF3ZU v1r1 validator.  It does not change the
1,428-decision population, instruction graphs, evidence ontology, causal image
inputs, candidate support, memory-required definition, retrieval rule, model,
training schedule, or scientific gates.

The v1r1 artifact remains immutable at 1,176 PASS and 252 FAIL.  V1r2 writes a
new artifact tree and treats every retained v1r1 response as a read-only input.

## 2. Information boundary

V1r2 may read only the sealed target-blind evidence requests, merged
instruction graphs, causal replay manifests, v1r1 response records, and their
contract errors.  It must not open candidate targets, route outcomes, utility,
model performance, public splits, future observations, or any result artifact.

There are no manual labels or human adjudication.  All annotations remain
provisional with `human_verified=false` and `gold=false`.

## 3. Fixed recovery partition

The 1,176 v1r1 PASS response files are copied byte-for-byte.  Exactly the 252
request IDs in the sealed v1r1 failure manifest are classified before any new
response.  The partition is fixed from contract violations alone:

* `MECHANICAL_REPAIR`: a response has no invalid `OBSERVED` historical source
  and no semantic value longer than 500 code points.  These responses may be
  canonicalized deterministically.
* `FIXED_REANNOTATION`: a response contains an `OBSERVED` historical source
  outside `0 <= source_step < decision_step`, or a semantic value longer than
  500 code points.  These cases cannot be safely fixed by clamping, downgrading,
  or truncating because doing so would alter the claimed visual evidence.

The sealed parent bundle yields 163 mechanical repairs and 89 fixed
reannotations.  This split must not be recomputed from subgroup support or
model performance.

### 3.1 Mechanical rules

The following rules are exhaustive and are sealed before canonical outputs
exist:

1. A missing `instruction_atom_id` may be filled only when the response has the
   expected number of rows, exactly one row lacks the field, and exactly one
   graph atom ID is otherwise absent.  The unique absent ID is used.
2. Only the three already observed non-contract keys may be dropped:
   `relative_heading_rad`, `informative_value`, and `standard_deviation`.
3. The declared implication `relevant => active` is closed by setting
   `active_for_current_ranking := active_for_current_ranking OR
   relevant_to_current_ranking`.
4. A non-`OBSERVED` historical status cannot carry a source; its
   `source_step` is set to null.
5. `semantic_value` is stripped; it must already be no longer than 500 code
   points.

No other field may change.  Unknown schema deviations fail closed.  Every
canonical response must pass the unchanged v1 evidence validator, including
atom coverage, candidate bindings, confidence classes, and causality.

### 3.2 One-time fixed reannotation

Each of the 89 reannotation requests is sent exactly once to the same fixed
`qwen3.8-max` model at temperature zero, with the same instruction graph,
causal historical storyboard, current panorama, and role-blind candidates.
The only prompt change is an explicit restatement of the already existing
output contract:

* emit exactly one row per graph atom and only the declared keys;
* `relevant` implies `active`;
* historical `OBSERVED` requires one source in `[0, decision_step)`;
* evidence visible only at the current step is not historical `OBSERVED`;
* non-observed history has a null source;
* semantic values are short factual states of at most 500 code points, with no
  chain-of-thought or action recommendation.

The previous invalid annotation is not included in the new prompt.  There is
no prompt search, per-error prompt, model search, semantic retry, or
post-response canonicalization.  Invalid new output is retained and stops
v1r2.

## 4. Sealing and immutability

Before canonicalization, v1r2 seals:

* method, runner, and regression-test inventories at a committed source SHA;
* the v1 and v1r1 method/protocol inventories;
* target-blind population, replay, instruction, evidence-request, and v1r1
  annotation manifests;
* all 1,428 parent response inventories and the exact 252 failure identities;
* a per-response repair-input ledger, fixed 163/89 partition, and exhaustive
  operation vocabulary;
* the fixed reannotation prompt and one-logical-call limit.

The output root is independent.  Existing deterministic outputs are accepted
only when byte-identical to recomputation; no historical file is overwritten.

## 5. Evidence freeze and inherited support gate

Only a complete 1,428/1,428 validated bundle may enter deterministic evidence
memory materialization.  The original `memory_required()` definition and fixed
support gate remain unchanged:

```text
memory-required decisions >= 50
memory-required raw scenes >= 10
```

The freeze writes a target-blind provenance record and a pre-training support
audit.  If support is insufficient, status is
`MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL` and execution stops before target
access or training.  V1r2 may not change labels, definitions, ontology,
population, or thresholds to rescue support.

## 6. Authorization and stop rules

V1r2 authorizes only the 89 sealed annotation calls, deterministic response
recovery, evidence-memory freeze, and target-blind audits.  It never authorizes
or launches training.

Any parent hash drift, unknown schema issue, incomplete coverage, validator
failure, memory causality failure, or support failure stops the revision.  A
technical failure is recorded as
`MF3ZU_V1R2_EVIDENCE_ANNOTATION_TECHNICAL_FAIL`; insufficient subgroup support
is recorded separately as `MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL`.

Neither condition may be repaired inside v1r2.
