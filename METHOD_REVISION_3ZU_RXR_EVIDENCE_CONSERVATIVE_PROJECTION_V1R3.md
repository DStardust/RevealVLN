# MF3ZU v1r3 — unsupported-history conservative projection

## 1. Purpose

Identifier:

`mf3zu_rxr_evidence_annotation_conservative_projection_v1r3`

MF3ZU v1r2 remains immutable at 1,416/1,428 valid evidence responses.  Its 12
failed responses passed every schema, identity, activity, candidate-binding,
length, and confidence-class check, but 20 atoms claimed historical
`OBSERVED` evidence without a legal source in `[0, decision_step)`.

V1r3 is a final, outcome-blind conservative projection for exactly those 12
sealed request IDs.  It is not a provider retry, annotation-model change,
population change, or scientific-model change.

## 2. Fixed projection

The 1,416 v1r2 PASS response files are reused byte-for-byte.  In the 12 failed
responses, and only for an atom satisfying both conditions:

```text
historical_status == OBSERVED
source_step is not an integer in [0, decision_step)
```

v1r3 applies:

```text
historical_status := AMBIGUOUS
source_step := null
```

This projection expresses that the retained annotation does not establish a
causal historical observation.  It never clamps, guesses, or fabricates a
history step.  It is deliberately conservative: projected atoms cannot create
an evidence-memory record or a memory-required positive.

Every other field is byte-value preserved, including current status, active
and relevant flags, candidate bindings, and semantic text.  The complete
projected response must pass the unchanged v1 validator.  Unknown or additional
violations fail closed.

The preserved semantic text on a projected atom is retained only as unverified
audit text.  Because its historical status is not `OBSERVED`, it may not enter
records, retrieval, or candidate memory features.

## 3. Boundaries

V1r3 uses only sealed target-blind requests, instruction graphs, causal replay
manifests, v1r2 responses, and v1r2 contract errors.  It does not call any
provider and must not read candidate targets, outcomes, utility, model
performance, future observations, or public splits.  Human review remains
skipped; `human_verified=false` and `gold=false`.

The projection rules, exact 12 response identities and hashes, exact 20 atom
count, implementation files, parent protocols, and parent response bundle are
sealed before projected outputs exist.

## 4. Evidence freeze and stop rule

Only 1,428/1,428 validated responses may enter the unchanged deterministic
evidence-memory builder.  The original memory-required definition and support
gate remain fixed at at least 50 decisions and 10 raw scenes.

If support is insufficient, v1r3 records
`MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL` and stops before training or target
access.  The projection, subgroup definition, population, ontology, and support
threshold may not be changed to rescue the result.

V1r3 never launches or authorizes training.  A support PASS would require a
separate versioned training handoff; a support FAIL ends this recovery path.
