# MF3ZP v2 — Prefix-safe Qwen U/A/D reference observations

This is a versioned data-protocol correction, not a new policy or gate.
`MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json` (v1) is retained as an engineering
failure record.  Its collector treated some historical exact-switch
`base_trace.jsonl` files as complete native traces even though the target row
contained the old runner-up action.  v2 never makes that claim.

## Scientific scope

The fixed 150-event MF3ZO outcome-blind population is reused without changing
event selection, scenes, folds, or any outcome.  v2 only reconstructs causal
observations needed for a provisional, role-blind Qwen semantic reference
label.  It cannot authorize Probe A, TEAL collection, a deployment model, or a
public split.

## Trace provenance rule

For each episode the protocol records one of two immutable modes:

* `native_reference`: an independently recorded frozen-native trace is
  available (the R2R baseline-completion trace, an MF3ZL native trace, or an
  existing native RxR trace).
* `prefix_witness`: a legacy trace is known to be an exact one-switch run and
  is used only to witness actions strictly before the sealed decision step.
  Its target-step action is deliberately never compared or called native.

The v2 worker reruns the frozen native policy, exports observations only for
prefixes through the maximum sealed decision step, and verifies exact action
signatures for source indices `< decision_step`.  The assembler independently
checks the observed native identity and executable candidate set at every
sealed event.  Post-decision records are not annotation inputs.

## Annotation boundary

Qwen receives only the instruction, opaque candidate aliases, geometry-free
heading metadata, and ordered RGB views from the causal prefix.  It never
receives event targets, native/runner roles, utility, catastrophe, outcomes,
folds, or model predictions.  The two pinned multimodal snapshots, prompt,
temperature, response schema, and retry policy are sealed before responses.

The response is provisional reference supervision for candidate separation and
instruction evidence closure.  `target_in_set`, expiry, reveal, and
resolvability remain deterministic/programmatic or unavailable; no Qwen
confidence is treated as a calibrated probability.  Human audit is required
before any formal Oracle-Relevance claim.

## Stop rules

Any source-prefix mismatch, candidate identity drift, incomplete episode,
forbidden field, API schema failure, or public-split access fails closed.  A
future exploratory Qwen scout may read exact outcomes only after all responses
are frozen and must remain explicitly provisional.  No checkpoint or public
evaluation is produced by this revision.

The v1 protocol, source files, and artifacts are immutable historical records;
this document and the v2 protocol are not an amendment to `FROZEN_SPEC.md`.
