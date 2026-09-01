# MF3ZP v2r1 — Qwen annotation transport/schema repair

This is a bounded engineering repair for the provisional MF3ZP v2
annotation batch.  It is not a new method, a new event population, or a
scientific result.  The sealed v2 protocol, observations, requests, and
responses are retained unchanged.

## Observed failure

The v2 collector and prefix audit passed for all 149 episodes.  The fixed
annotation batch then produced 918 valid responses and 452 failures.  The
failures were transport errors and responses whose auxiliary `rationale`
string exceeded the implementation validator's 500-character bound.  No
target, outcome, or public-split field was read by the annotation process.

## Repair boundary

v2r1 reprocesses only the request/model pairs recorded as failed by the
sealed v2 annotation manifest.  Existing v2 PASS responses are read-only
inputs and are never overwritten.  The semantic response schema and its
strict validation (including `rationale` length <= 500) are unchanged.

The only request-level change is a fixed formatting addendum requiring a
short rationale (<=160 characters), exact JSON keys, and no markdown.  This
addresses a serialization/formatting failure; it does not alter the U/A/D
questions or expose any outcome information.

Transport uses a bounded retry schedule and lower concurrency than the
original batch.  HTTP status, parser, and schema errors are recorded as
typed summaries without persisting provider bodies or credentials.

## Scientific limits

The repaired responses remain `qwen_provisional_unverified` data.  A merged
annotation set may be used only for the explicitly exploratory scout.  It
does not constitute formal human-verified Probe A evidence, does not create
a checkpoint, and does not authorize any public split.

The v2 artifact and protocol are immutable historical records.  If the
repair cannot complete, the result is a repair/data-readiness failure; no
schema relaxation, client-side label fabrication, threshold tuning, or
additional event collection is allowed.
