# MF3ZP v2r1 scout correction — event-local causal prefixes

This is a bounded audit/scout correction on top of the immutable MF3ZP v2
and v2r1 annotation records.  It does not change the event population, the
Qwen semantic questions, or any model/policy.

## Causal correction

The v2 assembler exported the observation stream through the maximum sealed
decision step of an episode.  When one episode contains events at different
steps, that can leave a small number of request rows after an earlier event's
own decision step.  Those rows are not used here.  The scout input is
restricted by the pre-existing event-local rule:

``prefix_step <= event.decision_step``

The filtered set is sealed before any exact outcome is opened.  Existing v2
and v2r1 responses are read-only; a response is accepted only if it passes
the original v2 response validator.  No response field is repaired in the
client and no semantic label is imputed.

## Scientific scope

The resulting scout remains exploratory Qwen-provisional supervision.  Exact
outcomes are opened only after all filtered responses have been verified, and
the resulting Probe-A numbers are not human-verified formal evidence.  No
checkpoint, deployment authorization, or public split is produced.

The original v2/v2r1 protocols and all response files remain immutable.
