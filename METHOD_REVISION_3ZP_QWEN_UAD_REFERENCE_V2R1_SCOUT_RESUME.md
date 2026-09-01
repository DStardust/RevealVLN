# MF3ZP repaired scout resume correction

The sealed repaired-scout protocol stopped before opening outcomes because
its new wrapper expected U/A/D values to be enum objects while the immutable
MF3ZP reference helper returns the same states as strings.  This correction
adapts only that representation at the wrapper boundary (`str` plus a
read-only `.value` property); it does not change state derivation, response
validation, features, folds, or probe criteria.

The original repaired-scout protocol and all response/observation artifacts
remain immutable.  The compatibility wrapper is sealed separately before
the resumed scout.  The run remains exploratory, Qwen-provisional, and
non-authorizing: no checkpoint or public split is produced.
