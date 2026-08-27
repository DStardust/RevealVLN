# Net-Advantage ensemble feasibility revision V5.14

The sealed V5.13.1 single-model robustness gate failed before any `val_seen` or
`val_unseen` evaluation was launched. All three members nevertheless transferred
above chance to the untouched R2R-train scene holdout (AUC 0.772--0.800). The failure
was confined to an extreme-tail operating point: only 0--11 of 3,064 holdout events
activated per member, so realized sparse net varied around zero across random
initializations.

V5.14 replaces checkpoint alternatives with one deterministic deep ensemble. It
averages the three member probabilities and their positive-gain predictions, then
applies the unchanged online causal penalty. The ensemble construction is fixed;
calibration data selects its threshold, and the unchanged internal scene holdout is a
method-development gate. The aggregation was selected after a post-hoc feasibility
calculation on that R2R-train-only holdout, so its result is not treated as independent
confirmatory evidence or as a paper benchmark result. Full `val_seen` and `val_unseen`
benchmark validation remained unopened when V5.14 was sealed.

V5.14 does not change the frozen REE, ECOG, OPP, datasets, labels, task metrics, or
benchmark success criteria. It strengthens deployment stability and resolves the
previous mismatch between one selected deployment checkpoint and three evaluation
checkpoint alternatives. The failed V5.13.1 training result remains preserved as
negative engineering evidence.

V5.14.1 is a schema-only correctness revision made after all `val_seen` episode runs
finished but before their metrics were successfully aggregated and before any
`val_unseen` access. It adds `training_lock.seeds` as an exact alias of
`training_lock.member_seeds`, which the already-frozen evaluator expected. No method,
matrix, seed, threshold, gate, episode output, or metric value changed.
