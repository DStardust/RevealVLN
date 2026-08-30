# MF3ZL-RCSP v1r1 — R2R Dense Instruction-Variant Expansion

This is a versioned data-support revision of `mf3zl_rcsp_v1`. The RCSP model,
utility, proposal hierarchy, action identity checks, scene folds, and public
split prohibition are unchanged.

The parent v1 collection completed its sealed population but produced 253
combined R2R exact one-switch events, below the pre-registered 300-event
support gate. v1 is retained as `TRAIN_DATA_SUPPORT_FAIL`; its result is not
overwritten and its outcomes are not used to choose routes.

v1r1 adds one deterministic, outcome-blind population: every remaining R2R
train instruction episode belonging to a trajectory whose canonical
representative was selected by v1, excluding the already collected canonical
episode and any historically consumed episode. Selection uses only train
payload metadata (scene, trajectory, split, reference-path length, and
instruction identity hash). It does not inspect proposal scores, features,
metrics, labels, or prior model errors, and it does not stop adaptively.

Each selected episode receives a native shadow run. Only proposal events
observed in that shadow are sealed as targets; each target receives an
independent exact one-switch treatment replay with an identical prefix and a
single declared action change. The parent v1 artifacts remain immutable.

The expansion is restricted to the same 39 development scenes. No consumed
confirmation scene, `val_seen`, `val_unseen`, `test`, or `test_challenge`
payload is accessed. A combined support audit is required before any future
RCSP fit; this revision itself does not authorize confirmation or public
evaluation.

The extension is complete-population, fail-closed, and outcome-blind. If the
combined R2R count still does not reach the existing 300-event gate, the
revision records `TRAIN_DATA_SUPPORT_FAIL`; the gate is not lowered after
observing outcomes.
