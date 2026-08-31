# MF3ZL-RCSP RxR long-training diagnostic

This is a single, pre-registered development diagnostic following the
`mf3zl_rcsp_rxr_probe_v1_1` result.  It tests optimization duration only:

* `training_steps` is fixed at **2400** (three times the v1/v1.1 value of
  800);
* architecture, loss, risk constraints, scene folds, seeds, weight-decay
  grid, decision rule, and all source rows are unchanged;
* no duration sweep, threshold sweep, outcome-based stopping, or public split
  access is allowed;
* no checkpoint is produced or authorized.

The v1.1 result remains immutable and this run is a new consumed development
diagnostic.  A failure here means that simply training longer is not a
credible fix; further changes require a separately specified algorithmic
revision rather than repeated duration tuning.
