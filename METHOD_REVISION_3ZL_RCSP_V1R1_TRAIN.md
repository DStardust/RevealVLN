# MF3ZL-RCSP v1r1 expanded-data training

This protocol trains the frozen RCSP v1.1 zero-relative-delta correction on
the complete, audited development population.  It combines the previously
sealed 249 canonical rows and 1001 parent dense exact rows with the 290
outcome-blind v1r1 R2R variant exact rows, deduplicating only the complete
`(dataset, scene, episode, decision_step)` identity.

The algorithm is unchanged from the versioned v1.1 correction: rank-4
relative semantic policy, magnitude-weighted preference loss, domain
catastrophic constraints, scene-disjoint nested fitting, and fixed
`switch_logit > 0` deployment boundary.  The 800-step setting is retained so
this run isolates the effect of expanded data; the separate 2400-step RxR
diagnostic is not mixed into this result.

This is train-development only.  It cannot authorize confirmation, val_seen,
val_unseen, or test, and a failed fit must not emit a checkpoint.
