# MF2 Post-Excursion Q Training V4.8

Status: preregistered train-only engineering gate; not a paper result.

The frozen ETP-R1 feature producer and V4.7 reached-state manifest feed one
small `PostExcursionQHead`. The head predicts the additional normalized task
cost of `CONTINUE` and `BACKTRACK` after a branch excursion. No backbone,
waypoint, REE, branch-Q, fusion, or return-controller parameter is updated.

The existing scene partition is retained: `sha256(scene_id) mod 6 == 1` is an
internal development split and every other train scene is optimization data.
Three fixed seeds are selected independently by minimum native internal
development loss. Optimization uses equal-weight Smooth-L1 losses for the two
costs and their difference, plus weight 0.25 margin ranking with margin 0.1.
Exact action-cost ties are excluded from ranking and decision accuracy.

The preregistered engineering hurdle requires at least two seeds to beat the
train-median action selector in both strict-action accuracy and mean action
regret, and at least two seeds to improve each cost MAE over its corresponding
train median. Failure is retained and does not authorize threshold tuning on
evaluation splits.
