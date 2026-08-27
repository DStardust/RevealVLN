# MF2 Post-Excursion BACKTRACK Supervision V4.6

Status: versioned implementation protocol; train-only engineering stage.

## Purpose

V4.5 executes a return selected by an external policy but does not predict
when to return. V4.6 adds the missing state-conditioned supervision. For each
declared branch in a train event, the frozen controller first attempts the
already-sealed decision-state-to-branch route. Only observations available at
the reached branch state enter the model input.

## Causal input boundary

The input for one reached branch contains:

- the existing frozen ETP-R1 instruction embedding;
- the causal history through the sealed decision prefix;
- the saved checkpoint history token;
- the selected branch token at the decision prefix;
- one frozen, front-facing ETP-R1 panorama/history token at the reached state;
- the pooled declared-candidate token at that state; and
- the elapsed excursion action count normalized by the frozen target-route
  denominator.

It does not contain the target branch id, return rollout, future observation,
navmesh, simulator pose, oracle distance, or any Gold/development/unseen
payload. Offline target identity is stored only beside the action-cost labels.

## Counterfactual actions and sunk-cost boundary

The decision point is after the selected branch has been reached, so the
outbound excursion is a sunk cost and is excluded from both labels.

- `CONTINUE`: zero additional branch-entry loss for the target branch and the
  frozen wrong-commitment penalty 5.0 for a non-target branch.
- `BACKTRACK`: bounded frozen-controller cost for `reached state -> saved Q ->
  target branch`. Returning after entering the target branch additionally
  receives the frozen missed-opportunity penalty 5.0.

Controller failures remain at bounded cost 5.0. A failed outbound route has no
reached-state observation and remains in the evidence ledger with
`trainable=false`; it is never silently filtered or represented by a synthetic
zero observation.

## Model contract

The next stage may train only a small head over these frozen embeddings. It
predicts non-negative `CONTINUE` and `BACKTRACK` cost-to-go with Huber
regression and within-example action ranking. Checkpoint selection, REE/Q
fusion coefficient, ETP-R1, waypoint proposal, and return controller remain
frozen.

This stage is not a paper result and does not authorize Gold, val_seen,
val_unseen, test, or test_challenge access.
