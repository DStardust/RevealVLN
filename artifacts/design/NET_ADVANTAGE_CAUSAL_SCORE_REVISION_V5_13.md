# Net-Advantage causal-score correctness revision V5.13

This version does not change the frozen REE, ECOG, OPP, dataset, or benchmark claims.
It corrects the deployment boundary of the auxiliary sparse Net-Advantage trigger.

The pilot model was trained from causal embeddings, but its threshold score subtracted
an offline geodesic trial cost available only to the label builder. That threshold is
therefore diagnostic and is rejected by the online loader. The full-data training run
instead calibrates the same score that inference can compute:

`p_better * predicted_positive_gain - (1 - p_better) * 2 * online_candidate_distance`.

`online_candidate_distance` is the checkpoint-to-candidate Euclidean distance already
present in the causal ETP graph. No reference path, future frame, task metric, or
counterfactual rollout value enters inference.

V5.13 uses this head conservatively: it may approve or veto a non-native proposal from
V5.6, but it cannot introduce a branch not proposed by the frozen causal candidate set.
The reversible topology module remains a separately reported ablation and stays in the
main method only if its preregistered non-regression gate passes.
