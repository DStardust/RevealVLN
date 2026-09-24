"""One weighted order statistic on reserved training houses, no score/threshold sweep."""
import torch
def fit_bias(candidate_margin,reference_margin,groups):
    index=groups['negative_indices'];mask=groups['negative_mask'];w=groups['trajectory_weights']
    candidate=candidate_margin[index].masked_fill(~mask,-1e9).max(1).values
    reference=reference_margin[index].masked_fill(~mask,-1e9).max(1).values
    budget=float((w*(reference>0)).sum());before=float((w*(candidate>0)).sum());threshold=0.
    if before>budget+1e-12:
        above=0.
        for value in sorted(set(candidate[candidate>0].tolist()),reverse=True):
            mass=float(w[candidate==value].sum())
            if above+mass>budget+1e-12:threshold=value;break
            above+=mass
        assert threshold>0
    reduction=threshold+1e-5 if threshold>0 else 0.
    return dict(bias_reduction=reduction,order_statistic_threshold=threshold,reference_risk_budget=budget,candidate_risk_before=before,calibration_trajectories=len(w),numerical_tie_reserve=1e-5,method='weighted order statistic; no trajectory replay or threshold grid')
