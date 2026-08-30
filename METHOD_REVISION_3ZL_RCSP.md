# Method Revision 3ZL: Risk-Constrained Counterfactual Switch Policy

Revision: `mf3zl_rcsp_v1`  
Status before collection: **pre-sealed train-development revision**

This versioned correctness/novelty revision does not modify
`FROZEN_SPEC.md`. MF3ZK-NP v9 and MF3ZK-DSR v1 remain immutable consumed
negative evidence. Their old 52-episode confirmation cohort is not reused.

## Frozen boundary

The ETP-R1 policy and visual-language backbone, MF3V proposal ranker, MF3ZG
core/expansion hierarchy, native and frozen-runner action identities, and the
single-switch intervention budget remain frozen. RCSP can only abstain (execute
the native action) or authorize that exact runner-up. The utility is unchanged:

\[
U=0.50\,nDTW+0.25\,SDTW+0.25\,SPL,
\qquad y=U(\tau^{a_r})-U(\tau^{a_n}).
\]

A catastrophe remains `y <= -0.10`. No public split, old confirmation outcome,
future observation, route geometry, scene identity, benchmark identity, or
proposal-tier identity is a model input.

## Outcome-blind dense exact replay

Before fitting RCSP, the complete unused train-route population is sealed in
the 39 already consumed development scenes. One deterministic representative
instruction is selected per `(dataset, scene, trajectory)` when the reference
path has at least four points; RxR is restricted to the already established
English languages (`en-US`, `en-IN`). Every trajectory present in the earlier
MF3ZD/MF3ZF RxR selections or MF3ZK R2R selection is excluded. Selection never
reads task metrics, proposal scores, margins, previous model errors, or outcome
labels and does not stop adaptively.

Each selected episode first receives a native shadow rollout. MF3ZG observes
the first core opportunity and, after abstention, the first later expansion
opportunity without changing an action. Each observed event then receives its
own fresh treatment rollout that abstains at all prior proposals, switches only
at the sealed identity, and returns control to frozen ETP-R1. Assembly requires
the same episode, exact pre-switch trace prefix, native/runner round trips,
matching causal features, one controller-authorized switch, and finite paired
metrics. Different episodes are never paired.

The data gate is fixed at at least 300 unique exact events and 30 development
scenes per domain with zero conflicting identities. It is evaluated only after
the complete sealed population has run. Failure is
`TRAIN_DATA_SUPPORT_FAIL`; untouched scenes are never added to repair it.

## Relative semantic switch policy

For frozen instruction, strictly causal history, native-action, and runner-up
embeddings `I,H,N,R`, let `D=R-N`. After row-wise L2 normalization,

\[
C=\sigma(\alpha)\bar I+(1-\sigma(\alpha))\bar H,
\quad
s=b+w_p^\top p+(AC)^\top(B\bar D),
\]

where `A,B` have fixed rank 4. `p` contains exactly ten causal policy scalars:
step, MF3V score, native margin, minimum/median/robust advantage, ensemble MAD,
two cold-start statistics, and candidate count. Deployment is fixed:

\[
\text{switch iff }s>0.
\]

There is no searched decision threshold.

## Utility-consistent preference and risk constraint

With `z=1[y>0]` and fixed domain-scene-episode-event weights `w`,

\[
L_{pref}=\frac{\sum_i w_i|y_i|\,BCEWithLogits(s_i,z_i)}
                  {\sum_iw_i|y_i|}.
\]

For each domain, its training-partition ungated catastrophic rate is `r_d^0`.
With `pi=sigmoid(s)`, training constrains

\[
g_d=\sum_{i\in d}w_i\pi_i(c_i-r_d^0)\le0
\]

through projected primal-dual optimization. Hard scientific evaluation still
uses actual selected catastrophic rates. The constraint, catastrophe boundary,
rank, optimizer, primal/dual learning rates, steps, seeds, and zero decision
boundary are frozen before outcomes.

## Nested development protocol

The existing five whole-MP3D-scene outer folds are reused, with four whole-scene
inner folds. Shared RxR/R2R scenes remain in the same fold. The sole selectable
hyperparameter is weight decay in `{1e-4,1e-3,1e-2}`. Candidates share folds,
initialization seeds, and full-batch ordering. Feasible candidates are selected
by the lowest inner-OOF utility-weighted preference loss, never maximum realized
utility.

Failure is mandatory for any provenance/action invariant breach, unavailable
fold prediction, fold/domain zero intervention, nonpositive domain deployed
utility, nonpositive leave-one-selected-scene utility, selected catastrophic
rate above ungated or matched simple controls, or failure to beat both
fold/domain-budget-matched low-native-margin and high-MF3V-score controls.

Only a complete nested train-development pass may produce an RCSP checkpoint.
The trainer cannot authorize confirmation or public evaluation.

## Pre-registered controls

If the semantic RCSP produces complete outer OOF predictions, the same expanded
exact dataset is used for frozen DSR v1, RCSP-28D, RCSP without catastrophic
constraints, RxR-only, R2R-only, pooled/separate-tier diagnostics, and global
plus fold/domain-matched proposal-side baselines. Negative arms and
intention-to-intervene outcomes are retained.

## Prohibitions

This revision does not modify DSR, ETP-R1, MF3V, MF3ZG, or `FROZEN_SPEC.md`; does
not tune utility/catastrophe/decision thresholds; does not use non-exact pairs;
does not access `val_seen`, `val_unseen`, test, or test-challenge; and does not
integrate a deployment gate before train-development passes.
