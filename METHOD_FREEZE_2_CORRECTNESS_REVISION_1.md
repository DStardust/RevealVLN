# RevealNav Method-Freeze-2 Correctness Revision 1

**Revision ID:** `MF2-CR1`  
**Date:** 2026-08-24  
**Status:** approved for Phase-0C feasibility implementation only; not yet the
canonical method freeze; training and semantic annotation remain forbidden.  
**Canonical document:** `FROZEN_SPEC.md` remains byte-for-byte unchanged until
all Phase-0C gates in this document pass.

## 1. Why this revision is necessary

The accepted Phase-0B engineering evidence invalidates two assumptions of the
current implementation, but not the high-level research question.

1. The frozen ETP-R1 runtime constructs twelve RGB and twelve depth headings
   using `get_camera_orientations12()`. It is a 360-degree panoramic candidate
   frontend, although the method claim concerns candidates that evolve under
   finite ego-FOV observation.
2. ETP-R1 exposes all retained ghosts in its global action set and removes a
   ghost only after it is consumed. MP3D is static and reversible. Therefore a
   threshold-free physical last-safe time is normally absent.
3. Low-level `MOVE`/`TURN` operations use `step_without_obs`; they do not
   consume Habitat's nominal 5000-step episode counter. The effective ETP-R1
   cap is 25 high-level decisions, and one such decision can execute an entire
   multi-node backtrack path.
4. The accepted first-five witness produced zero unique expiry boundaries and
   evaluated candidate-endpoint-to-checkpoint return, not actual-prefix to a
   fixed target branch/saved option.
5. After candidate identity closure, all 50 traces are complete, but the
   current offline target proposal has zero episodes with three consecutive
   identical raw/lineage waypoint IDs. A persistent semantic branch cannot be
   equated with a raw GraphMap waypoint ID.

Machine-readable evidence:

- `artifacts/runtime/phase0_correctness/CANDIDATE_IDENTITY_AUDIT.json`
- `artifacts/runtime/phase0_correctness/IDENTITY_V3_RERUN_SUMMARY.json`
- `artifacts/runtime/phase0_correctness/TX_FEASIBILITY_AUDIT.json`

The T_X diagnostic admits 100 reached reference-route turns. When only the
allowed return distance changes from 2m to 5m to 10m, unique observed expiry
counts change from 79 to 49 to 23, while right-censoring changes from 14 to 50
to 76. These are post-hoc engineering diagnostics, not benchmark results. They
show that expiry is resource-conditioned rather than intrinsic.

## 2. Corrected scientific object: Reveal–Option-Cost frontier

For a fixed semantic branch event with target region (R(b^*)), event
checkpoint (v_e), actual prefix state (x_t), and a controller (pi_c),
define controller costs in counted low-level actions:

\[
C_t^{\mathrm{direct}} =
\begin{cases}
C_{\pi_c}(x_t\rightarrow R(b^*)) & b^*\text{ is causally exposed at }t,\\
+\infty & \text{otherwise},
\end{cases}
\]

\[
C_t^{\mathrm{save}} =
C_{\pi_c}(x_t\rightarrow v_e)+C_{\pi_c}(v_e\rightarrow R(b^*)),
\qquad
C_t^*=\min(C_t^{\mathrm{direct}},C_t^{\mathrm{save}}).
\]

For an explicitly declared remaining budget (B_t), the corrected expiry is

\[
T_X(B)=\max\{t:C_t^*\le B_t\text{ and the witnessed sequence is safe}\}.
\]

The primary supervision target is the full cost/frontier (C_t^*), including
right censoring and controller failure. `T_X(B)` is a derived deployment
quantity. The benchmark must not select (B) after observing model results.

`T_X(B)` is a **last-passage label**, not an online first-passage stopping
time.  A safe--unsafe--safe sequence has a unique last safe prefix when an
observed infeasible suffix follows it; this is reported as re-entry rather
than rejected as non-unique.  The benchmark therefore publishes four
disjoint statuses for every event, controller and budget:
`UNIQUE_LAST_SAFE_MONOTONE`, `UNIQUE_LAST_SAFE_WITH_REENTRY`,
`RIGHT_CENSORED`, and `NEVER_FEASIBLE`.  Re-entry is never filtered from the
denominator.  At the final observed prefix, a feasible frontier is
right-censored and must not be converted into an observed expiry.

Because a last-passage target depends on the future, the model must also
predict the causal current-feasibility variable

\[
z_t(B)=\mathbb{1}[C_t^*\le B_t],
\]

or its calibrated probability.  Online decisions consume only this causal
current-feasibility prediction and the cost distribution; offline
`T_X(B)` is used for supervision, diagnosis and evaluation.  The method must
not claim that the last-safe label itself is observable online.

To compare across event scale, also report

\[
b_t = B_t / \max(C_{\pi_c}(v_e\rightarrow R(b^*)),5),
\]

where the denominator floor of five low-level actions rejects numerically
degenerate event legs. Fixed reporting points are (b\in\{1.5,2,3,4\}) plus
the budget-curve AUC over ([1,4]). Raw unnormalized action costs are always
released with the labels. No one point is designated as the favorable main
operating point.

This revision does **not** claim irreversible dynamics in MP3D. It claims
resource-bounded option loss under evolving, causally exposed candidates.

## 3. Causal ego-FOV candidate protocol

### 3.1 Oracle Ego-FOV (first feasibility protocol)

- Sensor HFOV is 63 degrees.
- Only branches intersecting the current causal view and passing the shared
  traversability/controller check enter the current executable set.
- Navmesh, future route and hidden views are used only by offline label and
  rollout workers, never by the evaluated model.
- `INSPECT` is a physical counted turn; the new heading becomes available only
  after that action.

This protocol isolates whether Reveal–Option-Cost events exist before risking
an automatic-front-end confound.

### 3.2 Frozen Automatic Ego-FOV

- Keep the accepted Hong et al. waypoint predictor weights and ETP-R1
  controller/topology weights frozen.
- Replace simultaneous panoramic acquisition with a causal view buffer.
  Initially only the current 63-degree view is populated. `INSPECT` actions
  physically rotate the agent and populate additional slots.
- Hidden RGB/depth views are never rendered for model inference. Missing
  feature slots use a fixed zero/missing token and an explicit mask; candidate
  logits outside acquired headings are forced to negative infinity before
  NMS/action selection.
- The same causal buffer and mask must feed both the candidate frontend and
  the frozen policy features. Masking only REE while ETP-R1 still consumes the
  panorama is forbidden.
- A hidden-view perturbation test must prove bit-identical exposed candidates,
  logits and actions before the corresponding `INSPECT` acquisition.

The adapter is deterministic, shared by every matched-input baseline, and is
not presented as a learned candidate-generation contribution. Unmodified
panoramic ETP-R1 may be reported only as a different-sensor reference.

## 4. Branch identity contract

Numeric graph localization uses the accepted
`persistent-branch-identity/v3-engineering` protocol:

- preserve ETP-R1's deterministic nearest-ID localization;
- retain every within-radius ID and exact ranked distance in the hash chain;
- retain executed ghost-to-node lineage evidence;
- never promote numeric uniqueness to semantic branch identity.

A Reveal Event uses a semantic branch track, not a raw waypoint ID. The Phase
0C oracle labeler defines a target navmesh exit region. The automatic protocol
associates causal candidates over time using only observed appearance,
relative bearing/range and controller odometry. Any track involved in a
multi-match that cannot be semantically resolved is excluded, not guessed.

The mapping from numeric waypoint proposals to a semantic exit region is
many-to-one. Multiple ETP proposals may occupy different angle/range bins yet
represent the same executable exit; all such proposals and their v3 evidence
are retained as the branch's proposal set. This within-region multiplicity is
reported but is not semantic ambiguity. A track is fail-closed only when a
proposal/track is compatible with more than one fixed exit region under the
declared separation margin, or when no causal proposal supports the region.
No arbitrary representative is used to manufacture semantic uniqueness.

The persistence requirement remains (K=3) consecutive causal prefixes. It
is not weakened because the raw waypoint-ID probe failed.

## 5. Model consequences

The three-part method remains recognizable:

1. **REE becomes budget-conditioned.** It predicts reveal variables, causal
   current feasibility (z_t(B)), and a calibrated distribution/quantiles for
   (C_t^*). Last-passage labels are derived offline for declared budgets and
   are not described as first-passage hazards. U/A/D remains an interpretable
   readout.
2. **ECOG remains selective option memory.** A node stores an executable
   controller reference and predicted option-cost frontier; it is valuable
   only when it lowers future task loss under one or more declared budgets.
3. **OPP remains expected-loss minimization.** OPV is still the difference of
   predicted Q values with and without a checkpoint. Cost-frontier predictions
   make the resource contract explicit rather than threshold-induced.

The model input receives remaining counted budget and causal action history.
It still receives no future frame, target truth, navmesh or simulator pose.

## 6. Low-level controller accounting

Every `MOVE_FORWARD`, `TURN_LEFT`, and `TURN_RIGHT` executed inside FOLLOW,
INSPECT, EXPLORE or multi-node BACKTRACK increments the common budget. A
high-level ETP action cannot hide an unbounded backtrack inside one step.

For every counterfactual label, report separately:

- low-level action count (primary budget);
- path length;
- wall-clock/controller calls;
- collision count and controller failure;
- Oracle-controller and frozen-controller costs.

All methods share the same speed, action definitions, collision behavior and
budget ledger.

## 7. Phase-0C gates

No training or semantic review packet is authorized until all gates pass.

1. **Causal sensor gate:** hidden-view perturbation has exactly zero influence
   before acquisition; all low-level turns are counted.
2. **Oracle event gate:** on the frozen 50 RxR-train queue, the oracle ego-FOV
   probe yields at least 15 K=3 persistent, nondegenerate provisional branch
   events across at least 10 scenes. This is a feasibility floor, not a paper
   dataset claim.
3. **Cost witness gate:** every admitted event has reproducible raw
   (C_t^{direct}, C_t^{save}, C_t^*) evidence under both Oracle and frozen
   controllers, or an explicit controller-failure/censoring status.
4. **Budget-frontier gate:** at least 60% of admitted events have a unique
   (T_X(B)) for at least two of the four fixed normalized budgets; report the
   full budget sensitivity, split monotone from re-entry cases, and do not
   discard right-censored or never-feasible budgets.
5. **Nontrivial timing gate:** at least 25% of admitted events satisfy either
   (T_R>0), (T_R>T_X(B)), or a checkpoint changes the feasible-budget set.
6. **Identity gate:** all numeric mappings are v3-verifiable; semantic branch
   ambiguity is zero among admitted events.
7. **Boundary/regression gate:** RxR train only, network attempts zero,
   checkpoints/environment unchanged, no frozen source modification, no
   reserve release, and at least 8 GiB free.

Failure of Gates 1–3 is a method NO-GO for the current CVPR direction. Failure
of Gates 4–5 means the public static indoor setting does not support the
expiry/option claim strongly enough; do not compensate by adding modules.

## 8. What remains frozen and what is not claimed

Unchanged for this feasibility revision:

- public-first datasets (RxR-CE-en primary, R2R-CE validation);
- accepted ETP-R1 checkpoint as the initial backbone;
- REE/ECOG/OPP decomposition and pre-error option-preservation thesis;
- train/val/test isolation, three-seed/statistical requirements, main
  baselines, and the publication gates in `FROZEN_SPEC.md`;
- no large-scale pretraining or online RFT in the first implementation.

Explicit non-claims:

- no intrinsic physical irreversibility in MP3D;
- no validated Reveal Event or T_X yet;
- no training authorization;
- no CVPR competitiveness or acceptance guarantee;
- no change to the canonical `FROZEN_SPEC.md` until Phase-0C acceptance.
