# SecondThought: Current Method, Evidence, and Research Direction

**Revision:** `secondthought_method_and_results_v1`
**Date:** 2026-09-03
**Status:** exploratory novelty revision; not a frozen submission protocol
**Working title:** *SecondThought: Risk-Controlled Proposal Deliberation for Frozen Vision-Language Navigation Agents*

**Chinese version:** [`METHOD_NOVELTY_REVISION_SECONDTHOUGHT_V1_ZH.md`](METHOD_NOVELTY_REVISION_SECONDTHOUGHT_V1_ZH.md)

## 0. Relationship to the Frozen Project

This document records a separate candidate-correction research line. It does
not modify, supersede, or reinterpret [`FROZEN_SPEC.md`](FROZEN_SPEC.md), whose
Method-Freeze-2 claims, modules, datasets, protocols, and gates remain
immutable.

The candidate-correction artifacts summarized here are not currently present
in this self-contained repository. Exact numbers below are preserved from the
reported experimental record, but have not been re-derived from local artifacts
in this workspace. Before any paper submission, every retained result must be
reproduced inside this project from independently acquired authoritative assets,
with sealed protocols, immutable records, code hashes, checkpoint provenance,
and scene-disjoint split manifests.

## 1. Executive Summary

The experiments establish a narrow but coherent empirical chain:

1. Frozen ETP-R1 usually generates the correct local candidate: the gold
   candidate is in its Top-3 for approximately 99.4--99.7% of exact-rankable
   decisions.
2. Its own score distribution identifies many risky decisions: Top-1
   probability reaches AUROC 0.803 for separating wrong from correct Top-1
   predictions, and the highest-risk 20% captures 47.4% of all Top-1 errors.
3. The remaining problem is not candidate generation but reliable selection
   among a small set of plausible candidates.
4. Independent linear scoring, pairwise comparison, current-frame raw evidence,
   ResNet features, frozen DINOv2 features, and simple short-history pooling did
   not produce a reliable positive correction effect.
5. A relational Top-3 Transformer trained on the expanded dataset is the only
   tested selector that changes the result from negative to slightly positive.
   Its formal held-scene gain is nevertheless far below the pre-sealed gate.
6. The decisive bottleneck is intervention harm: the selector repairs many
   baseline errors but breaks nearly as many originally correct decisions.

The resulting method hypothesis is therefore not merely "rerank uncertain
candidates." It is:

> A frozen navigation policy should be overridden only when a relational
> deliberator has both an alternative proposal and calibrated evidence that the
> intervention is more likely to rescue an error than to damage a correct
> decision.

## 2. Problem Formulation

At timestep \(t\), a frozen navigation policy produces a proposal set
\(\mathcal C_t=\{c_1,\ldots,c_K\}\), native scores \(s_t\), and a baseline
decision

\[
b_t=\arg\max_{c_i\in\mathcal C_t}s_t(c_i).
\]

The correction system may inspect only deployment-time information:

- the current observation;
- the instruction;
- current proposal representations and native scores;
- proposal geometry or action semantics;
- frozen policy states and executed actions from the past;
- no future frame, outcome, shortest-path distance, scene ID, or episode ID.

It outputs either `KEEP` or a replacement proposal \(r_t\). The scientific
objective is net correction, not reranker accuracy in isolation:

\[
u_t=
\mathbf 1[r_t=y_t]-\mathbf 1[b_t=y_t]
\in\{-1,0,+1\},
\]

where \(y_t\) is used only as training/evaluation supervision. The three cases
are:

- \(+1\): rescue, or wrong \(\rightarrow\) correct;
- \(0\): neutral intervention;
- \(-1\): harm, or correct \(\rightarrow\) wrong.

The system should intervene only when

\[
P(u_t=+1\mid x_t)-\lambda P(u_t=-1\mid x_t)>\tau,
\]

where \(\lambda\) encodes the cost of damaging an already correct action and
\(\tau\) is selected without held-scene leakage.

## 3. Current Method

### 3.1 Frozen proposal policy

The base policy remains completely frozen. ETP-R1 is the discovery backbone in
the completed probes; neither its candidate generator nor its checkpoint is
updated.

### 3.2 Uncertainty trigger

The current trigger uses low native Top-1 probability. Under five-fold
raw-scene-disjoint evaluation, each fold derives the fixed 20% training-risk
threshold from training scenes and applies it unchanged to held-out scenes.
Untriggered decisions retain the base action exactly.

### 3.3 Relational Top-3 deliberator

Only triggered decisions are reconsidered. The strongest tested selector is a
small Transformer over the three highest-scoring proposals. Its available
evidence includes:

- frozen instruction and navigation/history representations;
- frozen candidate or proposal representation;
- frozen visual evidence where available;
- native ETP score;
- relative heading and distance;
- proposal masks and relational comparison across the Top-3.

The same scorer is shared across proposal positions, and its output is used as
a residual correction to the native ranking. All frozen-policy encoders remain
unchanged. This relational processing is important: models that score each
candidate independently did not show the same scaling behavior.

### 3.4 Proposed counterfactual utility gate

The next method component is a separate intervention gate trained on
out-of-fold proposal outcomes. It predicts `RESCUE`, `NEUTRAL`, and `HARM`, or
equivalently estimates the expected utility above. Its purpose is not to find a
new candidate; it decides whether to trust the deliberator or preserve the
frozen policy decision.

To prevent leakage, utility labels for an example must be generated by a
selector that did not train on that example or its raw scene. Thresholds and
costs must be fixed on nested training folds before held-scene evaluation.

### 3.5 General proposal interface

The implementation should be refactored away from ETP-specific tensor widths
into a `FrozenPolicyAdapter` exposing:

```text
instruction evidence
current observation evidence
last K policy/history states
K executable proposals
native proposal logits
proposal geometry or action semantics
STOP proposal, when available
proposal-to-environment executor
```

For graph navigation agents, proposals are waypoint or topology nodes. For
modern VLA agents, proposals are executable action tokens. Agent-specific input
projections are allowed, but the uncertainty trigger, relational deliberator,
utility objective, and evaluation protocol should remain shared.

## 4. Completed Evidence

### 4.1 STOP/termination branch

#### Oracle headroom: PASS

Revision: `stop_oracle_headroom_v1`

| Quantity | Result |
|---|---:|
| Train-development episodes | 156 |
| Baseline successes / failures | 134 / 22 |
| Baseline SR | 85.90% |
| FALSE_STOP | 10 |
| MISSED_STOP | 1 |
| PURE_NAVIGATION_FAILURE | 11 |
| Oracle-recovered successes | 10 |
| Oracle SR | 92.31% |
| Absolute SR headroom | +6.41 percentage points |
| Recoverable fraction of baseline failures | 45.45% |

Result: `STOP_ORACLE_HEADROOM_PASS`. Termination mistakes contain meaningful
oracle value, dominated by false STOP rather than missed STOP.

#### Tiny STOP veto: FAIL

Revision: `tiny_stop_veto_v1`

The collected set contained 137 STOP proposals: 127 valid and only 10 invalid.
The held-scene STOP-invalid recall was 0.60, below the fixed 0.70 gate. The small
number of invalid examples was insufficient for a reliable scene-disjoint
verifier, so the branch stopped before claiming rollout improvement.

Result: `TINY_STOP_VETO_FAIL`. The oracle opportunity remains real, but the
first supervised verifier is not validated.

### 4.2 Candidate availability and reranking headroom

Revision: `candidate_rerank_headroom_v1`

| Quantity | Result |
|---|---:|
| Exact-rankable decisions | 1,428 |
| Top-1 correct / wrong | 1,179 / 249 |
| Top-1 accuracy | 82.56% |
| Top-3 accuracy | 99.72% |
| Top-1 errors with gold in Top-3 | 245 / 249 |
| Top-3-recoverable error rate | 98.39% |
| Scenes covered by recoverable errors | 54 / 59 |

Result: `CANDIDATE_RERANK_HEADROOM_PASS`. The candidate generator is not the
primary bottleneck in this population. Almost every Top-1 error is locally
recoverable if the system can choose correctly within its existing Top-3.

### 4.3 Existing secondary signals

Revision: `candidate_secondary_signal_probe_v1`

Frozen instruction-candidate compatibility and available local-score signals
did not reach the sealed 65% gold-preference requirement on the fixed 245-error
population.

Result: `SECONDARY_SIGNAL_PROBE_FAIL`. No existing scalar signal justified a
zero-training reranker.

### 4.4 Candidate uncertainty

Revision: `candidate_uncertainty_headroom_v1`

On the fixed 1,428-decision population, native Top-1 probability was the best
single uncertainty signal:

- wrong-versus-correct AUROC: **0.803**;
- highest-risk 20% captured **47.4%** of all Top-1 errors.

Result: `CANDIDATE_UNCERTAINTY_HEADROOM_PASS`. The frozen policy often knows
when a decision is risky, even though uncertainty alone does not identify the
correct alternative.

### 4.5 Selective correction probes

All probes used the same five-fold raw-scene-disjoint principle and changed
only triggered Top-3 decisions.

| Probe | Evidence/structure | Result | Scientific implication |
|---|---|---|---|
| Independent linear Top-3 | Frozen candidate features, score, geometry | FAIL | Candidate-wise separability is insufficient |
| Pairwise MLP | Explicit candidate-vs-candidate differences | FAIL | Pairwise structure alone does not solve the error |
| Raw current evidence MLP | Candidate-facing current visual evidence | FAIL | Re-reading only the current observation is insufficient |
| Frozen ResNet arm | Alternative frozen visual feature | No positive held-scene gain | Conventional backbone replacement is insufficient |
| Frozen DINOv2 arm | Stronger frozen self-supervised visual feature | No positive held-scene gain | Better static visual representation alone is insufficient |
| Simple temporal pooling | Short frozen history | FAIL | Merely averaging a short history is insufficient |
| Relational Top-3 Transformer | Joint Top-3 comparison with frozen evidence | Small positive, gate FAIL | Relational reasoning is learnable but not yet safe |

The exact per-fold ResNet and DINOv2 metrics are not reproduced here because
their local result artifacts are absent from this repository. Their defensible
current conclusion is limited to the observed lack of positive held-scene gain.

### 4.6 Expanded-data relational Transformer

The expanded collection changed the evidence scale substantially:

| Quantity | Result |
|---|---:|
| Episodes | 6,446 |
| Exact-rankable decisions | 65,031 |
| Raw scenes | 59 |
| Frozen ETP Top-1 errors | 12,248 |
| Decisions with gold in Top-3 | 64,654 |
| Gold-in-Top-3 rate | 99.42% |

Formal epoch-20 held-scene result:

| Metric | Frozen ETP | Selective Transformer | Change |
|---|---:|---:|---:|
| Triggered exact-candidate accuracy | 55.170% | 55.639% | +0.469 pp |
| Global exact-candidate accuracy | 81.166% | 81.260% | +0.094 pp |
| Wrong \(\rightarrow\) correct | -- | 1,489 | -- |
| Correct \(\rightarrow\) wrong | -- | 1,428 | -- |
| Net corrections | -- | +61 | -- |
| Recovery/new-error ratio | -- | 1.043 | -- |

This fails the predeclared requirement of at least +5 pp triggered gain, +1.5 pp
global gain, and a recovery/new-error ratio of at least 1.5.

Result: the formal model is **FAIL**, despite a positive sign.

For diagnosis only, the best observed checkpoint was epoch 11:

- triggered gain: +0.953 pp;
- global gain: +0.191 pp;
- net corrections: +124;
- recovery/new-error ratio: 1.103.

This post-hoc checkpoint also fails every substantive improvement gate and must
not be presented as a selected primary result.

### 4.7 Scalar intervention-confidence diagnostic

A scalar derived from the Transformer advantage over the native ETP Top-1 did
not separate beneficial from harmful interventions:

- benefit-versus-harm AUROC: 0.538;
- within the selected highest-risk 20%, benefit/harm ratio: 0.975;
- net corrections: -2.

Therefore, a threshold on the reranker advantage is not a sufficient safety
gate. The proposed utility gate must learn intervention outcomes explicitly and
must be evaluated out of fold.

## 5. What the Experiments Establish

### Supported conclusions

1. **Proposal coverage is high.** In both the 1,428-decision and expanded
   populations, almost all gold actions are already in Top-3.
2. **Risk localization is feasible.** Native probability carries a strong
   wrong-versus-correct signal.
3. **Static feature strength is not the central missing ingredient.** Replacing
   ResNet with DINOv2 did not turn the correction problem positive.
4. **Relational structure matters.** The Top-3 Transformer is the only tested
   family with a reproducible positive sign after scaling the data.
5. **Safe intervention is the bottleneck.** Recoveries and newly introduced
   errors remain almost balanced.

### Unsupported conclusions

The current evidence does **not** establish that:

- selective correction improves full navigation SR or SPL;
- the method generalizes beyond ETP-R1;
- the proposed utility gate can predict rescue versus harm;
- DINOv2 or any new backbone solves the problem after additional tuning;
- a tiny STOP verifier is ready for deployment;
- the current result is sufficient for a CVPR-level empirical claim.

## 6. Revised Novelty Position

An uncertainty-triggered second reasoning stage is no longer a defensible
standalone novelty claim. [AdaNav](https://www.microsoft.com/en-us/research/publication/adanav-adaptive-reasoning-with-uncertainty-for-vision-language-navigation/)
uses action uncertainty to trigger adaptive reasoning. [AwareVLN](https://github.com/GWxuan/AwareVLN)
switches between sparse `[REASON]` and `[ACT]` modes at key nodes. [ATENA](https://github.com/kuai-lab/NeurIPS25_att_vln)
performs uncertainty-aware test-time adaptation on DUET and ETPNav backbones.

The more defensible hypothesis is the combination of:

1. **frozen-policy post-hoc correction**, without retraining the base agent;
2. **proposal-space abstraction** across graph nodes and VLA action tokens;
3. **counterfactual intervention utility**, explicitly separating rescue from
   harm rather than predicting action correctness alone;
4. **risk-controlled abstention**, preserving the native action whenever the
   expected intervention value is not positive;
5. a possible **unified MOVE/STOP verification view**, if the STOP branch later
   receives adequate invalid-example support.

The modern agent evaluations are evidence of generality, not separate
contributions to be counted independently.

## 7. Modern-Agent Validation Plan

Older DUET/ScaleVLN-style agents may remain controlled legacy baselines, but
they should not be the headline evidence for a 2027 submission.

| Role | Agent | Why it is relevant | Interface/cost assessment |
|---|---|---|---|
| Engineering bridge | [TagaVLM, ICRA 2026](https://github.com/APEX-BJUT/Taga-VLM) | Qwen2-0.5B/7B, public weights, topology-aware global actions | Closest modern match to Top-K proposal correction; medium cost, but DUET-derived |
| Primary modern target | [StreamVLN, ICRA 2026](https://github.com/InternRobotics/StreamVLN) | LLaVA-Video with SlowFast streaming context; R2R/RxR support and public checkpoint | Strong test of temporal correction; requires action-token adapter |
| Direct competitor/stress test | [AwareVLN, CVPR 2026](https://github.com/GWxuan/AwareVLN) | Llama-3 8B + SigLIP with explicit sparse reasoning/action switching | Important comparison, but overlaps selective reasoning and is not a neutral base |
| Generalist extension | [OctoNav-R1, CVPR 2026](https://github.com/buaa-colalab/OctoNav-R1) | VLA policy producing low-level action sequences in continuous environments | Broadest generality test and highest integration cost |

Recommended order:

1. Run a zero-training headroom audit on TagaVLM to validate the generic
   proposal adapter cheaply.
2. Treat StreamVLN as the primary modern cross-agent experiment.
3. Compare against AwareVLN/AdaNav as contemporary selective-reasoning work.
4. Add OctoNav-R1 only after the method produces material positive results on
   the first two bases.

For each new agent, perform a bounded train-development audit before training:

- deterministic replay and executable proposal correspondence;
- exact gold proposal/action availability;
- Top-1/Top-3 coverage;
- uncertainty AUROC;
- recoverable-error count and raw-scene coverage.

An agent should not enter the expensive correction experiment unless its native
proposal space has meaningful recoverable headroom.

## 8. Required Next Experiment

The next feasibility gate should test whether explicit intervention utility can
reduce harm on the already collected expanded population.

### Fixed comparison

```text
A. Frozen ETP-R1
B. Uncertainty-triggered relational Transformer
C. B + out-of-fold counterfactual utility gate
```

### Required leakage control

For every outer held-scene fold:

1. fit the proposal selector only on outer-training scenes;
2. generate selector choices for those scenes through inner scene-disjoint
   out-of-fold predictions;
3. derive rescue/neutral/harm labels from those out-of-fold choices;
4. train the utility gate on those labels;
5. freeze its operating point before evaluating the outer held scenes.

### Primary measurements

- triggered and global exact-candidate accuracy;
- wrong \(\rightarrow\) correct;
- correct \(\rightarrow\) wrong;
- net corrections;
- recovery/new-error ratio;
- intervention coverage;
- scene-by-scene degradation;
- expected calibration error for rescue and harm probabilities.

No full rollout or modern-agent port is warranted until the gate materially
improves the recovery/new-error trade-off. If explicit out-of-fold utility
prediction also fails, the lightweight frozen-policy correction direction
should be stopped rather than rescued with a larger Transformer.

## 9. Intended Paper Contribution Stack, Conditional on Positive Results

If the utility gate and modern-agent experiments succeed, the paper can make a
coherent three-part contribution:

1. **Diagnosis:** a large-scale study showing the proposal-coverage/risk-control
   paradox in frozen VLN agents: the correct alternative is usually present and
   errors are detectable, yet naive correction causes comparable harm.
2. **Method:** a risk-controlled relational proposal deliberator optimized for
   counterfactual rescue-versus-harm utility.
3. **Generality:** one correction abstraction spanning waypoint-candidate
   policies, streaming VLM action policies, and potentially STOP decisions.

These claims remain conditional. The current positive Transformer result is
evidence of learnability, not yet evidence of a successful navigation method.
