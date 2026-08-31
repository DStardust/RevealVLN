# MF3ZN-TUAD v1 — Temporal UAD Counterfactual Intervention Policy

## Status and family boundary

This document pre-registers `mf3zn_tuad_v1` and its counterfactual dataset
`mf3zn_teal_v1`.  It is a new causal-state and action-support revision, not a
new threshold, loss, or architecture search over MF3ZM-CAR.

The completed MF3ZM result is immutable.  Its pre-registered stop rule is
represented by the following machine-readable invariant in
`revealnav_mf3/tuad_protocol.py`:

```python
FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED = True
```

Consequently, CAR v2, RCSP v2, CAR2, threshold or catastrophe sweeps, new
single-snapshot heads, new loss weighting, extra weight decays, longer fitting,
and outcome-adaptive relaxation of the CAR gates are outside this revision.

## Scientific hypothesis

The fixed hypothesis is:

> Irreversible intervention value is identifiable from how causal evidence and
> executable candidate availability evolve before a decision—especially the
> U/A/D state and reveal/expiry dynamics—not from the instantaneous decision
> representation alone.

MF3ZN continues to study exactly one irreversible intervention.  It does not
restore ECOG, multi-branch exploration, or topology search.

## Strict causal boundary

For a decision at step `t`, an inference record may contain only information
observed at steps `j <= t`: frozen ETP checkpoint/history embeddings, the
native and executable candidate identities and embeddings, policy scores and
margins, candidate birth/death and rank history, persistence statistics, MF3V
score and uncertainty trajectories, candidate count, and the instruction
embedding.

Inference schema and tensor builders must reject treatment outcomes, utility
targets, oracle labels, `future_*` fields, observations after `t`, navmesh,
simulator pose, and future route state.  Oracle supervision is stored in a
physically separate `TemporalOracleLabel`.  Changing an oracle label or a
treatment outcome must not change an inference tensor.

## Native-inclusive exact action support

Before any treatment outcome exists, every eligible decision seals this
action support:

\[
\mathcal A_t = \{a_{native}, a^{(1)}, a^{(2)}\},
\]

where `a^(1)` and `a^(2)` are the two highest frozen-policy-ranked executable
non-native actions.  If only one exists, the support has two total actions.
The action list is never selected using utility, catastrophe, CAR errors, or a
treatment result.

The fixed `native_margin` feature is non-negative and equals the absolute
difference between the native policy score and the highest-ranked executable
non-native policy score.  The matched low-margin control uses this exact
quantity.

The collection plan also inventories, by project-relative path, byte count,
and SHA-256, the strict causal-temporal record list, frozen continuation
controller, and native-baseline manifest.  Validation rereads the hashed causal
snapshot source and rebuilds the complete plan; the embedded action list alone
is never trusted.  Before sealing, the record list is parsed through the strict
causal schema, its source-universe commitment must equal the protocol's
canonical development commitment, and its decision identities, terminal step,
native action, and executable support must exactly match the action snapshots.
Its path, byte count, and SHA-256 must also exactly equal the causal-probe
provenance whose temporal summaries passed Audit B; a same-universe replacement
record list is not authorized.
Production training must tensorize the inventoried strict causal records and
may not accept an independently assembled tensor shard.

Each TEAL arm preserves the exact native prefix, changes exactly one action at
the target step, then uses the frozen continuation with zero second
intervention.  Native and treatment arms must agree on prefix SHA-256; the
treatment action must round-trip to an ID in the target-step executable set;
and pairing never crosses episodes.

Every native and treatment arm has a separate, hashed, project-local task-
metric source bound to dataset, scene, episode, decision step, prefix, and
action identity.  It records exactly `success`, `spl`, `ndtw`, and `sdtw`, and
asserts that task metrics were not read by the controller and no public split
was accessed.  The collection validator, not the worker or training input,
recomputes

\[
U = 0.50\,\mathrm{nDTW} + 0.25\,\mathrm{sDTW} + 0.25\,\mathrm{SPL},
\qquad \Delta U(a)=U(a)-U(a_{native}).
\]

The native delta is exactly zero and an alternative is catastrophic exactly
when `delta_utility <= -0.10`.  The exact-lattice audit commits each
`(lattice_id, action_id)` source, metric vector, delta, utility, and catastrophe
with SHA-256.  Downstream training/evaluation must consume those recomputed
audit outcomes rather than caller-supplied utility or catastrophe arrays.

## U/A/D and reveal/expiry semantics

The oracle record stores target-in-set, candidate-separated, and
decisive-evidence-closed sequences, a reveal interval, an expiry step, and
resolvability.  U/A/D is derived deterministically:

- U: the target branch is not in the declared candidate set;
- A: the target is present, but separation or decisive evidence is not closed;
- D: target presence, separation, and decisive evidence remain closed for the
  fixed stability run.

No independent U/A/D classifier may bypass this definition.  U/A/D transitions
are monotone except for the separately pre-registered occlusion-reset case.

## Identifiability gate before collection

This method, architecture, audit definition, and both stop rules are sealed
before any audit result.  All three audits below must pass in RxR and R2R
separately before `mf3zn_teal_v1` treatment collection is authorized.
The sealed source inventory covers the method, causal schema, oracle labels,
fixed temporal summaries, Stage-1 model, native-anchored value head, exact
lattice, identifiability and selection logic, protocol implementation, protocol
sealer, identifiability-audit entrypoint, lattice collection/validation
entrypoint, and fixed OOF training entrypoint.

Audits A and B are bound to the immutable MF3ZM source universe: exactly 1,540
events in 39 raw MP3D scenes, with 997 RxR and 543 R2R events.  The source CAR
protocol and its canonical identity commitment are included in the MF3ZN
source inventory; a smaller, larger, rebalanced, or identity-drifted cohort
cannot authorize collection.

At audit runtime the original sealed MF3ZM loader revalidates its source-file
hashes and recomputes the canonical population.  Each audit `event_id` must
equal the fixed SHA-256 of canonical JSON over dataset, scene, episode, and
decision step in that exact canonical row order; merely copying the population
digest or matching aggregate counts cannot substitute another cohort.

### Audit A — oracle relevance

Use the existing 1,540 exact native-versus-runner events and five raw-MP3D-
scene folds.  Compare a fixed low-capacity current-observable control with the
same fixed probe augmented only by oracle U/A/D, `T_R - t`, and `T_X - t`.
Both use the same fixed ridge/additive form and Huber prediction loss; there is
no probe search.

For each domain, oracle relevance passes only when OOF `delta_Huber > 0` and a
raw-scene-cluster bootstrap 95% lower bound is strictly positive.  Domains may
not compensate for each other.

### Audit B — causal observability

On causal prefixes from the same events, compare a fixed snapshot probe at `t`
with one fixed temporal summary: score and margin slope, candidate birth and
expiry counts, native and runner persistence, rank-switch count, checkpoint
embedding drift, instruction/history alignment drift, and prefix candidate-set
Jaccard.

For each domain, temporal history must improve U/A/D macro-F1 and reduce both
reveal NLL and expiry NLL.  The raw-scene bootstrap 95% lower bound for every
oriented improvement must be strictly positive.  No post-hoc absolute F1
threshold is allowed.

### Audit C — label validity

Use one fixed scene-balanced 300-event manual-review pilot.  U/A/D agreement
must have Cohen's kappa at least `0.65`, and evidence-closure agreement must
have kappa at least `0.70`.

If any audit fails, record `TEMPORAL_ORACLE_RELEVANCE_FAIL`,
`TEMPORAL_CAUSAL_OBSERVABILITY_FAIL`, or `TEMPORAL_LABEL_VALIDITY_FAIL` and
apply Stop A: stop temporal-UAD intervention research before collecting a new
treatment.

Collection never trusts a copied PASS JSON: it re-runs the source-sealed
deterministic identifiability entrypoint from the PASS artifact's hashed causal,
oracle, and review provenance and requires an exact result match.

## Fixed model after identifiability PASS

There is no architecture grid and no joint end-to-end training.

Stage 1 is one causal GRU with hidden size 64 over frozen per-step projections.
It predicts target-in-set, separation, evidence-closure, reveal hazard, and
expiry hazard.  U/A/D is derived from those quantities under the frozen rule.
Stage 1 cannot consume intervention utility and is frozen before Stage 2.
Its loss is the unweighted mean of five masked binary cross-entropy terms: the
three factors and the two discrete-time hazards.

Stage 2 is one fixed native-anchored action-value head trained with Huber loss:

\[
q_\phi(z_t,a) = \widehat{U(a)-U(a_{native})}, \qquad
q_\phi(z_t,a_{native}) = 0.
\]

The native value bypasses the network.  Deployment is the native-inclusive
argmax over the sealed action support.  There is no switch threshold,
catastrophe threshold, risk head, quantile head, or calibration sweep.
Catastrophic risk is a scientific evaluation gate, not a tuned multiplier.
The action-value head has one hidden layer of width 64 with GELU activation.

## Development protocol and controls

The architecture, losses, feature definitions, history semantics, and action
support are fixed.  Weight decay is fixed to zero; there is no weight-decay
grid.  Stage 1 and Stage 2 each train for exactly 200 epochs with Adam learning
rate `1e-3`; Stage 2 uses Huber delta `1.0`.  Three fixed seeds
`(20260831, 20260832, 20260833)` are always fitted.  Their inference outputs
are combined by elementwise median for the declared ensemble and are reported
individually; they are never used for seed selection.  Five whole raw-MP3D-
scene folds directly produce complete OOF predictions; there is no inner
model-selection loop.

The only causal-attribution controls are:

- `TUAD-full`;
- `current-only`;
- `temporal-no-UAD-supervision`;
- `oracle-UAD` (diagnostic upper bound only);
- `runner-only-support`;
- `frozen-native`;
- fold/domain-budget-matched `matched-high-proposal-score`;
- fold/domain-budget-matched `matched-low-native-margin`;
- fold/domain-budget-matched `matched-random` (sanity only).

No control can authorize public access or replace a failed mainline.

## One-shot scientific gates

Before any unseen evaluation, all causal invariants and the exact-lattice audit
must pass, no raw scene or episode lattice may cross folds, no public split may
have been accessed, and five-fold OOF must be complete.

For RxR and R2R separately:

- total selected utility must be strictly positive;
- every fold/domain total utility must be non-negative;
- minimum leave-one-selected-scene-out total utility must be strictly positive;
- catastrophic rate must not exceed the stronger matched simple baseline;
- `tuad_full - current_only` utility must be positive with a raw-scene
  bootstrap 95% lower bound greater than zero;
- `tuad_full` must outperform `temporal_no_uad_supervision`;
- at matched fold/domain intervention budgets, `tuad_full` must outperform the
  stronger of `high_proposal_score` and `low_native_margin`.

The stronger simple baseline is pre-defined per domain as the matched baseline
with greater OOF total utility, breaking an exact tie in favor of
`high_proposal_score`.  All domains and all controls are reported.

If complete development fails any gate after identifiability passes, apply
Stop B:

> Stop learned irreversible-intervention policy development on this consumed
> development universe.

Stop B forbids TUAD v2, another temporal architecture, a different history
window or U/A/D definition, new thresholds, losses, risk heads, weight decays,
or outcome-driven tuning on these 39 scenes.  Further work must change the
data/sensor information or benchmark protocol.

## Authorization boundary

The protocol-sealer entrypoint may only seal or verify this protocol.  The
separately source-sealed scientific entrypoints are the identifiability audit,
gated lattice seal/validation, and gated fixed OOF training.  The
identifiability audit is train-development-only.  No collection is authorized
until all three audit records pass, and no training is authorized until the
sealed exact lattice passes its audit.  This revision contains no confirmation,
`val_seen`, `val_unseen`, `test`, `test_challenge`, or public-evaluation
entrypoint.
