# MF3ZP-REVEALSKILL v1

Revision: `mf3zp_revealskill_v1`

Status: versioned feasibility revision. It does not modify `FROZEN_SPEC.md`,
does not reopen the stopped single-decision gate family, and does not authorize
public-split access.

## Hypothesis

RevealSkill asks whether an irreversible navigation commitment should be
delayed until the option-specific decisive evidence chain is resolved, while
preserving still-returnable alternatives. Frozen ETP-R1 remains the navigation
substrate:

```text
instruction -> decisive evidence DAG -> per-constraint U/A/D
            -> evidence memory -> option memory -> Reveal/Expiry
            -> evidence-preserving high-level skill -> frozen ETP-R1
```

Qwen is restricted to instruction decomposition and causal visual grounding.
It never receives rewards, rollout outcomes, task metrics, policy errors, or a
request to choose a navigation action.

The preannotation API model identifier is fixed to `qwen3.8-max`; no fallback
model or rolling-alias substitution is permitted.

## Evidence graph and U/A/D

An instruction is a DAG of minimal constraints with kinds `ENTITY`,
`RELATION`, `DIRECTION`, `ORDINAL`, `TEMPORAL_ORDER`, `EXCLUSION`, and `GOAL`.
Each option uses the transitive dependency closure of the constraints marked
decisive for that option. There is no instruction-level U/A/D state.

For constraint k at prefix t, S denotes instantiation, G distinguishability,
and E decisive semantic resolution. The state is U when S=0, A when S=1 but G
or E is 0, and D only after S=G=E=1 for exactly K=3 consecutive prefixes.
Option readiness is U if any decisive constraint is U, D if all are D, and A
otherwise. Soft readiness is a feature only and cannot open COMMIT.

The active evidence frontier contains unresolved constraints whose dependencies
are all D. Constraint Reveal is the first stable D prefix; option Reveal is the
maximum Reveal time over its decisive chain. Interval annotations remain
intervals. Expiry is the last prefix where a safe frozen-controller sequence can
preserve or recover the option; it is computed by Habitat/controller tooling,
never guessed by Qwen.

## Memory and high-level skills

Evidence memory stores only instruction-relevant observation hashes, regions,
candidate bindings, semantic support, and temporal validity. Option memory
binds ETP topology/candidates to unresolved evidence dependencies and
returnability. Neither memory may store future observations, simulator oracle
state, task outcomes, or rewards.

The fixed high-level action set is `FOLLOW`, `INSPECT`, `EXPLORE(b)`,
`BACKTRACK(v)`, `COMMIT(b)`, and `STOP`. U/A options cannot be committed; D
options may be committed. Skills execute exclusively through the frozen ETP-R1
controller and may not teleport.

## Pilot and scientific gates

The first pilot is exactly 300 outcome-blind Reveal Events: 150 R2R and 150
RxR, selected by a predeclared scene-balanced deterministic rule from train-only
development observations. Consumed confirmation scenes and every public split
are excluded.

1. Label validity: two blinded reviewers plus adjudication, with derived U/A/D
   agreement at least 0.65 and evidence-closure agreement at least 0.70.
2. Oracle headroom: in each domain, Oracle RevealSkill must reduce premature
   commitment rate by at least 25% and improve fixed navigation utility, with
   raw-scene bootstrap lower 95% bounds above zero.
3. REE learnability: evidence memory must improve per-constraint U/A/D macro-F1,
   Reveal NLL, and Expiry NLL over temporal-only in both domains, again with
   positive raw-scene bootstrap lower bounds.
4. Full development: Full RevealSkill must improve utility and PCR in both
   domains without clear SR/SPL regression, and beat the frozen specified
   ablations.

Failure at any gate stops this revision. No threshold, loss-weight,
regularization, architecture, or event-definition search is permitted after
results. Even complete development PASS requires a separate, newly sealed
public-evaluation protocol.

## Fixed implementation values

- U/A/D stability: K=3.
- Evidence-memory option budget: M=8.
- Bounded high-level counterfactual horizon: H=8 skills.
- Temporal backbone: existing frozen-form GRU design, hidden size 64.
- Constraint readout: one fixed width-64 layer.
- REE loss: equal 1/6 weights for set, separation, evidence, Reveal, Expiry,
  and valid-interval monotonicity.
- Utility: 0.50 nDTW + 0.25 SDTW + 0.25 SPL.
- Development bootstrap: raw MP3D scene clusters, 10,000 replicates, fixed
  seed 20260901.
