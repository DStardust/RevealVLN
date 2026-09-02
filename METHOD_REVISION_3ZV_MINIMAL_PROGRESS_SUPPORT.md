# MF3ZV — Minimal Instruction-Progress Support Audit v1

Revision: `mf3zv_minimal_progress_support_v1`

MF3ZV is a train-development-only, support-only audit. It does not train a model,
execute navigation, read outcome metrics, or access a public split. It asks whether
the smallest useful instruction-progress state can be defined causally before any
progress-memory learner is proposed.

## Frozen scope

Only two families are allowed:

1. `ORDINAL`: an ordinal token (`first`, `second`, `third`, `next`, or `another`)
   bound to a navigational occurrence such as a turn, doorway, hall, room, or stair.
2. `PASSED_LANDMARK`: an explicit causal instruction to pass or proceed beyond a
   visual landmark or region.

Each instruction contributes at most its earliest mechanically proposed atom. No
instruction graph, UAD, Reveal/Expiry, option memory, returnability, oracle, model,
or navigation policy is part of this revision.

## Fixed stages and stop rules

The order is `LANGUAGE → ATOM → STATE → LOCAL_TARGET → FINAL_SUPPORT`.

The Q1 atom gate requires at least 50 valid atoms and valid coverage at least 0.70
within the outcome-blind, scene-balanced review cohort (at most 50 R2R and 50 RxR).
Only a Q1 pass permits Q2.

The Q2 state gate requires at least 40 causally observable transitions, coverage at
least 0.70 among Q1-valid episodes, and at least 15 raw MP3D scenes. `UNKNOWN` is a
valid negative support result and is never replaced with route truth or future
observations. Only a Q2 pass permits Q3.

For Q3, a domain is eligible for a later separately versioned probe only if it has
at least 30 exact same-prefix native local targets over at least 10 raw scenes.
R2R and RxR are evaluated independently; a single-domain pass is explicitly only a
single-domain feasibility result.

## Scientific boundaries

Selection and annotation may use train instructions, strictly causal observations,
candidate histories, and exact native action identity in the current dynamic
candidate set. They may not use success, reward, SPL, nDTW, SDTW, utility,
counterfactual deltas, CAR results, public splits, future frames/candidates, route
truth, shortest-path targets, or cross-episode matching.

All copied input assets are physical project-local copies with SHA-256 provenance.
The fixed lexical audit is recorded as AI-assisted and is not human gold.

Even a pass produces no checkpoint and authorizes no navigation or public
evaluation. A later model probe, if allowed, must be separately versioned as
`MF3ZW_PROGRESS_STATE_PROBE_V1`.

