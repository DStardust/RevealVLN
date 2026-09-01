# MF3ZP Codex-Proxy REE v1

Revision: `mf3zp_codex_proxy_ree_v1`

This is an exploratory engineering run requested while human review is not
available.  It does not modify or satisfy the sealed single-expert or formal
three-reviewer protocols.

## Labels

- Population: the frozen 80-event single-expert scout selection.
- S/G/E source: the already cached, outcome-blind `qwen3.8-max` evidence
  records for every reviewed prefix.
- DEC source: one deterministic dependency/evidence rule fixed in code before
  training.  It selects the deepest currently evidenced graph nodes, includes
  their dependency closure, and treats earlier-only dependencies as
  prerequisites.
- Provenance: `AI_PROXY`; never `human`, `expert`, `gold`, or `adjudicated`.
- No new Qwen request is made.

The proxy labels may diagnose whether the frozen temporal implementation can
learn the cached semantic annotations.  They cannot establish annotation
validity, Oracle Headroom, REE scientific PASS, or public-evaluation authority.

## Fixed training probe

- Raw MP3D scene-disjoint five-fold OOF evaluation.
- Arms: current snapshot and strictly causal temporal prefixes.
- Backbone: the existing fixed `TemporalRevealExpiryEncoder`, GRU hidden 64.
- Target: per-constraint S/G/E only.  Expiry is unavailable and is not
  fabricated.
- Epochs: 250; Adam learning rate `1e-3`; weight decay `1e-4`.
- Seed: `20260901`; no model, threshold, seed, or hyperparameter search.
- Prediction boundary: probability `>= 0.5`.
- No deployment checkpoint is written.

The result is descriptive.  A positive proxy result means only that temporal
features improve prediction of provisional machine labels on held scenes.  A
negative result does not invalidate manually verified RevealSkill labels.

All public split flags remain false.
