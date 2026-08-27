# V5.11 / V5.12 offline failure analysis

- V5.11 direct interventions are not a viable main result: mean SPL benefit is -0.147246.
- V5.12 return mechanics mostly execute, but branch selection remains harmful: 58 successful returns versus 1 failed, while mean SPL benefit is -0.144445.
- Reversibility alone does not recover task performance: V5.12 minus V5.11 mean SPL benefit changes by 0.002800 and nDTW by -0.010158.
- The next intervention should therefore be a causal pre-action veto trained on branch-level net advantage, while V5.6 remains the validated proposal backbone.

This analysis uses previously opened val-seen development results only; it makes no fresh-confirmation or paper-performance claim.
