# MF3ZP-REVEALSKILL v1.1 annotation correctness erratum

This is a pre-label, outcome-blind correctness revision to the sealed v1
annotation validator. It changes no event, instruction graph, prompt, model,
scientific gate, U/A/D rule, or downstream policy.

The v1 validator incorrectly required `resolved => instantiated AND
distinguishable`. The frozen definition instead treats instantiation (S),
distinguishability (G), and semantic resolution (E) as independent factors.
Only deterministic D derivation requires S=G=E for K=3 consecutive prefixes.
Past evidence may also remain resolved in Evidence Memory after its entity is no
longer visible in the current frame.

v1.1 therefore validates the three booleans independently. It keeps exact
constraint-set checks, candidate-ID membership, ordered finite bounding boxes,
causal image-index bounds, and fixed K=3. Evidence explanations may be empty
only when all three factors are false and may not exceed 2000 characters.

The correction is sealed before rerunning the exact set of v1-invalid request
IDs. It may not read outcomes, alter factor values, or normalize malformed
candidate/bbox/image references. Remaining structural failures require a
separately sealed format-only repair or remain missing.
