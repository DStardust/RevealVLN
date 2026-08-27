# Net-Advantage evaluation correctness revision V5.13.1

V5.13 correctly removed the offline score from deployment, but its evaluation
protocol named a separate reversible group even though both that group and the main
V5.6-derived controller already inherited ECOG checkpointed exploration and return.

V5.13.1 keeps the method fixed and repairs only the ablation semantics. The main
variant is V5.6 Full OPP, including ECOG, plus the causal Net-Advantage veto. Its
reversibility ablation uses the same proposal and veto but suppresses every ECOG
exploration proposal and delegates to the native ETP action. The two executable paths
therefore differ at exactly the return-capable intervention boundary.

The deterministic ETP-R1 control is executed once per episode and broadcast across
the three treatment seeds during paired statistics. Every learned group uses the
three locked training seeds. This avoids redundant deterministic simulator runs while
preserving episode-level paired comparisons and disclosing the reuse in the protocol.

No val-unseen payload is opened by this revision. Full training must pass provenance,
split-isolation, and learnability gates before val-unseen selection or execution.
