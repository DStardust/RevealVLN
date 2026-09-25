# CPU mechanism probe V3

All six final3000 heads, all64 recorded DEV trajectories. No new GPU use, optimizer updates, autonomous navigation, new hyperparameters or outcome-based selection.

Compare actual head readout against zero added memory on identical recorded actor features/base logits. This diagnoses readout dependence and supervised action changes. LOCAL zeroing also removes its current-observation embedding, so it is not a history-only intervention. The final-query input gradient separately checks whether older inputs can influence the added branch; it is sensitivity evidence, not a causal navigation gain.

The matched unseen evaluation continues independently. Diagnostics do not change its checkpoints, source lock, protocol, denominator or scheduler.
