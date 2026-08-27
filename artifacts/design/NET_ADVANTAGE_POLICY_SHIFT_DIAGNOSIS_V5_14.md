# V5.14 policy-shift diagnosis

V5.14 completed all 10,114 preregistered `val_seen` runs (778 episodes, five
groups, and three treatment seeds) after 13 CUDA-OOM jobs and two interrupted tail
jobs were rerun. The final matrix has zero missing or failed runs. This is diagnostic
development evidence, not a paper result; `val_unseen` was not opened.

The scientific gate failed. Official ETP-R1 reached Success 0.7571, SPL 0.6731, and
nDTW 0.7428. V5.6 reached 0.7528, 0.6696, and 0.7422. V5.6 plus Net-Advantage was
bit-equivalent in task metrics to ETP-R1, so its paired differences were exactly zero
and could not satisfy a strictly positive directional or bootstrap gate. It did,
however, improve mean SPL over V5.6 by 0.0035; that confidence interval crossed zero.

The result is explained by deployment coverage, not by an episode or metric failure.
The main controller made 304 Net-Advantage decisions and vetoed all 304. The
Net-Advantage-only ablation made 20,211 decisions and also approved none. The frozen
ensemble threshold was +3.641 m. Among 268 main decisions with complete causal score
traces, the best candidate score ranged from -5.259 m to +1.558 m; 28 were above zero
but none reached the frozen threshold. The remaining decisions failed closed because
a required causal input was unavailable. Lowering the threshold to zero is not an
admissible repair: on the R2R-train calibration and internal-development partitions,
that rule produced negative realized net benefit.

This demonstrates a candidate-distribution mismatch. The 46,743 training rows cover
all aligned ETP alternatives, whereas deployment scores only alternatives proposed by
the much sparser V5.6 policy. Absolute extreme-tail calibration did not transfer even
though member ranking AUC transferred inside the R2R-train scene split.

## Required V5.15 development gate

1. Preserve V5.14, its protocol, and its negative result unchanged.
2. Collect V5.6 policy-induced proposal rows on R2R `train` only. Run the controller
   in causal shadow or active collection mode and label proposals with the existing
   route-consistent counterfactual procedure; no validation payload or metric may be
   used for fitting or threshold selection.
3. Reuse the existing candidate-level ensemble as representation initialization, but
   learn or calibrate selective risk on the policy-induced distribution. The online
   rule must retain the explicit wrong-trial cost and must fail closed on missing
   causal inputs.
4. Keep train/calibration/internal-development scenes disjoint. Require positive
   realized net benefit and nontrivial coverage on both policy-induced calibration
   and internal-development scenes, plus a prespecified uncertainty/risk bound.
5. Seal a versioned V5.15 protocol before rerunning complete `val_seen`. Treat that
   rerun as development evidence. Open `val_unseen` only if the sealed `val_seen`
   gate passes without another post-hoc operating-point search.

Post-hoc selection of a threshold from the completed V5.14 `val_seen` scores is
forbidden. Exact equality to ETP-R1 is a safe fallback result, not a positive method
result and not evidence for the paper's main claim.
