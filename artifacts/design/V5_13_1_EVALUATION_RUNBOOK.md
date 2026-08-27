# V5.14 ensemble evaluation runbook

The runner refuses every evaluation unless the V5.14 full R2R-train ensemble result
passes its learnability, split-isolation, manifest, checkpoint-size, checkpoint-SHA,
and three-seed gates. In particular, `val_unseen prepare` fails before reading the
validation payload when that training result is absent or failed.

## Automatic handoff

The detached handoff supervisor waits for the existing full collection pipeline,
validates the generated labels and three checkpoints, runs complete `val_seen`, and
launches `val_unseen` only when the sealed `val_seen` main gate passes:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=scripts:. .envs/etpr1/bin/python \
  scripts/watch_r2r_v5_13_1_handoff.py launch
```

Its combined live monitor is:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=scripts:. .envs/etpr1/bin/python \
  scripts/watch_r2r_v5_13_1_handoff.py monitor
```

The supervisor is independent of the launching terminal and Codex session. It stops
fail-closed on a data, training, launch, or `val_seen` scientific-gate failure and
records the exact terminal stage in `HANDOFF_STATE.json`.

## Development diagnostic

After full training passes, launch complete `val_seen` in a detached process:

```bash
cd /mnt/data_nas/deeprobotics/daiyang/vla
PYTHONNOUSERSITE=1 PYTHONPATH=scripts:. .envs/etpr1/bin/python \
  scripts/run_r2r_v5_13_1_paired.py launch \
  --split val_seen --gpus 0,1,2,3,4,5,6,7
```

Monitor it without attaching to the orchestrator:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=scripts:. .envs/etpr1/bin/python \
  scripts/monitor_r2r_v5_13_1_paired.py --split val_seen
```

If one or more jobs fail, diagnose their per-job stderr logs and issue the same
`launch` command again. Completed provenance-matching summaries are retained; failed
or interrupted run directories are moved under `interrupted/`, and every attempt is
recorded in `JOB_ATTEMPTS.json`.

## Primary R2R result

Only after the training operating point remains locked, launch the complete authorized
`val_unseen` split by replacing `val_seen` with `val_unseen` in the two commands above.
No test or challenge split is supported by the runner.

Each successful detached run automatically aggregates all five groups, computes
10,000 paired episode-bootstrap replicates, and writes:

- `R2R_V5_13_1_PAIRED_RESULT.json`
- `tables/R2R_V5_13_1_GROUP_METRICS.csv`
- `tables/R2R_V5_13_1_CONTROLLER_METRICS.csv`
- `tables/R2R_V5_13_1_PAPER_TABLES.md`

The official ETP-R1 baseline is deterministic and runs once per episode. It is
explicitly broadcast across the three treatment seeds only inside paired statistics.
