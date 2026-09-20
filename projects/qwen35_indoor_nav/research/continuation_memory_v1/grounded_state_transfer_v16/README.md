# Q35N V16

Single shared-data B1/B2/Ours experiment. Evidence baseline:
`754d574bc98fd49fe30b2be76e30cbc53f0f25c3`. V15 code/results and its 87 UNKNOWN
labels are read-only. This directory contains implementation and actual pilot
attempts; it is not a claim that nine models or 720 continuations are complete.

Read `REPORT_ZH.md` for current measured progress. `PROTOCOL.json` is the current
pre-training registration; each run retains its own protocol and source lock.
Generator corrections before TEST selection are recorded as new attempts.
No TEST method score has been used to choose houses, templates or a checkpoint.

The active registered run is `runs/v16_formal_001`; independent service:
`q35n-v16-formal-20260920-01.service`. It reuses the certified FIT16/DEV2
physical assets from `runs/v16_007`. `promote_pilot.py` runs all registered
stages and publishes the resulting outcome (including failures) to the V16
GitHub branch. The service is independent of this chat. No additional launch
is needed while it is running. `GITHUB_HANDOFF.json` points to the latest
uploaded evidence snapshot, which is distinct from live server progress.

```bash
systemctl status q35n-v16-formal-20260920-01.service
tail -n 100 "$V16/standalone_jobs/v16-formal-20260920-01/job.log"
cat "$V16/runs/v16_formal_001/STATUS.json"
```

Snapshots contain directly readable summaries, an indexed log archive and
final V16 heads when produced. If the archive is split, concatenate the
ordered `EVIDENCE_LOGS.tar.gz.part*` files before extraction. Each part and
archived file has a SHA256 in its snapshot `INDEX.json`. Raw observation
arrays, base weights, feature caches and optimizer checkpoint binaries stay
on the server; their paths and hashes are explicitly catalogued. Their
absence from GitHub is not evidence that they were loaded by a web reviewer.

## Entry points

```bash
ROOT="$(git rev-parse --show-toplevel)"
PY="$ROOT/.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3"
V16="$ROOT/projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16"
JOB="v16-grounded-state-$(date -u +%Y%m%d-%H%M%S)"
printf '%s\n' "$JOB" > "$V16/LAST_JOB.txt"
"$PY" -I -S -B "$V16/standalone.py" start "$JOB" -- \
  "$PY" -I -B "$V16/pipeline.py" --config "$V16/PROTOCOL.json" \
  --run-id "$JOB" --phase all
"$PY" -I -S -B "$V16/standalone.py" status "$JOB"
```

`pipeline.py` implements `preflight`, `collect`, `features`, `train`, `evaluate`,
`review`, and `all`. An existing run requires `--resume`; resume checks immutable
configuration, source, data/cache/model bindings. Use a new service name with the
same run ID. Never restart an old completed launcher.

`standalone.py` uses the caller's UID/GID in a systemd service, closes stdin, and
records PPID/cgroup/exit status. It survives the submitting terminal or Codex
session ending; it cannot promise survival of power loss, OOM or administrator
termination. Only the task's verified owned process groups may be signalled.

## Implemented paths

- `build_manifest.py`, `collect.py`, `build_data.py`, `audit_data.py`: local asset
  split, real Habitat histories/recoveries, executed suffix labels, dense causal
  cuts, raw semantic/RGB reconstruction, exact-state sufficiency and provenance.
- `encoder.py`, `numerical_pilot.py`, `extract_features.py`: the frozen best4k
  loader and one cache/live forward implementation, golden-input diagnosis and
  resumable causal feature chunks. Future queries stay outside the encoder.
- `objective.py`, `train.py`: shared 8×64 architecture and saved seed initials,
  three auxiliary choices, full recurrent gradients, FIT-only schedule and
  optimizer/RNG checkpoints. The base encoder has zero optimizer updates.
- `select_action.py`: method logits argmax, shared by training metrics and
  rollout/review. Native STOP conflicts are logged, without the old hard guard.
- `continuation_service.py`, `evaluate_continuations.py`: private physical
  checker, full history replay, dynamic 80×9 registry, complete-group seals.
- `evaluator_v16.py`, `review.py`, `diagnose.py`: legacy and prospective safe
  endpoints, full planned denominators, missing-label bounds, event and memory
  intervention diagnostics. Missing matched controls remain unidentifiable.

CPU checks:

```bash
CUDA_VISIBLE_DEVICES='' "$ROOT/projects/qwen35_indoor_nav/.envs/q35n_qwen_g2_v1/bin/python3" \
  -I -B "$V16/test_v16.py"
```

CPU tests, golden forwards, family certificates and model efficacy are separate
claims. An incomplete 26-family dataset cannot be presented as the fixed 720-run
experiment. A positive auxiliary reader score is not closed-loop method gain.
