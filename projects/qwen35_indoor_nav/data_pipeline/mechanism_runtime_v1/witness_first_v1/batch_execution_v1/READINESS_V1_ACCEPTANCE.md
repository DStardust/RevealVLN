# Readiness V1 CPU acceptance

Status: CPU_READY_FOR_MAIN_AGENT_REVIEW. Fourteen simulated CPU tests passed;
no GPU query, simulator, or worker was launched by this implementation task.

Entry point: import `readiness_v1.py` and call `run_main(batch_path)`, or execute
the module with the fresh batch directory as its sole positional argument.
The batch must be a direct child of this directory named `batch_N` or
`batch_NrN`, with a pre-created `run_v1/EXECUTION_CONFIG.json`. It must have no
prior supervisor lock, process record, or launch receipt. GPU 1 and GPU 2 use
the unchanged shared UUID map and config validation.

Only the first GPU query is adapted: at most 60 seconds of genuine samples,
each checked by the original resource guard, until an actual utilization-zero
sample is observed. Samples and errors are durably appended to
`READINESS_SAMPLES.jsonl`. No value is synthesized or recycled. Later queries
call the original function. The exact original `shared.supervisor_source`
adapter, original assertions and cleanup remain in effect. The launcher does
not stop any process. The original 3,900-second supervision and bounded cleanup
are unchanged; the additional admission wait has its own 60-second bound.

Every admitted new launch receives `LAUNCH_RESERVATION.json` and an independent
`LAUNCH_RESULT.json`, including preworker exceptions. Invalid/out-of-scope or
previously attempted directories are rejected without writing into them.
Filesystem failures cannot guarantee durable error reporting. An original
supervisor return is not a scientific, data-quality, or worker-success claim.

Tests cover busy-to-idle, immediate idle, subsequent original queries, timeout,
non-resettable failure, external-resource rejection, query errors, clipped
subprocess timeout, late-idle rejection, unchanged snapshots, unchanged source
guards and hashes, preworker receipts, and preserved previous attempts.

Frozen implementation SHA256:
`6ad5b369845917a57963ce2fb4718bf2580b8d87a408c8f467cb70ddc61b7600`.

Frozen tests SHA256:
`5e7b46a0362dc6edec561a6247fb3646bc939bb3ea6007399e4a18c52f86a7d9`.

Read-only dependency locks:

- `shared.py`: `3b717bdf07a53ce06191a2445a20e86982073c17b43412b30b1332c8e01a53b0`
- `../../compact_loop_v2/run.py`: `9aabce69787bc717f6283b7679fef3c2f811018f0456741dc51516ebe6e986ca`

The previous batch_01 remains PREWORKER_EXIT_UNOBSERVED. No original GPU error
is inferred. A fresh retry needs its own main-agent-frozen config and approval.
