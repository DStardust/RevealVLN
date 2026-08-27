#!/usr/bin/env python3
"""Run hindsight event localization over the frozen scale-v2 route census."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import run_rxr_hindsight_event_factory as base


ROOT = Path("/mnt/daiyang/vla")
QUEUE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/scale_v2/"
    "RXR_SCALE_V2_ROUTE_CENSUS.json"
)
RUNTIME = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
PROMPT = ROOT / (
    "artifacts/phase1/rxr_train_expansion/contract/"
    "RXR_HINDSIGHT_EVENT_LOCATOR_PROMPT_V3.md"
)
SCHEMA = ROOT / (
    "artifacts/phase1/rxr_train_expansion/contract/"
    "RXR_HINDSIGHT_EVENT_LOCATOR_SCHEMA_V3.json"
)
OUT_DIR = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/hindsight_factory"
COUNT = 2_971
EXPECTED = {
    QUEUE: "3a5e1d03620b1e993a1039c95d55ba423338490ca36fc1f681586e38d43fd6b6",
    RUNTIME: "f06b2ef4dc947ca15d6c4a5a3d629c9212328f4cbdd38a13bed9c5c1fc224a94",
    PROMPT: "96401dff92ab6a3c72066601dd434852e14bf9db38445a6ef929a3e01fde1623",
    SCHEMA: "122e0d880be1786bdf0ce5bb9558bc27dfc8af7871ecc9d9b656d16490611ca4",
}
COMMITMENT = "d64663ebb28bc776eed7a26682dec75a7712dfb405bb8158b6ca6c84c37e98d8"


base.QUEUE = QUEUE
base.RUNTIME = RUNTIME
base.PROMPT = PROMPT
base.SCHEMA = SCHEMA
base.OUT_DIR = OUT_DIR
base.MEDIA_DIR = OUT_DIR / "storyboards"
base.RESULT_DIR = OUT_DIR / "results"
base.RUN_DIR = OUT_DIR / "runs"
base.TMP_DIR = OUT_DIR / "tmp"
base.EXPECTED = EXPECTED


def scale_evidence(record: dict, validation_feedback: dict | None = None) -> dict:
    media = [record["global_storyboard"]] + record["chunk_storyboards"]
    return {
        "revision": "rxr-scale-v2-hindsight-event-request/1-nonthinking",
        "queue_sha256": EXPECTED[QUEUE],
        "selection_commitment_sha256": COMMITMENT,
        "prompt_sha256": EXPECTED[PROMPT],
        "schema_sha256": EXPECTED[SCHEMA],
        "model": base.MODEL,
        "enable_thinking": base.ENABLE_THINKING,
        "reasoning_effort": "none",
        "temperature": 0,
        "expansion_order": record["expansion_order"],
        "episode_id": record["episode_id"],
        "trajectory_id": record["trajectory_id"],
        "instruction_sha256": record["instruction_sha256"],
        "trace_pose_action_sha256": record["trace_pose_action_sha256"],
        "timeline_prefix_indices": record["timeline_prefix_indices"],
        "timeline_frame_ids": record["timeline_frame_ids"],
        "deterministic_segments": record["deterministic_segments"],
        "media": [
            {key: row[key] for key in ("path", "bytes", "sha256", "pixels")}
            for row in media
        ],
        "complete_future_trajectory_used_offline": True,
        "validation_retry_feedback": validation_feedback,
    }


base.evidence = scale_evidence


def scale_existing(queue_row: dict):
    directory = base.RESULT_DIR / (
        f"order{queue_row['expansion_order']:04d}_ep{queue_row['episode_id']}"
    )
    for path in sorted(directory.glob("attempt_*.json"), reverse=True):
        try:
            value = json.loads(path.read_text())
            if (
                value.get("status")
                in {
                    "VALID_MLLM_PROPOSAL",
                    "INVALID_MLLM_PROPOSAL",
                    "REQUEST_OR_VALIDATION_FAILURE",
                }
                and value.get("expansion_order") == queue_row["expansion_order"]
                and value.get("episode_id") == queue_row["episode_id"]
                and value.get("request_evidence", {}).get("queue_sha256")
                == EXPECTED[QUEUE]
                and all(
                    base.safe_media(row)
                    for row in value.get("request_evidence", {}).get("media", [])
                )
            ):
                return path, value
            if (
                value.get("status") == "FACTORY_INPUT_FAILURE"
                and value.get("queue_sha256") == EXPECTED[QUEUE]
                and value.get("expansion_order") == queue_row["expansion_order"]
                and value.get("episode_id") == queue_row["episode_id"]
            ):
                return path, value
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return None


base.valid_existing = scale_existing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=24)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index")
    for path, expected in EXPECTED.items():
        if not path.is_file() or path.is_symlink() or base.sha256_file(path) != expected:
            raise RuntimeError("scale-v2 event-factory source drift: " + str(path))

    queue = json.loads(QUEUE.read_text())
    if (
        queue.get("status") != "SCALE_V2_ROUTE_CENSUS_FROZEN"
        or queue.get("selection_commitment_sha256") != COMMITMENT
        or len(queue.get("candidates", [])) != COUNT
    ):
        raise RuntimeError("scale-v2 route census contract failure")
    selected = [
        row
        for row in queue["candidates"]
        if row["scale_v2_order"] % args.shard_count == args.shard_index
    ]
    plan = [
        {
            "scale_v2_order": row["scale_v2_order"],
            "expansion_order": row["expansion_order"],
            "episode_id": row["episode_id"],
            "trajectory_id": row["trajectory_id"],
        }
        for row in selected
    ]
    if not args.execute:
        output = {
            "status": "DRY_RUN_PASS_NO_NETWORK",
            "schema_version": "revealnav-rxr-scale-v2-hindsight-shard-dry-run/1",
            "queue_sha256": EXPECTED[QUEUE],
            "selection_commitment_sha256": COMMITMENT,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "gpu": args.gpu,
            "jobs": plan,
            "network_calls_made": 0,
            "training_authorized": False,
        }
        path = base.RUN_DIR / f"shard_{args.shard_index:02d}_dry_run.json"
        base.atomic_json(path, output)
        print(json.dumps({"status": output["status"], "jobs": len(plan)}, indent=2))
        return 0

    wanted = {row["episode_id"] for row in selected}
    with gzip.open(RUNTIME, "rt", encoding="utf-8") as stream:
        episodes = {
            str(row["episode_id"]): row
            for row in json.load(stream)["episodes"]
            if str(row["episode_id"]) in wanted
        }
    if set(episodes) != wanted:
        raise RuntimeError("scale-v2 runtime episode closure failure")
    key = base.transport.read_secret()
    prompt = PROMPT.read_text()
    rows = []
    for index, queue_row in enumerate(selected, 1):
        path, result, reused = base.execute_one(
            queue_row,
            episodes[queue_row["episode_id"]],
            args.gpu,
            prompt,
            key,
        )
        rows.append(
            {
                "scale_v2_order": queue_row["scale_v2_order"],
                "expansion_order": queue_row["expansion_order"],
                "episode_id": queue_row["episode_id"],
                "status": result["status"],
                "reused": reused,
                "path": str(path.relative_to(ROOT)),
                "sha256": base.sha256_file(path),
            }
        )
        print(
            f"[{index}/{len(selected)}] v2-{queue_row['scale_v2_order']:04d} "
            f"ep{queue_row['episode_id']} {result['status']}"
            + (" (existing)" if reused else ""),
            flush=True,
        )

    counts = Counter(row["status"] for row in rows)
    failures = sum(
        counts.get(name, 0)
        for name in (
            "FACTORY_INPUT_FAILURE",
            "INVALID_MLLM_PROPOSAL",
            "REQUEST_OR_VALIDATION_FAILURE",
        )
    )
    status = "PASS_WITH_FAIL_CLOSED_FAILURES" if failures else "PASS"
    output = {
        "schema_version": "revealnav-rxr-scale-v2-hindsight-shard/1",
        "status": status,
        "queue_sha256": EXPECTED[QUEUE],
        "selection_commitment_sha256": COMMITMENT,
        "prompt_sha256": EXPECTED[PROMPT],
        "schema_sha256": EXPECTED[SCHEMA],
        "model": base.MODEL,
        "enable_thinking": False,
        "reasoning_effort": "none",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "gpu": args.gpu,
        "job_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "results": rows,
        "replacement_samples_created": 0,
        "future_trajectory_used_offline_only": True,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    path = base.RUN_DIR / f"shard_{args.shard_index:02d}.json"
    base.atomic_json(path, output)
    print(json.dumps({"status": status, "jobs": len(rows), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
