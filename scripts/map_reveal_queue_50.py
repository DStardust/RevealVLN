#!/usr/bin/env python3
"""Stage 3: map the frozen 50-item RxR train screening queue to the runtime
payload used by the ETP-R1 checkpoint.

Reads:
  artifacts/phase0/rxr_train_screening_seed20260822.json   (frozen queue)
  third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz
                                                           (runtime payload)

The queue was screened from the canonical public RxR_VLNCE_v0 payload
(data/phase0/raw/...); the runtime payload is the XLM-R-encoded variant.
This mapper verifies per-item identity across both representations:
episode_id, instruction_id, trajectory_id, scene basename, language and the
SHA-256 of the instruction text.  Only en-US/en-IN items are accepted.

Writes artifacts/phase0/REVEAL_QUEUE_50_MAPPING.json.

Exit codes: 0 = 50/50 unique mapping, 1 = mismatch (full diagnostic report is
still written), 2 = structural error.
"""

import gzip
import hashlib
import json
import os
import sys

PROJECT_ROOT = "/mnt/daiyang/vla"
QUEUE_PATH = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                          "rxr_train_screening_seed20260822.json")
RUNTIME_PAYLOAD = os.path.join(
    PROJECT_ROOT,
    "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/train/"
    "train_guide.json.gz")
CANONICAL_PAYLOAD = os.path.join(
    PROJECT_ROOT,
    "data/phase0/raw/rxr_vlnce_v0/train/train_guide.json.gz")
OUT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                        "REVEAL_QUEUE_50_MAPPING.json")
MAPPING_RULE_VERSION = "reveal-queue-mapping/v1"
ALLOWED_LANGUAGES = {"en-US", "en-IN"}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def scene_base(scene_id):
    return os.path.splitext(os.path.basename(scene_id))[0]


def main():
    with open(QUEUE_PATH) as fh:
        queue = json.load(fh)
    samples = queue["samples"]
    if len(samples) != 50:
        print(json.dumps({"status": "ERROR",
                          "reason": "queue does not contain 50 samples",
                          "count": len(samples)}))
        return 2

    with gzip.open(RUNTIME_PAYLOAD, "rt") as fh:
        runtime_eps = {e["episode_id"]: e
                       for e in json.load(fh)["episodes"]}
    with gzip.open(CANONICAL_PAYLOAD, "rt") as fh:
        canonical_eps = {e["episode_id"]: e
                         for e in json.load(fh)["episodes"]}

    items = []
    mismatch_count = 0
    duplicate_count = 0
    seen_episodes = set()
    language_violations = 0
    for order, item in enumerate(samples):
        eid = str(item["episode_id"])
        mismatches = []
        rt = runtime_eps.get(eid)
        cn = canonical_eps.get(eid)
        if rt is None:
            mismatches.append("episode_id absent from runtime payload")
        if cn is None:
            mismatches.append("episode_id absent from canonical payload")

        inst_hash_queue = hashlib.sha256(
            item["instruction"].encode("utf-8")).hexdigest()
        rec = {
            "queue_order": order,
            "episode_id": eid,
            "instruction_id": str(item.get("instruction_id")),
            "trajectory_id": str(item.get("trajectory_id")),
            "scene_id": item.get("scene_id"),
            "language": item.get("language"),
            "split": item.get("split"),
            "queue_seed": queue.get("sampling", {}).get("seed"),
            "queue_design": queue.get("sampling", {}).get("design"),
            "instruction_sha256_queue": inst_hash_queue,
        }
        if item.get("language") not in ALLOWED_LANGUAGES:
            language_violations += 1
            mismatches.append("language not in {en-US, en-IN}")
        if item.get("split") != "train":
            mismatches.append("source split is not train")
        if eid in seen_episodes:
            duplicate_count += 1
            mismatches.append("duplicate episode_id within queue")
        seen_episodes.add(eid)

        if rt is not None:
            rt_inst = rt["instruction"]
            checks = {
                "instruction_id": (str(rt_inst.get("instruction_id"))
                                   == str(item.get("instruction_id"))),
                "trajectory_id": (str(rt.get("trajectory_id"))
                                  == str(item.get("trajectory_id"))),
                "scene_id": (scene_base(rt["scene_id"])
                             == item.get("scene_id")),
                "language": (rt_inst.get("language")
                             == item.get("language")),
                "instruction_sha256": (
                    hashlib.sha256(
                        rt_inst["instruction_text"].encode("utf-8")
                    ).hexdigest() == inst_hash_queue),
            }
            rec["runtime_identity"] = {
                "instruction_id": str(rt_inst.get("instruction_id")),
                "trajectory_id": str(rt.get("trajectory_id")),
                "scene_id": scene_base(rt["scene_id"]),
                "language": rt_inst.get("language"),
                "instruction_sha256": hashlib.sha256(
                    rt_inst["instruction_text"].encode("utf-8")).hexdigest(),
                "start_position": rt.get("start_position"),
                "reference_path_points": len(rt.get("reference_path") or []),
            }
            rec["runtime_field_checks"] = checks
            for k, ok in checks.items():
                if not ok:
                    mismatches.append("runtime field mismatch: " + k)
        if cn is not None:
            cn_inst = cn["instruction"]
            cn_ok = (
                str(cn_inst.get("instruction_id"))
                == str(item.get("instruction_id"))
                and str(cn.get("trajectory_id"))
                == str(item.get("trajectory_id"))
                and scene_base(cn["scene_id"]) == item.get("scene_id")
                and cn_inst.get("language") == item.get("language")
                and hashlib.sha256(
                    cn_inst["instruction_text"].encode("utf-8")
                ).hexdigest() == inst_hash_queue)
            rec["canonical_identity_consistent"] = cn_ok
            if not cn_ok:
                mismatches.append("canonical payload identity mismatch")
        else:
            rec["canonical_identity_consistent"] = None

        rec["mapped"] = rt is not None and not mismatches
        rec["mismatches"] = mismatches
        if mismatches:
            mismatch_count += 1
        items.append(rec)

    mapped_count = sum(1 for r in items if r["mapped"])
    report = {
        "mapping_rule_version": MAPPING_RULE_VERSION,
        "mapping_rules": [
            "queue order preserved exactly (no resampling, no reordering)",
            "episode_id matched as string against runtime payload episode ids",
            "identity fields compared: instruction_id, trajectory_id, "
            "scene basename, language, instruction SHA-256",
            "runtime payload is the XLM-R-encoded variant consumed by the "
            "checkpoint; canonical payload is the screened public variant; "
            "both must agree with the queue item",
            "only en-US/en-IN accepted; only split=train accepted",
            "any mismatch is recorded per item and counted; mismatches do "
            "not trigger resampling",
        ],
        "queue": {
            "path": os.path.relpath(QUEUE_PATH, PROJECT_ROOT),
            "sha256": sha256_file(QUEUE_PATH),
            "sample_count": len(samples),
            "sampling": queue.get("sampling"),
        },
        "runtime_payload": {
            "path": os.path.relpath(RUNTIME_PAYLOAD, PROJECT_ROOT),
            "sha256": sha256_file(RUNTIME_PAYLOAD),
            "bytes": os.path.getsize(RUNTIME_PAYLOAD),
            "episode_count": len(runtime_eps),
        },
        "canonical_payload": {
            "path": os.path.relpath(CANONICAL_PAYLOAD, PROJECT_ROOT),
            "sha256": sha256_file(CANONICAL_PAYLOAD),
            "bytes": os.path.getsize(CANONICAL_PAYLOAD),
            "episode_count": len(canonical_eps),
        },
        "mapped_count": mapped_count,
        "mismatch_count": mismatch_count,
        "duplicate_count": duplicate_count,
        "language_violation_count": language_violations,
        "unique_mapping_50_of_50": mapped_count == 50
        and mismatch_count == 0 and duplicate_count == 0,
        "items": items,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({
        "status": "PASS" if report["unique_mapping_50_of_50"] else "FAIL",
        "mapped_count": mapped_count,
        "mismatch_count": mismatch_count,
        "duplicate_count": duplicate_count,
        "out": os.path.relpath(OUT_PATH, PROJECT_ROOT),
    }, indent=2))
    return 0 if report["unique_mapping_50_of_50"] else 1


if __name__ == "__main__":
    sys.exit(main())
