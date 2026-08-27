#!/usr/bin/env python3
"""Stage 6: assemble the 50-item human review packet.

Builds:
  artifacts/phase0/REVIEW_PACKET_50.json
  artifacts/phase0/REVIEW_PACKET_50.csv
  artifacts/phase0/review_packet_50/PRIVATE_DO_NOT_DISTRIBUTE.txt
(REVIEW_GUIDE.md is written separately and its SHA recorded here.)

Strict non-fabrication rules:
  - every human judgment field is left empty/null;
  - annotation_status=PENDING, reviewed=false, candidate_valid=null for all
    rows;
  - machine outputs (chains, proposals, witnesses) are presented strictly as
    review aids and are never marked validated;
  - instruction texts come from the frozen queue artifact and are included
    only because a human cannot judge decisive constraints without reading
    the instruction; the whole packet is marked PRIVATE_DO_NOT_DISTRIBUTE
    because MP3D written authorization is not recorded.
"""

import csv
import hashlib
import json
import os
import sys

PROJECT_ROOT = "/mnt/daiyang/vla"
QUEUE_PATH = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                          "rxr_train_screening_seed20260822.json")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                            "REVEAL_QUEUE_50_MAPPING.json")
UNITS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase0", "review_units")
PACKET_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                          "review_packet_50")
MEDIA_DIR = os.path.join(PACKET_DIR, "private_media")
GUIDE_PATH = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                          "REVIEW_GUIDE.md")
OUT_JSON = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                        "REVIEW_PACKET_50.json")
OUT_CSV = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                       "REVIEW_PACKET_50.csv")
WITNESS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                            "phase0_reveal_closure", "witness",
                            "WITNESS_RETURN_EXPIRY_FIRST5.json")

HUMAN_FIELDS = [
    "reviewer_id",
    "review_timestamp",
    "candidate_valid",
    "branch_event_visible",
    "target_branch_confirmed",
    "target_branch_id",
    "decisive_constraint_present",
    "decisive_constraint_types_direction",
    "decisive_constraint_types_ordinal",
    "decisive_constraint_types_exclusion",
    "decisive_constraint_types_temporal",
    "decisive_constraint_types_landmark",
    "decisive_constraint_types_other",
    "reveal_interval_start",
    "reveal_interval_end",
    "unique_expiry_confirmed",
    "expiry_prefix",
    "return_witness_confirmed",
    "candidate_identity_stable",
    "rejection_reason",
    "reviewer_notes",
]


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def normalized_media_file(name):
    """Return a project-relative path while confining media to MEDIA_DIR."""
    candidate = (name if os.path.isabs(name)
                 else os.path.join(PACKET_DIR, name))
    real_candidate = os.path.realpath(candidate)
    real_media_dir = os.path.realpath(MEDIA_DIR)
    if os.path.commonpath([real_candidate, real_media_dir]) != real_media_dir:
        raise ValueError("media path escapes private media dir: %r" % name)
    if not os.path.isfile(real_candidate) or os.path.islink(real_candidate):
        raise ValueError("media is missing/non-regular/symlink: %r" % name)
    return os.path.relpath(real_candidate, PROJECT_ROOT)


def main():
    with open(QUEUE_PATH) as fh:
        queue = json.load(fh)
    queue_by_order = {i: s for i, s in enumerate(queue["samples"])}
    with open(MAPPING_PATH) as fh:
        mapping = json.load(fh)
    witness = {}
    if os.path.isfile(WITNESS_PATH):
        with open(WITNESS_PATH) as fh:
            witness = json.load(fh)
    witness_by_episode = {
        str(e["episode_id"]): e for e in witness.get("episodes", [])
    }

    rows = []
    units_meta = []
    media_manifest = []
    for order, item in enumerate(mapping["items"]):
        eid = str(item["episode_id"])
        unit_path = os.path.join(UNITS_DIR,
                                 "unit_%02d_ep%s.json" % (order, eid))
        unit = None
        if os.path.isfile(unit_path):
            with open(unit_path) as fh:
                unit = json.load(fh)
        q = queue_by_order.get(order, {})
        witness_item = witness_by_episode.get(eid)
        row = {
            "row_order": order,
            "unit_id": "unit_%02d_ep%s" % (order, eid),
            "episode_id": eid,
            "instruction_id": str(item.get("instruction_id")),
            "trajectory_id": str(item.get("trajectory_id")),
            "scene_id": item.get("scene_id"),
            "language": item.get("language"),
            "source_split": item.get("split"),
            "screening_triggers": ",".join(q.get("triggers", [])),
            "instruction_sha256": item.get("instruction_sha256_queue"),
            "instruction_text_for_review": q.get("instruction", ""),
            "unit_file": os.path.relpath(unit_path, PROJECT_ROOT)
            if unit else None,
            "chain_file": unit["run"]["chain_file"] if unit else None,
            "chain_root": unit["run"]["chain_root"] if unit else None,
            "prefix_count": unit["run"]["prefix_count"] if unit else None,
            "collect_status": unit["run"]["collect_status"] if unit else None,
            "media_files": [normalized_media_file(m["file"]) for m in
                            (unit["media"] if unit else [])],
            # machine proposals shown as review aids only
            "target_proposal_status": ",".join(
                p["proposal_status"] for p in
                unit["target_branch_proposals"]) if unit else None,
            "witness_file": (witness_item.get("file")
                             if witness_item else None),
            "expiry_proposal_status": (
                witness_item.get("expiry_proposal_status")
                if witness_item else None),
            "expiry_prefix_proposal": (
                witness_item.get("expiry_prefix")
                if witness_item else None),
            "annotation_status": "PENDING",
            "reviewed": False,
        }
        for f in HUMAN_FIELDS:
            row[f] = None
        row["candidate_valid"] = None
        rows.append(row)
        if unit:
            units_meta.append({
                "unit_id": unit["unit_id"],
                "episode_id": unit["episode_id"],
                "prefix_count": unit["run"]["prefix_count"],
                "collect_status": unit["run"]["collect_status"],
                "chain_root": unit["run"]["chain_root"],
                "media": unit["media"],
            })
            for m in unit["media"]:
                media_manifest.append({
                    "unit_id": unit["unit_id"],
                    "file": normalized_media_file(m["file"]),
                    "sha256": m["sha256"],
                    "bytes": m["bytes"],
                    "source_prefix_hash": m["source_prefix_hash"],
                    "content": m["content"],
                })

    reviewed_true = sum(1 for r in rows if r["reviewed"] is True)
    total_media_bytes = sum(m["bytes"] for m in media_manifest)
    human_prefilled = any(
        r.get(field) is not None for r in rows for field in HUMAN_FIELDS
    )
    all_rows_pending = all(
        r["annotation_status"] == "PENDING"
        and r["reviewed"] is False
        and all(r.get(field) is None for field in HUMAN_FIELDS)
        for r in rows
    )
    media_per_unit_ok = all(len(u["media"]) <= 12 for u in units_meta)
    packet_ok = (
        len(rows) == 50
        and reviewed_true == 0
        and all_rows_pending
        and not human_prefilled
        and total_media_bytes <= 250 * 1024 * 1024
        and media_per_unit_ok
        and witness.get("validated_tx_count", 0) == 0
    )
    packet = {
        "packet": "REVIEW_PACKET_50",
        "status": "PASS" if packet_ok else "FAIL",
        "generated_for": "frozen 50-item RxR train screening queue "
                         "(seed 20260822; no resampling)",
        "reviewed": False,
        "row_count": len(rows),
        "reviewed_true_count": reviewed_true,
        "all_rows_pending": all_rows_pending,
        "human_fields_prefilled": human_prefilled,
        "human_fields": HUMAN_FIELDS,
        "rows": rows,
        "units": units_meta,
        "media": {
            "dir": "artifacts/phase0/review_packet_50/private_media/",
            "image_count": len(media_manifest),
            "total_bytes": total_media_bytes,
            "limit_bytes": 250 * 1024 * 1024,
            "format": "224x224 JPEG, quality 85, front RGB observations "
                      "only; no depth; no video; no raw-resolution RGB",
            "manifest": media_manifest,
            "per_unit_at_most_12": media_per_unit_ok,
        },
        "witness": {
            "path": (os.path.relpath(WITNESS_PATH, PROJECT_ROOT)
                     if witness else None),
            "sha256": sha256_file(WITNESS_PATH) if witness else None,
            "engineering_only": True,
            "validated_tx_count": witness.get("validated_tx_count", 0),
        },
        "privacy": {
            "mp3d_written_authorization_recorded": False,
            "distribution": "PRIVATE_DO_NOT_DISTRIBUTE: no upload, "
                            "submission or sharing of this packet or its "
                            "media is permitted until dataset authorization "
                            "provenance is recorded",
        },
        "guide": {
            "path": "artifacts/phase0/REVIEW_GUIDE.md",
            "sha256": sha256_file(GUIDE_PATH) if os.path.isfile(GUIDE_PATH)
            else None,
        },
        "inputs": {
            "queue": {"path": "artifacts/phase0/"
                              "rxr_train_screening_seed20260822.json",
                      "sha256": sha256_file(QUEUE_PATH)},
            "mapping": {"path": "artifacts/phase0/REVEAL_QUEUE_50_MAPPING.json",
                        "sha256": sha256_file(MAPPING_PATH)},
        },
        "non_conclusions": {
            "validated_events": 0,
            "valid_candidates": 0,
            "unique_expiry_events": 0,
            "provisional_expiry_counted_as_validated": False,
        },
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(packet, fh, indent=2)

    fieldnames = (["row_order", "unit_id", "episode_id", "instruction_id",
                   "trajectory_id", "scene_id", "language", "source_split",
                   "screening_triggers", "instruction_sha256",
                   "instruction_text_for_review", "unit_file", "chain_file",
                   "chain_root", "prefix_count", "collect_status",
                   "media_files", "target_proposal_status",
                   "witness_file", "expiry_proposal_status",
                   "expiry_prefix_proposal",
                   "annotation_status", "reviewed"] + HUMAN_FIELDS)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["media_files"] = ";".join(r["media_files"])
            w.writerow(r)

    txt = (
        "PRIVATE - DO NOT DISTRIBUTE\n"
        "\n"
        "This review packet contains Matterport3D-derived scene media and "
        "RxR instruction texts.\n"
        "No written MP3D authorization for this project is recorded. Until "
        "authorization provenance is recorded, this packet and its media "
        "must not be uploaded, submitted, shared or redistributed. Internal "
        "human review use only, inside /mnt/daiyang/vla.\n"
        "\n"
        "All annotation fields are intentionally blank. reviewed=false and "
        "annotation_status=PENDING for every row until a human reviewer "
        "fills them. Machine outputs (prefix chains, target-branch "
        "proposals, return witnesses) are review aids only and are not "
        "validated events.\n"
    )
    with open(os.path.join(PACKET_DIR,
                           "PRIVATE_DO_NOT_DISTRIBUTE.txt"), "w") as fh:
        fh.write(txt)

    print(json.dumps({
        "packet": os.path.relpath(OUT_JSON, PROJECT_ROOT),
        "csv": os.path.relpath(OUT_CSV, PROJECT_ROOT),
        "row_count": len(rows),
        "reviewed_true_count": reviewed_true,
        "all_rows_pending": packet["all_rows_pending"],
        "media_count": len(media_manifest),
        "media_total_bytes": total_media_bytes,
    }, indent=2))
    return 0 if packet["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
