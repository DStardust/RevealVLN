#!/usr/bin/env python3
"""Independent closure validator for the 50-trajectory CR5 expansion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from PIL import Image


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
REVIEW = BASE / "human_review_fast"
OUT = BASE / "CR5_QUEUE50_EXPANSION_ACCEPTANCE.json"
EXPECTED = {
    "FROZEN_SPEC.md": "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    "PHASE0_PROTOCOL.md": "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/CR5_QUEUE50_HINDSIGHT_INPUTS.json": "8e00000ee306369e305c53d580444e1ac3228a6e94c3c424d84f9db5d16ea151",
    "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/CR5_QUEUE50_HINDSIGHT_INPUTS_ACCEPTANCE.json": "8729061a065a37a7d54a93035d27781a4e999c11f7fffedf582ab5373e341fb7",
    "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/CR5_QUEUE50_HINDSIGHT_ACCEPTED_RUN.json": "7bc014b084324152083dde4467fb7f849455c830a855cd6826cca35ff5b00692",
    "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/CR5_QUEUE50_HINDSIGHT_AGGREGATED.json": "46f4e592b8b15e16df9641351b5d08a0d1a9fe6b59def4959b34775b8b469612",
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json": "6b70a70e5eb1e25f9522b30209eb56dc2efbf6457377a1aabefdeca6886aee72",
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS_ACCEPTANCE.json": "73133f833c2d8a64c58d834e6fa063ed38349a306302753c01e1f02cfb734af0",
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/CR5_QUEUE50_PRIMARY_MULTIVIEW_RUN.json": "c07a6c233bd05d0f72546d25e82fc9c36f6af2b17dfe310eed73a68f81cf28fb",
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/CR5_QUEUE50_PRIMARY_MULTIVIEW_RETRY_RUN.json": "55ac2f8b2618f9f7e6656da1c6cdd1f89bc6c1f8bc18c52860228d7189a4a714",
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/CR5_QUEUE50_PRIMARY_MULTIVIEW_ACCEPTED_RUN.json": "0f5b643612ad1a52b12aaa12d3d26b06b5dc7b288cfbc4f435f98fd3c5b81ead",
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/CR5_QUEUE50_PRIMARY_MACHINE_PRESCREEN.json": "99b8b6f070b73bc65bb6a24268b1c2c436edd08e1789b487807c98cfee648a0f",
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/CR5_QUEUE50_DIRECTED_GEOMETRY.json": "46609126537ddb9c4936bc93d683dd06243b91203d6bf2f7c03f30cca7deb850",
    "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/CR5_QUEUE50_CONTROLLER_EXECUTION.json": "567039afac8f53141b9f1d2114ee79a47611ca7e68b1eefc9d2ea40d72eff574",
    "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/CR5_QUEUE50_FAST_REVIEW_MANIFEST.json": "6d75c96075d5746dc90a19c7d4a59941b17ab81e1d7a4cfa6480c607fb089017",
    "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/CR5_QUEUE50_FAST_REVIEW_TEMPLATE.jsonl": "6a84af6ee4e23e30c914d4ebb3b341d93700676562cabb3712d13eca11f7d2a8",
    "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/CR5_QUEUE50_AUTO_REJECTED.json": "14f549c8d0c73628335fa673b433593f7152fb6b6dd8a0abd074134b7c218403",
    "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/审核说明.md": "406a36f6305aaea8c839ff1604ffb9b799e4966dd966ac75537f6c8469f18a06",
    "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/CR5_QUEUE50_REVIEWER.html": "27d96636a0fa8903f8924215997796c76da0ea121fe8c7b050867dba15513df5",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(relative: str):
    return json.loads((ROOT / relative).read_text())


def check(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(message)
    checks.append(message)


def main() -> int:
    checks = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        check(path.is_file() and not path.is_symlink(),
              "regular source: " + relative, checks)
        check(sha256_file(path) == expected,
              "SHA source: " + relative, checks)

    hindsight = load(
        "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/"
        "CR5_QUEUE50_HINDSIGHT_INPUTS.json")
    accepted = load(
        "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/"
        "CR5_QUEUE50_PRIMARY_MULTIVIEW_ACCEPTED_RUN.json")
    geometry = load(
        "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/"
        "CR5_QUEUE50_DIRECTED_GEOMETRY.json")
    controller = load(
        "artifacts/phase0/phase0c_cr5_queue50/multiview_primary/"
        "CR5_QUEUE50_CONTROLLER_EXECUTION.json")
    manifest = load(
        "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/"
        "CR5_QUEUE50_FAST_REVIEW_MANIFEST.json")
    rejected = load(
        "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/"
        "CR5_QUEUE50_AUTO_REJECTED.json")
    check(hindsight["episode_count"] == 50,
          "50 hindsight episodes", checks)
    check(len({row["episode_id"] for row in hindsight["episodes"]}) == 50,
          "50 unique hindsight episodes", checks)
    check(accepted["status"] == "PASS" and
          accepted["accepted_count"] == 50,
          "50 accepted offline branch proposals", checks)
    geometry_counts = geometry["status_counts"]
    check(geometry_counts == {
        "GEOMETRY_PASS_CONTROLLER_REQUIRED": 38,
        "GEOMETRY_REJECT": 12,
    }, "geometry 38 pass / 12 reject", checks)
    controller_counts = controller["status_counts"]
    check(controller_counts == {
        "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED": 34,
        "CONTROLLER_REJECT": 4,
    }, "controller 34 pass / 4 reject", checks)
    check(manifest["screened_trajectory_count"] == 50 and
          manifest["full_review_board_count"] == 34 and
          manifest["automatic_reject_count"] == 16,
          "review closure 50 = 34 + 16", checks)
    review_ids = {row["event_id"] for row in manifest["items"]}
    reject_ids = {row["event_id"] for row in rejected["events"]}
    all_ids = {row["event_id"] for row in accepted["events"]}
    check(not review_ids & reject_ids and review_ids | reject_ids == all_ids,
          "review/reject disjoint exact event closure", checks)

    for item in manifest["items"]:
        path = ROOT / item["board_path"]
        check(path.is_file() and not path.is_symlink(),
              "regular review board: " + item["event_id"], checks)
        check(sha256_file(path) == item["board_sha256"],
              "review board SHA: " + item["event_id"], checks)
        with Image.open(path) as image:
            check(image.size == (3600, 1800),
                  "review board dimensions: " + item["event_id"], checks)
        check(bool(item["instruction_text"]) and
              item["human_status"] == "PENDING" and
              item["training_label"] is False,
              "unlabeled review item: " + item["event_id"], checks)

    lines = (REVIEW / "CR5_QUEUE50_FAST_REVIEW_TEMPLATE.jsonl").read_text(
        encoding="utf-8").splitlines()
    template = [json.loads(line) for line in lines]
    check(len(template) == 34 and
          {row["event_id"] for row in template} == review_ids,
          "34-line human template exact closure", checks)
    for row in template:
        check(row["reviewer_type"] == "HUMAN" and
              row["reviewer_id"] is None and
              row["final_label"] is None and
              all(row[key] is None for key in (
                  "two_distinct_executable_exits",
                  "alternative_is_not_incoming_closed_or_duplicate",
                  "instruction_uniquely_selects_target",
                  "decision_center_and_temporal_order_are_reasonable")),
              "empty human template: " + row["event_id"], checks)

    reserve = ROOT / ".disk_reserve"
    reserve_files = sorted(reserve.glob("reserve_10G_*.bin"))
    check(len(reserve_files) == 19, "19 reserve files remain", checks)
    for path in reserve_files:
        stat = path.lstat()
        check(path.is_file() and not path.is_symlink()
              and stat.st_size == 10_737_418_240,
              "reserve intact: " + path.name, checks)
    secret = ROOT / ".secret/qwen_api_key"
    if secret.is_file():
        key = secret.read_text().strip()
        for relative in EXPECTED:
            path = ROOT / relative
            if path.suffix in {".json", ".jsonl", ".md", ".html"}:
                check(not key or key not in path.read_text(errors="ignore"),
                      "secret absent: " + relative, checks)
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    check(free_bytes >= 8 * 1024 ** 3, "free disk >= 8 GiB", checks)

    output = {
        "manifest": "MF2-CR5 queue50 expansion acceptance",
        "revision": "cr5-queue50-expansion-acceptance/1",
        "status": "ENGINEERING_PASS_HUMAN_AND_CAUSAL_GATES_PENDING",
        "screened_trajectories": 50,
        "offline_branch_proposals_accepted": 50,
        "geometry_pass": 38,
        "controller_pass": 34,
        "full_human_review_boards": 34,
        "automatic_rejects_not_human_labels": 16,
        "checks_passed": len(checks),
        "free_bytes": free_bytes,
        "frozen_50_item_human_protocol_satisfied": False,
        "causal_prefix_gate_completed": False,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "checks_passed": output["checks_passed"],
        "free_bytes": free_bytes,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
