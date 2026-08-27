#!/usr/bin/env python3
"""Fail-closed acceptance for the queue50 target-route geometry revision."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
OLD = BASE / "multiview_primary/CR5_QUEUE50_DIRECTED_GEOMETRY.json"
NEW = BASE / "regrounding_v2/CR5_QUEUE50_TARGET_ROUTE_GEOMETRY_V2.json"
SUBSET = BASE / "regrounding_v2/CR5_QUEUE50_REGROUNDED_CANDIDATES.json"
CONTROLLER = BASE / "regrounding_v2/CR5_QUEUE50_REGROUNDED_CONTROLLER.json"
Q36_GEOMETRY = (
    BASE / "human_review_fast/CR5_QUEUE50_Q36_CORRECTED_GEOMETRY.json")
Q36_CONTROLLER = (
    BASE / "human_review_fast/CR5_QUEUE50_Q36_CORRECTED_CONTROLLER.json")
AUTH = (ROOT / "artifacts/upstream/matterport3d/"
        "MP3D_ACCESS_AUTHORIZATION_ATTESTATION.json")
OUT = BASE / "regrounding_v2/CR5_TARGET_ROUTE_GEOMETRY_V2_ACCEPTANCE.json"
EXPECTED = {
    OLD: "46609126537ddb9c4936bc93d683dd06243b91203d6bf2f7c03f30cca7deb850",
    NEW: "7a0044ab458f130d74b37331904ddad379f2270cdb4dbc0ef6213cab09df9fa0",
    SUBSET: "309670e690d15c4af1f2924cdf3a93c3f9225804f256ba10f065852fcea78b50",
    CONTROLLER: "03413c917d019280e07c3c8caef16a63ce0a880f180b5e1128b50a627dc3c1f9",
    Q36_GEOMETRY: "80ee7482df1cfd821fa1984c1e4cbf8d88d777ce8d1b750ca118712345b9fea3",
    Q36_CONTROLLER: "e835f838e3111c497643d598db2f434d36bb76a01790b92a3cdc8ff0c2f878df",
    AUTH: "d840d2edde2049c1dccdf3c4bc696deed4bd79354cf49931087086a03900fcad",
    ROOT / "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    ROOT / "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
}
RECOVERED = {
    "q02_ep37248_hv02",
    "q29_ep41108_hv01",
    "q36_ep1049_hv05",
    "q44_ep38032_hv04",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    checks = []

    def check(condition: bool, name: str) -> None:
        require(condition, "acceptance failure: " + name)
        checks.append(name)

    for path, expected in EXPECTED.items():
        check(path.is_file() and not path.is_symlink(),
              "safe source: " + str(path.relative_to(ROOT)))
        check(sha256_file(path) == expected,
              "pinned SHA: " + str(path.relative_to(ROOT)))

    old_doc = load(OLD)
    new_doc = load(NEW)
    subset_doc = load(SUBSET)
    controller_doc = load(CONTROLLER)
    old = {row["event_id"]: row for row in old_doc["events"]}
    new = {row["event_id"]: row for row in new_doc["events"]}
    controllers = {row["event_id"]: row
                   for row in controller_doc["events"]}
    check(set(old) == set(new) and len(new) == 50,
          "50-event exact closure")
    check(old_doc["thresholds"] == new_doc["thresholds"],
          "all numeric geometry thresholds unchanged")
    check(new_doc["target_direction_policy"]["name"] ==
          "official_reference_future_diagnostic",
          "official future is target authority")
    check(new_doc["training_authorized"] is False,
          "full v2 geometry does not authorize training")

    old_pass = {event_id for event_id, row in old.items()
                if row["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"}
    check(len(old_pass) == 38, "legacy pass count is 38")
    check(all(new[event_id]["status"] ==
              "GEOMETRY_PASS_CONTROLLER_REQUIRED" for event_id in old_pass),
          "no legacy pass regressed")
    for event_id in old_pass:
        for field in ("navmesh", "trace", "alternative",
                      "alternative_search"):
            check(old[event_id][field] == new[event_id][field],
                  "legacy geometry unchanged: %s/%s" % (event_id, field))
        old_target = old[event_id]["target"]
        check(all(new[event_id]["target"][key] == value
                  for key, value in old_target.items()),
              "legacy target geometry unchanged: " + event_id)

    recovered = {event_id for event_id in new
                 if old[event_id]["status"] == "GEOMETRY_REJECT"
                 and new[event_id]["status"] ==
                 "GEOMETRY_PASS_CONTROLLER_REQUIRED"}
    check(recovered == RECOVERED, "exact recovered event set")
    check({row["event_id"] for row in subset_doc["events"]} == RECOVERED,
          "controller subset exact closure")
    check(subset_doc["training_authorized"] is False,
          "candidate subset does not authorize training")

    for event_id in ("q12_ep46758_hv01", "q24_ep28644_hv04"):
        check(new[event_id]["status"] == "GEOMETRY_REJECT"
              and new[event_id]["failures"] ==
              ["NO_DISTINCT_EXECUTABLE_ALTERNATIVE"],
              "direction-only suppression removed but no branch invented: "
              + event_id)
    check("TARGET_REFERENCE_ROUTE_SHORTER_THAN_1_75M" in
          new["q07_ep57660_hv03"]["failures"],
          "short reference route remains hard reject")
    for event_id in ("q33_ep30793_hv02", "q43_ep23895_hv05"):
        check("TARGET_VERTICAL_MOTION_MISMATCH" in
              new[event_id]["failures"],
              "vertical mismatch remains hard reject: " + event_id)
    for event_id in ("q17_ep34158_hv05", "q24_ep28644_hv04"):
        check(new[event_id]["alternative"] is None,
              "human-confirmed no-alternative remains rejected: " + event_id)

    q36_new = new["q36_ep1049_hv05"]
    q36_old_fix = load(Q36_GEOMETRY)["events"][0]
    for field in ("Q", "B_star_at_1m", "T_star_at_1_75m"):
        check(q36_new["target"][field] == q36_old_fix["target"][field],
              "q36 target reproduces one-off correction: " + field)
    for field in ("branch_id", "B_i_at_1m", "T_i_at_1_75m",
                  "navmesh_shortest_path_length_m", "distinctness",
                  "path_samples", "search_score"):
        check(q36_new["alternative"][field] ==
              q36_old_fix["alternative"][field],
              "q36 alternative reproduces one-off correction: " + field)
    check(controllers["q36_ep1049_hv05"] ==
          load(Q36_CONTROLLER)["events"][0],
          "q36 controller reproduces prior accepted replay")

    check(set(controllers) == RECOVERED,
          "controller result exact recovered closure")
    passed_controller = {event_id for event_id, row in controllers.items()
                         if row["status"] ==
                         "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"}
    check(passed_controller == {
        "q02_ep37248_hv02", "q29_ep41108_hv01", "q36_ep1049_hv05"},
        "exact controller-pass set")
    q44 = controllers["q44_ep38032_hv04"]
    check(q44["status"] == "CONTROLLER_REJECT"
          and all(row["collision_count"] == 2
                  for row in q44["alternative"]["replays"]),
          "q44 collision failure stays rejected")
    check(controller_doc["training_authorized"] is False,
          "controller output does not authorize training")
    check(all(row["training_label"] is False
              for row in subset_doc["events"] + controller_doc["events"]),
          "no recovered event promoted to training label")
    serialized = json.dumps(new_doc, ensure_ascii=False)
    check(all(token not in serialized for token in
              ("val_unseen", "test_challenge", '"dataset_split": "test"')),
          "no forbidden split reference")

    reserves = sorted((ROOT / ".disk_reserve").glob("reserve_10G_*.bin"))
    check(len(reserves) == 19 and all(
        path.is_file() and not path.is_symlink()
        and path.stat().st_size == 10_737_418_240 for path in reserves),
        "19 reserve files untouched")
    auth = load(AUTH)
    check(auth["status"] == "USER_CONFIRMED_AUTHORIZED"
          and auth["handling_boundary"]["redownload_required"] is False,
          "MP3D authorization attestation accepted")

    output = {
        "manifest": "MF2-CR5 target-route geometry v2 acceptance",
        "revision": "cr5-target-route-geometry-v2-acceptance/1",
        "status": "PASS",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "checks_passed": len(checks),
        "legacy_passes_preserved": len(old_pass),
        "geometry_recovered": sorted(recovered),
        "controller_passed": sorted(passed_controller),
        "controller_rejected": ["q44_ep38032_hv04"],
        "scientific_interpretation": {
            "fixed_failure_mode": (
                "MLLM target-direction mismatch no longer suppresses "
                "alternative search because target geometry is supplied by "
                "the official reference future."),
            "unchanged_hard_gates": [
                "reference future length", "vertical compatibility",
                "incoming retrace", "3-D branch distinctness",
                "early remerge", "discrete controller execution"],
            "new_training_labels": 0,
        },
        "next_gate": (
            "unbiased RxR-train expansion; recovered controller passes still "
            "require causal and label gates"),
        "network_calls_made": 0,
        "forbidden_split_payloads_opened": 0,
        "training_authorized": False,
    }
    temporary = OUT.with_name(OUT.name + ".part")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({
        "status": output["status"],
        "checks_passed": output["checks_passed"],
        "geometry_recovered": output["geometry_recovered"],
        "controller_passed": output["controller_passed"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
