#!/usr/bin/env python3
"""Build the private, unreviewed language packet for 35 CR1 events."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys

import cv2
import numpy as np


ROOT = "/mnt/daiyang/vla"
SCRIPTS = os.path.join(ROOT, "scripts")
HABSIM = os.path.join(ROOT, "third_party", "habitat-sim")
for path in (SCRIPTS, HABSIM):
    if path not in sys.path:
        sys.path.insert(0, path)
from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402


AUTO = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                    "AUTOMATIC_SEMANTIC_MULTIPLICITY_ADJUDICATION.json")
AUTO_RAW = os.path.join(ROOT, "artifacts", "runtime",
                        "phase0_correctness",
                        "AUTOMATIC_SEMANTIC_CANDIDATE_GATE.json")
SEMANTIC = os.path.join(ROOT, "artifacts", "runtime",
                        "phase0_correctness",
                        "ORACLE_SEMANTIC_BRANCH_TRACK_AUDIT.json")
COST = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                    "PHASE0C_COST_FRONTIER_ADJUDICATION.json")
PROBE = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                     "PHASE0C_ORACLE_LOWLEVEL_PROBE.json")
OLD_PACKET = os.path.join(ROOT, "artifacts", "phase0",
                          "REVIEW_PACKET_50.json")
RXR_TRAIN = os.path.join(
    ROOT, "third_party", "ETP-R1", "data", "datasets",
    "RxR_VLNCE_v0_enc_xlmr", "train", "train_guide.json.gz")
MP3D = os.path.join(ROOT, "third_party", "ETP-R1", "data",
                    "scene_datasets", "mp3d")
PACKET_DIR = os.path.join(ROOT, "artifacts", "phase0",
                          "phase0c_language_review_35")
MEDIA_DIR = os.path.join(PACKET_DIR, "private_media")
OUT = os.path.join(PACKET_DIR, "PHASE0C_LANGUAGE_REVIEW_35.json")
CSV_OUT = os.path.join(PACKET_DIR, "PHASE0C_LANGUAGE_REVIEW_35.csv")
GUIDE = os.path.join(PACKET_DIR, "REVIEW_GUIDE.md")
PRIVATE = os.path.join(PACKET_DIR, "PRIVATE_DO_NOT_DISTRIBUTE.txt")
EXPECTED = {
    AUTO: "e2dfba0b25f7df3cfcc4082567d95d897860595a1b6e0bf46bbe81846f696d3a",
    AUTO_RAW: "13797692e69847392b572f17f0559f36b685ec84b10051fc14c9f26c13ad2f7b",
    SEMANTIC: "e4b570dc9cdbe317d28b57507f1f74b9a16f92c8350810beb6b0f4dacd9df6a4",
    COST: "43481d408358322a826f9769e269b38115ba0cacb794d2de377aaae4b6b12551",
    PROBE: "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac",
}
HUMAN_FIELDS = [
    "reviewer_id", "review_timestamp", "branch_dependent_instruction",
    "instruction_clause", "target_branch_matches_instruction",
    "causal_reveal_confirmed", "semantic_track_confirmed",
    "cost_expiry_interpretation_confirmed", "candidate_valid",
    "rejection_reason", "reviewer_notes",
]


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_sim(scene):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.join(MP3D, scene, scene + ".glb")
    sim_cfg.gpu_device_id = 0
    rgb = habitat_sim.SensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb.resolution = [224, 224]
    rgb.position = [0.0, 0.88, 0.0]
    rgb.hfov = 63.0
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = [rgb]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg,
                                                           [agent_cfg]))
    navmesh = os.path.join(MP3D, scene, scene + ".navmesh")
    if not sim.pathfinder.load_nav_mesh(navmesh):
        sim.close()
        raise RuntimeError("navmesh load failed")
    return sim


def render(sim, state):
    import habitat_sim
    from scipy.spatial.transform import Rotation

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.asarray(sim.pathfinder.snap_point(
        state["position"]), dtype="float32")
    agent_state.rotation = Rotation.from_rotvec(
        [0.0, float(state["heading"]), 0.0]).as_quat()
    sim.get_agent(0).set_state(agent_state, True)
    return sim.get_sensor_observations()["rgb"][..., :3].copy()


def label_frame(rgb, text):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (223, 26), (0, 0, 0), -1)
    cv2.putText(bgr, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def plan_panel(trace, prefixes, target):
    panel = np.full((224, 224, 3), 248, dtype=np.uint8)
    points = np.asarray([[row["position"][0], row["position"][2]]
                         for row in trace], dtype=np.float64)
    target_points = np.asarray([
        [target["directed_start_q"][0], target["directed_start_q"][2]],
        [target["directed_end_q"][0], target["directed_end_q"][2]]])
    all_points = np.vstack([points, target_points])
    low, high = all_points.min(0), all_points.max(0)
    span = np.maximum(high - low, 1e-3)

    def xy(point):
        normalized = (np.asarray(point) - low) / span
        return (int(12 + normalized[0] * 200),
                int(211 - normalized[1] * 187))

    for a, b in zip(points[:-1], points[1:]):
        cv2.line(panel, xy(a), xy(b), (175, 175, 175), 1,
                 cv2.LINE_AA)
    cv2.arrowedLine(panel, xy(target_points[0]), xy(target_points[1]),
                    (0, 150, 0), 3, cv2.LINE_AA, tipLength=0.12)
    colors = [(160, 160, 160), (0, 150, 255), (0, 90, 255), (0, 0, 255)]
    for index, prefix in enumerate(prefixes):
        cv2.circle(panel, xy(points[prefix]), 4, colors[index], -1)
    cv2.rectangle(panel, (0, 0), (223, 26), (0, 0, 0), -1)
    cv2.putText(panel, "offline route/exit review aid", (4, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1,
                cv2.LINE_AA)
    return panel


def main():
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise SystemExit("input SHA drift: " + path)
    os.makedirs(MEDIA_DIR, exist_ok=True)
    auto = json.load(open(AUTO))
    auto_raw = json.load(open(AUTO_RAW))
    semantic = json.load(open(SEMANTIC))
    cost = json.load(open(COST))
    probe = json.load(open(PROBE))
    old = json.load(open(OLD_PACKET))
    auto_ids = {event["provisional_event_id"] for event in auto["events"]
                if event["adjudicated_status"].startswith("TRACKED_K3")}
    cost_map = {event["provisional_event_id"]: event
                for event in cost["events"]}
    eligible = sorted(event_id for event_id in auto_ids
                      if cost_map[event_id]["controllers"][
                          "frozen_shortest_path_compat"][
                              "passes_two_budget_gate"])
    if len(eligible) != 35:
        raise SystemExit("review eligibility cardinality drift")
    semantic_map = {event["provisional_event_id"]: event
                    for event in semantic["events"]}
    probe_map = {event["provisional_event_id"]: event
                 for event in probe["events"]}
    auto_raw_map = {event["provisional_event_id"]: event
                    for event in auto_raw["events"]}
    instruction_rows = {str(row["episode_id"]): row for row in old["rows"]}
    episode_ids = {str(semantic_map[event_id]["episode_id"])
                   for event_id in eligible}
    with gzip.open(RXR_TRAIN, "rt") as fh:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(fh)["episodes"]
                    if str(item["episode_id"]) in episode_ids}

    by_scene = {}
    for event_id in eligible:
        by_scene.setdefault(semantic_map[event_id]["scene_id"], []).append(
            event_id)
    rows, media_manifest = [], []
    row_order = 0
    for scene, event_ids in sorted(by_scene.items()):
        sim = build_sim(scene)
        try:
            trace_cache = {}
            for event_id in event_ids:
                semantic_event = semantic_map[event_id]
                episode_id = str(semantic_event["episode_id"])
                if episode_id not in trace_cache:
                    trace_cache[episode_id] = build_lowlevel_trace(
                        sim.pathfinder, episodes[episode_id])
                trace = trace_cache[episode_id]
                reveal = int(probe_map[event_id]["candidate_reveal_prefix"])
                prefixes = [max(0, reveal - 1), reveal, reveal + 1,
                            reveal + 2]
                if max(prefixes) >= len(trace):
                    raise RuntimeError("review prefix outside trace")
                panels = []
                labels = ["pre-reveal", "D1", "D2", "D3"]
                individual = []
                for prefix, label in zip(prefixes, labels):
                    rgb = render(sim, trace[prefix])
                    bgr = label_frame(rgb, "%s p%d" % (label, prefix))
                    name = "%03d_%s_%s.jpg" % (row_order, event_id, label)
                    path = os.path.join(MEDIA_DIR, name)
                    if not cv2.imwrite(path, bgr,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                        raise RuntimeError("failed to write private frame")
                    relative = os.path.relpath(path, ROOT)
                    individual.append(relative)
                    media_manifest.append({"path": relative,
                                           "sha256": sha256_file(path),
                                           "bytes": os.path.getsize(path)})
                    panels.append(bgr)
                panels.append(plan_panel(
                    trace, prefixes, semantic_event["target_exit_region"]))
                contact = np.concatenate(panels, axis=1)
                contact_name = "%03d_%s_contact.jpg" % (row_order, event_id)
                contact_path = os.path.join(MEDIA_DIR, contact_name)
                if not cv2.imwrite(contact_path, contact,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                    raise RuntimeError("failed to write contact sheet")
                contact_relative = os.path.relpath(contact_path, ROOT)
                media_manifest.append({"path": contact_relative,
                                       "sha256": sha256_file(contact_path),
                                       "bytes": os.path.getsize(contact_path)})
                instruction = instruction_rows[episode_id]
                frozen = cost_map[event_id]["controllers"][
                    "frozen_shortest_path_compat"]
                row = {
                    "row_order": row_order,
                    "event_id": event_id,
                    "episode_id": episode_id,
                    "scene_id": scene,
                    "instruction_id": instruction["instruction_id"],
                    "trajectory_id": instruction["trajectory_id"],
                    "language": instruction["language"],
                    "instruction_sha256": instruction["instruction_sha256"],
                    "instruction_text_for_private_review":
                        instruction["instruction_text_for_review"],
                    "screening_triggers": instruction["screening_triggers"],
                    "semantic_branch_id":
                        semantic_event["semantic_branch_id"],
                    "target_exit_region":
                        semantic_event["target_exit_region"],
                    "reference_turn_index": probe_map[event_id][
                        "reference_turn_index"],
                    "turn_angle_deg": probe_map[event_id]["turn_angle_deg"],
                    "causal_prefixes": {"pre_reveal": prefixes[0],
                                        "d1": prefixes[1], "d2": prefixes[2],
                                        "d3": prefixes[3]},
                    "automatic_track_status": next(
                        event["adjudicated_status"] for event in auto["events"]
                        if event["provisional_event_id"] == event_id),
                    "automatic_prefix_records": auto_raw_map[event_id][
                        "prefix_records"],
                    "frozen_cost_frontiers": frozen["frontiers"],
                    "cost_unique_last_safe_budget_count": frozen[
                        "unique_last_safe_budget_count"],
                    "private_media": individual,
                    "private_contact_sheet": contact_relative,
                    "annotation_status": "PENDING_HUMAN_REVIEW",
                    "reviewed": False,
                }
                for field in HUMAN_FIELDS:
                    row[field] = None
                rows.append(row)
                row_order += 1
        finally:
            sim.close()

    if len(rows) != 35 or any(row["reviewed"] is not False or
                              any(row[field] is not None
                                  for field in HUMAN_FIELDS)
                              for row in rows):
        raise SystemExit("non-fabrication gate failed")
    output = {
        "packet": "MF2-CR1 Phase0C language review 35",
        "revision": "phase0c-language-review/1",
        "status": "PASS_PENDING_HUMAN_REVIEW",
        "distribution": "PRIVATE_DO_NOT_DISTRIBUTE",
        "selection": "intersection of 38 automatic causal K3 semantic "
                     "tracks and frozen-controller unique T_X at >=2 fixed "
                     "budgets; no resampling",
        "row_count": len(rows),
        "reviewed_true_count": 0,
        "all_rows_pending": True,
        "human_fields_prefilled": False,
        "human_fields": HUMAN_FIELDS,
        "media_file_count": len(media_manifest),
        "media_total_bytes": sum(item["bytes"] for item in media_manifest),
        "inputs": {os.path.relpath(path, ROOT): sha256_file(path)
                   for path in EXPECTED},
        "rows": rows,
        "media_manifest": media_manifest,
        "non_conclusions": {"human_validated_events": 0,
                            "full_gate6_pass": False,
                            "training_authorized": False,
                            "distribution_authorized": False},
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    csv_fields = ["row_order", "event_id", "episode_id", "scene_id",
                  "instruction_id", "language", "screening_triggers",
                  "semantic_branch_id", "private_contact_sheet", "reviewed"]
    csv_fields += HUMAN_FIELDS
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    guide_text = """# MF2-CR1 private language review guide

This packet has 35 machine-screened RxR-train events. No row is currently a
validated Reveal Event. Review the instruction, four causal front views and
the offline route/exit panel. Set `reviewed=true` only after filling every
human field.

Accept only if the instruction genuinely requires the directed target branch,
the D1--D3 sequence causally reveals it, the semantic proposal set represents
one exit (not two alternatives), and the resource-conditioned expiry is a
meaningful last-passage label. Reject uncertain rows; do not infer from future
frames. The media and Matterport-derived geometry remain private and must not
be distributed without authorization.
"""
    with open(GUIDE, "w") as fh:
        fh.write(guide_text)
    with open(PRIVATE, "w") as fh:
        fh.write("PRIVATE Matterport-derived review material. Do not "
                 "distribute. Human fields are entirely unreviewed.\n")
    print(json.dumps({
        "status": output["status"], "rows": len(rows),
        "reviewed": 0, "media_files": output["media_file_count"],
        "media_bytes": output["media_total_bytes"],
        "json": os.path.relpath(OUT, ROOT),
        "json_sha256": sha256_file(OUT),
        "csv": os.path.relpath(CSV_OUT, ROOT),
        "csv_sha256": sha256_file(CSV_OUT),
    }, indent=2))


if __name__ == "__main__":
    main()

