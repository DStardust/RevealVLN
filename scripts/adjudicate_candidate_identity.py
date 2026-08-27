#!/usr/bin/env python3
"""Read-only Phase-0B adjudication of ETP-R1 persistent candidate identity.

This script does not run Habitat, load a checkpoint, parse a dataset payload,
or modify a frozen source file.  It reconstructs the exact sequential
GraphMap localization state from the already accepted, hash-chained 50-episode
prefix traces.  Positions in those traces are quantized to 1 mm, so all
distance conclusions explicitly carry a worst-case quantization bound.

The purpose is narrow:
  * determine whether the earlier rule "more than one ID within loc_noise"
    exposed a real nearest-ID tie or merely rejected a deterministic nearest
    match that upstream GraphMap actually uses;
  * verify ghost -> newly visited node lineage for cross-prefix branch IDs;
  * produce a correctness decision for the identity protocol only.  This is
    not a RevealEvent, T_R, T_X, or training acceptance gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter


ROOT = "/mnt/daiyang/vla"
OUT_DIR = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness")
OUT_PATH = os.path.join(OUT_DIR, "CANDIDATE_IDENTITY_AUDIT.json")

MAPPING_PATH = os.path.join(ROOT, "artifacts", "phase0",
                            "REVEAL_QUEUE_50_MAPPING.json")
STAGE4_PATH = os.path.join(
    ROOT, "artifacts", "runtime", "phase0_reveal_closure",
    "STAGE4_TRACE_GENERATION_SUMMARY.json")
ACCEPTANCE_PATH = os.path.join(
    ROOT, "artifacts", "runtime",
    "PHASE0_REVEAL_CLOSURE_MAIN_ACCEPTANCE.json")
GRAPH_UTILS_PATH = os.path.join(
    ROOT, "third_party", "ETP-R1", "vlnce_baselines", "models",
    "graph_utils.py")

EXPECTED_SHA256 = {
    "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    "artifacts/runtime/PHASE0_REVEAL_CLOSURE_MAIN_ACCEPTANCE.json":
        "36c2fb2bce69b8ebc337e0d2192c731c52dba33d8f0b5fe781ffa2a53783b435",
    "artifacts/runtime/phase0_reveal_closure/STAGE4_TRACE_GENERATION_SUMMARY.json":
        "7f3fe38842c38acfe856a19a1feac0aabf134e8cd487f9630a01eaac6e4d7ee9",
    "artifacts/phase0/REVEAL_QUEUE_50_MAPPING.json":
        "fe8dfd9c3af01a67a28035787fdfbe4844ca68c47ca1c7c9f363361c6c331ece",
    "artifacts/phase0/evidence_current.json":
        "430f73ec5752783aa553c2c2f4fe4128247a93fa29bee0ae454c97f5547d9ce1",
}

TRACE_SCHEMA = "reveal-prefix-trace/1"
GENESIS = hashlib.sha256(
    b"RevealNav-Phase0-Reveal-Prefix-Genesis-v1").hexdigest()

# Each stored xyz coordinate is rounded to the nearest 1 mm.  A point has at
# most sqrt(3)*0.5 mm L2 error.  A reconstructed distance has at most twice
# that error.  A difference of two reconstructed distances has at most four
# point-error terms.  Add 4e-5 m numerical slack and round upward.
POINT_ERROR_M = math.sqrt(3.0) * 0.0005
DISTANCE_ERROR_M = 2.0 * POINT_ERROR_M
MARGIN_ERROR_M = 2.0 * DISTANCE_ERROR_M
ROBUST_MARGIN_M = 0.0035
LOC_NOISE_M = 0.5


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def euclid(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


def mean(points):
    return [sum(float(p[j]) for p in points) / len(points)
            for j in range(3)]


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def verify_chain(chain):
    problems = []
    previous = GENESIS
    for i, rec in enumerate(chain):
        if rec.get("schema_version") != TRACE_SCHEMA:
            problems.append("record_%d_schema" % i)
        if rec.get("previous_record_hash") != previous:
            problems.append("record_%d_previous_hash" % i)
        body = {k: v for k, v in rec.items() if k != "current_record_hash"}
        recomputed = hashlib.sha256(
            canonical_json(body).encode("utf-8")).hexdigest()
        if rec.get("current_record_hash") != recomputed:
            problems.append("record_%d_current_hash" % i)
        previous = rec.get("current_record_hash")
    return not problems and bool(chain), previous, problems


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Preserve the earlier/raw source ID as the canonical lineage
            # root; this is only an audit label and never model input.
            self.parent[rb] = ra


def margin_bin(margin):
    if margin <= ROBUST_MARGIN_M:
        return "quantization_unresolved_le_3.5mm"
    if margin <= 0.02:
        return "very_close_3.5_to_20mm"
    if margin <= 0.05:
        return "close_20_to_50mm"
    if margin <= 0.10:
        return "moderate_50_to_100mm"
    return "well_separated_gt_100mm"


def ordered_distances(qpos, positions):
    return [(key, euclid(qpos, pos)) for key, pos in positions.items()]


def audit_episode(item, stage_item):
    order = int(item["queue_order"])
    eid = str(item["episode_id"])
    unit_path = os.path.join(
        ROOT, "artifacts", "phase0", "review_units",
        "unit_%02d_ep%s.json" % (order, eid))
    unit = load_json(unit_path)
    run_dir = os.path.join(ROOT, unit["run"]["run_dir"])
    chain_path = os.path.join(run_dir, "reveal_prefix_chain.jsonl")
    chain = load_jsonl(chain_path)
    chain_ok, root, chain_problems = verify_chain(chain)

    problems = []
    if not os.path.realpath(chain_path).startswith(ROOT + os.sep):
        problems.append("chain_outside_project")
    if os.path.islink(chain_path):
        problems.append("chain_is_symlink")
    if unit["run"].get("chain_sha256") != sha256_file(chain_path):
        problems.append("unit_chain_sha256_mismatch")
    if unit["run"].get("chain_root") != root:
        problems.append("unit_chain_root_mismatch")
    if stage_item.get("chain_root") != root:
        problems.append("stage4_chain_root_mismatch")
    if int(unit.get("queue_order")) != order or str(unit.get("episode_id")) != eid:
        problems.append("unit_identity_mismatch")

    nodes = {}
    ghosts = {}
    ghost_counter = 0
    uf = UnionFind()
    ambiguity_details = []
    localization_boundary_uncertainties = []
    all_mapping_count = 0
    reconstructed_mapping_count = 0
    lineage = []

    for k, rec in enumerate(chain):
        graph = rec["graph"]
        all_mapping_count += len(graph["mappings"])

        # A selected ghost is deleted before the following update_graph call.
        # Aligning to the recorded pre-state captures that exact transition.
        expected_pre_nodes = list(graph["pre_node_ids"])
        expected_pre_ghosts = list(graph["pre_ghost_ids"])
        if list(nodes.keys()) != expected_pre_nodes:
            problems.append("prefix_%d_pre_node_order:%s!=%s" %
                            (k, list(nodes.keys()), expected_pre_nodes))
        removed = [g for g in ghosts if g not in expected_pre_ghosts]
        for g in removed:
            del ghosts[g]
        if list(ghosts.keys()) != expected_pre_ghosts:
            problems.append("prefix_%d_pre_ghost_reconstruction:%s!=%s" %
                            (k, list(ghosts.keys()), expected_pre_ghosts))

        cur_vp = str(rec["cur_vp"])
        nodes[cur_vp] = list(rec["agent_pose"]["position_q"])
        uf.add(cur_vp)

        # Explicit executed-ghost -> new-node lineage across adjacent
        # prefixes.  It is deterministic evidence, not a semantic label.
        if k > 0:
            selected = chain[k - 1]["action"].get("selected_vp")
            if isinstance(selected, str) and selected.startswith("g"):
                was_post = selected in chain[k - 1]["graph"]["post_ghost_ids"]
                removed_now = selected not in expected_pre_ghosts
                position_delta = euclid(
                    chain[k - 1]["action"]["post_position_q"],
                    rec["agent_pose"]["position_q"])
                valid = was_post and removed_now and position_delta <= 0.002
                lineage.append({
                    "from_ghost": selected,
                    "to_node": cur_vp,
                    "from_prefix": k - 1,
                    "to_prefix": k,
                    "post_to_next_pose_delta_m": round(position_delta, 6),
                    "valid": valid,
                })
                if valid:
                    uf.union(selected, cur_vp)
                else:
                    problems.append("prefix_%d_invalid_ghost_node_lineage" % k)

        for mapping in sorted(graph["mappings"],
                              key=lambda x: int(x["cand_index"])):
            i = int(mapping["cand_index"])
            qpos = rec["candidate_positions_q"][i]
            node_dists = ordered_distances(qpos, nodes)
            node_within = [(pid, dis) for pid, dis in node_dists
                           if dis <= LOC_NOISE_M + DISTANCE_ERROR_M]
            node_within_strict_q = [(pid, dis) for pid, dis in node_dists
                                    if dis <= LOC_NOISE_M]
            node_nearest = min(node_dists, key=lambda x: x[1]) \
                if node_dists else (None, math.inf)

            tier = None
            distances = None
            if node_nearest[1] <= LOC_NOISE_M + DISTANCE_ERROR_M:
                tier = "node"
                distances = node_dists
            else:
                ghost_means = {g: mean(points) for g, points in ghosts.items()}
                ghost_dists = ordered_distances(qpos, ghost_means)
                ghost_nearest = min(ghost_dists, key=lambda x: x[1]) \
                    if ghost_dists else (None, math.inf)
                if ghost_nearest[1] <= LOC_NOISE_M + DISTANCE_ERROR_M:
                    tier = "ghost"
                    distances = ghost_dists

            if tier == "node":
                expected_kind = "node"
            elif tier == "ghost":
                expected_kind = "ghost_merged"
            else:
                expected_kind = "ghost_created"

            # Bound-aware reconstruction can only be uncertain for a point
            # very close to the 0.5 m boundary.  Recorded mapping semantics
            # resolve which tier upstream actually used; flag rather than
            # silently forcing a conflicting reconstructed tier.
            if mapping["kind"] != expected_kind:
                # Retry exact quantized boundary before declaring a problem.
                exact_node = node_nearest[1] <= LOC_NOISE_M
                if exact_node:
                    quantized_kind = "node"
                else:
                    ghost_means = {g: mean(points)
                                   for g, points in ghosts.items()}
                    gd = ordered_distances(qpos, ghost_means)
                    exact_ghost = bool(gd) and min(d for _, d in gd) <= LOC_NOISE_M
                    quantized_kind = "ghost_merged" if exact_ghost \
                        else "ghost_created"
                if mapping["kind"] != quantized_kind:
                    # Exact positions used by upstream are not retained;
                    # 1-mm quantization can move a point across the 0.5-m
                    # tier boundary.  Classify such cases as bounded
                    # uncertainty, not reconstruction corruption.
                    if quantized_kind == "node":
                        boundary_distance = node_nearest[1]
                        boundary_tier = "node"
                    else:
                        ghost_means = {g: mean(points)
                                       for g, points in ghosts.items()}
                        gd = ordered_distances(qpos, ghost_means)
                        boundary_distance = min(
                            (d for _, d in gd), default=math.inf)
                        boundary_tier = "ghost"
                    if abs(boundary_distance - LOC_NOISE_M) <= \
                            DISTANCE_ERROR_M + 1e-6:
                        localization_boundary_uncertainties.append({
                            "prefix_index": k,
                            "cand_index": i,
                            "upstream_kind": mapping["kind"],
                            "quantized_reconstruction_kind": quantized_kind,
                            "boundary_tier": boundary_tier,
                            "quantized_boundary_distance_m":
                                round(boundary_distance, 6),
                            "distance_error_bound_m": DISTANCE_ERROR_M,
                        })
                    else:
                        problems.append(
                            "prefix_%d_cand_%d_kind:%s!=%s" %
                            (k, i, mapping["kind"], quantized_kind))

            if mapping["kind"] == "node":
                distances = node_dists
            elif mapping["kind"] == "ghost_merged":
                ghost_means = {g: mean(points) for g, points in ghosts.items()}
                distances = ordered_distances(qpos, ghost_means)
            else:
                distances = []

            if distances:
                ordered = sorted(enumerate(distances),
                                 key=lambda z: (z[1][1], z[0]))
                nearest_id, nearest_d = ordered[0][1]
                target_ok = str(mapping["target"]) == str(nearest_id)
                stored_distance_ok = abs(float(mapping["distance"]) - nearest_d) \
                    <= DISTANCE_ERROR_M + 1e-6
                sorted_d = sorted(d for _, d in distances)
                target_margin = (sorted_d[1] - sorted_d[0]
                                 if len(sorted_d) > 1 else math.inf)
                if not target_ok and target_margin > ROBUST_MARGIN_M:
                    problems.append("prefix_%d_cand_%d_nearest_target" % (k, i))
                if not stored_distance_ok:
                    problems.append("prefix_%d_cand_%d_distance" % (k, i))
            else:
                nearest_id, nearest_d = mapping["target"], 0.0
                target_ok = mapping["kind"] == "ghost_created"
                stored_distance_ok = float(mapping["distance"]) == 0.0

            recorded_matches = list(mapping.get("matches_within_loc_noise", []))
            ambiguity = next(
                (a for a in graph.get("ambiguous", [])
                 if int(a["cand_index"]) == i), None)
            if ambiguity is not None:
                relevant = node_dists if ambiguity["tier"] == "node" else \
                    ordered_distances(qpos, {g: mean(points)
                                             for g, points in ghosts.items()})
                listed = [str(x) for x in ambiguity["within"]]
                selected = [(str(pid), dis) for pid, dis in relevant
                            if str(pid) in listed]
                selected.sort(key=lambda x: x[1])
                if set(listed) != set(recorded_matches):
                    problems.append("prefix_%d_cand_%d_ambiguity_match_set" %
                                    (k, i))
                if len(selected) < 2:
                    problems.append("prefix_%d_cand_%d_missing_second_match" %
                                    (k, i))
                    margin = 0.0
                    second_id = None
                    second_d = None
                else:
                    margin = selected[1][1] - selected[0][1]
                    second_id, second_d = selected[1]
                robust = margin > ROBUST_MARGIN_M
                ambiguity_details.append({
                    "prefix_index": k,
                    "cand_index": i,
                    "tier": ambiguity["tier"],
                    "within_ids": listed,
                    "upstream_target": str(mapping["target"]),
                    "quantized_nearest_id": str(selected[0][0])
                    if selected else None,
                    "quantized_nearest_distance_m": round(selected[0][1], 6)
                    if selected else None,
                    "quantized_second_id": second_id,
                    "quantized_second_distance_m": round(second_d, 6)
                    if second_d is not None else None,
                    "nearest_second_margin_m": round(margin, 6),
                    "margin_bin": margin_bin(margin),
                    "robust_unique_under_1mm_quantization": robust,
                    "upstream_target_matches_quantized_nearest":
                        bool(selected and str(mapping["target"]) ==
                             str(selected[0][0])),
                    "policy_action_selected_same_persistent_id":
                        rec["action"].get("selected_vp") == mapping["target"],
                    "scope_note": "numeric nearest-ID identity only; does "
                                  "not establish semantic branch identity",
                })

            if mapping["kind"] == "ghost_created":
                expected_id = "g%d" % ghost_counter
                if mapping["target"] != expected_id:
                    problems.append("prefix_%d_cand_%d_ghost_counter:%s!=%s" %
                                    (k, i, mapping["target"], expected_id))
                    # Advance past observed ID to keep the audit useful.
                    try:
                        ghost_counter = int(str(mapping["target"])[1:])
                    except ValueError:
                        pass
                ghosts[str(mapping["target"])] = [list(qpos)]
                uf.add(str(mapping["target"]))
                ghost_counter += 1
            elif mapping["kind"] == "ghost_merged":
                target = str(mapping["target"])
                if target not in ghosts:
                    problems.append("prefix_%d_cand_%d_missing_ghost" % (k, i))
                    ghosts[target] = []
                ghosts[target].append(list(qpos))
            reconstructed_mapping_count += 1

        if list(nodes.keys()) != list(graph["post_node_ids"]):
            problems.append("prefix_%d_post_node_order" % k)
        if list(ghosts.keys()) != list(graph["post_ghost_ids"]):
            problems.append("prefix_%d_post_ghost_order" % k)

    robust_count = sum(d["robust_unique_under_1mm_quantization"]
                       for d in ambiguity_details)
    target_match_count = sum(
        d["upstream_target_matches_quantized_nearest"]
        for d in ambiguity_details)
    lineage_valid = sum(x["valid"] for x in lineage)
    return {
        "queue_order": order,
        "episode_id": eid,
        "scene_id": item["scene_id"],
        "stage4_status": stage_item["status"],
        "chain_file": os.path.relpath(chain_path, ROOT),
        "chain_sha256": sha256_file(chain_path),
        "chain_root": root,
        "chain_valid": chain_ok,
        "chain_problems": chain_problems,
        "prefix_count": len(chain),
        "mapping_count": all_mapping_count,
        "reconstructed_mapping_count": reconstructed_mapping_count,
        "identity_ambiguity_count": len(ambiguity_details),
        "localization_boundary_uncertainty_count":
            len(localization_boundary_uncertainties),
        "robust_unique_count": robust_count,
        "upstream_nearest_target_match_count": target_match_count,
        "ghost_to_node_lineage_count": len(lineage),
        "valid_ghost_to_node_lineage_count": lineage_valid,
        "ghost_to_node_lineage": lineage,
        "ambiguity_details": ambiguity_details,
        "localization_boundary_uncertainties":
            localization_boundary_uncertainties,
        "audit_problems": problems,
        "reconstruction_pass": (chain_ok and not problems and
                                reconstructed_mapping_count ==
                                all_mapping_count),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    preflight = []
    for rel, expected in EXPECTED_SHA256.items():
        path = os.path.join(ROOT, rel)
        actual = sha256_file(path)
        preflight.append({"path": rel, "expected_sha256": expected,
                          "actual_sha256": actual,
                          "pass": actual == expected})
    acceptance = load_json(ACCEPTANCE_PATH)
    mapping = load_json(MAPPING_PATH)
    stage4 = load_json(STAGE4_PATH)
    preconditions = (
        all(x["pass"] for x in preflight)
        and acceptance.get("status") == "ACCEPTED_ENGINEERING_BATCH"
        and acceptance.get("overall_phase0_decision") == "NO_GO"
        and acceptance.get("training_authorized") is False
        and mapping.get("mapped_count") == 50
        and stage4.get("total_units") == 50
        and stage4.get("counts") == {"ok": 16, "ambiguous": 34,
                                     "failed": 0}
    )
    stage_by_order = {int(x["queue_order"]): x for x in stage4["items"]}
    episodes = []
    if preconditions:
        for item in mapping["items"]:
            episodes.append(audit_episode(
                item, stage_by_order[int(item["queue_order"])]))

    details = [d for ep in episodes for d in ep["ambiguity_details"]]
    margin_bins = Counter(d["margin_bin"] for d in details)
    tier_counts = Counter(d["tier"] for d in details)
    ambiguous_episodes = [ep for ep in episodes
                          if ep["stage4_status"] == "AMBIGUOUS"]
    all_reconstructed = (len(episodes) == 50 and
                         all(ep["reconstruction_pass"] for ep in episodes))
    all_ambiguous_have_evidence = (
        len(ambiguous_episodes) == 34 and
        all(ep["identity_ambiguity_count"] > 0
            for ep in ambiguous_episodes))
    all_robust = bool(details) and all(
        d["robust_unique_under_1mm_quantization"] for d in details)
    all_robust_target_match = bool(details) and all(
        d["upstream_target_matches_quantized_nearest"]
        for d in details if d["robust_unique_under_1mm_quantization"])
    all_lineage_valid = all(
        ep["valid_ghost_to_node_lineage_count"] ==
        ep["ghost_to_node_lineage_count"] for ep in episodes)

    if not preconditions or not all_reconstructed:
        decision = "NO_GO_AUDIT_INTEGRITY"
    elif not all_robust_target_match:
        decision = "NO_GO_UPSTREAM_NEAREST_MISMATCH"
    elif not all_robust:
        decision = "REVISION_REQUIRED_QUANTIZATION_UNRESOLVED_TIES"
    elif not all_lineage_valid:
        decision = "REVISION_REQUIRED_LINEAGE_FAILURE"
    else:
        decision = "REVISION_JUSTIFIED_USE_UPSTREAM_NEAREST_WITH_LINEAGE"

    output = {
        "gate": "phase0b_candidate_identity_correctness_adjudication",
        "revision": "candidate-identity-audit/1",
        "status": "PASS" if decision.startswith("REVISION_JUSTIFIED")
                  else "FAIL",
        "decision": decision,
        "scope": "read-only numeric persistent-ID correctness audit over "
                 "the accepted 50 RxR-train engineering traces",
        "preflight": {
            "pass": preconditions,
            "files": preflight,
            "acceptance_status": acceptance.get("status"),
            "phase0_decision_preserved":
                acceptance.get("overall_phase0_decision"),
            "training_authorized_preserved":
                acceptance.get("training_authorized"),
            "graph_utils_path": os.path.relpath(GRAPH_UTILS_PATH, ROOT),
            "graph_utils_sha256": sha256_file(GRAPH_UTILS_PATH),
        },
        "pre_registered_numeric_contract": {
            "upstream_rule": "GraphMap._localize: insertion-order scan, "
                             "strictly smallest Euclidean distance, accept "
                             "iff min_distance <= loc_noise",
            "loc_noise_m": LOC_NOISE_M,
            "stored_position_resolution_m": 0.001,
            "point_l2_error_bound_m": POINT_ERROR_M,
            "single_distance_error_bound_m": DISTANCE_ERROR_M,
            "nearest_second_margin_error_bound_m": MARGIN_ERROR_M,
            "robust_unique_threshold_m": ROBUST_MARGIN_M,
            "important_boundary": "robust numeric nearest-ID does not by "
                                  "itself establish semantic branch identity",
        },
        "counts": {
            "episodes": len(episodes),
            "fully_reconstructed_episodes": sum(
                ep["reconstruction_pass"] for ep in episodes),
            "stage4_ok_episodes": sum(
                ep["stage4_status"] == "OK" for ep in episodes),
            "stage4_ambiguous_episodes": len(ambiguous_episodes),
            "ambiguous_episodes_with_distance_evidence": sum(
                ep["identity_ambiguity_count"] > 0
                for ep in ambiguous_episodes),
            "all_mappings": sum(ep["mapping_count"] for ep in episodes),
            "reconstructed_mappings": sum(
                ep["reconstructed_mapping_count"] for ep in episodes),
            "multiple_within_radius_candidates": len(details),
            "robust_unique_nearest_candidates": sum(
                d["robust_unique_under_1mm_quantization"] for d in details),
            "upstream_target_matches_quantized_nearest": sum(
                d["upstream_target_matches_quantized_nearest"]
                for d in details),
            "localization_boundary_uncertainties": sum(
                ep["localization_boundary_uncertainty_count"]
                for ep in episodes),
            "policy_selected_same_ambiguous_mapping_target": sum(
                d["policy_action_selected_same_persistent_id"]
                for d in details),
            "ghost_to_node_lineages": sum(
                ep["ghost_to_node_lineage_count"] for ep in episodes),
            "valid_ghost_to_node_lineages": sum(
                ep["valid_ghost_to_node_lineage_count"] for ep in episodes),
            "tier_counts": dict(tier_counts),
            "margin_bins": dict(margin_bins),
        },
        "gates": {
            "all_50_reconstructed": all_reconstructed,
            "all_34_ambiguous_have_distance_evidence":
                all_ambiguous_have_evidence,
            "all_robust_upstream_targets_match_nearest":
                all_robust_target_match,
            "all_nearest_matches_robust_to_1mm_quantization": all_robust,
            "all_executed_ghost_to_node_lineages_valid": all_lineage_valid,
        },
        "adjudication": {
            "earlier_rule": "FAIL if more than one persistent ID lies "
                            "within loc_noise=0.5 m",
            "finding": "The earlier rule is tested against the upstream "
                       "deterministic nearest-ID contract.  A revision is "
                       "justified only if every rejected mapping has a "
                       "unique nearest target outside the complete 1 mm "
                       "quantization uncertainty bound and all cross-prefix "
                       "executed-ghost lineages are valid.",
            "recommended_identity_protocol_if_all_gates_pass": {
                "version": "persistent-branch-identity/v2-proposal",
                "localization": "use frozen upstream nearest Euclidean ID "
                                "within 0.5 m; preserve dictionary insertion "
                                "order for exact ties",
                "fail_closed_when": [
                    "nearest/second-nearest margin <= 0.0035 m in a "
                    "1-mm-quantized audit trace",
                    "collector mirror differs from upstream post-state",
                    "executed ghost cannot be linked to the next node by "
                    "deletion plus <=2 mm post/next-pose agreement",
                    "semantic reviewer cannot identify a stable physical "
                    "branch even though numeric localization is unique",
                ],
                "lineage": "union the executed selected ghost with the next "
                           "created node; retain raw IDs and transition "
                           "evidence alongside the canonical lineage ID",
            },
        },
        "episodes": episodes,
        "non_conclusions": {
            "semantic_branch_identity_established": False,
            "reveal_event_validity_established": False,
            "unique_tx_established": False,
            "human_review_started": False,
            "training_authorized": False,
            "frozen_spec_modified": False,
            "dataset_payload_parsed": False,
            "checkpoint_loaded": False,
        },
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"],
        "decision": decision,
        "counts": output["counts"],
        "output": os.path.relpath(OUT_PATH, ROOT),
        "output_sha256": sha256_file(OUT_PATH),
    }, indent=2))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
