#!/usr/bin/env python3
"""Reveal Prefix candidate trace collector (harness-only instrumentation).

Label: phase0 reveal-prefix engineering.  No frozen source file is modified.

Loads the accepted rxr_en_worker module (gym Discrete(0) shim, network deny
guard, high-level action trace recorder) and ADDS three observation-only
hooks:

  A. Waypoint output hook
     Wraps ETP.forward for mode == "waypoint" without altering outputs.
     Records per high-level prefix: episode identity, quantized agent pose
     summary (via the subsequent identify_node inputs), cand_angles,
     cand_distances, cand_img_idxes, candidate count, candidate feature
     shape/dtype/hash, and strict SHA-256 + shape/dtype of every observation
     tensor (raw RGB/depth are NEVER written).

  B. GraphMap hook
     Wraps GraphMap.identify_node and GraphMap.update_graph.  Records local
     candidate positions, current node id, persistent node/ghost ids before
     and after the merge, and the candidate-to-persistent-id mapping
     computed with the SAME loc_noise=0.5 merge semantics as upstream
     (mirrored sequentially and cross-checked against the real post-merge
     graph state).  If a local candidate can map to more than one persistent
     id, or the mirror disagrees with upstream post-state, the episode event
     generation FAILS CLOSED with status AMBIGUOUS (no guessing).

     The three tiers are kept strictly separate:
       - the current waypoint candidate set (per-prefix local proposals),
       - the ETP-R1 persistent graph memory (nodes + ghosts),
       - any future TopoReveal ECOG (NOT implemented or implied here).

  C. Action/return trace
     Reuses the accepted VLNCEDaggerEnv.step trace: policy-selected branch
     (ghost_vp/front_vp or stop_vp), low-level back path length, tryout,
     done/reward, plus post-execution agent pose (RXRENG_POSE_FILE) and
     episode identity metadata (RXRENG_EPISODE_META_FILE).

Hash chain: one JSONL line per high-level prefix with schema_version,
previous_record_hash, current_record_hash, observation hash, candidate set
hash, action hash and graph mapping hash.  The first record uses the fixed
genesis hash.  Canonicalization rules are fixed a priori (never tuned from
results): positions quantized to 1e-3 m, headings to 1e-3 rad, candidate
angles/distances rounded to 1e-9, tensor hashes over raw bytes.  Candidate
ORDER is upstream order (NMS output nonzero() row-major order); it is not
reordered.

The validator scripts/validate_reveal_prefix_trace.py recomputes the whole
chain from genesis from scratch.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time

PROJECT_ROOT = "/mnt/daiyang/vla"
ETPR1_ROOT = os.path.join(PROJECT_ROOT, "third_party", "ETP-R1")
HABITAT_LAB_ROOT = os.path.join(PROJECT_ROOT, "third_party", "habitat-lab")
HABITAT_SIM_ROOT = os.path.join(PROJECT_ROOT, "third_party", "habitat-sim")

SCHEMA_VERSION = "reveal-prefix-trace/1"
GENESIS_HASH = hashlib.sha256(
    b"RevealNav-Phase0-Reveal-Prefix-Genesis-v1").hexdigest()

for _p in (HABITAT_SIM_ROOT, HABITAT_LAB_ROOT, ETPR1_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load the accepted worker module: installing it applies the gym shim,
# the network deny guard and the action trace recorder at top level.
_worker_path = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                            "rxr_en_worker.py")
_spec = importlib.util.spec_from_file_location("rxr_en_worker", _worker_path)
WORKER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WORKER)

CKPT_RXR = WORKER.CKPT_RXR
JOINT_PRETRAINED = WORKER.JOINT_PRETRAINED
CKPT_R2R = os.path.join(
    ETPR1_ROOT, "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)

# task family -> frozen run config + checkpoint + eval overrides
TASK_FAMILIES = {
    "rxr": {
        "exp_config": "run_rxr/iter_train.yaml",
        "ckpt": CKPT_RXR,
        "extra_opts": [
            "IL.RECOLLECT_TRAINER.gt_file",
            "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        ],
        "languages": True,
    },
    "r2r": {
        "exp_config": "run_r2r/iter_train.yaml",
        "ckpt": CKPT_R2R,
        "extra_opts": [],
        "languages": False,
    },
}


# --------------------------------------------------------------------------
# canonical hashing helpers
# --------------------------------------------------------------------------
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def q3(x):
    """Quantize a float to 1e-3 (positions/headings)."""
    return round(float(x), 3)


def q9(x):
    """Round candidate angles/distances to 1e-9."""
    return round(float(x), 9)


def tensor_record(t):
    import torch

    if not isinstance(t, torch.Tensor):
        return None
    arr = t.detach().cpu().contiguous()
    raw = arr.numpy().tobytes()
    return {
        "sha256": sha256_bytes(raw),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


# --------------------------------------------------------------------------
# collector state (parent trainer process)
# --------------------------------------------------------------------------
COLLECT = {
    "active": False,
    "waypoint_records": [],
    "identify_records": [],
    "graph_records": [],
    "status": "OK",
    "status_reason": None,
    "media_staging": None,
    "identity_protocol": "persistent-branch-identity/v1",
    "identity_margin_m": None,
}

IDENTITY_PROTOCOLS = {
    "v1": {
        "version": "persistent-branch-identity/v1",
        "rule": "fail_closed_if_more_than_one_id_within_loc_noise",
        "nearest_second_margin_m": None,
    },
    "v2": {
        "version": "persistent-branch-identity/v2-engineering",
        "rule": "use_upstream_nearest_id; fail_closed_only_if_multiple_"
                "within_loc_noise_and_nearest_second_margin_le_0.0035m",
        # The bound is pre-registered by candidate-identity-audit/1: four
        # point quantization terms for positions stored at 1 mm resolution.
        "nearest_second_margin_m": 0.0035,
    },
    "v3": {
        "version": "persistent-branch-identity/v3-engineering",
        "rule": "preserve frozen upstream deterministic nearest-ID choice; "
                "record every within-radius distance in the hash chain; "
                "do not truncate the engineering trace for semantic "
                "ambiguity, which remains an event-validator/reviewer gate",
        "nearest_second_margin_m": None,
    },
}


def fail_closed(reason):
    if COLLECT["status"] == "OK":
        COLLECT["status"] = "AMBIGUOUS"
        COLLECT["status_reason"] = reason
        print("[collector] FAIL CLOSED: %s" % reason, flush=True)


# --------------------------------------------------------------------------
# A. waypoint output hook
# --------------------------------------------------------------------------
def install_waypoint_hook():
    from vlnce_baselines.models.R1Policy import ETP

    original_forward = ETP.forward
    if getattr(original_forward, "_reveal_hooked", False):
        return

    def hooked_forward(self, *args, **kwargs):
        outputs = original_forward(self, *args, **kwargs)
        mode = kwargs.get("mode")
        if mode != "waypoint" or args:
            return outputs
        if not COLLECT["active"] or COLLECT["status"] != "OK":
            return outputs
        try:
            observations = kwargs.get("observations") or {}
            obs_hashes = {}
            for key in sorted(observations.keys()):
                rec = tensor_record(observations[key])
                if rec is not None:
                    obs_hashes[key] = rec
            cand_out = {
                "count": None,
                "angles": None,
                "distances": None,
                "img_idxes": None,
                "features": [],
            }
            if isinstance(outputs, dict) and "cand_angles" in outputs:
                # single-env batch: record env 0
                ang = outputs["cand_angles"][0]
                dis = outputs["cand_distances"][0]
                idx = outputs["cand_img_idxes"][0]
                cand_out["count"] = len(ang)
                cand_out["angles"] = [q9(a) for a in ang]
                cand_out["distances"] = [q9(d) for d in dis]
                cand_out["img_idxes"] = [int(i) for i in idx]
                rgb = outputs.get("cand_rgb")
                dep = outputs.get("cand_depth")
                if rgb is not None:
                    cand_out["features"].append(
                        dict(kind="cand_rgb", **tensor_record(rgb[0])))
                if dep is not None:
                    cand_out["features"].append(
                        dict(kind="cand_depth", **tensor_record(dep[0])))
            COLLECT["waypoint_records"].append({
                "obs_hashes": obs_hashes,
                "candidates": cand_out,
            })
            # optional media staging (front RGB JPEG only; raw tensors are
            # never written)
            if COLLECT["media_staging"] and "rgb" in observations:
                try:
                    import cv2
                    import numpy as np
                    import torch

                    rgb_t = observations["rgb"][0]
                    img = rgb_t.detach().cpu().numpy().astype(np.uint8)
                    bgr = img[:, :, ::-1]
                    ok, buf = cv2.imencode(
                        ".jpg", bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ok:
                        k = len(COLLECT["waypoint_records"]) - 1
                        with open(os.path.join(
                                COLLECT["media_staging"],
                                "prefix_%03d.jpg" % k), "wb") as fh:
                            fh.write(buf.tobytes())
                except Exception as exc:  # noqa: BLE001
                    print("[collector] media staging skipped: %r" % exc,
                          flush=True)
        except Exception as exc:  # noqa: BLE001
            fail_closed("waypoint hook error: %r" % exc)
        return outputs

    hooked_forward._reveal_hooked = True
    ETP.forward = hooked_forward


# --------------------------------------------------------------------------
# B. GraphMap hook
# --------------------------------------------------------------------------
def _mirror_localize(qpos, pos_dict, loc_noise):
    """Reproduce GraphMap._localize nearest-match + collect all matches
    within loc_noise (for ambiguity detection).  Insertion-order ties."""
    import numpy as np

    min_dis = 10000.0
    min_vp = None
    within = []
    for kvp, kpos in pos_dict.items():
        dis = float(((np.asarray(qpos) - np.asarray(kpos)) ** 2).sum() ** 0.5)
        if dis <= loc_noise:
            within.append((kvp, dis))
        if dis < min_dis:
            min_dis = dis
            min_vp = kvp
    nearest = min_vp if (min_vp is not None and min_dis <= loc_noise) else None
    return nearest, within, min_dis


def install_graphmap_hook():
    import numpy as np
    from vlnce_baselines.models import graph_utils

    original_identify = graph_utils.GraphMap.identify_node
    original_update = graph_utils.GraphMap.update_graph
    if getattr(original_update, "_reveal_hooked", False):
        return

    def hooked_identify_node(self, cur_pos, cur_ori, cand_ang, cand_dis):
        cur_vp, cand_vp, cand_pos = original_identify(
            self, cur_pos, cur_ori, cand_ang, cand_dis)
        if COLLECT["active"] and COLLECT["status"] == "OK":
            try:
                from vlnce_baselines.models.graph_utils import (
                    heading_from_quaternion)

                COLLECT["identify_records"].append({
                    "cur_vp": str(cur_vp),
                    "cand_vp": [str(v) for v in cand_vp],
                    "cur_pos_q": [q3(x) for x in np.asarray(cur_pos)],
                    "heading_q": q3(heading_from_quaternion(cur_ori)),
                    "cand_pos_q": [[q3(x) for x in p] for p in cand_pos],
                })
            except Exception as exc:  # noqa: BLE001
                fail_closed("identify hook error: %r" % exc)
        return cur_vp, cand_vp, cand_pos

    def hooked_update_graph(self, prev_vp, step_id,
                            cur_vp, cur_pos, cur_embeds,
                            cand_vp, cand_pos, cand_embeds,
                            cand_real_pos):
        active = COLLECT["active"] and COLLECT["status"] == "OK"
        mirror = None
        pre_snapshot = None
        if active:
            try:
                pre_snapshot = {
                    "node_ids": list(self.node_pos.keys()),
                    "ghost_ids": list(self.ghost_pos.keys()),
                    "ghost_pos": {gvp: [np.asarray(p).copy()
                                        for p in plist]
                                  for gvp, plist in self.ghost_pos.items()},
                    "ghost_cnt": int(self.ghost_cnt),
                }
                mirror = _mirror_mapping(self, cur_vp, cur_pos, cand_pos)
            except Exception as exc:  # noqa: BLE001
                fail_closed("graph mirror error: %r" % exc)

        original_update(self, prev_vp, step_id,
                        cur_vp, cur_pos, cur_embeds,
                        cand_vp, cand_pos, cand_embeds,
                        cand_real_pos)

        if active and mirror is not None:
            try:
                record = _crosscheck_and_record(
                    self, mirror, pre_snapshot, cur_vp, cur_pos, cand_vp,
                    cand_pos, step_id)
                record["ambiguous"] = mirror["ambiguous"]
                COLLECT["graph_records"].append(record)
            except Exception as exc:  # noqa: BLE001
                fail_closed("graph crosscheck error: %r" % exc)
        elif active:
            fail_closed("mirror missing after update_graph")

    def _mirror_mapping(gmap, cur_vp, cur_pos, cand_pos):
        """Sequential mirror of upstream update_graph merge semantics with
        the same loc_noise; returns per-candidate mapping + ambiguity info.
        Raises on any candidate with >1 plausible persistent target."""
        node_live = dict(gmap.node_pos)
        node_live[cur_vp] = np.asarray(cur_pos)
        ghost_mean_live = {k: np.asarray(v).copy()
                           for k, v in gmap.ghost_mean_pos.items()}
        ghost_pos_live = {k: [np.asarray(p).copy() for p in v]
                          for k, v in gmap.ghost_pos.items()}
        ghost_cnt = int(gmap.ghost_cnt)
        loc_noise = float(gmap.loc_noise)
        mappings = []
        ambiguous = []
        multiple_matches = []
        for i, cpos in enumerate(cand_pos):
            cpos = np.asarray(cpos)
            nearest_node, nodes_within, _ = _mirror_localize(
                cpos, node_live, loc_noise)
            if nearest_node is not None:
                if len(nodes_within) > 1:
                    ranked = sorted(nodes_within, key=lambda x: x[1])
                    evidence = {
                        "cand_index": i,
                        "tier": "node",
                        "within": sorted(v for v, _ in nodes_within),
                        "nearest_second_margin_m": round(
                            float(ranked[1][1] - ranked[0][1]), 9),
                        "ranked_matches": [
                            {"persistent_id": str(v),
                             "distance_m": round(float(d), 9)}
                            for v, d in ranked],
                    }
                    multiple_matches.append(evidence)
                    if (COLLECT["identity_protocol"].endswith("/v1") or
                            (COLLECT["identity_protocol"].endswith(
                                "/v2-engineering") and
                             evidence["nearest_second_margin_m"] <=
                             COLLECT["identity_margin_m"])):
                        ambiguous.append(evidence)
                mappings.append({
                    "cand_index": i,
                    "kind": "node",
                    "target": nearest_node,
                    "distance": round(float(min(
                        d for _, d in nodes_within)), 9),
                    "matches_within_loc_noise": sorted(
                        v for v, _ in nodes_within),
                })
                continue
            nearest_ghost, ghosts_within, _ = _mirror_localize(
                cpos, ghost_mean_live, loc_noise)
            if nearest_ghost is not None:
                if len(ghosts_within) > 1:
                    ranked = sorted(ghosts_within, key=lambda x: x[1])
                    evidence = {
                        "cand_index": i,
                        "tier": "ghost",
                        "within": sorted(v for v, _ in ghosts_within),
                        "nearest_second_margin_m": round(
                            float(ranked[1][1] - ranked[0][1]), 9),
                        "ranked_matches": [
                            {"persistent_id": str(v),
                             "distance_m": round(float(d), 9)}
                            for v, d in ranked],
                    }
                    multiple_matches.append(evidence)
                    if (COLLECT["identity_protocol"].endswith("/v1") or
                            (COLLECT["identity_protocol"].endswith(
                                "/v2-engineering") and
                             evidence["nearest_second_margin_m"] <=
                             COLLECT["identity_margin_m"])):
                        ambiguous.append(evidence)
                mappings.append({
                    "cand_index": i,
                    "kind": "ghost_merged",
                    "target": nearest_ghost,
                    "distance": round(float(min(
                        d for _, d in ghosts_within)), 9),
                    "matches_within_loc_noise": sorted(
                        v for v, _ in ghosts_within),
                })
                ghost_pos_live[nearest_ghost].append(cpos.copy())
                ghost_mean_live[nearest_ghost] = np.mean(
                    ghost_pos_live[nearest_ghost], axis=0)
            else:
                gvp = "g%d" % ghost_cnt
                ghost_cnt += 1
                ghost_pos_live[gvp] = [cpos.copy()]
                ghost_mean_live[gvp] = cpos.copy()
                mappings.append({
                    "cand_index": i,
                    "kind": "ghost_created",
                    "target": gvp,
                    "distance": 0.0,
                    "matches_within_loc_noise": [],
                })
        if ambiguous:
            fail_closed("candidate maps to multiple persistent ids: %s"
                        % canonical_json(ambiguous))
        return {"mappings": mappings, "ambiguous": ambiguous,
                "multiple_matches": multiple_matches,
                "ghost_cnt_post": ghost_cnt}

    def _crosscheck_and_record(gmap, mirror, pre_snapshot, cur_vp, cur_pos,
                               cand_vp, cand_pos, step_id):
        ok = True
        problems = []
        # Build the exact ordered ghost-position lists the upstream merge
        # semantics must have produced, then verify them element-wise.
        expected_ghost_pos = {
            gvp: [p.copy() for p in pre_snapshot["ghost_pos"].get(gvp, [])]
            for gvp in pre_snapshot["ghost_ids"]
        }
        for m in sorted(mirror["mappings"], key=lambda x: x["cand_index"]):
            gvp = m["target"]
            if m["kind"] == "ghost_created":
                expected_ghost_pos.setdefault(gvp, []).insert(
                    0, np.asarray(cand_pos[m["cand_index"]]).copy())
            elif m["kind"] == "ghost_merged":
                expected_ghost_pos.setdefault(gvp, []).append(
                    np.asarray(cand_pos[m["cand_index"]]).copy())
        for m in mirror["mappings"]:
            i = m["cand_index"]
            if m["kind"] == "node":
                if not gmap.graph_nx.has_edge(cur_vp, m["target"]):
                    ok = False
                    problems.append("node edge missing for cand %d" % i)
            else:
                if cur_vp not in gmap.ghost_fronts.get(m["target"], []):
                    ok = False
                    problems.append("front %s absent for ghost %s (cand %d)"
                                    % (cur_vp, m["target"], i))
        for gvp, expected in expected_ghost_pos.items():
            actual = gmap.ghost_pos.get(gvp)
            if actual is None:
                ok = False
                problems.append("ghost %s missing in upstream state" % gvp)
                continue
            if len(actual) != len(expected):
                ok = False
                problems.append("ghost %s position count mismatch: "
                                "upstream=%d mirror=%d"
                                % (gvp, len(actual), len(expected)))
                continue
            for a, b in zip(actual, expected):
                if not np.allclose(np.asarray(a), b, atol=0.0, rtol=0.0):
                    ok = False
                    problems.append("ghost %s position sequence mismatch"
                                    % gvp)
                    break
        if len(gmap.ghost_pos) != len(expected_ghost_pos):
            ok = False
            problems.append("ghost id set mismatch: upstream=%s mirror=%s"
                            % (sorted(gmap.ghost_pos.keys()),
                               sorted(expected_ghost_pos.keys())))
        if int(gmap.ghost_cnt) != mirror["ghost_cnt_post"]:
            ok = False
            problems.append("ghost_cnt mismatch: upstream=%d mirror=%d"
                            % (int(gmap.ghost_cnt),
                               mirror["ghost_cnt_post"]))
        if not ok:
            fail_closed("mirror/upstream crosscheck failed: %s"
                        % "; ".join(problems))
        return {
            "step_id": int(step_id),
            "cur_vp": str(cur_vp),
            "loc_noise": float(gmap.loc_noise),
            "pre_node_ids": pre_snapshot["node_ids"],
            "pre_ghost_ids": pre_snapshot["ghost_ids"],
            "post_node_ids": list(gmap.node_pos.keys()),
            "post_ghost_ids": list(gmap.ghost_pos.keys()),
            "mappings": mirror["mappings"],
            "multiple_matches": mirror.get("multiple_matches", []),
            "crosscheck_ok": ok,
            "problems": problems,
        }

    hooked_identify_node._reveal_hooked = True
    hooked_update_graph._reveal_hooked = True
    graph_utils.GraphMap.identify_node = hooked_identify_node
    graph_utils.GraphMap.update_graph = hooked_update_graph


# --------------------------------------------------------------------------
# hash chain assembly
# --------------------------------------------------------------------------
def build_chain(meta):
    """Assemble the per-prefix hash chain from staged hook records."""
    wp = COLLECT["waypoint_records"]
    idr = COLLECT["identify_records"]
    gr = COLLECT["graph_records"]
    trace = meta.get("trace") or []
    poses = meta.get("poses") or []
    counts = {"waypoint": len(wp), "identify": len(idr), "graph": len(gr),
              "action": len(trace), "pose": len(poses)}
    n = min(counts.values())
    alignment_ok = (counts["waypoint"] == counts["identify"]
                    == counts["graph"] == counts["action"])
    if not alignment_ok and COLLECT["status"] == "OK":
        fail_closed("hook record count mismatch: %s" % canonical_json(counts))
        n = 0
    chain = []
    prev_hash = GENESIS_HASH
    for k in range(n):
        w = wp[k]
        ident = idr[k]
        g = gr[k]
        act = trace[k]
        pose = poses[k] if k < len(poses) else None
        cand_identity = {
            "count": w["candidates"]["count"],
            "angles": w["candidates"]["angles"],
            "distances": w["candidates"]["distances"],
            "img_idxes": w["candidates"]["img_idxes"],
        }
        candidate_set_hash = sha256_bytes(
            canonical_json(cand_identity).encode("utf-8"))
        feature_hash = sha256_bytes(canonical_json(
            w["candidates"]["features"]).encode("utf-8"))
        observation_hash = sha256_bytes(
            canonical_json(w["obs_hashes"]).encode("utf-8"))
        graph_mapping_hash = sha256_bytes(canonical_json({
            "cur_vp": g["cur_vp"],
            "mappings": g["mappings"],
            "loc_noise": g["loc_noise"],
        }).encode("utf-8"))
        post = (pose or {}).get("post")
        action_payload = {
            "act": act.get("act"),
            "cur_vp": act.get("cur_vp"),
            "tryout": act.get("tryout"),
            "back_path_len": act.get("back_path_len"),
            "selected_vp": act.get("ghost_vp") or act.get("stop_vp"),
            "front_vp": act.get("front_vp"),
            "done": act.get("done"),
            "reward_q": q9(act.get("reward", 0.0)),
            "post_position_q": ([q3(x) for x in post["position"]]
                                if post else None),
            "post_heading_q": (q3(post["heading"]) if post else None),
        }
        action_hash = sha256_bytes(
            canonical_json(action_payload).encode("utf-8"))
        record = {
            "schema_version": SCHEMA_VERSION,
            "high_level_step": k,
            "agent_pose": {
                "position_q": ident["cur_pos_q"],
                "heading_q": ident["heading_q"],
            },
            "cur_vp": ident["cur_vp"],
            "candidate_vp_ids": ident["cand_vp"],
            "candidate_positions_q": ident["cand_pos_q"],
            "candidates": cand_identity,
            "candidate_feature_hash": feature_hash,
            "observation_hashes": w["obs_hashes"],
            "graph": {
                "cur_vp": g["cur_vp"],
                "loc_noise": g["loc_noise"],
                "pre_node_ids": g["pre_node_ids"],
                "pre_ghost_ids": g["pre_ghost_ids"],
                "post_node_ids": g["post_node_ids"],
                "post_ghost_ids": g["post_ghost_ids"],
                "mappings": g["mappings"],
                "ambiguous": g.get("ambiguous", []),
                "multiple_matches": g.get("multiple_matches", []),
                "identity_protocol": COLLECT["identity_protocol"],
                "identity_nearest_second_margin_m":
                    COLLECT["identity_margin_m"],
                "crosscheck_ok": g["crosscheck_ok"],
            },
            "action": action_payload,
            "observation_hash": observation_hash,
            "candidate_set_hash": candidate_set_hash,
            "action_hash": action_hash,
            "graph_mapping_hash": graph_mapping_hash,
            "previous_record_hash": prev_hash,
        }
        record["current_record_hash"] = sha256_bytes(
            canonical_json(record).encode("utf-8"))
        chain.append(record)
        prev_hash = record["current_record_hash"]
    return chain, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val_seen"])
    ap.add_argument("--task", required=True, choices=["rxr", "r2r"])
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gate-out", required=True)
    ap.add_argument("--media-staging", default=None)
    ap.add_argument("--identity-protocol", choices=sorted(IDENTITY_PROTOCOLS),
                    default="v1")
    args = ap.parse_args()
    if args.task not in TASK_FAMILIES:
        raise SystemExit("unknown task family")
    run_dir = os.path.abspath(args.run_dir)
    gate_out = os.path.abspath(args.gate_out)
    if not run_dir.startswith(PROJECT_ROOT) or \
            not gate_out.startswith(PROJECT_ROOT):
        raise SystemExit("run-dir/gate-out must stay inside the workspace")
    os.makedirs(run_dir, exist_ok=True)
    identity_contract = IDENTITY_PROTOCOLS[args.identity_protocol]
    COLLECT["identity_protocol"] = identity_contract["version"]
    COLLECT["identity_margin_m"] = identity_contract[
        "nearest_second_margin_m"]

    trace_path = os.path.join(run_dir, "trace.jsonl")
    pose_path = os.path.join(run_dir, "pose.jsonl")
    meta_path = os.path.join(run_dir, "episode_meta.json")
    for p in (trace_path, pose_path, meta_path):
        open(p, "w").close()
    os.environ["RXRENG_TRACE_FILE"] = trace_path
    os.environ["RXRENG_POSE_FILE"] = pose_path
    os.environ["RXRENG_EPISODE_META_FILE"] = meta_path
    if args.media_staging:
        os.makedirs(args.media_staging, exist_ok=True)
        COLLECT["media_staging"] = args.media_staging

    install_waypoint_hook()
    install_graphmap_hook()
    COLLECT["active"] = True

    os.chdir(ETPR1_ROOT)
    from etpr1_compat import configure_project_cache_env  # noqa: E402

    configure_project_cache_env()

    fam = TASK_FAMILIES[args.task]
    langs = "['en-US','en-IN']"
    argv = [
        "run.py",
        "--exp_name", args.exp_name,
        "--run-type", "eval",
        "--exp-config", fam["exp_config"],
        "EVAL.SPLIT", args.split,
        "TASK_CONFIG.DATASET.SPLIT", args.split,
    ]
    if fam["languages"]:
        argv += [
            "EVAL.LANGUAGES", langs,
            "TASK_CONFIG.DATASET.LANGUAGES", langs,
        ]
    argv += [
        "EVAL.EPISODE_ID", "['%s']" % args.episode_id,
        "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", fam["ckpt"],
        "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", JOINT_PRETRAINED,
        "IL.back_algo", "control",
    ]
    argv += fam["extra_opts"]
    argv += [
        "INFERENCE.SPLIT", args.split,
        "GPU_NUMBERS", "1",
        "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]",
        "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0",
        "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "VIDEO_OPTION", "[]",
        "TENSORBOARD_DIR", gate_out + "/tensorboard/",
        "CHECKPOINT_FOLDER", gate_out + "/checkpoints/",
        "RESULTS_DIR", gate_out + "/results/",
    ]

    summary = {
        "collector": "collect_reveal_prefixes.py",
        "schema_version": SCHEMA_VERSION,
        "genesis_hash": GENESIS_HASH,
        "episode_id": args.episode_id,
        "split": args.split,
        "task": args.task,
        "exp_name": args.exp_name,
        "argv": argv,
        "instrumentation": [
            "waypoint hook: ETP.forward(mode='waypoint') outputs + obs hashes",
            "graph hook: GraphMap.identify_node/update_graph mirror+crosscheck",
            "action trace: VLNCEDaggerEnv.step (accepted R2R-gate recorder)",
            "post-execution pose records (env child)",
        ],
        "canonicalization": {
            "positions_m": 1e-3,
            "headings_rad": 1e-3,
            "candidate_angles_distances": 1e-9,
            "tensor_hashes": "sha256 over raw bytes",
            "candidate_order": "upstream NMS nonzero() row-major order",
        },
        "identity_contract": identity_contract,
    }

    sys.argv = argv
    import run  # frozen ETP-R1 entrypoint  # noqa: E402

    t0 = time.monotonic()
    exit_status = "OK"
    error_text = None
    try:
        run.main()
    except BaseException as exc:  # noqa: BLE001
        exit_status = "EXCEPTION"
        error_text = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        COLLECT["active"] = False
        summary["exit_status"] = exit_status
        summary["error"] = error_text
        summary["wall_time_s"] = round(time.monotonic() - t0, 3)
        summary["peak_rss_self_kib"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss

        def read_jsonl(path):
            try:
                with open(path) as fh:
                    return [json.loads(ln) for ln in fh if ln.strip()]
            except OSError:
                return []

        trace = read_jsonl(trace_path)
        poses = read_jsonl(pose_path)
        meta = {"trace": trace, "poses": poses}
        episode_meta = {}
        try:
            with open(meta_path) as fh:
                lines = [json.loads(ln) for ln in fh if ln.strip()]
            if lines:
                episode_meta = lines[0]
        except (OSError, json.JSONDecodeError):
            pass
        summary["episode_meta"] = episode_meta

        chain, counts = build_chain(meta)
        chain_path = os.path.join(run_dir, "reveal_prefix_chain.jsonl")
        with open(chain_path, "w") as fh:
            for rec in chain:
                fh.write(json.dumps(rec) + "\n")
        root = chain[-1]["current_record_hash"] if chain else None
        summary["hook_record_counts"] = counts
        summary["prefix_count"] = len(chain)
        summary["chain_root"] = root
        summary["collect_status"] = COLLECT["status"]
        summary["collect_status_reason"] = COLLECT["status_reason"]
        with open(os.path.join(run_dir, "COLLECT_SUMMARY.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        with open(os.path.join(run_dir, "CHAIN_ROOT.txt"), "w") as fh:
            fh.write((root or "EMPTY") + "\n")


if __name__ == "__main__":
    main()
