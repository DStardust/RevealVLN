#!/usr/bin/env python3
"""Parallel, pinned-GPU v2/v3 rerun of 34 v1-truncated RxR train traces.

The input set is exactly the accepted Stage-4 items with status AMBIGUOUS;
there is no resampling.  Each task uses run_reveal_collect.py in eval mode
with persistent-branch-identity/v2-engineering.  Existing accepted v1 runs
are read-only and every v2 run is written to phase0_correctness/collect_v2.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import threading
from queue import Queue


ROOT = "/mnt/daiyang/vla"
PYBIN = os.path.join(ROOT, ".envs", "etpr1", "bin", "python")
RUNNER = os.path.join(ROOT, "scripts", "run_reveal_collect.py")
VALIDATOR = os.path.join(ROOT, "scripts", "validate_reveal_prefix_trace.py")
STAGE4 = os.path.join(ROOT, "artifacts", "runtime",
                      "phase0_reveal_closure",
                      "STAGE4_TRACE_GENERATION_SUMMARY.json")
AUDIT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                     "CANDIDATE_IDENTITY_AUDIT.json")
EXPECTED_STAGE4_SHA = \
    "7f3fe38842c38acfe856a19a1feac0aabf134e8cd487f9630a01eaac6e4d7ee9"
EXPECTED_AUDIT_SHA = \
    "b4a815d41c830b748db18f9f8cabfd7001240870e841506456ace45ee4e4b9fb"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def gpu_inventory():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,"
         "memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    result = []
    for line in out.strip().splitlines():
        i, free, used, total = [int(x.strip()) for x in line.split(",")]
        result.append({"index": i, "free_mib": free, "used_mib": used,
                       "total_mib": total})
    return result


def run_one(item, gpu_queue, print_lock, protocol, out_root, gate_out):
    order = int(item["queue_order"])
    eid = str(item["episode_id"])
    run_dir = os.path.join(out_root, "rpc%s_%02d_ep%s" %
                          (protocol, order, eid))
    os.makedirs(run_dir, exist_ok=True)
    gpu = gpu_queue.get()
    try:
        cmd = [
            PYBIN, RUNNER,
            "--episode-id", eid,
            "--split", "train",
            "--task", "rxr",
            "--exp-name", "rpc%s_%02d_ep%s" % (protocol, order, eid),
            "--run-dir", run_dir,
            "--gate-out", gate_out,
            "--identity-protocol", protocol,
            "--gpu-index", str(gpu),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        with open(os.path.join(run_dir, "batch_runner_stdout.log"), "w") as fh:
            fh.write(proc.stdout)
        with open(os.path.join(run_dir, "batch_runner_stderr.log"), "w") as fh:
            fh.write(proc.stderr)
        with print_lock:
            print("[%02d] ep%s gpu=%d rc=%d" %
                  (order, eid, gpu, proc.returncode), flush=True)
        return {"queue_order": order, "episode_id": eid,
                "gpu_index_assigned": gpu, "returncode": proc.returncode,
                "run_dir": run_dir}
    finally:
        gpu_queue.put(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,3,4,7")
    ap.add_argument("--protocol", choices=["v2", "v3"], default="v2")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="validate already completed standardized run dirs")
    args = ap.parse_args()
    protocol = args.protocol
    out_root = os.path.join(ROOT, "artifacts", "runtime",
                            "phase0_correctness", "collect_" + protocol)
    gate_out = os.path.join(ROOT, "artifacts", "runtime",
                            "phase0_correctness", "collect_gate_" + protocol)
    summary_path = os.path.join(
        ROOT, "artifacts", "runtime", "phase0_correctness",
        "IDENTITY_%s_RERUN_SUMMARY.json" % protocol.upper())
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit("GPU list must be nonempty and unique")
    inventory = gpu_inventory()
    by_id = {g["index"]: g for g in inventory}
    bad = [g for g in gpu_ids if g not in by_id or
           by_id[g]["free_mib"] < 10240]
    if bad:
        raise SystemExit("requested GPUs unavailable or below 10 GiB: %s" % bad)
    if sha256_file(STAGE4) != EXPECTED_STAGE4_SHA:
        raise SystemExit("Stage-4 baseline SHA drift")
    if sha256_file(AUDIT) != EXPECTED_AUDIT_SHA:
        raise SystemExit("candidate identity audit SHA drift")
    audit = load_json(AUDIT)
    if audit.get("decision") != \
            "REVISION_REQUIRED_QUANTIZATION_UNRESOLVED_TIES":
        raise SystemExit("unexpected identity adjudication decision")
    stage4 = load_json(STAGE4)
    selected = [x for x in stage4["items"] if x["status"] == "AMBIGUOUS"]
    if len(selected) != 34:
        raise SystemExit("expected exactly 34 v1-ambiguous inputs")

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(gate_out, exist_ok=True)
    gpu_queue = Queue()
    for gpu in gpu_ids:
        gpu_queue.put(gpu)
    print_lock = threading.Lock()
    raw = []
    if args.aggregate_only:
        for item in selected:
            order = int(item["queue_order"])
            eid = str(item["episode_id"])
            run_dir = os.path.join(out_root, "rpc%s_%02d_ep%s" %
                                   (protocol, order, eid))
            orch_path = os.path.join(run_dir, "ORCHESTRATOR_RESULT.json")
            orch = load_json(orch_path) if os.path.isfile(orch_path) else {}
            raw.append({"queue_order": order, "episode_id": eid,
                        "gpu_index_assigned": orch.get("gpu_index"),
                        "returncode": orch.get("returncode", 999),
                        "run_dir": run_dir})
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(run_one, item, gpu_queue, print_lock,
                                   protocol, out_root, gate_out)
                       for item in selected]
            for future in concurrent.futures.as_completed(futures):
                raw.append(future.result())

    raw.sort(key=lambda x: x["queue_order"])
    results = []
    for item in raw:
        run_dir = item["run_dir"]
        collect_path = os.path.join(run_dir, "COLLECT_SUMMARY.json")
        orch_path = os.path.join(run_dir, "ORCHESTRATOR_RESULT.json")
        chain_path = os.path.join(run_dir, "reveal_prefix_chain.jsonl")
        old_dir = os.path.join(
            ROOT, "artifacts", "runtime", "phase0_reveal_closure", "collect",
            "rpc50_%02d_ep%s" % (item["queue_order"], item["episode_id"]))
        old_chain_path = os.path.join(old_dir, "reveal_prefix_chain.jsonl")
        problems = []
        if item["returncode"] != 0:
            problems.append("runner_nonzero")
        for p in (collect_path, orch_path, chain_path, old_chain_path):
            if not os.path.isfile(p) or os.path.islink(p):
                problems.append("missing_or_symlink:" + os.path.relpath(p, ROOT))
        collect = load_json(collect_path) if os.path.isfile(collect_path) else {}
        orch = load_json(orch_path) if os.path.isfile(orch_path) else {}
        chain_valid = False
        v2_chain = []
        if os.path.isfile(chain_path):
            verify = subprocess.run(
                [PYBIN, VALIDATOR, "verify", "--chain", chain_path],
                cwd=ROOT, capture_output=True, text=True)
            chain_valid = verify.returncode == 0
            if not chain_valid:
                problems.append("hash_chain_invalid")
            v2_chain = load_jsonl(chain_path)
        old_chain = load_jsonl(old_chain_path) \
            if os.path.isfile(old_chain_path) else []
        common = min(len(old_chain), len(v2_chain))
        common_equal = common == len(old_chain) and all(
            old_chain[k]["candidates"] == v2_chain[k]["candidates"]
            and old_chain[k]["graph"]["mappings"] ==
                v2_chain[k]["graph"]["mappings"]
            and old_chain[k]["action"] == v2_chain[k]["action"]
            for k in range(common))
        if not common_equal:
            problems.append("v1_v2_common_prefix_behavior_drift")
        if collect.get("identity_contract", {}).get("version") != \
                "persistent-branch-identity/%s-engineering" % protocol:
            problems.append("wrong_identity_contract")
        multiple_match_count = 0
        exact_evidence_count = 0
        margins = []
        tiers = []
        for rec in v2_chain:
            mappings = {int(m["cand_index"]): m
                        for m in rec["graph"]["mappings"]}
            multi = rec["graph"].get("multiple_matches", [])
            by_cand = {int(m["cand_index"]): m for m in multi}
            expected_multi = {
                int(m["cand_index"]) for m in mappings.values()
                if len(m.get("matches_within_loc_noise", [])) > 1
            }
            if set(by_cand) != expected_multi:
                problems.append("multiple_match_evidence_set_mismatch")
            for cand_i, evidence in by_cand.items():
                multiple_match_count += 1
                margins.append(float(evidence[
                    "nearest_second_margin_m"]))
                tiers.append(evidence["tier"])
                ranked = evidence.get("ranked_matches") or []
                mapping = mappings[cand_i]
                exact_ok = (
                    len(ranked) >= 2
                    and [x["distance_m"] for x in ranked] == sorted(
                        x["distance_m"] for x in ranked)
                    and set(x["persistent_id"] for x in ranked) == set(
                        mapping["matches_within_loc_noise"])
                    and ranked[0]["persistent_id"] == mapping["target"]
                    and abs(float(ranked[0]["distance_m"]) -
                            float(mapping["distance"])) <= 1e-9
                    and abs((float(ranked[1]["distance_m"]) -
                             float(ranked[0]["distance_m"])) -
                            float(evidence[
                                "nearest_second_margin_m"])) <= 2e-9
                )
                exact_evidence_count += int(exact_ok)
                if protocol == "v3" and not exact_ok:
                    problems.append("v3_exact_ranked_evidence_invalid")
        if orch.get("network_attempts_all_processes") != 0 or \
                orch.get("network_guard_child_evidence_ok") is not True:
            problems.append("network_guard_failure")
        if orch.get("gpu_memory_returned_within_band") is not True:
            problems.append("gpu_memory_not_returned")
        results.append({
            "queue_order": item["queue_order"],
            "episode_id": item["episode_id"],
            "scene_id": next(x["scene_id"] for x in selected
                             if int(x["queue_order"]) == item["queue_order"]),
            "gpu_index": orch.get("gpu_index"),
            "returncode": item["returncode"],
            "collect_status": collect.get("collect_status"),
            "collect_status_reason": collect.get("collect_status_reason"),
            "v1_prefix_count": len(old_chain),
            "v2_prefix_count": len(v2_chain),
            "prefixes_recovered": len(v2_chain) - len(old_chain),
            "v1_v2_common_prefix_behavior_equal": common_equal,
            "chain_valid": chain_valid,
            "chain_root": collect.get("chain_root"),
            "chain_sha256": sha256_file(chain_path)
            if os.path.isfile(chain_path) else None,
            "network_attempts": orch.get("network_attempts_all_processes"),
            "gpu_memory_returned":
                orch.get("gpu_memory_returned_within_band"),
            "run_dir": os.path.relpath(run_dir, ROOT),
            "multiple_match_count": multiple_match_count,
            "exact_ranked_evidence_count": exact_evidence_count,
            "minimum_nearest_second_margin_m": min(margins)
            if margins else None,
            "multiple_match_tier_counts": {
                "node": tiers.count("node"),
                "ghost": tiers.count("ghost"),
            },
            "problems": problems,
        })

    counts = {
        "selected_v1_ambiguous": len(results),
        "rerun_protocol": protocol,
        "rerun_ok": sum(x["collect_status"] == "OK" for x in results),
        "rerun_ambiguous": sum(x["collect_status"] == "AMBIGUOUS"
                               for x in results),
        "rerun_failed_or_missing": sum(bool(x["problems"])
                                       for x in results),
        "v1_total_prefixes": sum(x["v1_prefix_count"] for x in results),
        "rerun_total_prefixes": sum(x["v2_prefix_count"] for x in results),
        "prefixes_recovered": sum(x["prefixes_recovered"] for x in results),
        "full_50_traces_after_reusing_16_v1_ok":
            16 + sum(x["collect_status"] == "OK" for x in results),
        "multiple_match_candidates_recorded": sum(
            x["multiple_match_count"] for x in results),
        "exact_ranked_evidence_records": sum(
            x["exact_ranked_evidence_count"] for x in results),
        "node_multiple_matches": sum(
            x["multiple_match_tier_counts"]["node"] for x in results),
        "ghost_multiple_matches": sum(
            x["multiple_match_tier_counts"]["ghost"] for x in results),
    }
    integrity = (len(results) == 34 and
                 all(not x["problems"] for x in results) and
                 all(x["collect_status"] in {"OK", "AMBIGUOUS"}
                     for x in results))
    summary = {
        "gate": "phase0b_identity_%s_parallel_rerun" % protocol,
        "revision": "persistent-branch-identity/%s-engineering" % protocol,
        "status": "ENGINEERING_PASS" if integrity else "FAIL",
        "scientific_status": "IDENTITY_TRACE_CLOSURE_COMPLETE"
        if counts["rerun_ambiguous"] == 0 and integrity else
        "IDENTITY_TRACE_CLOSURE_PARTIAL",
        "input_contract": {
            "stage4_path": os.path.relpath(STAGE4, ROOT),
            "stage4_sha256": sha256_file(STAGE4),
            "candidate_identity_audit_path": os.path.relpath(AUDIT, ROOT),
            "candidate_identity_audit_sha256": sha256_file(AUDIT),
            "selection": "exactly the 34 accepted Stage-4 AMBIGUOUS items; "
                         "frozen order; no resampling",
        },
        "parallel_execution": {
            "gpu_ids": gpu_ids,
            "max_workers": len(gpu_ids),
            "preflight_inventory": inventory,
            "one_explicit_physical_gpu_per_active_task": True,
        },
        "counts": counts,
        "results": results,
        "non_conclusions": {
            "semantic_branch_identity_established": False,
            "reveal_event_validity_established": False,
            "unique_tx_established": False,
            "training_authorized": False,
            "human_fields_filled": False,
            "frozen_spec_modified": False,
            "val_unseen_or_test_used": False,
        },
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": summary["status"],
        "scientific_status": summary["scientific_status"],
        "counts": counts,
        "summary": os.path.relpath(summary_path, ROOT),
        "summary_sha256": sha256_file(summary_path),
    }, indent=2))
    return 0 if integrity else 1


if __name__ == "__main__":
    raise SystemExit(main())
