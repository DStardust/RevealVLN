#!/usr/bin/env python3
"""Matched real-model test for the post-INSPECT two-view buffer state."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time


ROOT = "/mnt/daiyang/vla"
PYTHON = os.path.join(ROOT, ".envs", "etpr1", "bin", "python")
WORKER = os.path.join(ROOT, "scripts", "causal_frontend_model_worker.py")
BASE = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                    "causal_multiview_integration")
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "CAUSAL_FRONTEND_MULTIVIEW_INTEGRATION.json")
PHYSICAL_GATE = os.path.join(ROOT, "artifacts", "runtime",
                             "phase0_correctness",
                             "PHYSICAL_INSPECT_ACQUISITION_GATE.json")
EXPECTED_PHYSICAL_SHA = \
    "d76362431b05a962b0569915f82d45db1fe05e014afe878c34d3f8c5e8f0d93a"
GPUS = {"original": 4, "adversarial": 5}


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_variant(variant):
    run_dir = os.path.join(BASE, variant)
    os.makedirs(run_dir, exist_ok=True)
    netguard = os.path.join(run_dir, "netguard.jsonl")
    open(netguard, "w").close()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPUS[variant]),
               PYTHONPATH=os.pathsep.join([
                   ROOT, os.path.join(ROOT, "third_party", "ETP-R1"),
                   os.path.join(ROOT, "third_party", "habitat-lab"),
                   os.path.join(ROOT, "third_party", "habitat-sim")]),
               PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_HOME=os.path.join(ROOT, ".cache", "torch"),
               HF_HOME=os.path.join(ROOT, ".cache", "huggingface"),
               TRANSFORMERS_CACHE=os.path.join(
                   ROOT, ".cache", "huggingface", "transformers"),
               CLIP_DOWNLOAD_ROOT=os.path.join(ROOT, ".cache", "clip"),
               RXREN_NETGUARD_FILE=netguard,
               CR1_HIDDEN_VARIANT=variant, CR1_RUN_DIR=run_dir,
               CR1_ACQUIRED_SLOTS="0,11")
    cmd = [PYTHON, WORKER, "--episode-id", "43629", "--split", "train",
           "--task", "rxr", "--exp-name", "cr1_two_view_" + variant,
           "--run-dir", run_dir, "--gate-out",
           os.path.join(run_dir, "gate_out"), "--identity-protocol", "v3"]
    start = time.monotonic()
    with open(os.path.join(run_dir, "stdout.log"), "w") as stdout, \
            open(os.path.join(run_dir, "stderr.log"), "w") as stderr:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout,
                              stderr=stderr)
    attempts, child = 0, False
    with open(netguard) as fh:
        for line in fh:
            record = json.loads(line)
            attempts = max(attempts, int(record.get("attempts") or 0))
            child |= (record.get("role") == "env_child_step" and
                      record.get("guard_installed") is True)
    return {"variant": variant, "gpu": GPUS[variant],
            "returncode": proc.returncode,
            "wall_time_s": round(time.monotonic() - start, 3),
            "network_attempts": attempts, "child_guard": child}


def main():
    physical_sha = sha256_file(PHYSICAL_GATE)
    if EXPECTED_PHYSICAL_SHA.startswith("TO_BE_"):
        raise SystemExit("pin physical gate SHA before running")
    if physical_sha != EXPECTED_PHYSICAL_SHA:
        raise SystemExit("physical acquisition gate SHA drift")
    physical = json.load(open(PHYSICAL_GATE))
    if physical.get("status") != "PASS" or physical["measurements"][
            "acquired_relative_slots_after_turn"] != [0, 11]:
        raise SystemExit("physical acquisition prerequisite failed")
    os.makedirs(BASE, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(run_variant, ("original", "adversarial")))
    runs.sort(key=lambda value: value["variant"])
    if not all(value["returncode"] == 0 for value in runs):
        print(json.dumps(runs, indent=2))
        raise SystemExit("two-view matched run failed")

    data = {}
    for variant in ("original", "adversarial"):
        base = os.path.join(BASE, variant)
        data[variant] = {
            "model": read_jsonl(os.path.join(base,
                                              "CAUSAL_MODEL_TRACE.jsonl")),
            "actions": read_jsonl(os.path.join(base, "trace.jsonl")),
            "graph": read_jsonl(os.path.join(
                base, "reveal_prefix_chain.jsonl")),
            "perturb": json.load(open(os.path.join(
                base, "PERTURBATION_EVIDENCE.json"))),
            "summary": json.load(open(os.path.join(base,
                                                    "COLLECT_SUMMARY.json"))),
        }
    a, b = data["original"], data["adversarial"]
    paired_perturb = zip(a["perturb"]["records"],
                         b["perturb"]["records"])
    perturb_exercised = len(a["perturb"]["records"]) > 0 and all(
        not x["source_differs_from_original"] and
        y["source_differs_from_original"] and
        x["source_before_mask_aggregate_sha256"] !=
        y["source_before_mask_aggregate_sha256"]
        for x, y in paired_perturb)
    checks = {
        "physical_turn_yields_slots_0_11": True,
        "both_real_model_runs_ok": all(
            value["summary"].get("exit_status") == "OK" and
            value["summary"].get("collect_status") == "OK"
            for value in data.values()),
        "two_acquired_slots_used_every_waypoint_call": all(
            record.get("acquired_slots") == [0, 11]
            for record in a["model"] if record["mode"] == "waypoint"),
        "remaining_hidden_views_adversarially_changed": perturb_exercised,
        "model_chain_bit_exact": a["model"] == b["model"],
        "action_chain_bit_exact": a["actions"] == b["actions"],
        "graph_chain_bit_exact": a["graph"] == b["graph"],
        "network_zero": all(value["network_attempts"] == 0 and
                            value["child_guard"] for value in runs),
    }
    passed = all(checks.values())
    output = {
        "gate": "mf2_cr1_post_inspect_two_view_real_model",
        "revision": "causal-multiview-integration/1",
        "status": "PASS" if passed else "FAIL",
        "decision": "POST_INSPECT_TWO_VIEW_PASS" if passed else
                    "POST_INSPECT_TWO_VIEW_NO_GO",
        "physical_acquisition_input": {"path": os.path.relpath(
            PHYSICAL_GATE, ROOT), "sha256": physical_sha},
        "acquired_relative_slots": [0, 11],
        "runs": runs,
        "checks": checks,
        "counts": {"model_records": len(a["model"]),
                   "action_records": len(a["actions"]),
                   "graph_records": len(a["graph"])},
        "non_conclusions": {
            "learned_inspect_policy": False,
            "semantic_branch_validated": False,
            "training_authorized": False,
            "benchmark_result": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({"status": output["status"],
                      "decision": output["decision"], "checks": checks,
                      "runs": runs, "counts": output["counts"],
                      "output": os.path.relpath(OUT, ROOT),
                      "output_sha256": sha256_file(OUT)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
