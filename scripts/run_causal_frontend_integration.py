#!/usr/bin/env python3
"""Parallel matched-run orchestrator for the real causal frontend gate."""

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
                    "causal_model_integration")
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "CAUSAL_FRONTEND_MODEL_INTEGRATION.json")
EPISODE = "43629"
GPUS = {"original": 1, "adversarial": 3}
CHECKPOINTS = {
    "rxr_final": (
        "third_party/ETP-R1/data/logs/checkpoints/release_rxr_grpo/store/"
        "ckpt.iter1320.pth",
        "3796c9c94ff8674b8cfe99f2b4aab0f4b391f0d4c9c1e167e4736b3848f27821"),
    "joint": (
        "third_party/ETP-R1/pretrained/r2r_rxr_ce/"
        "mlm.sap_habitat_depth/store2/model_step_367500.pt",
        "203fe62cc22c63261a5c5b6a3638bc52fd3b08a7f09dd31d8539bf2beab6c3cf"),
    "waypoint_hfov63": (
        "third_party/ETP-R1/data/wp_pred/check_cwp_bestdist_hfov63",
        "6796087c9a37c845fa6094002bcbfeccec29dd5d37916f472c52fc90d843f56e"),
}


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        .encode()).hexdigest()


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_variant(variant):
    run_dir = os.path.join(BASE, variant)
    gate_out = os.path.join(run_dir, "gate_out")
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
               CR1_HIDDEN_VARIANT=variant, CR1_RUN_DIR=run_dir)
    cmd = [PYTHON, WORKER, "--episode-id", EPISODE, "--split", "train",
           "--task", "rxr", "--exp-name", "cr1_causal_" + variant,
           "--run-dir", run_dir, "--gate-out", gate_out,
           "--identity-protocol", "v3"]
    started = time.monotonic()
    with open(os.path.join(run_dir, "stdout.log"), "w") as stdout, \
            open(os.path.join(run_dir, "stderr.log"), "w") as stderr:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout,
                              stderr=stderr)
    attempts = 0
    guard_installed = False
    with open(netguard) as fh:
        for line in fh:
            record = json.loads(line)
            attempts = max(attempts, int(record.get("attempts") or 0))
            guard_installed |= (record.get("role") == "env_child_step" and
                                record.get("guard_installed") is True)
    return {"variant": variant, "gpu": GPUS[variant],
            "returncode": proc.returncode,
            "wall_time_s": round(time.monotonic() - started, 3),
            "network_attempts": attempts,
            "child_network_guard_evidenced": guard_installed}


def main():
    os.makedirs(BASE, exist_ok=True)
    verified = {}
    for name, (relative, expected) in CHECKPOINTS.items():
        observed = sha256_file(os.path.join(ROOT, relative))
        if observed != expected:
            raise SystemExit("checkpoint SHA drift: " + name)
        verified[name] = {"path": relative, "sha256": observed}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(run_variant, ("original", "adversarial")))
    runs.sort(key=lambda item: item["variant"])
    if not all(item["returncode"] == 0 for item in runs):
        print(json.dumps(runs, indent=2))
        raise SystemExit("matched model run failed")

    artifacts = {}
    for variant in ("original", "adversarial"):
        run_dir = os.path.join(BASE, variant)
        artifacts[variant] = {
            "model_trace": read_jsonl(os.path.join(
                run_dir, "CAUSAL_MODEL_TRACE.jsonl")),
            "action_trace": read_jsonl(os.path.join(run_dir,
                                                     "trace.jsonl")),
            "reveal_chain": read_jsonl(os.path.join(
                run_dir, "reveal_prefix_chain.jsonl")),
            "perturbation": json.load(open(os.path.join(
                run_dir, "PERTURBATION_EVIDENCE.json"))),
            "summary": json.load(open(os.path.join(
                run_dir, "COLLECT_SUMMARY.json"))),
        }
    left, right = artifacts["original"], artifacts["adversarial"]
    model_exact = left["model_trace"] == right["model_trace"]
    actions_exact = left["action_trace"] == right["action_trace"]
    chains_exact = left["reveal_chain"] == right["reveal_chain"]
    original_perturb = left["perturbation"]["records"]
    adversarial_perturb = right["perturbation"]["records"]
    perturbation_exercised = (
        len(original_perturb) > 0 and
        len(original_perturb) == len(adversarial_perturb) and
        all(a["original_hidden_aggregate_sha256"] ==
            b["original_hidden_aggregate_sha256"] and
            a["source_differs_from_original"] is False and
            b["source_differs_from_original"] is True and
            a["source_before_mask_aggregate_sha256"] !=
            b["source_before_mask_aggregate_sha256"]
            for a, b in zip(original_perturb, adversarial_perturb)))
    expected_modes = {"waypoint", "panorama", "navigation"}
    modes = {record["mode"] for record in left["model_trace"]}
    policy_chain_complete = expected_modes <= modes and len(
        left["action_trace"]) > 0
    summaries_ok = all(
        item["summary"].get("exit_status") == "OK" and
        item["summary"].get("collect_status") == "OK"
        for item in artifacts.values())
    network_zero = all(item["network_attempts"] == 0 and
                       item["child_network_guard_evidenced"]
                       for item in runs)
    checks = {
        "checkpoint_provenance_verified": True,
        "matched_runs_exit_and_collect_ok": summaries_ok,
        "adversarial_hidden_perturbation_exercised":
            perturbation_exercised,
        "waypoint_panorama_navigation_chain_present":
            policy_chain_complete,
        "candidate_panorama_and_global_logits_bit_exact": model_exact,
        "policy_action_trace_bit_exact": actions_exact,
        "reveal_graph_chain_bit_exact": chains_exact,
        "network_attempts_zero_with_child_guard": network_zero,
    }
    passed = all(checks.values())
    output = {
        "gate": "mf2_cr1_real_model_hidden_view_integration",
        "revision": "causal-model-integration/1",
        "status": "PASS" if passed else "FAIL",
        "decision": "REAL_MODEL_HIDDEN_VIEW_NONINTERFERENCE_PASS" if passed
                    else "REAL_MODEL_HIDDEN_VIEW_NO_GO",
        "scope": "RxR-train episode engineering integration; shared "
                 "single-front mask; no INSPECT policy learned or claimed",
        "episode_id": EPISODE,
        "physical_gpus": GPUS,
        "checkpoints": verified,
        "runs": runs,
        "checks": checks,
        "comparison": {
            "model_trace_records": len(left["model_trace"]),
            "action_records": len(left["action_trace"]),
            "reveal_chain_records": len(left["reveal_chain"]),
            "model_trace_canonical_sha256":
                canonical_sha(left["model_trace"]),
            "action_trace_canonical_sha256":
                canonical_sha(left["action_trace"]),
            "reveal_chain_canonical_sha256":
                canonical_sha(left["reveal_chain"]),
        },
        "non_conclusions": {
            "multi_view_inspect_runtime_complete": False,
            "semantic_branch_validated": False,
            "training_authorized": False,
            "benchmark_result": False,
            "val_unseen_or_test_used": False,
            "frozen_source_modified": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "checks": checks, "runs": runs,
        "comparison": output["comparison"],
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
