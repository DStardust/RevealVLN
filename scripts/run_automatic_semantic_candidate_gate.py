#!/usr/bin/env python3
"""Seven-GPU orchestration and acceptance for automatic semantic tracks."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from collections import Counter


ROOT = "/mnt/daiyang/vla"
PYTHON = os.path.join(ROOT, ".envs", "etpr1", "bin", "python")
WORKER = os.path.join(ROOT, "scripts",
                      "automatic_semantic_candidate_worker.py")
BASE = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                    "automatic_semantic_shards")
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "AUTOMATIC_SEMANTIC_CANDIDATE_GATE.json")
GPUS = (1, 2, 3, 4, 5, 6, 7)
SHARDS = len(GPUS)
MIN_EVENTS = 15
MIN_SCENES = 10
CHECKPOINTS = {
    "rxr_final": (
        "third_party/ETP-R1/data/logs/checkpoints/release_rxr_grpo/store/"
        "ckpt.iter1320.pth",
        "3796c9c94ff8674b8cfe99f2b4aab0f4b391f0d4c9c1e167e4736b3848f27821"),
    "joint": (
        "third_party/ETP-R1/pretrained/r2r_rxr_ce/"
        "mlm.sap_habitat_depth/store2/model_step_367500.pt",
        "203fe62cc22c63261a5c5b6a3638bc52fd3b08a7f09dd31d8539bf2beab6c3cf"),
    "waypoint": (
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


def run_shard(index):
    gpu = GPUS[index]
    output = os.path.join(BASE, "shard_%02d.json" % index)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=os.pathsep.join([
                   ROOT, os.path.join(ROOT, "third_party", "ETP-R1"),
                   os.path.join(ROOT, "third_party", "habitat-lab"),
                   os.path.join(ROOT, "third_party", "habitat-sim"),
                   os.path.join(ROOT, "scripts")]),
               PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_HOME=os.path.join(ROOT, ".cache", "torch"),
               HF_HOME=os.path.join(ROOT, ".cache", "huggingface"),
               TRANSFORMERS_CACHE=os.path.join(
                   ROOT, ".cache", "huggingface", "transformers"),
               CLIP_DOWNLOAD_ROOT=os.path.join(ROOT, ".cache", "clip"))
    cmd = [PYTHON, WORKER, "--shard-index", str(index),
           "--shard-count", str(SHARDS), "--output", output]
    started = time.monotonic()
    with open(os.path.join(BASE, "shard_%02d.stdout" % index), "w") as o, \
            open(os.path.join(BASE, "shard_%02d.stderr" % index), "w") as e:
        proc = subprocess.run(cmd, cwd=os.path.join(ROOT, "third_party",
                                                   "ETP-R1"), env=env,
                              stdout=o, stderr=e)
    return {"shard": index, "gpu": gpu, "returncode": proc.returncode,
            "wall_time_s": round(time.monotonic() - started, 3),
            "output": os.path.relpath(output, ROOT)}


def main():
    os.makedirs(BASE, exist_ok=True)
    provenance = {}
    for name, (relative, expected) in CHECKPOINTS.items():
        observed = sha256_file(os.path.join(ROOT, relative))
        if observed != expected:
            raise SystemExit("checkpoint SHA drift: " + name)
        provenance[name] = {"path": relative, "sha256": observed}
    with concurrent.futures.ThreadPoolExecutor(max_workers=SHARDS) as pool:
        runs = list(pool.map(run_shard, range(SHARDS)))
    runs.sort(key=lambda value: value["shard"])
    if not all(value["returncode"] == 0 for value in runs):
        print(json.dumps(runs, indent=2))
        raise SystemExit("automatic semantic shard failed")
    events, shard_summaries = [], []
    for run in runs:
        data = json.load(open(os.path.join(ROOT, run["output"])))
        if data.get("network_attempts") != 0:
            raise SystemExit("network attempt in shard")
        events.extend(data["events"])
        shard_summaries.append(data["counts"])
    events.sort(key=lambda value: value["provisional_event_id"])
    ids = [event["provisional_event_id"] for event in events]
    if len(ids) != 90 or len(set(ids)) != 90:
        raise SystemExit("automatic event cardinality mismatch")
    tracked = [event for event in events if event["status"] == "TRACKED_K3"]
    scenes = {event["scene_id"] for event in tracked}
    reasons = Counter(reason for event in events
                      for reason in event["reasons"])
    floor = len(tracked) >= MIN_EVENTS and len(scenes) >= MIN_SCENES
    ambiguity_zero = all(not event["reasons"] for event in tracked)
    passed = floor and ambiguity_zero
    output = {
        "gate": "mf2_cr1_automatic_candidate_semantic_track",
        "revision": "automatic-semantic-candidate/1",
        "status": "PASS" if passed else "FAIL",
        "decision": "AUTOMATIC_SEMANTIC_TRACK_SUBGATE_PASS" if passed else
                    "AUTOMATIC_SEMANTIC_TRACK_NO_GO",
        "scope": "RxR-train frozen queue, machine-geometric events only, "
                 "single-front causal automatic ETP-R1 waypoint frontend",
        "fixed_gate": {"minimum_tracked_k3_events": MIN_EVENTS,
                       "minimum_scenes": MIN_SCENES,
                       "no_resampling": True,
                       "no_threshold_tuning": True},
        "checkpoint_provenance": provenance,
        "parallel_execution": {"physical_gpus": list(GPUS),
                               "runs": runs,
                               "shard_summaries": shard_summaries},
        "counts": {"machine_geometric_inputs": len(events),
                   "automatic_tracked_k3": len(tracked),
                   "automatic_tracked_scenes": len(scenes),
                   "not_tracked": len(events) - len(tracked),
                   "failure_reasons": dict(reasons)},
        "gates": {"event_scene_floor_pass": floor,
                  "ambiguity_zero_among_tracked": ambiguity_zero,
                  "automatic_candidate_semantic_subgate_pass": passed,
                  "language_branch_dependence_pass": False,
                  "full_gate6_pass": False},
        "events": events,
        "non_conclusions": {
            "language_validated_events": 0,
            "human_review_performed": False,
            "learned_inspect_policy": False,
            "training_authorized": False,
            "benchmark_result": False,
            "val_unseen_or_test_used": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({"status": output["status"],
                      "decision": output["decision"],
                      "counts": output["counts"],
                      "gates": output["gates"],
                      "runs": runs,
                      "output": os.path.relpath(OUT, ROOT),
                      "output_sha256": sha256_file(OUT)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

