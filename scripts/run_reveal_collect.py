#!/usr/bin/env python3
"""Orchestrator for Reveal Prefix collector runs (single writer, serial GPU).

Selects the freest GPU (accepted R2R-gate policy: max free memory with
>= 10 GiB free, ties by lowest index), records parent-side GPU baseline and
post-exit probe, enforces the offline environment and network-guard counter
file, and launches scripts/collect_reveal_prefixes.py as a subprocess.

Used by Stage 2 collector engineering validation and Stage 4 batch
generation.  Writes ORCHESTRATOR_RESULT.json into each run directory.
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = "/mnt/daiyang/vla"
ETPR1_ROOT = os.path.join(PROJECT_ROOT, "third_party", "ETP-R1")
PROJECT_PYTHONPATH = os.pathsep.join([
    ETPR1_ROOT,
    os.path.join(PROJECT_ROOT, "third_party", "habitat-lab"),
    os.path.join(PROJECT_ROOT, "third_party", "habitat-sim"),
])
PYBIN = os.path.join(PROJECT_ROOT, ".envs", "etpr1", "bin", "python")
COLLECTOR = os.path.join(PROJECT_ROOT, "scripts",
                         "collect_reveal_prefixes.py")


def query_gpus():
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,memory.free,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, fr, used, total = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "free_mib": int(fr),
                     "used_mib": int(used), "total_mib": int(total)})
    return gpus


def gpu_mem_used(idx):
    out = subprocess.run(
        ["nvidia-smi", "--id=%d" % idx, "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def select_gpu(min_free_mib=10240):
    gpus = query_gpus()
    eligible = [g for g in gpus if g["free_mib"] >= min_free_mib]
    if not eligible:
        raise SystemExit(json.dumps({"status": "FAIL",
                                     "reason": "no GPU with >= 10 GiB free",
                                     "gpus": gpus}))
    return sorted(eligible, key=lambda g: (-g["free_mib"], g["index"]))[0], gpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val_seen"])
    ap.add_argument("--task", required=True, choices=["rxr", "r2r"])
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gate-out", required=True)
    ap.add_argument("--media-staging", default=None)
    ap.add_argument("--identity-protocol", choices=["v1", "v2", "v3"],
                    default="v1")
    ap.add_argument("--gpu-index", type=int, default=None,
                    help="optional explicit physical GPU; still requires "
                         ">=10 GiB free")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    if args.gpu_index is None:
        gpu, all_gpus = select_gpu()
    else:
        all_gpus = query_gpus()
        matches = [g for g in all_gpus if g["index"] == args.gpu_index]
        if len(matches) != 1 or matches[0]["free_mib"] < 10240:
            raise SystemExit(json.dumps({
                "status": "FAIL",
                "reason": "explicit GPU absent or has <10 GiB free",
                "requested_gpu": args.gpu_index,
                "gpus": all_gpus,
            }))
        gpu = matches[0]
    baseline = gpu_mem_used(gpu["index"])
    netguard_file = os.path.join(run_dir, "netguard.jsonl")
    open(netguard_file, "w").close()

    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": PROJECT_PYTHONPATH,
        "TORCH_HOME": os.path.join(PROJECT_ROOT, ".cache", "torch"),
        "HF_HOME": os.path.join(PROJECT_ROOT, ".cache", "huggingface"),
        "TRANSFORMERS_CACHE": os.path.join(
            PROJECT_ROOT, ".cache", "huggingface", "transformers"),
        "CLIP_DOWNLOAD_ROOT": os.path.join(PROJECT_ROOT, ".cache", "clip"),
        "RXREN_NETGUARD_FILE": netguard_file,
    })
    cmd = [PYBIN, COLLECTOR,
           "--episode-id", args.episode_id,
           "--split", args.split,
           "--task", args.task,
           "--exp-name", args.exp_name,
           "--run-dir", run_dir,
           "--gate-out", os.path.abspath(args.gate_out),
           "--identity-protocol", args.identity_protocol]
    if args.media_staging:
        cmd += ["--media-staging", os.path.abspath(args.media_staging)]

    meta = {
        "cmd": cmd,
        "gpu_selection": {"selected": gpu, "all_gpus": all_gpus,
                          "policy": "max free memory >= 10 GiB, ties by "
                                    "lowest index"},
        "gpu_baseline_used_mib": baseline,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(run_dir, "LAUNCH_META.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    t0 = time.monotonic()
    with open(os.path.join(run_dir, "collector_stdout.log"), "wb") as o, \
            open(os.path.join(run_dir, "collector_stderr.log"), "wb") as e:
        proc = subprocess.run(cmd, cwd=ETPR1_ROOT, env=env,
                              stdout=o, stderr=e)
    rc = proc.returncode
    wall = time.monotonic() - t0
    time.sleep(5)
    post_exit = gpu_mem_used(gpu["index"])

    per_pid = {}
    child_guard = {}
    with open(netguard_file) as fh:
        for ln in fh:
            try:
                rec = json.loads(ln)
                pid = rec["pid"]
                if rec.get("attempts") is not None:
                    per_pid[pid] = max(per_pid.get(pid, 0),
                                       int(rec["attempts"]))
                if rec.get("role") == "env_child_step":
                    child_guard[pid] = bool(rec.get("guard_installed"))
            except (ValueError, KeyError, json.JSONDecodeError):
                pass

    collect_summary = {}
    cs_path = os.path.join(run_dir, "COLLECT_SUMMARY.json")
    if os.path.isfile(cs_path):
        with open(cs_path) as fh:
            collect_summary = json.load(fh)

    result = {
        "exp_name": args.exp_name,
        "episode_id": args.episode_id,
        "split": args.split,
        "task": args.task,
        "identity_protocol": args.identity_protocol,
        "returncode": rc,
        "wall_time_s": round(wall, 3),
        "gpu_index": gpu["index"],
        "gpu_baseline_used_mib": baseline,
        "gpu_post_exit_used_mib": post_exit,
        "gpu_memory_returned_within_band": abs(post_exit - baseline) <= 1500,
        "network_attempts_all_processes": sum(per_pid.values()),
        "network_guard_child_evidence_ok": bool(child_guard) and
            all(child_guard.values()),
        "collect_status": collect_summary.get("collect_status"),
        "prefix_count": collect_summary.get("prefix_count"),
        "chain_root": collect_summary.get("chain_root"),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(run_dir, "ORCHESTRATOR_RESULT.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
