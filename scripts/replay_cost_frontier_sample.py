#!/usr/bin/env python3
"""Independent multi-GPU replay of a deterministic Phase-0C cost sample.

The lexicographically first event from each of the first three distinct
scenes in the immutable full witness is recomputed in fresh spawned
processes.  Acceptance requires byte-equivalent canonical event records,
including every prefix, controller outcome, action count and action hash.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import multiprocessing
import os
import sys


ROOT = "/mnt/daiyang/vla"
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from phase0c_cost_frontier_witness import scene_worker  # noqa: E402


RAW = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHASE0C_COST_FRONTIER_WITNESS.json")
PROBE = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                     "PHASE0C_ORACLE_LOWLEVEL_PROBE.json")
RXR_TRAIN = os.path.join(
    ROOT, "third_party", "ETP-R1", "data", "datasets",
    "RxR_VLNCE_v0_enc_xlmr", "train", "train_guide.json.gz")
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHASE0C_COST_FRONTIER_REPLAY.json")
EXPECTED_RAW_SHA = \
    "9b59ea9b7b9995aeb604b00587dd79a3af863ea51b16ddfe805b3f719f1a16d1"
EXPECTED_PROBE_SHA = \
    "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac"
GPUS = (1, 3, 4)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    if sha256_file(RAW) != EXPECTED_RAW_SHA:
        raise SystemExit("raw witness SHA drift")
    if sha256_file(PROBE) != EXPECTED_PROBE_SHA:
        raise SystemExit("probe SHA drift")
    with open(RAW) as fh:
        raw = json.load(fh)
    with open(PROBE) as fh:
        probe = json.load(fh)

    # Selection is fixed before seeing replay outcomes: lexicographic event
    # order, taking the first three distinct scenes.
    selected = []
    scenes = set()
    for event in sorted(raw["events"],
                        key=lambda item: item["provisional_event_id"]):
        if event["scene_id"] in scenes:
            continue
        selected.append(event)
        scenes.add(event["scene_id"])
        if len(selected) == 3:
            break
    selected_ids = {event["provisional_event_id"] for event in selected}
    probe_events = {event["provisional_event_id"]: event
                    for event in probe["events"]
                    if event["provisional_event_id"] in selected_ids}
    episode_ids = {str(event["episode_id"]) for event in selected}
    with gzip.open(RXR_TRAIN, "rt") as fh:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(fh)["episodes"]
                    if str(item["episode_id"]) in episode_ids}
    if len(probe_events) != 3 or len(episodes) != 3:
        raise SystemExit("replay input closure failed")

    payloads = []
    for index, event in enumerate(selected):
        event_id = event["provisional_event_id"]
        episode_id = str(event["episode_id"])
        payloads.append({
            "scene": event["scene_id"],
            "events": [probe_events[event_id]],
            "gpu_index": GPUS[index],
            "episodes": {episode_id: episodes[episode_id]},
        })

    os.environ.setdefault("GLOG_minloglevel", "2")
    results = []
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=ctx) as pool:
        future_map = {pool.submit(scene_worker, payload): payload
                      for payload in payloads}
        for future in concurrent.futures.as_completed(future_map):
            payload = future_map[future]
            result = future.result()
            if len(result.get("events", [])) != 1:
                raise RuntimeError("unexpected replay worker cardinality")
            replayed = result["events"][0]
            original = next(x for x in selected if
                            x["provisional_event_id"] ==
                            replayed["provisional_event_id"])
            results.append({
                "provisional_event_id": replayed["provisional_event_id"],
                "scene_id": replayed["scene_id"],
                "gpu_index": payload["gpu_index"],
                "prefix_count": replayed["prefix_count"],
                "original_event_canonical_sha256": canonical_sha(original),
                "replay_event_canonical_sha256": canonical_sha(replayed),
                "exact_canonical_match": canonical_sha(original) ==
                                         canonical_sha(replayed),
            })
    results.sort(key=lambda item: item["provisional_event_id"])
    passed = len(results) == 3 and all(
        item["exact_canonical_match"] for item in results)
    output = {
        "gate": "mf2_cr1_cost_frontier_determinism_replay",
        "revision": "cost-frontier-replay/1",
        "status": "PASS" if passed else "FAIL",
        "decision": "DETERMINISM_SAMPLE_PASS" if passed else
                    "DETERMINISM_SAMPLE_NO_GO",
        "selection_rule": "lexicographic first event in each of the first "
                          "three distinct scenes",
        "input": {
            "raw_witness_sha256": sha256_file(RAW),
            "oracle_lowlevel_probe_sha256": sha256_file(PROBE),
            "rxr_split": "train",
        },
        "fresh_processes": 3,
        "physical_gpu_indices": list(GPUS),
        "results": results,
        "boundaries": {
            "checkpoint_loaded": False,
            "observations_materialized": False,
            "training_performed": False,
            "val_unseen_or_test_used": False,
            "network_used": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "results": results,
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
