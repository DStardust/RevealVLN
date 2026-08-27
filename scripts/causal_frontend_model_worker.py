#!/usr/bin/env python3
"""Real ETP-R1 episode worker for the MF2-CR1 hidden-view gate.

The worker wraps (without editing) the accepted frozen ETP-R1 waypoint,
panorama and navigation path.  One run keeps hidden raw sensors unchanged;
the matched negative-control run deterministically corrupts every hidden raw
sensor before the same causal zero boundary.  Only downstream hashes and
aggregate perturbation hashes are written—never images or tensor values.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch


ROOT = "/mnt/daiyang/vla"
ETPR1 = os.path.join(ROOT, "third_party", "ETP-R1")
HABLAB = os.path.join(ROOT, "third_party", "habitat-lab")
HABSIM = os.path.join(ROOT, "third_party", "habitat-sim")
for path in (ROOT, ETPR1, HABLAB, HABSIM):
    if path not in sys.path:
        sys.path.insert(0, path)

# Habitat VectorEnv uses forkserver and re-imports this file as __mp_main__.
# Install the previously accepted runtime shims at module import time so the
# Discrete(0) compatibility fix, network guard and action recorder exist in
# the actual environment child—not only in the parent trainer.
_accepted_worker_path = os.path.join(ROOT, "artifacts", "runtime",
                                     "rxr_en_worker.py")
_accepted_spec = importlib.util.spec_from_file_location(
    "cr1_accepted_runtime_shims", _accepted_worker_path)
_accepted_worker = importlib.util.module_from_spec(_accepted_spec)
_accepted_spec.loader.exec_module(_accepted_worker)

from revealnav_cr1.causal_frontend import (  # noqa: E402
    apply_raw_view_mask,
    causal_vp_feature_variable,
    filter_waypoint_outputs,
    sensor_view_slot,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def aggregate_tensors(items):
    digest = hashlib.sha256()
    for key, value in sorted(items):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adversarial_like(value):
    if value.is_floating_point():
        return torch.full_like(value, 0.73125)
    return torch.full_like(value, min(torch.iinfo(value.dtype).max, 231))


def hash_candidates(outputs):
    records = []
    for batch_index, angles in enumerate(outputs["cand_angles"]):
        records.append({
            "angles": [round(float(x), 9) for x in angles],
            "distances": [round(float(x), 9) for x in
                          outputs["cand_distances"][batch_index]],
            "source_slots": [int(x) for x in
                             outputs["causal_candidate_source_idxes"]
                             [batch_index]],
            "rgb_sha256": tensor_sha(outputs["cand_rgb"][batch_index]),
            "depth_sha256": tensor_sha(
                outputs["cand_depth"][batch_index]),
        })
    return hashlib.sha256(canonical(records).encode()).hexdigest(), records


def install_causal_hooks(variant, trace_path, perturb_path):
    from vlnce_baselines.models.R1Policy import ETP
    from vlnce_baselines import ss_trainer_ETP_R1

    original_forward = ETP.forward
    counters = {"waypoint": 0, "panorama": 0, "navigation": 0}
    perturb_records = []
    acquired_slots = sorted({int(value) % 12 for value in
                             os.environ.get("CR1_ACQUIRED_SLOTS", "0")
                             .split(",")})
    if not acquired_slots:
        raise ValueError("at least one acquired slot is required")

    def append_trace(record):
        with open(trace_path, "a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def causal_forward(self, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode == "waypoint" and not args:
            observations = kwargs.get("observations") or {}
            first = next(value for key, value in observations.items()
                         if sensor_view_slot(key) is not None and
                         isinstance(value, torch.Tensor))
            mask = torch.zeros((int(first.shape[0]), 12), dtype=torch.bool,
                               device=first.device)
            mask[:, acquired_slots] = True
            hidden = [(key, value) for key, value in observations.items()
                      if sensor_view_slot(key) is not None and
                      sensor_view_slot(key) not in acquired_slots and
                      isinstance(value, torch.Tensor)]
            original_hidden_sha = aggregate_tensors(hidden)
            perturbed = [(key, adversarial_like(value))
                         for key, value in hidden]
            adversarial_hidden_sha = aggregate_tensors(perturbed)
            source_hidden_sha = (adversarial_hidden_sha if
                                 variant == "adversarial" else
                                 original_hidden_sha)
            masked = apply_raw_view_mask(observations, mask, variant)
            kwargs = dict(kwargs)
            kwargs["observations"] = masked
            raw_output = original_forward(self, *args, **kwargs)
            output = filter_waypoint_outputs(raw_output, mask)
            candidate_sha, candidate_records = hash_candidates(output)
            masked_views = [(key, value) for key, value in masked.items()
                            if sensor_view_slot(key) is not None and
                            isinstance(value, torch.Tensor)]
            append_trace({
                "mode": "waypoint",
                "call_index": counters["waypoint"],
                "acquired_slots": acquired_slots,
                "masked_raw_aggregate_sha256":
                    aggregate_tensors(masked_views),
                "candidate_aggregate_sha256": candidate_sha,
                "candidate_counts": [len(x["angles"])
                                     for x in candidate_records],
                "pano_rgb_sha256": tensor_sha(output["pano_rgb"]),
                "pano_depth_sha256": tensor_sha(output["pano_depth"]),
            })
            perturb_records.append({
                "call_index": counters["waypoint"],
                "hidden_tensor_count": len(hidden),
                "original_hidden_aggregate_sha256": original_hidden_sha,
                "adversarial_hidden_aggregate_sha256":
                    adversarial_hidden_sha,
                "source_before_mask_aggregate_sha256": source_hidden_sha,
                "source_differs_from_original": source_hidden_sha !=
                                                original_hidden_sha,
            })
            counters["waypoint"] += 1
            return output

        output = original_forward(self, *args, **kwargs)
        if mode == "panorama":
            embeds, masks = output
            append_trace({
                "mode": "panorama",
                "call_index": counters["panorama"],
                "embeds_sha256": tensor_sha(embeds),
                "masks_sha256": tensor_sha(masks),
                "token_count": int(masks.sum().item()),
            })
            counters["panorama"] += 1
        elif mode == "navigation":
            logits = output["global_logits"]
            append_trace({
                "mode": "navigation",
                "call_index": counters["navigation"],
                "global_logits_sha256": tensor_sha(logits),
                "argmax": [int(x) for x in torch.argmax(logits, dim=1)
                           .detach().cpu().tolist()],
            })
            counters["navigation"] += 1
        return output

    def causal_vp(self, outputs):
        return causal_vp_feature_variable(outputs, self.device)

    ETP.forward = causal_forward
    ss_trainer_ETP_R1.RLTrainer._vp_feature_variable = causal_vp

    def finalize():
        with open(perturb_path, "w") as fh:
            json.dump({
                "variant": variant,
                "acquired_slots": acquired_slots,
                "boundary": "perturb hidden raw sensors, then apply shared "
                            "single-front causal mask before encoders",
                "waypoint_calls": counters["waypoint"],
                "records": perturb_records,
            }, fh, indent=2)
            fh.write("\n")
    return finalize


def main():
    variant = os.environ.get("CR1_HIDDEN_VARIANT")
    run_dir = os.environ.get("CR1_RUN_DIR")
    if variant not in ("original", "adversarial"):
        raise SystemExit("CR1_HIDDEN_VARIANT must be original/adversarial")
    if not run_dir or not os.path.realpath(run_dir).startswith(ROOT + os.sep):
        raise SystemExit("CR1_RUN_DIR must be inside the workspace")
    os.makedirs(run_dir, exist_ok=True)
    trace_path = os.path.join(run_dir, "CAUSAL_MODEL_TRACE.jsonl")
    perturb_path = os.path.join(run_dir, "PERTURBATION_EVIDENCE.json")
    open(trace_path, "w").close()

    # Loading this module installs the already accepted Discrete(0), network
    # deny and env-action trace shims. Its main function is not called here.
    collector_path = os.path.join(ROOT, "scripts",
                                  "collect_reveal_prefixes.py")
    spec = importlib.util.spec_from_file_location(
        "cr1_base_collector", collector_path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)

    finalize = install_causal_hooks(variant, trace_path, perturb_path)
    # The collector installs its observation-only wrapper outside our causal
    # wrapper, so it records the causally filtered candidate output while the
    # frozen ETP implementation remains byte-for-byte untouched.
    try:
        collector.main()
    finally:
        finalize()


if __name__ == "__main__":
    main()
