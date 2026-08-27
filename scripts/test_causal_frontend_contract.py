#!/usr/bin/env python3
"""Deterministic unit/negative-control gate for the MF2-CR1 causal adapter."""

import hashlib
import json
import math
import os

import numpy as np
import torch

from revealnav_cr1.causal_frontend import (
    CausalPoseViewBuffer,
    apply_raw_view_mask,
    causal_vp_feature_variable,
    filter_waypoint_outputs,
)


ROOT = "/mnt/daiyang/vla"
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "CAUSAL_FRONTEND_CONTRACT_GATE.json")


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy()
                          .tobytes()).hexdigest()


def main():
    generator = torch.Generator().manual_seed(20260824)
    observations = {}
    for modality, shape, dtype in (
            ("rgb", (1, 8, 8, 3), torch.uint8),
            ("depth", (1, 8, 8, 1), torch.float32)):
        for slot in range(12):
            key = modality if slot == 0 else "%s_%d" % (
                modality, slot * 30)
            if dtype == torch.uint8:
                value = torch.randint(0, 256, shape, generator=generator,
                                      dtype=dtype)
            else:
                value = torch.rand(shape, generator=generator, dtype=dtype)
            observations[key] = value
    observations["instruction"] = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([True] + [False] * 11)
    masked_a = apply_raw_view_mask(observations, mask, "original")
    masked_b = apply_raw_view_mask(observations, mask, "adversarial")
    raw_equal = bool(all(torch.equal(masked_a[key], masked_b[key])
                         for key in observations if key.startswith((
                             "rgb", "depth"))))
    hidden_zero = bool(all(int(torch.count_nonzero(masked_a[key])) == 0
                           for key in observations
                           if key.startswith(("rgb_", "depth_"))))
    front_preserved = (torch.equal(masked_a["rgb"], observations["rgb"])
                       and torch.equal(masked_a["depth"],
                                       observations["depth"]))

    pano_rgb = torch.arange(12 * 4, dtype=torch.float32).reshape(1, 12, 4)
    pano_depth = torch.arange(12 * 3, dtype=torch.float32).reshape(1, 12, 3)
    angles = [0.0, math.radians(20), math.radians(40),
              2 * math.pi - math.radians(20)]
    outputs = {
        "cand_rgb": [pano_rgb[0, [0, 1, 1, 11]].clone()],
        "cand_depth": [pano_depth[0, [0, 1, 1, 11]].clone()],
        "cand_angle_fts": [torch.arange(16, dtype=torch.float32)
                            .reshape(4, 4)],
        "cand_img_idxes": [np.asarray([0, 1, 1, 11])],
        "cand_angles": [angles],
        "cand_distances": [[1.0, 1.25, 1.5, 1.75]],
        "pano_rgb": pano_rgb,
        "pano_depth": pano_depth,
        "pano_angle_fts": torch.arange(48, dtype=torch.float32)
                               .reshape(12, 4),
        "pano_img_idxes": np.arange(12),
    }
    filtered = filter_waypoint_outputs(outputs, mask)
    candidate_angles = filtered["cand_angles"][0]
    candidate_gate = bool(
        len(candidate_angles) == 3 and
        all(source == 0 for source in
            filtered["causal_candidate_source_idxes"][0]) and
        int(torch.count_nonzero(filtered["pano_rgb"][0, 1:])) == 0 and
        int(torch.count_nonzero(filtered["pano_depth"][0, 1:])) == 0)
    vp = causal_vp_feature_variable(filtered, torch.device("cpu"))
    explicit_transformer_mask = (
        vp["view_lens"].tolist() == [3] and
        tuple(vp["rgb_fts"].shape) == (1, 3, 4) and
        vp["nav_types"].tolist() == [[1, 1, 1]])

    # A deterministic downstream head stands in only for the algebraic unit
    # contract. The separate integration gate uses the real accepted model.
    weight = torch.arange(4, dtype=torch.float32)
    logits = (vp["rgb_fts"] * weight).sum(-1)
    action = int(torch.argmax(logits, dim=1)[0])

    state = CausalPoseViewBuffer()
    state.reset_pose("pose0", heading_slot=0)
    initial_mask = state.relative_mask().nonzero().flatten().tolist()
    state.turn(1)
    after_turn_mask = state.relative_mask().nonzero().flatten().tolist()
    state.move("pose1")
    after_move_mask = state.relative_mask().nonzero().flatten().tolist()
    state_gate = (initial_mask == [0] and after_turn_mask == [0, 11] and
                  after_move_mask == [0] and state.counted_actions == 2 and
                  state.lowlevel_turn_count == 1 and
                  state.lowlevel_move_count == 1)

    checks = {
        "hidden_raw_perturbation_bit_exact_after_boundary": raw_equal,
        "all_hidden_raw_slots_zero": hidden_zero,
        "front_raw_slot_preserved": front_preserved,
        "continuous_hfov_candidate_filter_and_resourcing": candidate_gate,
        "hidden_pano_tokens_absent_from_transformer":
            explicit_transformer_mask,
        "physical_turn_move_state_machine_counted": state_gate,
    }
    passed = all(checks.values())
    output = {
        "gate": "mf2_cr1_causal_frontend_contract",
        "revision": "causal-frontend-contract/1",
        "status": "PASS" if passed else "FAIL",
        "decision": "CONTINUE_TO_REAL_MODEL_INTEGRATION" if passed else
                    "CAUSAL_FRONTEND_CONTRACT_NO_GO",
        "checks": checks,
        "synthetic_deterministic_head": {
            "scope": "algebraic unit contract only",
            "logits_sha256": tensor_sha(logits),
            "action": action,
        },
        "state_machine_evidence": {
            "initial_relative_slots": initial_mask,
            "after_one_physical_turn_relative_slots": after_turn_mask,
            "after_move_relative_slots": after_move_mask,
            "lowlevel_turn_count": state.lowlevel_turn_count,
            "lowlevel_move_count": state.lowlevel_move_count,
        },
        "non_conclusions": {
            "real_waypoint_checkpoint_tested": False,
            "real_policy_logits_tested": False,
            "real_policy_action_tested": False,
            "phase0c_gate1_pass": False,
            "training_authorized": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({"status": output["status"],
                      "decision": output["decision"],
                      "checks": checks,
                      "output": os.path.relpath(OUT, ROOT)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
