"""Shared causal-view boundary for the MF2-CR1 automatic frontend.

The accepted ETP-R1 model receives twelve overlapping 63-degree cameras at
30-degree yaw increments.  This adapter keeps the tensor schema compatible
while ensuring that only physically acquired headings can affect waypoint
prediction, panorama embeddings, GraphMap features, or policy actions.

No model weights live here and no frozen upstream source is modified.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Set

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


NUM_VIEWS = 12
VIEW_STEP_DEG = 30
HFOV_DEG = 63.0
_VIEW_RE = re.compile(r"^(rgb|depth)_(30|60|90|120|150|180|210|240|270|300|330)$")


def _angle_slot(angle_deg: int) -> int:
    if angle_deg % VIEW_STEP_DEG:
        raise ValueError("view angle must be a multiple of 30 degrees")
    return (angle_deg // VIEW_STEP_DEG) % NUM_VIEWS


def sensor_view_slot(key: str):
    """Return ETP-R1's counter-clockwise view slot for a raw sensor key."""
    if key in ("rgb", "depth"):
        return 0
    match = _VIEW_RE.match(key)
    return _angle_slot(int(match.group(2))) if match else None


def _normalize_mask(acquired_view_mask, batch_size, device):
    mask = torch.as_tensor(acquired_view_mask, dtype=torch.bool,
                           device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    if tuple(mask.shape) != (batch_size, NUM_VIEWS):
        raise ValueError("acquired_view_mask must have shape [12] or [B,12]")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every batch item must contain an acquired view")
    return mask


def apply_raw_view_mask(
        observations: Mapping[str, object], acquired_view_mask,
        hidden_variant: str = "original") -> Dict[str, object]:
    """Mask hidden raw sensors before either visual encoder executes.

    ``hidden_variant='adversarial'`` first replaces hidden raw values with a
    deterministic dtype-safe perturbation, then applies the exact same zero
    boundary.  Identical downstream results between the two variants are the
    required hidden-view noninterference control.
    """
    tensor = next((value for key, value in observations.items()
                   if sensor_view_slot(key) is not None and
                   isinstance(value, torch.Tensor)), None)
    if tensor is None:
        raise ValueError("no ETP-R1 RGB/depth view tensor found")
    batch_size = int(tensor.shape[0])
    mask = _normalize_mask(acquired_view_mask, batch_size, tensor.device)
    if hidden_variant not in ("original", "adversarial"):
        raise ValueError("unknown hidden perturbation variant")
    result = dict(observations)
    for key, value in observations.items():
        slot = sensor_view_slot(key)
        if slot is None or not isinstance(value, torch.Tensor):
            continue
        if int(value.shape[0]) != batch_size:
            raise ValueError("inconsistent raw view batch size")
        visible = mask[:, slot]
        expand = visible.reshape(batch_size, *([1] * (value.ndim - 1)))
        source = value
        if hidden_variant == "adversarial":
            if value.is_floating_point():
                adversarial = torch.full_like(value, 0.73125)
            else:
                limit = torch.iinfo(value.dtype).max
                adversarial = torch.full_like(value, min(limit, 231))
            source = torch.where(expand, value, adversarial)
        # The boundary itself is identical for both variants. Hidden raw
        # pixels/depth never reach either encoder.
        result[key] = torch.where(expand, source, torch.zeros_like(value))
    result["causal_view_mask"] = mask
    return result


def _circular_distance_rad(angle, center):
    return abs((float(angle) - float(center) + math.pi) %
               (2.0 * math.pi) - math.pi)


def _acquired_centers(mask_row: torch.Tensor):
    return [slot * VIEW_STEP_DEG / 180.0 * math.pi
            for slot in torch.nonzero(mask_row, as_tuple=False)
            .flatten().tolist()]


def _select_rows(tensor, keep):
    if tensor is None:
        return None
    index = torch.as_tensor(keep, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index)


def filter_waypoint_outputs(outputs: Mapping[str, object],
                            acquired_view_mask,
                            hfov_deg: float = HFOV_DEG):
    """Apply the shared acquisition mask to candidates and pano features.

    Candidate membership is decided from the continuous candidate angle,
    not its coarse image index. Candidate appearance/depth is sourced only
    from the nearest acquired camera. Hidden panorama slots become zeros and
    are separately excluded by :func:`causal_vp_feature_variable`.
    """
    if not isinstance(outputs.get("pano_rgb"), torch.Tensor) or \
            not isinstance(outputs.get("pano_depth"), torch.Tensor):
        raise ValueError("waypoint output lacks pano tensors")
    batch_size = int(outputs["pano_rgb"].shape[0])
    mask = _normalize_mask(acquired_view_mask, batch_size,
                           outputs["pano_rgb"].device)
    result = dict(outputs)
    pano_rgb = outputs["pano_rgb"].clone()
    pano_depth = outputs["pano_depth"].clone()
    pano_rgb[~mask] = 0
    pano_depth[~mask.to(pano_depth.device)] = 0
    result["pano_rgb"] = pano_rgb
    result["pano_depth"] = pano_depth

    result_cand = {name: [] for name in (
        "cand_rgb", "cand_depth", "cand_angle_fts", "cand_img_idxes",
        "cand_angles", "cand_distances")}
    source_slots_all = []
    half_fov = math.radians(float(hfov_deg)) / 2.0
    for batch_index in range(batch_size):
        angles = list(outputs["cand_angles"][batch_index])
        centers = _acquired_centers(mask[batch_index])
        keep, sources = [], []
        for index, angle in enumerate(angles):
            distances = [_circular_distance_rad(angle, center)
                         for center in centers]
            source_local = min(range(len(centers)),
                               key=lambda k: (distances[k], k))
            if distances[source_local] <= half_fov + 1e-12:
                keep.append(index)
                sources.append(_angle_slot(round(
                    centers[source_local] / math.pi * 180.0)))

        result_cand["cand_angles"].append([angles[i] for i in keep])
        result_cand["cand_distances"].append(
            [outputs["cand_distances"][batch_index][i] for i in keep])
        result_cand["cand_angle_fts"].append(_select_rows(
            outputs["cand_angle_fts"][batch_index], keep))
        # Candidate visual features are explicitly re-sourced from acquired
        # camera features, never from a hidden candidate image index.
        source_index_rgb = torch.as_tensor(
            sources, dtype=torch.long, device=pano_rgb.device)
        source_index_depth = source_index_rgb.to(pano_depth.device)
        result_cand["cand_rgb"].append(
            pano_rgb[batch_index].index_select(0, source_index_rgb))
        result_cand["cand_depth"].append(
            pano_depth[batch_index].index_select(0, source_index_depth))
        result_cand["cand_img_idxes"].append(np.asarray(sources,
                                                        dtype=np.int64))
        source_slots_all.append(sources)
    result.update(result_cand)
    result["causal_view_mask"] = mask
    result["causal_candidate_source_idxes"] = source_slots_all
    result["causal_frontend_revision"] = "mf2-cr1-causal-view/1"
    return result


def _pad_tensors(tensors: Sequence[torch.Tensor]):
    if not tensors:
        raise ValueError("cannot pad empty tensor list")
    max_len = max(int(item.shape[0]) for item in tensors)
    trailing = tuple(tensors[0].shape[1:])
    out = tensors[0].new_zeros((len(tensors), max_len) + trailing)
    for index, tensor in enumerate(tensors):
        out[index, :tensor.shape[0]] = tensor
    return out


def causal_vp_feature_variable(outputs: Mapping[str, object], device=None):
    """Create ETP panorama inputs without materializing missing slots.

    ``view_lens`` is the explicit transformer mask: only candidate tokens and
    acquired non-candidate view tokens are present. No zero-filled hidden
    pano token, positional feature, or nav type reaches ``forward_panorama``.
    """
    mask = outputs.get("causal_view_mask")
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
        raise ValueError("causal_view_mask missing from waypoint outputs")
    if device is None:
        device = outputs["pano_rgb"].device
    rgb_rows, depth_rows, loc_rows, nav_rows, lengths = [], [], [], [], []
    for batch_index in range(mask.shape[0]):
        candidate_count = len(outputs["cand_angles"][batch_index])
        candidate_sources = set(int(x) for x in
                                outputs["causal_candidate_source_idxes"]
                                [batch_index])
        extra_slots = [int(slot) for slot in torch.nonzero(
            mask[batch_index], as_tuple=False).flatten().tolist()
                       if int(slot) not in candidate_sources]
        rgb_parts = [outputs["cand_rgb"][batch_index]]
        dep_parts = [outputs["cand_depth"][batch_index]]
        loc_parts = [outputs["cand_angle_fts"][batch_index]]
        nav = [1] * candidate_count
        if extra_slots:
            rgb_parts.append(outputs["pano_rgb"][batch_index, extra_slots])
            dep_parts.append(outputs["pano_depth"][batch_index, extra_slots])
            angle_features = outputs["pano_angle_fts"]
            if not isinstance(angle_features, torch.Tensor):
                angle_features = torch.as_tensor(angle_features)
            loc_parts.append(angle_features[extra_slots].to(
                outputs["cand_angle_fts"][batch_index].device))
            nav.extend([0] * len(extra_slots))
        rgb = torch.cat(rgb_parts, dim=0)
        dep = torch.cat(dep_parts, dim=0)
        loc = torch.cat(loc_parts, dim=0)
        if rgb.shape[0] == 0:
            raise ValueError("causal panorama must contain at least one token")
        rgb_rows.append(rgb)
        depth_rows.append(dep)
        loc_rows.append(loc)
        nav_rows.append(torch.as_tensor(nav, dtype=torch.long))
        lengths.append(len(nav))
    return {
        "rgb_fts": _pad_tensors(rgb_rows),
        "dep_fts": _pad_tensors(depth_rows),
        "loc_fts": _pad_tensors(loc_rows).to(device),
        "nav_types": pad_sequence(nav_rows, batch_first=True).to(device),
        "view_lens": torch.as_tensor(lengths, dtype=torch.long,
                                      device=device),
    }


@dataclass
class CausalPoseViewBuffer:
    """Pure state machine for counted physical view acquisition at one pose."""

    heading_slot: int = 0
    pose_token: object = None
    acquired_world_slots: Set[int] = field(default_factory=set)
    lowlevel_turn_count: int = 0
    lowlevel_move_count: int = 0

    def reset_pose(self, pose_token, heading_slot=0):
        self.pose_token = pose_token
        self.heading_slot = int(heading_slot) % NUM_VIEWS
        self.acquired_world_slots = {self.heading_slot}

    def observe_front(self):
        self.acquired_world_slots.add(self.heading_slot)

    def turn(self, signed_steps):
        steps = int(signed_steps)
        direction = 1 if steps >= 0 else -1
        for _ in range(abs(steps)):
            self.heading_slot = (self.heading_slot + direction) % NUM_VIEWS
            self.lowlevel_turn_count += 1
            self.observe_front()

    def move(self, new_pose_token):
        self.lowlevel_move_count += 1
        self.reset_pose(new_pose_token, self.heading_slot)

    def relative_mask(self):
        """Acquired world headings reindexed relative to current body yaw."""
        mask = torch.zeros(NUM_VIEWS, dtype=torch.bool)
        for world_slot in self.acquired_world_slots:
            mask[(world_slot - self.heading_slot) % NUM_VIEWS] = True
        return mask

    @property
    def counted_actions(self):
        return self.lowlevel_turn_count + self.lowlevel_move_count

