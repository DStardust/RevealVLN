#!/usr/bin/env python3
"""Collect one outcome-blind frozen-native temporal observation episode.

The worker writes only strictly causal RGB panoramas, frozen embeddings,
candidate identities/scores, and action traces.  It does not receive an event
target, read task metrics, or execute an intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for value in (ROOT, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import rxr_uad_controller_worker_mf3 as base  # noqa: E402


R2R_CHECKPOINT = base.ETPR1 / (
    "data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"
)
RXR_CHECKPOINT = base.RXR_CHECKPOINT
JOINT_PRETRAINED = base.JOINT_PRETRAINED
PANORAMA_YAWS = tuple(range(0, 360, 30))
JPEG_QUALITY = 90
SCHEMA = "revealnav-mf3zp-frozen-native-observation/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: list[dict]) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as stream:
        for value in values:
            stream.write(json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ) + "\n")
    os.replace(temporary, path)


def _new_directory(path: Path) -> Path:
    resolved = path.resolve()
    if (
        ROOT not in resolved.parents
        or resolved.exists()
        or resolved.is_symlink()
    ):
        raise RuntimeError("MF3ZP worker output must be a fresh project-local path")
    resolved.mkdir(parents=True)
    return resolved


def _source_trace(path: Path) -> tuple[Path, list[dict], str]:
    resolved = path.resolve()
    if (
        ROOT not in resolved.parents
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise RuntimeError("source native trace is not a regular project-local file")
    rows = [
        json.loads(line)
        for line in resolved.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("source native trace is empty")
    return resolved, rows, sha256_file(resolved)


def _action_signature(row: dict) -> tuple:
    return (
        int(row["act"]),
        str(row.get("ghost_vp")),
        str(row.get("front_vp")),
        int(row.get("back_path_len", 0)),
        bool(row.get("tryout", False)),
    )


def _sensor_key(yaw: int) -> str:
    return "rgb" if yaw == 0 else f"rgb_{yaw}"


class PanoramaCapture:
    def __init__(self, media_dir: Path) -> None:
        self.media_dir = media_dir
        self.records: list[dict] = []

    def capture(self, observations: dict, waypoint: dict) -> None:
        step = len(self.records)
        views = []
        for yaw in PANORAMA_YAWS:
            key = _sensor_key(yaw)
            value = observations.get(key)
            if not isinstance(value, torch.Tensor) or value.shape[0] != 1:
                raise RuntimeError(f"missing RGB panorama sensor: {key}")
            image = value[0].detach().cpu().numpy()
            if image.ndim != 3 or image.shape[2] < 3:
                raise RuntimeError(f"invalid RGB panorama tensor: {key}")
            views.append(np.asarray(image[..., :3], dtype=np.uint8))
        height, width = views[0].shape[:2]
        if any(value.shape[:2] != (height, width) for value in views):
            raise RuntimeError("panorama view shapes differ")
        candidate_indices = [int(value) for value in waypoint["cand_img_idxes"][0]]
        candidate_angles = [float(value) for value in waypoint["cand_angles"][0]]
        candidate_distances = [float(value) for value in waypoint["cand_distances"][0]]
        if not (
            len(candidate_indices)
            == len(candidate_angles)
            == len(candidate_distances)
        ):
            raise RuntimeError("waypoint candidate metadata is misaligned")
        markers: dict[int, list[str]] = {}
        for index, view_index in enumerate(candidate_indices):
            if not 0 <= view_index < len(PANORAMA_YAWS):
                raise RuntimeError("candidate panorama index is out of range")
            markers.setdefault(view_index, []).append(f"L{index:02d}")

        header = 34
        panes = []
        for index, (yaw, rgb) in enumerate(zip(PANORAMA_YAWS, views)):
            pane = np.zeros((height + header, width, 3), dtype=np.uint8)
            pane[header:] = rgb[..., ::-1]
            label = f"yaw {yaw:03d}"
            if index in markers:
                label += " | " + ",".join(markers[index])
            cv2.putText(
                pane,
                label,
                (7, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panes.append(pane)
        rows = [
            np.concatenate(panes[index:index + 4], axis=1)
            for index in range(0, len(panes), 4)
        ]
        sheet = np.concatenate(rows, axis=0)
        path = self.media_dir / f"prefix_{step:03d}.jpg"
        temporary = path.with_name(path.stem + ".part.jpg")
        if not cv2.imwrite(
            str(temporary),
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        ):
            raise RuntimeError("failed to encode panorama contact sheet")
        os.replace(temporary, path)
        self.records.append({
            "schema_version": "revealnav-mf3zp-panorama/1",
            "step": step,
            "frame_id": f"P{step:03d}",
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "view_yaws_deg": list(PANORAMA_YAWS),
            "pixels": [int(sheet.shape[1]), int(sheet.shape[0])],
            "local_candidates": [
                {
                    "local_marker": f"L{index:02d}",
                    "view_index": candidate_indices[index],
                    "view_yaw_deg": PANORAMA_YAWS[candidate_indices[index]],
                    "relative_angle_rad": candidate_angles[index],
                    "distance_m": candidate_distances[index],
                }
                for index in range(len(candidate_indices))
            ],
        })


class FrozenNativeObserver:
    policy_fusion_features = False

    def __init__(self, arrays_dir: Path) -> None:
        self.arrays_dir = arrays_dir
        self.instruction: torch.Tensor | None = None
        self.history: torch.Tensor | None = None
        self.records: list[dict] = []

    def record_language(self, embedding: torch.Tensor, mask: torch.Tensor) -> None:
        self.instruction = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def record_panorama(self, embedding: torch.Tensor, mask: torch.Tensor) -> None:
        self.history = (
            (embedding * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp_min(1)
        )[0].detach()

    def navigation(self, kwargs: dict, result: dict) -> dict:
        if self.instruction is None or self.history is None:
            raise RuntimeError("language/history embedding is unavailable")
        step = len(self.records)
        ids = kwargs["gmap_vp_ids"][0]
        action_mask = [bool(value) for value in kwargs["gmap_masks"][0]]
        visited = [bool(value) for value in kwargs["gmap_visited_masks"][0]]
        current = base.current_local_action_indices(
            ids,
            action_mask,
            visited,
            base._LOCAL_ACTION_IDS[0],
        )
        logits = result["global_logits"]
        native_index = int(torch.argmax(logits[0]))
        forced_stop = (
            bool(base._NO_VP_LEFT[0])
            or step >= int(base._TRAINER.max_len) - 1
        )
        if forced_stop:
            native_index = 0
        candidate_ids = tuple(str(ids[index]) for index in current)
        action_embeddings = (
            kwargs["gmap_img_fts"][0, current].detach().cpu().float().numpy()
            if current
            else np.empty((0, 768), dtype=np.float32)
        )
        checkpoint = self.history.detach().cpu().float().numpy()
        instruction = self.instruction.detach().cpu().float().numpy()
        if (
            instruction.shape != (768,)
            or checkpoint.shape != (768,)
            or action_embeddings.shape != (len(current), 768)
            or not np.isfinite(instruction).all()
            or not np.isfinite(checkpoint).all()
            or not np.isfinite(action_embeddings).all()
        ):
            raise RuntimeError("MF3ZP embedding shape/value drift")
        position_features = (
            kwargs["gmap_pos_fts"][0, current].detach().cpu().float().numpy()
            if current
            else np.empty((0, 7), dtype=np.float32)
        )
        scores = (
            logits[0, current].detach().cpu().float().numpy()
            if current
            else np.empty((0,), dtype=np.float32)
        )
        relative_heading = [
            math.atan2(float(value[0]), float(value[1]))
            for value in position_features
        ]
        gmap = base._TRAINER.gmaps[0]
        positions = []
        for identity in candidate_ids:
            source = (
                gmap.ghost_mean_pos[identity]
                if identity.startswith("g")
                else gmap.node_pos[identity]
            )
            position = np.asarray(source, dtype=np.float32)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise RuntimeError("candidate position drift")
            positions.append(position)
        candidate_positions = (
            np.stack(positions)
            if positions
            else np.empty((0, 3), dtype=np.float32)
        )
        arrays_path = self.arrays_dir / f"prefix_{step:03d}.npz"
        temporary = arrays_path.with_name(arrays_path.name + ".part")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                instruction=instruction,
                checkpoint=checkpoint,
                action_embeddings=action_embeddings,
                candidate_positions=candidate_positions,
                position_features=position_features,
                policy_scores=scores,
            )
        os.replace(temporary, arrays_path)
        self.records.append({
            "schema_version": SCHEMA,
            "step": step,
            "candidate_action_ids": list(candidate_ids),
            "native_action_id": (
                None if native_index == 0 else str(ids[native_index])
            ),
            "candidate_relative_heading_rad": relative_heading,
            "arrays": {
                "path": str(arrays_path.relative_to(ROOT)),
                "bytes": arrays_path.stat().st_size,
                "sha256": sha256_file(arrays_path),
            },
            "action_changed": False,
        })
        return result


def install_panorama_hook(capture: PanoramaCapture) -> None:
    from vlnce_baselines.models.R1Policy import ETP

    original = ETP.forward

    def forward(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else None)
        if mode == "waypoint":
            capture.capture(kwargs["observations"], result)
        return result

    ETP.forward = forward


def run_argv(dataset: str, episode_id: str, output: Path) -> list[str]:
    if dataset == "RxR":
        config = "run_rxr/iter_train.yaml"
        checkpoint = RXR_CHECKPOINT
        extra = [
            "EVAL.LANGUAGES", "['en-US','en-IN']",
            "IL.RECOLLECT_TRAINER.gt_file",
            "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        ]
    elif dataset == "R2R":
        config = "run_r2r/iter_train.yaml"
        checkpoint = R2R_CHECKPOINT
        extra = []
    else:
        raise ValueError("unsupported MF3ZP dataset")
    return [
        "run.py",
        "--exp_name", f"mf3zp_{dataset.lower()}_{episode_id}",
        "--run-type", "eval",
        "--exp-config", config,
        "EVAL.SPLIT", "train",
        "TASK_CONFIG.DATASET.SPLIT", "train",
        *extra,
        "EVAL.EPISODE_ID", f"['{episode_id}']",
        "EVAL.EPISODE_COUNT", "1",
        "EVAL.CKPT_PATH_DIR", str(checkpoint),
        "EVAL.SAMPLE", "False",
        "MODEL.pretrained_path", str(JOINT_PRETRAINED),
        "IL.back_algo", "control",
        "INFERENCE.SPLIT", "train",
        "TASK_CONFIG.DATASET.SUFFIX", "''",
        "TASK_CONFIG.SEED", "100",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "GPU_NUMBERS", "1",
        "NUM_ENVIRONMENTS", "1",
        "SIMULATOR_GPU_IDS", "[0]",
        "TORCH_GPU_IDS", "[0]",
        "TORCH_GPU_ID", "0",
        "VIDEO_OPTION", "[]",
        "TENSORBOARD_DIR", str(output / "tensorboard"),
        "CHECKPOINT_FOLDER", str(output / "checkpoints"),
        "RESULTS_DIR", str(output / "results"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("R2R", "RxR"), required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-native-trace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="physical GPU selected by the parent orchestrator; the worker uses it as logical CUDA device 0",
    )
    args = parser.parse_args()

    if args.gpu_id < 0:
        raise SystemExit("--gpu-id must be non-negative")
    # This is set before the first CUDA context is created.  The orchestrator
    # also passes it in the child environment; keeping the check here makes a
    # manually launched worker fail closed instead of silently using another
    # device.
    requested_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if requested_gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    elif requested_gpu not in {str(args.gpu_id), "0"}:
        raise SystemExit("CUDA_VISIBLE_DEVICES disagrees with --gpu-id")

    source_path, source_actions, source_sha = _source_trace(
        args.source_native_trace
    )
    run_dir = _new_directory(args.run_dir)
    arrays_dir = run_dir / "arrays"
    media_dir = run_dir / "media"
    arrays_dir.mkdir()
    media_dir.mkdir()
    base_trace = run_dir / "base_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)

    observer = FrozenNativeObserver(arrays_dir)
    capture = PanoramaCapture(media_dir)
    base._CONTROLLER = observer
    base.install_hooks()
    install_panorama_hook(capture)

    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = run_argv(args.dataset, str(args.episode_id), output)
    summary = {
        "schema_version": "revealnav-mf3zp-observation-worker/1",
        "status": "RUNNING",
        "dataset": args.dataset,
        "split": "train",
        "episode_id": str(args.episode_id),
        "scene_id": args.scene_id,
        "public_split_access": False,
        "target_received": False,
        "task_metric_payload_read": False,
        "action_changed": False,
        "source_native_trace": {
            "path": str(source_path.relative_to(ROOT)),
            "bytes": source_path.stat().st_size,
            "sha256": source_sha,
        },
        "argv": argv,
    }
    sys.argv = argv
    started = time.monotonic()
    try:
        import run

        run.main()
        actual_actions = [
            json.loads(line)
            for line in base_trace.read_text().splitlines()
            if line.strip()
        ]
        if [_action_signature(value) for value in actual_actions] != [
            _action_signature(value) for value in source_actions
        ]:
            raise RuntimeError("frozen native replay action trace differs from source")
        if not (
            len(actual_actions)
            == len(observer.records)
            == len(capture.records)
        ):
            raise RuntimeError("MF3ZP observation stream cardinality drift")
        summary["status"] = "PASS"
        summary["source_native_replay_exact"] = True
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        records_path = run_dir / "causal_prefix_records.jsonl"
        media_path = run_dir / "panorama_manifest.jsonl"
        atomic_jsonl(records_path, observer.records)
        atomic_jsonl(media_path, capture.records)
        summary.update({
            "wall_time_seconds": round(time.monotonic() - started, 3),
            "peak_rss_self_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "prefix_records": len(observer.records),
            "panorama_records": len(capture.records),
            "causal_prefix_records": {
                "path": str(records_path.relative_to(ROOT)),
                "bytes": records_path.stat().st_size,
                "sha256": sha256_file(records_path),
            },
            "panorama_manifest": {
                "path": str(media_path.relative_to(ROOT)),
                "bytes": media_path.stat().st_size,
                "sha256": sha256_file(media_path),
            },
            "base_trace": {
                "path": str(base_trace.relative_to(ROOT)),
                "bytes": base_trace.stat().st_size,
                "sha256": sha256_file(base_trace),
            },
            "checkpoint_inventory": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (
                    R2R_CHECKPOINT if args.dataset == "R2R" else RXR_CHECKPOINT,
                    JOINT_PRETRAINED,
                )
            ],
            "no_outcome_or_target_input": True,
            "paper_result": False,
        })
        atomic_json(run_dir / "RUN_SUMMARY.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "dataset": args.dataset,
        "episode_id": str(args.episode_id),
        "prefix_records": len(observer.records),
        "run_dir": str(run_dir.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
