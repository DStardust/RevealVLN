#!/usr/bin/env python3
"""Remote migration smoke for the Alibaba Linux Habitat-Sim overlay."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path("/mnt/daiyang/vla")
OVERLAY = WORKSPACE / ".remote_runtime/habitat-sim"
SCENE = (
    WORKSPACE
    / "third_party/habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
)
OUTPUT = WORKSPACE / "artifacts/migration/REMOTE_VLA_SMOKE.json"
checks: list[dict[str, object]] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    checks.append({"name": name, "pass": bool(condition), "detail": str(detail)})
    print(f"{'PASS' if condition else 'FAIL'}: {name} | {detail}", flush=True)


try:
    import habitat_sim
    import magnum

    check("python_3_10", sys.version_info[:2] == (3, 10), sys.version.split()[0])
    check("torch_2_11_cu128", torch.__version__ == "2.11.0+cu128", torch.__version__)
    check("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        check("gpu_is_rtx_5090", "RTX 5090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0))
        tensor = torch.randn(512, 512, device="cuda")
        result = tensor @ tensor
        torch.cuda.synchronize()
        check("cuda_matmul_finite", bool(torch.isfinite(result).all().item()), result.shape)

    habitat_path = Path(habitat_sim.__file__).resolve()
    magnum_path = Path(magnum.__file__).resolve()
    check("habitat_sim_0_1_7", habitat_sim.__version__ == "0.1.7", habitat_sim.__version__)
    check("habitat_from_remote_overlay", habitat_path.is_relative_to(OVERLAY.resolve()), habitat_path)
    check("magnum_from_remote_overlay_env", ".remote_runtime/etpr1" in str(magnum_path), magnum_path)
    check("headless_build_without_cuda_interop", habitat_sim.cuda_enabled is False, habitat_sim.cuda_enabled)
    check("scene_present", SCENE.is_file(), SCENE)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(SCENE)
    sim_cfg.gpu_device_id = 0

    rgb = habitat_sim.SensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb.resolution = [64, 64]

    depth = habitat_sim.SensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth.resolution = [64, 64]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        observations = simulator.get_sensor_observations()
        rgb_frame = np.asarray(observations["rgb"])
        depth_frame = np.asarray(observations["depth"])
        check("rgb_frame_valid", rgb_frame.shape[:2] == (64, 64) and rgb_frame.dtype == np.uint8, rgb_frame.shape)
        check("depth_frame_valid", depth_frame.shape == (64, 64) and np.isfinite(depth_frame).any(), depth_frame.shape)
        second = simulator.step("move_forward")
        check("simulator_step_valid", "rgb" in second and "depth" in second, list(second))
    finally:
        simulator.close()
    check("simulator_closed", True)
except Exception as error:  # noqa: BLE001
    check("runtime_exception", False, repr(error))
    traceback.print_exc()

status = "PASS" if all(item["pass"] for item in checks) else "FAIL"
payload = {
    "status": status,
    "workspace": str(WORKSPACE),
    "overlay": str(OVERLAY),
    "checks": checks,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"REMOTE_VLA_SMOKE_{status}", flush=True)
raise SystemExit(0 if status == "PASS" else 1)
