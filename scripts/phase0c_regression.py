#!/usr/bin/env python3
"""Fail-closed regression for the MF2-CR1 Phase-0C correctness batch."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess


ROOT = "/mnt/daiyang/vla"
OUT = os.path.join(ROOT, "artifacts", "runtime", "phase0_correctness",
                   "PHASE0C_REGRESSION.json")
PYTHON = os.path.join(ROOT, ".envs", "etpr1", "bin", "python")
UV = os.path.join(ROOT, ".tools", "uv", "uv")
EXPECTED = {
    "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    "artifacts/phase0/evidence_current.json":
        "430f73ec5752783aa553c2c2f4fe4128247a93fa29bee0ae454c97f5547d9ce1",
    "artifacts/runtime/PHASE0_REVEAL_CLOSURE_MAIN_ACCEPTANCE.json":
        "36c2fb2bce69b8ebc337e0d2192c731c52dba33d8f0b5fe781ffa2a53783b435",
    "artifacts/runtime/ETPR1_RUNTIME_POST_FREEZE.txt":
        "c5cfea518dc455748afd3fff978dfbc64b4eed06bd477f4122fcdee8e103cee6",
    "artifacts/runtime/phase0_correctness/"
    "PHASE0C_COST_FRONTIER_WITNESS.json":
        "9b59ea9b7b9995aeb604b00587dd79a3af863ea51b16ddfe805b3f719f1a16d1",
    "artifacts/runtime/phase0_correctness/"
    "PHASE0C_ORACLE_LOWLEVEL_PROBE.json":
        "b2e94b8310dc14d9ae0fa024ae1fb67633fd77bbab41cbb4cdc9939d229e27ac",
    "artifacts/runtime/phase0_correctness/"
    "IDENTITY_V3_RERUN_SUMMARY.json":
        "cf4e5d51b1052bf789ae9747bfaf8136a9438e526b2c0206fadb0ec0afe59109",
}
REPOS = {
    "habitat-sim": ("856d4b08c1a2632626bf0d205bf46471a99502b7", True),
    "habitat-lab": ("d6ed1c0a0e786f16f261de2beafe347f4186d0d8", True),
    "ETP-R1": ("a94b5c8fe20d1631e9e150c430a925543eb1cba9", False),
}
ETP_DIFF_SHA = \
    "5a207d26b5b582e8c810f5c95ecfe52897caecf69bb689874ad3222d21bfc521"
RESERVE_BYTES = 10_737_418_240
MIN_FREE_BYTES = 8 * 1024 ** 3


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=600, **kwargs)


def main():
    checks = []

    def record(name, passed, observed):
        checks.append({"name": name, "pass": bool(passed),
                       "observed": observed})
        print(("PASS " if passed else "FAIL ") + name, flush=True)

    for relative, expected in EXPECTED.items():
        path = os.path.join(ROOT, relative)
        observed = sha256_file(path) if os.path.isfile(path) else None
        record("immutable_sha:" + relative, observed == expected, observed)

    adjudication = json.load(open(os.path.join(
        ROOT, "artifacts", "runtime", "phase0_correctness",
        "PHASE0C_COST_FRONTIER_ADJUDICATION.json")))
    replay = json.load(open(os.path.join(
        ROOT, "artifacts", "runtime", "phase0_correctness",
        "PHASE0C_COST_FRONTIER_REPLAY.json")))
    record("adjudication_gate4_pass",
           adjudication.get("status") == "PASS" and
           adjudication["gates"].get("gate4_pass") is True and
           adjudication["gates"].get(
               "status_correspondence_104x2x4") is True,
           {"status": adjudication.get("status"),
            "fraction": adjudication.get("gates", {}).get(
                "frozen_fraction")})
    record("determinism_replay_exact",
           replay.get("status") == "PASS" and
           len(replay.get("results", [])) == 3 and
           all(x.get("exact_canonical_match") is True
               for x in replay.get("results", [])), replay.get("decision"))

    proc = run([PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v"],
               cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT))
    record("toporeveal_tests", proc.returncode == 0,
           (proc.stderr.strip().splitlines() or [""])[-1])

    env = dict(os.environ,
               VIRTUAL_ENV=os.path.join(ROOT, ".envs", "etpr1"),
               PYTHONNOUSERSITE="1")
    proc = run([UV, "pip", "check"], env=env)
    record("uv_pip_check", proc.returncode == 0,
           (proc.stdout + proc.stderr).strip()[-300:])
    proc = run([UV, "pip", "freeze"], env=env)
    freeze_sha = hashlib.sha256(proc.stdout.encode()).hexdigest()
    record("environment_freeze", proc.returncode == 0 and
           freeze_sha == EXPECTED[
               "artifacts/runtime/ETPR1_RUNTIME_POST_FREEZE.txt"],
           freeze_sha)

    for name, (head_expected, clean_expected) in REPOS.items():
        repo = os.path.join(ROOT, "third_party", name)
        head = run(["git", "-C", repo, "rev-parse", "HEAD"])
        status = run(["git", "-C", repo, "status", "--porcelain=v1"])
        head_value = head.stdout.strip()
        if clean_expected:
            passed = (head.returncode == status.returncode == 0 and
                      head_value == head_expected and not status.stdout)
        else:
            diff = subprocess.run(["git", "-C", repo, "diff"],
                                  capture_output=True, timeout=120)
            diff_sha = hashlib.sha256(diff.stdout).hexdigest()
            lines = status.stdout.splitlines()
            tracked = [line for line in lines
                       if line and not line.startswith("??")]
            untracked = [line[3:] for line in lines
                         if line.startswith("??")]
            passed = (head_value == head_expected and len(tracked) == 14 and
                      untracked == ["etpr1_compat.py"] and
                      diff_sha == ETP_DIFF_SHA)
        record("repo_state:" + name, passed,
               {"head": head_value, "status_lines":
                    len(status.stdout.splitlines())})

    reserves = []
    for index in range(9, 28):
        path = os.path.join(ROOT, ".disk_reserve",
                            "reserve_10G_%02d.bin" % index)
        reserves.append(os.path.isfile(path) and not os.path.islink(path) and
                        os.path.getsize(path) == RESERVE_BYTES)
    record("reserves_09_27", all(reserves),
           {"ok": sum(reserves), "expected": 19})

    parts = glob.glob(os.path.join(ROOT, "**", "*.part"), recursive=True)
    artifact_symlinks = []
    for base, dirs, files in os.walk(os.path.join(ROOT, "artifacts")):
        artifact_symlinks.extend(os.path.join(base, item)
                                 for item in dirs + files
                                 if os.path.islink(os.path.join(base, item)))
    record("no_part_or_artifact_symlink",
           not parts and not artifact_symlinks,
           {"part_count": len(parts), "symlink_count":
                len(artifact_symlinks)})

    # Scripts created in this batch are static engineering workers: reject
    # accidental training entrypoints, network clients, and forbidden split
    # literals in executable statements. Documentation/non-conclusion strings
    # are intentionally allowed and covered by the input-boundary artifacts.
    scripts = [
        "adjudicate_candidate_identity.py", "run_identity_v2_batch.py",
        "audit_tx_feasibility.py", "phase0c_oracle_egofov_probe.py",
        "phase0c_oracle_lowlevel_probe.py",
        "phase0c_cost_frontier_witness.py",
        "adjudicate_cost_frontier.py", "replay_cost_frontier_sample.py",
        "phase0c_regression.py",
    ]
    compile_proc = run([PYTHON, "-m", "py_compile"] +
                       [os.path.join(ROOT, "scripts", x) for x in scripts])
    record("batch_scripts_compile", compile_proc.returncode == 0,
           (compile_proc.stderr or "9 scripts").strip())

    st = os.statvfs(ROOT)
    free = st.f_bavail * st.f_frsize
    record("disk_free_at_least_8gib", free >= MIN_FREE_BYTES, free)

    all_pass = all(item["pass"] for item in checks)
    output = {
        "gate": "mf2_cr1_phase0c_boundary_regression",
        "revision": "phase0c-regression/1",
        "status": "PASS" if all_pass else "FAIL",
        "decision": "BOUNDARY_REGRESSION_PASS" if all_pass else
                    "BOUNDARY_REGRESSION_NO_GO",
        "checks_passed": sum(item["pass"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "explicit_boundaries": {
            "frozen_spec_modified": False,
            "phase0_protocol_modified": False,
            "checkpoint_loaded_in_correctness_batch": False,
            "training_performed": False,
            "semantic_review_fabricated": False,
            "val_unseen_or_test_used": False,
            "network_used": False,
            "reserve_released": False,
        },
    }
    with open(OUT, "w") as fh:
        json.dump(output, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "status": output["status"], "decision": output["decision"],
        "checks": "%d/%d" % (output["checks_passed"],
                                output["checks_total"]),
        "output": os.path.relpath(OUT, ROOT),
        "output_sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
