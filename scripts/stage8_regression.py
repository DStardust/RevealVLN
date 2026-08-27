#!/usr/bin/env python3
"""Stage 8 regression: programmatic verification of the batch's 16-item
regression checklist.  Read-only; the two GPU smokes use the freest GPU.
Writes artifacts/runtime/phase0_reveal_closure/STAGE8_REGRESSION.json.
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = "/mnt/daiyang/vla"
ETPR1_ROOT = os.path.join(PROJECT_ROOT, "third_party", "ETP-R1")
PYBIN = os.path.join(PROJECT_ROOT, ".envs", "etpr1", "bin", "python")
UV_BIN = os.path.join(PROJECT_ROOT, ".tools", "uv", "uv")
OUT = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                   "phase0_reveal_closure", "STAGE8_REGRESSION.json")

EXPECTED = {
    "frozen_spec_sha256":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    "phase0_protocol_sha256":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    "main_acceptance_sha256":
        "02e121fba9033965df3ee2ca430287f60f036b3b07ad9f215aff6b15914805e5",
    "r2r_gate_sha256":
        "776a27f5c49ced929db9b27544943ccdf4bf87fb92ce46a2833641c172b8579b",
    "environment_freeze_sha256":
        "c5cfea518dc455748afd3fff978dfbc64b4eed06bd477f4122fcdee8e103cee6",
    "habitat_sim_head": "856d4b08c1a2632626bf0d205bf46471a99502b7",
    "habitat_lab_head": "d6ed1c0a0e786f16f261de2beafe347f4186d0d8",
    "etpr1_head": "a94b5c8fe20d1631e9e150c430a925543eb1cba9",
    "etpr1_tracked_diff_sha256":
        "5a207d26b5b582e8c810f5c95ecfe52897caecf69bb689874ad3222d21bfc521",
    "reserve_count": 19,
    "reserve_bytes": 10737418240,
    "min_free_bytes": 8 * 1024 ** 3,
}

REGISTERED_LARGE_ARTIFACTS = {
    os.path.join(PROJECT_ROOT, "artifacts", "upstream",
                 "ETP-R1-extra-files-86cacf29.zip"): 15238627709,
    os.path.join(PROJECT_ROOT, "artifacts", "upstream", "matterport3d",
                 "mp3d_habitat.zip"): 16085306031,
}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def check(report, name, ok, observed):
    report["checks"].append({"name": name, "pass": bool(ok),
                             "observed": str(observed)[:400]})
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    return bool(ok)


def query_gpus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, fr, used = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "free_mib": int(fr),
                     "used_mib": int(used)})
    return gpus


def main():
    report = {"stage": "stage8_regression", "checks": []}
    ok_all = True

    # 1. frozen docs unchanged
    ok_all &= check(report, "frozen_spec_sha",
                    sha256_file(os.path.join(PROJECT_ROOT, "FROZEN_SPEC.md"))
                    == EXPECTED["frozen_spec_sha256"], "sha recomputed")
    ok_all &= check(report, "phase0_protocol_sha",
                    sha256_file(os.path.join(PROJECT_ROOT,
                                             "PHASE0_PROTOCOL.md"))
                    == EXPECTED["phase0_protocol_sha256"], "sha recomputed")

    # 2. accepted R2R gate + main acceptance unchanged
    ma_path = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                           "R2R_VAL_SEEN_MAIN_ACCEPTANCE.json")
    ok_all &= check(report, "r2r_main_acceptance_sha",
                    sha256_file(ma_path)
                    == EXPECTED["main_acceptance_sha256"], "sha recomputed")
    ok_all &= check(report, "r2r_runtime_gate_sha",
                    sha256_file(os.path.join(
                        PROJECT_ROOT, "artifacts", "runtime",
                        "R2R_VAL_SEEN_RUNTIME_GATE.json"))
                    == EXPECTED["r2r_gate_sha256"], "sha recomputed")

    # 3. uv pip check + byte-identical freeze
    env = dict(os.environ, VIRTUAL_ENV=os.path.join(PROJECT_ROOT, ".envs",
                                                    "etpr1"))
    proc = subprocess.run([UV_BIN, "pip", "check"], env=env,
                          capture_output=True, text=True, timeout=300)
    ok_all &= check(report, "uv_pip_check_clean", proc.returncode == 0,
                    (proc.stdout + proc.stderr).strip()[-150:])
    proc = subprocess.run([UV_BIN, "pip", "freeze"], env=env,
                          capture_output=True, text=True, timeout=300)
    freeze_sha = hashlib.sha256(proc.stdout.encode()).hexdigest()
    ok_all &= check(report, "freeze_byte_identical",
                    freeze_sha == EXPECTED["environment_freeze_sha256"],
                    freeze_sha)

    # 4/5. GPU smokes on the freest GPU
    gpus = query_gpus()
    eligible = [g for g in gpus if g["free_mib"] >= 10240]
    gpu = sorted(eligible, key=lambda g: (-g["free_mib"], g["index"]))[0]
    smenv = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu["index"]),
                 PYTHONNOUSERSITE="1")
    proc = subprocess.run(
        [PYBIN, os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                             "etpr1_torch_smoke.py")],
        env=smenv, capture_output=True, text=True, timeout=600)
    ok_all &= check(report, "torch_gpu_smoke",
                    proc.returncode == 0 and "SMOKE_ALL_PASS" in proc.stdout,
                    "gpu %d" % gpu["index"])
    sim_env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [ETPR1_ROOT, os.path.join(PROJECT_ROOT, "third_party",
                                  "habitat-lab"),
         os.path.join(PROJECT_ROOT, "third_party", "habitat-sim")]),
        PYTHONNOUSERSITE="1")
    proc = subprocess.run(
        [PYBIN, os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                             "phase0_reveal_closure",
                             "stage0_habitat_smoke.py"),
         "--gpu", str(gpu["index"])],
        env=sim_env, capture_output=True, text=True, timeout=600)
    ok_all &= check(report, "habitat_sim_headless_smoke",
                    proc.returncode == 0
                    and "SIM_SMOKE_ALL_PASS" in proc.stdout,
                    "gpu %d" % gpu["index"])

    # 6. toporeveal tests (stdlib unittest runner; pytest absent by design)
    proc = subprocess.run(
        [PYBIN, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600,
        env=dict(os.environ, PYTHONPATH=PROJECT_ROOT))
    tests_ok = proc.returncode == 0
    ok_all &= check(report, "toporeveal_24_tests",
                    tests_ok, (proc.stderr.strip().splitlines() or [""])[-1])

    # 7. habitat repos clean
    def repo_state(repo):
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status",
                                "--porcelain=v1"],
                               capture_output=True, text=True,
                               check=True).stdout
        return head, dirty.strip()
    sim_head, sim_dirty = repo_state(os.path.join(PROJECT_ROOT,
                                                  "third_party",
                                                  "habitat-sim"))
    lab_head, lab_dirty = repo_state(os.path.join(PROJECT_ROOT,
                                                  "third_party",
                                                  "habitat-lab"))
    ok_all &= check(report, "habitat_sim_clean",
                    sim_head == EXPECTED["habitat_sim_head"]
                    and not sim_dirty, sim_head)
    ok_all &= check(report, "habitat_lab_clean",
                    lab_head == EXPECTED["habitat_lab_head"]
                    and not lab_dirty, lab_head)

    # 8. ETP-R1 head + dirty set + diff sha
    etp_head, _ = repo_state(ETPR1_ROOT)
    por = subprocess.run(["git", "-C", ETPR1_ROOT, "status",
                          "--porcelain=v1"],
                         capture_output=True, text=True,
                         check=True).stdout
    tracked = [ln for ln in por.splitlines() if ln and not
               ln.startswith("??")]
    untracked = [ln[3:] for ln in por.splitlines()
                 if ln.startswith("??")]
    diff = subprocess.run(["git", "-C", ETPR1_ROOT, "diff"],
                          capture_output=True, check=True).stdout
    diff_sha = hashlib.sha256(diff).hexdigest()
    ok_all &= check(report, "etpr1_head_dirty_diff",
                    etp_head == EXPECTED["etpr1_head"]
                    and len(tracked) == 14
                    and untracked == ["etpr1_compat.py"]
                    and diff_sha == EXPECTED["etpr1_tracked_diff_sha256"],
                    "head=%s tracked=%d untracked=%s" % (
                        etp_head[:12], len(tracked), untracked))

    # 9. reserves unchanged
    reserves = []
    for i in range(9, 28):
        p = os.path.join(PROJECT_ROOT, ".disk_reserve",
                         "reserve_10G_%02d.bin" % i)
        ok = (os.path.isfile(p) and not os.path.islink(p)
              and os.path.getsize(p) == EXPECTED["reserve_bytes"])
        reserves.append(ok)
    ok_all &= check(report, "reserves_19_unchanged", all(reserves),
                    "count ok=%d" % sum(reserves))

    # 10. workspace hygiene: no .part files, no symlinks in artifacts,
    #     no unexpected large files (>200 MiB) created by this batch
    part_files = glob.glob(PROJECT_ROOT + "/**/*.part", recursive=True)
    symlinks = []
    for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT,
                                                  "artifacts")):
        for name in files + dirs:
            p = os.path.join(root, name)
            if os.path.islink(p):
                symlinks.append(p)
    large = []
    for root, _dirs, files in os.walk(os.path.join(PROJECT_ROOT,
                                                   "artifacts")):
        for name in files:
            p = os.path.join(root, name)
            try:
                if os.path.getsize(p) > 200 * 1024 * 1024:
                    large.append((p, os.path.getsize(p)))
            except OSError:
                pass
    unexpected_large = [
        (p, size) for p, size in large
        if REGISTERED_LARGE_ARTIFACTS.get(p) != size
    ]
    registered_large_ok = all(
        os.path.isfile(p) and not os.path.islink(p)
        and os.path.getsize(p) == size
        for p, size in REGISTERED_LARGE_ARTIFACTS.items()
    )
    ok_all &= check(report, "workspace_hygiene",
                    not part_files and not symlinks
                    and not unexpected_large and registered_large_ok,
                    {"part": part_files[:3], "symlinks": symlinks[:3],
                     "unexpected_large": unexpected_large[:3],
                     "registered_large_count": len(large)} or "clean")

    # 11. forbidden splits never opened: scan selection/gate artifacts
    gate_txt = open(os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                                 "RXR_EN_RUNTIME_GATE.json")).read()
    opened_payloads = json.loads(gate_txt)["boundary_negatives_derived"][
        "opened_payloads"]
    forbidden_opened = any(("val_unseen" in p or "test_challenge" in p
                            or "/test/" in p) for p in opened_payloads)
    collect_split_violations = []
    for orch in glob.glob(os.path.join(
            PROJECT_ROOT, "artifacts", "runtime",
            "phase0_reveal_closure", "collect", "*",
            "ORCHESTRATOR_RESULT.json")):
        item = json.load(open(orch))
        if item.get("split") not in {"train", "val_seen"}:
            collect_split_violations.append(
                {"path": orch, "split": item.get("split")})
    ok_all &= check(report, "forbidden_splits_not_opened",
                    not forbidden_opened and not collect_split_violations,
                    {"runtime_opened_payloads": opened_payloads,
                     "collector_split_violations":
                         collect_split_violations[:3]})

    # 12/13. training/argv + network attempts across all recorded runs
    run_dirs = (glob.glob(os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                                       "rxr_en_gate", "runs", "*"))
                + glob.glob(os.path.join(PROJECT_ROOT, "artifacts",
                                         "runtime", "phase0_reveal_closure",
                                         "collect", "*")))
    bad_argv = []
    net_total = 0
    missing_child_guard_evidence = []
    for rd in run_dirs:
        orch = os.path.join(rd, "ORCHESTRATOR_RESULT.json")
        summ = os.path.join(rd, "RUN_SUMMARY.json")
        coll = os.path.join(rd, "COLLECT_SUMMARY.json")
        if os.path.isfile(orch):
            o = json.load(open(orch))
            net_total += int(o.get("network_attempts_all_processes") or 0)
            if o.get("network_guard_child_evidence_ok") is not True:
                missing_child_guard_evidence.append(orch)
        if os.path.isfile(summ):
            argv = json.load(open(summ)).get("argv") or []
            if "--run-type" not in argv or argv[
                    argv.index("--run-type") + 1] != "eval":
                bad_argv.append(rd)
        if os.path.isfile(coll):
            argv = json.load(open(coll)).get("argv") or []
            if "--run-type" not in argv or argv[
                    argv.index("--run-type") + 1] != "eval":
                bad_argv.append(rd)
    ok_all &= check(report, "no_training_argv_all_eval", not bad_argv,
                    bad_argv[:3] or "%d run dirs checked" % len(run_dirs))
    witness_path = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                                "phase0_reveal_closure", "witness",
                                "WITNESS_RETURN_EXPIRY_FIRST5.json")
    witness = json.load(open(witness_path))
    net_total += int(witness.get("network_attempts") or 0)
    ok_all &= check(report, "network_attempts_zero_all_runs",
                    net_total == 0 and not missing_child_guard_evidence,
                    "total=%d across %d run dirs + witness; missing child "
                    "guard=%d" % (net_total, len(run_dirs),
                                    len(missing_child_guard_evidence)))

    # 14. disk
    st = os.statvfs(PROJECT_ROOT)
    free = st.f_bavail * st.f_frsize
    ok_all &= check(report, "disk_free_at_least_8gib",
                    free >= EXPECTED["min_free_bytes"], free)

    # 15/16. review packet non-fabrication
    packet_path = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                               "REVIEW_PACKET_50.json")
    packet = {}
    if os.path.isfile(packet_path):
        packet = json.load(open(packet_path))
        rows = packet.get("rows") or []
        human_fields = packet.get("human_fields") or []
        rows_pending = (
            len(rows) == 50
            and len(human_fields) == 21
            and [r.get("row_order") for r in rows] == list(range(50))
            and all(r.get("reviewed") is False
                    and r.get("annotation_status") == "PENDING"
                    and all(r.get(f) is None for f in human_fields)
                    for r in rows)
        )
        ok_all &= check(report, "packet_all_reviewed_false",
                        packet.get("reviewed_true_count") == 0
                        and packet.get("all_rows_pending") is True
                        and packet["row_count"] == 50
                        and rows_pending,
                        "rows=%d" % packet["row_count"])
        ok_all &= check(report, "no_validated_events_claimed",
                        packet["non_conclusions"]["validated_events"] == 0
                        and packet["non_conclusions"]["unique_expiry_events"]
                        == 0, packet["non_conclusions"])
    else:
        ok_all &= check(report, "packet_all_reviewed_false", False,
                        "packet missing")

    # witness provisional non-counting
    if os.path.isfile(witness_path):
        w = witness
        provisional = sum(1 for e in w["episodes"]
                          if e.get("expiry_proposal_status") == "PROVISIONAL"
                          and e.get("expiry_prefix") is not None)
        ok_all &= check(report, "provisional_expiry_not_counted",
                        w.get("status") == "PASS"
                        and w.get("validated_tx_count") == 0
                        and w.get("observed_unique_expiry_prefix_count") == 0
                        and packet.get("non_conclusions", {}).get(
                            "validated_events") == 0
                        and packet.get("non_conclusions", {}).get(
                            "unique_expiry_events") == 0,
                        "provisional_expiry_prefixes=%d (not validated)"
                        % provisional)

    # Stage artifacts themselves must be internally accepted.  The
    # canonical evaluator intentionally remains NO-GO; this checks the
    # separate, SHA-cited technical refresh rather than promoting claims.
    refresh_path = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                                "PHASE0_TECHNICAL_EVIDENCE_REFRESH.json")
    refresh = json.load(open(refresh_path))
    ok_all &= check(report, "stage5_7_artifacts_fail_closed",
                    witness.get("status") == "PASS"
                    and packet.get("status") == "PASS"
                    and refresh.get("status") == "PASS"
                    and refresh["mandatory_holds"]["validated_event_count"]
                    == 0
                    and refresh["mandatory_holds"]["unique_expiry_count"]
                    == 0
                    and refresh["canonical_snapshot"][
                        "frozen_evaluator_exit_code_on_snapshot"] == 1,
                    "witness/packet/refresh accepted; canonical remains "
                    "NO-GO")

    report["status"] = "PASS" if ok_all else "FAIL"
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({"status": report["status"],
                      "failed": [c["name"] for c in report["checks"]
                                 if not c["pass"]]}, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
