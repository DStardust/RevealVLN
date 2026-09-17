"""Supplemental GPU1 XML guard; never signals an external or worker process.

Run only after main-agent admission. The sealed active run remains unchanged.
CPU helpers are independently testable; importing this file performs no I/O.
"""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
FEEDBACK = HERE.parent
LINE = FEEDBACK.parents[2]
PROJECT = LINE.parents[1]
RUN = FEEDBACK / "run_v1"
UUID = "GPU-734a5268-31fe-6452-105b-36cd08c3d9c8"
SUPERVISOR, WORKER = 3053374, 3053406
EXPECTED_START = {SUPERVISOR: 130817340, WORKER: 130817372}
EXPECTED_CMD = {
    SUPERVISOR: [".tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3", "-I", "-S", "-B",
                 "projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/feedback_generation_v1/run.py"],
    WORKER: [str(LINE / ".envs/q35n_habitat_v017_g0r/bin/python3"), "-I", "-B", str(FEEDBACK / "worker.py")],
}
EXPECTED_CWD = {SUPERVISOR: PROJECT, WORKER: LINE}


def mib(text):
    match = re.fullmatch(r"\s*(\d+)\s+MiB\s*", text or "")
    if not match:
        raise ValueError("missing or unsupported GPU memory field")
    return int(match.group(1))


def parse_xml(text):
    root = ET.fromstring(text)
    cards = root.findall("gpu")
    if len(cards) != 1 or cards[0].findtext("uuid") != UUID:
        raise ValueError("GPU UUID/count mismatch")
    card = cards[0]
    total = mib(card.findtext("fb_memory_usage/used"))
    container = card.find("processes")
    if container is None or (container.text or "").strip() not in ("",):
        raise ValueError("GPU process inventory missing")
    processes = {}
    for entry in container.findall("process_info"):
        pid = int(entry.findtext("pid"))
        if pid <= 0 or pid in processes:
            raise ValueError("invalid or duplicate GPU process")
        usage = entry.findtext("used_memory")
        if usage is None:
            usage = entry.findtext("used_gpu_memory")
        processes[pid] = {"memory_mib": mib(usage), "type": entry.findtext("type")}
    return {"uuid": UUID, "memory_mib": total, "processes": processes}


def assess(sample, owned_pids):
    external = {p: v["memory_mib"] for p, v in sample["processes"].items() if p not in owned_pids}
    external_total = sum(external.values())
    own_upper = sample["memory_mib"] - external_total
    reasons = []
    if any(value > 768 for value in external.values()):
        reasons.append("EXTERNAL_PROCESS_EXCEEDS_768_MIB")
    if external_total > 2048:
        reasons.append("EXTERNAL_TOTAL_EXCEEDS_2048_MIB")
    # Inconsistent/racing measurements fail closed instead of clamping to zero.
    if own_upper < 0 or sum(v["memory_mib"] for v in sample["processes"].values()) > sample["memory_mib"]:
        reasons.append("GPU_MEMORY_ACCOUNTING_INCONSISTENT")
    if own_upper >= 4096:
        reasons.append("OWN_CONSERVATIVE_UPPER_NOT_BELOW_4096_MIB")
    return {"pass": not reasons, "reasons": reasons, "external_mib": external,
            "external_total_mib": external_total, "own_conservative_upper_mib": own_upper}


def parse_stat(text):
    prefix, sep, suffix = text.rpartition(") ")
    if not sep:
        raise ValueError("invalid proc stat")
    fields = suffix.split()
    return {"pid": int(prefix.split(" (", 1)[0]), "ppid": int(fields[1]), "start_ticks": int(fields[19])}


def read_identity(pid):
    root = Path("/proc") / str(pid)
    before = parse_stat((root / "stat").read_text())
    cmd = [x.decode() for x in (root / "cmdline").read_bytes().split(b"\0") if x]
    cwd = os.readlink(root / "cwd")
    after = parse_stat((root / "stat").read_text())
    if before != after:
        raise ValueError("process identity changed while reading")
    return dict(before, cmd=cmd, cwd=cwd)


def validate_identity(identity, pid):
    if (identity["pid"] != pid or identity["start_ticks"] != EXPECTED_START[pid]
            or identity["cmd"] != EXPECTED_CMD[pid] or identity["cwd"] != str(EXPECTED_CWD[pid])):
        raise ValueError("owned process exact identity mismatch")
    if pid == WORKER and identity["ppid"] != SUPERVISOR:
        raise ValueError("worker no longer belongs to expected supervisor")
    return True


def is_owned_descendant(pid):
    """No unknown PID is subtracted as external if ancestry cannot be read."""
    seen = set()
    while pid > 1 and pid not in seen and len(seen) < 64:
        seen.add(pid)
        info = parse_stat((Path("/proc") / str(pid) / "stat").read_text())
        if pid in EXPECTED_START:
            return info["start_ticks"] == EXPECTED_START[pid]
        pid = info["ppid"]
    return False


def gpu_xml():
    return subprocess.check_output(["nvidia-smi", "-i", "1", "-q", "-x"], text=True, timeout=10)


def remaining_budget(started_unix, now_unix, now_boot, ticks_per_second):
    # Supervisor birth precedes its own started clock: this is conservative and
    # also prevents a wall-clock adjustment from extending the original budget.
    return min(started_unix + 3000 - now_unix,
               EXPECTED_START[SUPERVISOR] / ticks_per_second + 3000 - now_boot)


def main():
    process = json.loads((RUN / "PROCESS.json").read_text())
    if process["supervisor_pid"] != SUPERVISOR or process["pid"] != WORKER:
        raise ValueError("PROCESS.json belongs to a different run")
    started = process["started_unix"]
    if not isinstance(started, (int, float)) or not math.isfinite(started):
        raise ValueError("invalid original start time")
    deadline = started + 3000  # Original node cutoff, never reset at guard startup.
    def remaining():
        return remaining_budget(started, time.time(), time.clock_gettime(time.CLOCK_BOOTTIME), os.sysconf("SC_CLK_TCK"))
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("pidfd unavailable; refuse race-prone signalling fallback")
    supervisor_fd = None
    out = HERE / "execution_v1"
    out.mkdir(exist_ok=False)
    def save(name, value):
        with (out / name).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    def append(value):
        with (out / "SAMPLES.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    signalled = False
    signal_error = None
    error = None
    seen_owned = {SUPERVISOR, WORKER}
    def terminate_owned_supervisor(reason):
        nonlocal signalled, signal_error
        if signalled:
            return
        try:
            validate_identity(read_identity(SUPERVISOR), SUPERVISOR)
            if supervisor_fd is None:
                raise RuntimeError("no verified supervisor pidfd")
            signal.pidfd_send_signal(supervisor_fd, signal.SIGTERM)
            signalled = True
            append({"event": "SIGTERM_VERIFIED_OWN_SUPERVISOR", "reason": reason, "pid": SUPERVISOR})
        except Exception as exc:
            signal_error = repr(exc)
            append({"event": "NO_SIGNAL_IDENTITY_OR_PIDFD_FAILURE", "error": signal_error})
    try:
        if not (RUN / "SUPERVISOR_RESULT.json").exists():
            identity = read_identity(SUPERVISOR)
            validate_identity(identity, SUPERVISOR)
            supervisor_fd = os.pidfd_open(SUPERVISOR)
            validate_identity(read_identity(SUPERVISOR), SUPERVISOR)
            validate_identity(read_identity(WORKER), WORKER)
            save("ADMISSION.json", {"supervisor_identity": identity, "worker_identity": read_identity(WORKER),
                                     "deadline_unix": deadline, "guard_started_unix": time.time(), "pidfd": True})
        while not (RUN / "SUPERVISOR_RESULT.json").exists():
            try:
                if remaining() <= 0:
                    terminate_owned_supervisor("ORIGINAL_3000_SECOND_DEADLINE")
                raw = gpu_xml()
                sample = parse_xml(raw)
                owned = {p for p in sample["processes"] if is_owned_descendant(p)}
                seen_owned.update(owned)
                assessment = assess(sample, owned)
                append({"unix": time.time(), "original_elapsed": time.time() - started,
                        "sample": sample, "owned_gpu_pids": sorted(owned), "assessment": assessment})
                if not assessment["pass"]:
                    terminate_owned_supervisor(assessment["reasons"])
                if remaining() <= 0:
                    terminate_owned_supervisor("ORIGINAL_3000_SECOND_DEADLINE")
            except Exception as exc:
                error = repr(exc)
                append({"unix": time.time(), "error": error})
                terminate_owned_supervisor("MONITORING_FAILED_CLOSED")
            if signalled or error or signal_error:
                # Only cleanup waiting, never extra worker execution permission.
                for _ in range(8):
                    if (RUN / "SUPERVISOR_RESULT.json").exists():
                        break
                    time.sleep(5)
                break
            time.sleep(min(5, max(0.05, remaining())))
    except Exception as exc:
        error = repr(exc)
        terminate_owned_supervisor("ADMISSION_OR_GUARD_EXCEPTION")
    finally:
        try:
            final_xml = gpu_xml()
            with (out / "FINAL_GPU.xml").open("x") as stream:
                stream.write(final_xml)
            final = parse_xml(final_xml)
            remaining = sorted(set(final["processes"]) & seen_owned)
            save("RESULT.json", {"supervisor_result_present": (RUN / "SUPERVISOR_RESULT.json").exists(),
                                  "own_renderer_absent": not remaining, "remaining_owned_pids": remaining,
                                  "signal_sent_to_own_supervisor": signalled, "external_processes_signalled": 0,
                                  "worker_direct_signals": 0, "error": error, "signal_error": signal_error,
                                  "gpu_after": final, "original_deadline_unix": deadline})
        finally:
            if supervisor_fd is not None:
                os.close(supervisor_fd)


if __name__ == "__main__":
    main()
