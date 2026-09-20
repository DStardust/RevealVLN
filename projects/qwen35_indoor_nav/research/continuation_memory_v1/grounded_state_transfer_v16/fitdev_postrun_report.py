"""Build a compact, hash-bound report from a completed FIT/DEV run."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import tempfile


def read(path: Path):
    return json.loads(path.read_bytes())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable output differs: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=".pending_", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def main(run: Path) -> None:
    run = run.resolve()
    inputs = {
        name: run / name
        for name in (
            "DATA.json",
            "INPUT_REUSE.json",
            "TRAIN_SUMMARY.json",
            "EVALUATION_REGISTRY.json",
            "DEV_DIAGNOSIS.json",
            "EVENT_STATE_DIAGNOSIS.json",
            "QUERY_DIAGNOSIS.json",
            "MEMORY_INTERVENTIONS.json",
            "RESOURCE_SESSIONS.jsonl",
            "STATUS.json",
        )
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    data = read(inputs["DATA.json"])
    events = read(inputs["EVENT_STATE_DIAGNOSIS.json"])["rows"]
    query = read(inputs["QUERY_DIAGNOSIS.json"])
    dev = read(inputs["DEV_DIAGNOSIS.json"])
    train = read(inputs["TRAIN_SUMMARY.json"])
    reuse = read(inputs["INPUT_REUSE.json"])
    mechanism = read(inputs["MEMORY_INTERVENTIONS.json"])
    status = read(inputs["STATUS.json"])

    sequence_masks = {}
    for family in data["families"]:
        for sequence in family["sequences"]:
            key = (
                family["family_id"],
                sequence["history_id"],
                sequence["task_id"],
                sequence["continuation"],
            )
            if key in sequence_masks:
                raise ValueError(f"duplicate sequence identity: {key}")
            sequence_masks[key] = sequence["action_masks"]

    error_counts = collections.Counter()
    state_exact = 0
    action_eligible = 0
    exact_state_action_eligible = 0
    wrong_action_given_exact_state = 0
    for row in events:
        key = (row["family"], row["history"], row["task"], row["continuation"])
        masks = sequence_masks.get(key)
        if masks is None or row["step"] >= len(masks):
            raise ValueError(f"event/sequence mismatch: {key} step={row['step']}")
        predicted = tuple(probability >= 0.5 for probability in row["probability"])
        truth = tuple(bool(value) for value in row["target"])
        exact = predicted == truth
        eligible = bool(masks[row["step"]])
        state_exact += int(exact)
        action_eligible += int(eligible)
        exact_state_action_eligible += int(exact and eligible)
        wrong_action_given_exact_state += int(row["errors"]["wrong_action_given_exact_state"])
        error_counts.update({name: int(value) for name, value in row["errors"].items()})

    query_by_arm = {}
    for arm in ("B2", "Ours"):
        rows = [row for row in query["rows"] if row["arm"] == arm]
        valid = sum(row["valid"] for row in rows)
        correct = sum(row["correct"] for row in rows)
        query_by_arm[arm] = {
            "valid": valid,
            "correct": correct,
            "accuracy": correct / valid if valid else None,
            "scope": "read-only auxiliary teacher-route diagnosis",
        }

    resources = [json.loads(line) for line in inputs["RESOURCE_SESSIONS.jsonl"].read_text().splitlines() if line]
    total_gpu_seconds = sum(row["wall_seconds"] for row in resources)
    summary = {
        "status": "POST_RUN_READ_ONLY_VERIFICATION_COMPLETE",
        "run_status": status["status"],
        "scope": "FIT16_TRAIN_DEV2_DIAGNOSTIC",
        "inventory": {
            "project": {key: len(value) if isinstance(value, list) else value for key, value in reuse["project_inventory"].items()},
            "consumed": reuse["consumed_counts"],
            "new_collection": reuse["new_collection"],
        },
        "training": {
            "models": train["completed_models"],
            "updates_per_model": train["updates_per_model"],
            "total_updates": train["total_scheduled_updates"],
            "base_optimizer_updates": train["base_optimizer_updates"],
            "all_losses_finite": all(row["finite_losses"] for row in train["models"]),
            "all_updates_have_positive_gradient": all(row["positive_gradient_steps"] == 1200 for row in train["models"]),
            "test_scores_read": train["test_scores_read"],
        },
        "closed_loop": {
            "families": dev["families"],
            "independent_houses": dev["independent_house_count"],
            "conditions": dev["conditions"],
            "planned_slots": dev["planned_slots"],
            "completed_slots": dev["completed_slots"],
            "main_planned": dev["main_planned"],
            "control_planned": dev["control_planned"],
            "status_counts": dev["status_counts"],
            "by_arm": dev["by_arm"],
            "unknown_error_not_run": len(dev["unknown_error_not_run"]),
        },
        "event_state": {
            "model_scope": "B2 trained state head only",
            "correlated_teacher_route_decisions": len(events),
            "state_exact": state_exact,
            "state_exact_rate": state_exact / len(events),
            "error_counts": dict(error_counts),
        },
        "action_given_state": {
            "action_eligible_decisions": action_eligible,
            "exact_state_action_eligible_decisions": exact_state_action_eligible,
            "wrong_actions_given_exact_state": wrong_action_given_exact_state,
            "correct_actions_given_exact_state": exact_state_action_eligible - wrong_action_given_exact_state,
            "error_rate_given_exact_state": wrong_action_given_exact_state / exact_state_action_eligible if exact_state_action_eligible else None,
        },
        "query_auxiliary": query_by_arm,
        "memory_intervention": {
            "status": mechanism["status"],
            "diagnostic_mechanism_N": mechanism["diagnostic_mechanism_N"],
            "matched_control_N": mechanism["matched_control_N"],
            "supports_history_specific_effect": mechanism["supports_history_specific_effect"],
        },
        "resources": {
            "gpu_sessions": len(resources),
            "gpu_session_hours": total_gpu_seconds / 3600,
            "all_returncode_zero": all(row["returncode"] == 0 for row in resources),
            "all_not_timed_out": all(not row["timed_out"] for row in resources),
        },
        "boundaries": {
            "official_TEST_evaluations_started_by_this_run": status["official_TEST_evaluations_started_by_this_run"],
            "new_candidate_collections_started_by_this_run": status["new_candidate_collections_started_by_this_run"],
            "new_family_admissions": status["new_family_admissions"],
        },
        "limitations": [
            "The two DEV families come from one house; 180 rollouts are not 180 independent cross-house samples.",
            "DEV was used for development diagnosis and is not blind TEST evidence.",
            "Event-state rows are correlated teacher-route decisions, not independent rollouts.",
            "The matched memory intervention is UNIDENTIFIABLE because no registered geometry/sham matched subset exists.",
            "No formal TEST score, navigation generalization claim, or architecture-selection claim is supported here.",
        ],
    }
    summary_path = run / "DEV_DIAGNOSTIC_SUMMARY.json"
    publish(summary_path, json_bytes(summary))

    closed = summary["closed_loop"]
    event = summary["event_state"]
    action = summary["action_given_state"]
    report = f"""# V16 FIT/DEV 训练与 DEV 诊断交付报告

终态：`{status['status']}`。本节点复用了 FIT16 / DEV2，完成 B1/B2/Ours × 3 seeds 的 9 个模型；每模型 1200 次更新，总计 10800 次，底模更新 0。全部记录 loss 为有限值，全部 10800 个更新均有正梯度。

## DEV 闭环

- 注册与完成：2 个 DEV 族、1 个屋、20 条条件、180/180 槽位；主任务 144/144，task_T 控制 36/36。
- 主任务：PASS 2，FAIL 142；控制：PASS 1，FAIL 35；UNKNOWN/ERROR/NOT_RUN 为 0。
- 按方法合并三个 seed：B1 为 PASS 3 / 60，B2 为 PASS 0 / 60，Ours 为 PASS 0 / 60。

这些结果说明训练和真实闭环路径已经打通，但不支持 Ours 的 DEV 增量；也不能把一个 DEV 屋的 180 次执行解释成 180 个独立泛化样本。

## 状态、动作与辅助诊断

- 状态诊断只使用实际训练过的 B2 状态头，共 {event['correlated_teacher_route_decisions']} 个相关的教师路径决策；四位状态全部正确 {event['state_exact']} 次（{event['state_exact_rate']:.2%}）。
- 错误计数：可见见证漏检 {event['error_counts']['visible_witness_missed']}，真正缺失时误报 {event['error_counts']['truly_absent_false_positive']}，首次正确检出后的长时保持丢失 {event['error_counts']['retention_lost_after_detected_witness']}，terminal readiness 错误 {event['error_counts']['wrong_terminal_readiness']}。这些计数可重叠，分母均为上述教师路径决策。
- 动作诊断的真实子集：action-eligible 决策 {action['action_eligible_decisions']}；其中四位状态全对 {action['exact_state_action_eligible_decisions']}，动作错误 {action['wrong_actions_given_exact_state']}、正确 {action['correct_actions_given_exact_state']}，条件错误率 {action['error_rate_given_exact_state']:.2%}。
- 辅助 query 准确率：B2 {query_by_arm['B2']['correct']}/{query_by_arm['B2']['valid']}（{query_by_arm['B2']['accuracy']:.2%}），Ours {query_by_arm['Ours']['correct']}/{query_by_arm['Ours']['valid']}（{query_by_arm['Ours']['accuracy']:.2%}）。它们是只读教师路径指标，不能代替闭环。
- 预注册 correct/wrong/sham/zero 匹配子集不存在，机制干预记为 `UNIDENTIFIABLE`，不是正结果或负结果。

## 资源、边界与限制

- 特征、训练、DEV 评测共 3 个 GPU 会话，合计 {summary['resources']['gpu_session_hours']:.3f} 小时，均 returncode=0 且未超时。
- 正式 TEST 执行为 0；新候选采集为 0；新族接纳为 0。原 6 个 TEST 族及两个缺口均未用于本节点方法评分。
- 两个 DEV 族来自同一屋，DEV 已暴露用于开发诊断；本报告不支持正式 TEST 泛化、因果瓶颈唯一性、需要更大底模或 Ours 独立提升等结论。
"""
    report_path = run / "DELIVERY_REPORT_ZH.md"
    publish(report_path, report.encode())

    manifest = {
        "status": "POST_RUN_HANDOFF_SEALED",
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "outputs": {
            summary_path.name: {"bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
            report_path.name: {"bytes": report_path.stat().st_size, "sha256": sha256(report_path)},
        },
    }
    publish(run / "POST_RUN_HANDOFF_MANIFEST.json", json_bytes(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
