"""CPU-only inspection of saved training windows and installed projection code.

No model loading, optimizer updates, inference, simulator, source edits, or
checkpoint selection. The observed training prefix is fixed at update 1200.
"""
import ast
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
TRAIN = LINE / 'sft_acceptance/ordinary_history8_paired_train_r1'
ARMS = ('control_recent2', 'treatment_prefix8')
END = 1200


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect():
    windows = {}
    sources = {}
    for arm in ARMS:
        path = TRAIN / arm / 'run_001/PROGRESS.jsonl'
        # The file is append-only and live. Hash only complete selected lines,
        # not its changing tail. All selected windows were already completed.
        raw = path.read_bytes().splitlines(keepends=True)
        chosen = []
        rows = {}
        for line in raw:
            if not line.endswith(b'\n'):
                continue
            row = json.loads(line)
            if row['updates'] > END:
                continue
            assert row['arm'] == arm and row['updates'] not in rows
            rows[row['updates']] = row
            chosen.append(line)
        assert sorted(rows) == [1] + list(range(20, END + 1, 20))
        previous = 0
        for update in sorted(rows):
            matrix = rows[update]['metrics']['confusion']
            assert len(matrix) == 4 and all(len(r) == 4 for r in matrix)
            assert all(v >= 0 and float(v).is_integer() for r in matrix for v in r)
            assert sum(map(sum, matrix)) == (update - previous) * 96
            assert list(map(sum, matrix)) == rows[update]['metrics']['action_target_counts']
            previous = update
        windows[arm] = rows
        sources[arm] = dict(path=str(path), selected_prefix_lines=len(chosen),
                            selected_prefix_sha256=sha(b''.join(chosen)))
    assert all(windows[ARMS[0]][u]['metrics']['action_target_counts'] ==
               windows[ARMS[1]][u]['metrics']['action_target_counts']
               for u in windows[ARMS[0]])
    periods = []
    for lower, upper in ((0, 200), (200, 1000), (1000, 1200)):
        period = dict(first_update=lower + 1, last_update=upper, arms={})
        for arm in ARMS:
            rows = [r for u, r in windows[arm].items() if lower < u <= upper]
            matrix = [[int(sum(r['metrics']['confusion'][i][j] for r in rows))
                       for j in range(4)] for i in range(4)]
            support = list(map(sum, matrix))
            n = sum(support)
            assert n == (upper - lower) * 96
            predicted_stop = sum(r[3] for r in matrix)
            period['arms'][arm] = dict(decisions=n, confusion=matrix,
                accuracy=sum(matrix[i][i] for i in range(4)) / n,
                action_recall=[matrix[i][i] / support[i] if support[i] else None for i in range(4)],
                support=support, predicted_stop=predicted_stop,
                stop_precision=matrix[3][3] / predicted_stop if predicted_stop else None)
        periods.append(period)

    source = LINE / '.envs/q35n_qwen_g2_v1/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py'
    config_path = LINE / 'runtime/models/Qwen3.5-2B_15852e8/config.json'
    config = json.loads(config_path.read_text())['text_config']
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5GatedDeltaNet')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    calls = {n.func.attr for n in ast.walk(forward) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'self'}
    assert {'in_proj_qkv', 'out_proj'} <= calls
    h = config['hidden_size']
    kd = config['linear_key_head_dim'] * config['linear_num_key_heads']
    vd = config['linear_value_head_dim'] * config['linear_num_value_heads']
    nlinear = config['layer_types'].count('linear_attention')
    nfull = config['layer_types'].count('full_attention')
    assert (nlinear, nfull, h, kd, vd) == (18, 6, 2048, 2048, 2048)
    audit_path = LINE / 'reviews/Q35N_HISTORY_PAIR_R1_CHECKPOINTS/treatment_prefix8_STAGE1000.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'PASS_STAGE1000_NUMERICAL_ONLY'
    current = sum(r['coordinates'] for r in audit['actual_parameter_changes'].values())
    added = 8 * ((h + 2 * kd + vd) + (vd + h)) * nlinear
    assert current == 434180 and added == 1769472
    inventory = dict(config_sha256=sha(config_path.read_bytes()), source=str(source),
        source_sha256=sha(source.read_bytes()), checkpoint_audit_sha256=sha(audit_path.read_bytes()),
        full_attention_layers=nfull, frozen_linear_attention_layers=nlinear,
        actual_current_trainable_parameters=current,
        hypothetical_standard_rank8_linear_qkv_out_adapters_added_parameters=added,
        hypothetical_total_trainable_parameters=current + added,
        extra_fp32_weights_gradients_two_adam_moments_bytes=added * 16,
        activation_or_total_memory_estimate=None,
        runtime_adapter_reachability_verified=False,
        explanation='Native forward calls both projection modules, but this static inspection does not verify decorated runtime dispatch or actual gradients. No adapters created.',
        novelty_claim=False, training_authorized_by_this_inspection=False)
    return dict(status='PASS_SAVED_LOG_AND_STATIC_SOURCE_INSPECTION_ONLY', unix=time.time(),
        source_prefixes=sources, matched_target_counts=True, end_update=END,
        periods=periods, projection_inventory=inventory,
        limitations=['Changing models measured on training batches, not held-out validation or fixed-checkpoint inference.',
                     'Both arms start from the same two-frame-trained bridge; this tests finite-budget input adaptation, not history capacity from scratch.',
                     'No global pooled CE is reconstructed from window mean CE because weight sums were not saved.',
                     'No navigation gain, full-benchmark eligibility, or causal capacity bottleneck is established.'],
        new_training_updates=0, new_policy_forwards=0, new_simulator_actions=0,
        frozen_sources_modified=False)


if __name__ == '__main__':
    result = inspect()
    with (HERE / 'RESULT.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))
