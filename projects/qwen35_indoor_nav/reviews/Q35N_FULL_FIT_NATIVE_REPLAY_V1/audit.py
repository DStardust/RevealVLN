"""Independent CPU comparison to frozen original FIT logits; no policy calls."""
import importlib.util
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('native_cpu_audit_common', HERE / 'common.py')
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)


def compare(original, replayed, threshold):
    assert len(original) == len(replayed) == 4
    assert all(type(x) in (int, float) and math.isfinite(x) for x in original + replayed)
    error = max(abs(a - b) for a, b in zip(original, replayed))
    action_match = max(range(4), key=original.__getitem__) == max(range(4), key=replayed.__getitem__)
    return error, action_match, error <= threshold and action_match


def main():
    c.verify_lock()
    assert not (HERE / 'RESULT.json').exists()
    launch = c.read(HERE / 'LAUNCH_RESULT.json')
    assert launch['status'] == 'COMPLETE_TRANSPORT_ONLY'
    assert launch['cleanup']['exit_code'] == 0 and launch['cleanup']['exited']
    assert not launch['remaining'] and not launch['other_processes_signaled'] and not launch['holders_released']
    assert not launch['gpu_after']['GPU-734a5268-31fe-6452-105b-36cd08c3d9c8']['pids']
    plan = c.read(HERE / 'REPLAY_PROTOCOL.json')
    done = c.read(HERE / 'run_001/REPLAY_RESULT.json')
    loaded = c.read(HERE / 'run_001/MODEL_LOADED.json')
    old_loaded = c.read(LINE / 'closed_loop_bench/ordinary_stop_calibration_fit_v2/run_001/MODEL_LOADED.json')
    assert done['total_forward_decisions'] == 5519 and done['inputs'] == 5487
    assert done['fixture_forward_decisions'] == 32 and done['replay_forward_decisions'] == 5487
    assert done['backward_calls'] == done['optimizer_updates'] == done['simulator_actions'] == 0
    assert done['model_unchanged'] and done['trainable_fingerprint'] == loaded['trainable_sha256'] == old_loaded['trainable_sha256']
    assert loaded['checkpoint_sha256'] == old_loaded['checkpoint_sha256'] == plan['checkpoint_sha256']
    gate = c.read(HERE / 'run_001/BATCH_PARITY_GATE.json')
    assert gate['selected_batch_size'] == 1 and gate['comparison_batch_size_forced'] == 1
    assert len(gate['checks']) == 2
    path = HERE / 'run_001/REPLAY_LOGITS.jsonl'
    assert c.sha(path) == done['output_sha256']
    predictions = [json.loads(x) for x in path.read_text().splitlines()]
    labels = [json.loads(x) for x in (LINE / 'sft_acceptance/ordinary_stop_row_v12/SUPERVISION_ONLY.jsonl').read_text().splitlines()]
    assert len(predictions) == len(labels) == 5487
    errors = []
    mismatches = []
    for i, (p, label) in enumerate(zip(predictions, labels)):
        assert set(p) == {'index', 'record_id', 'logits'} and p['index'] == i and p['record_id'] == label['record_id']
        expected = label['occurrences'][0]['original_logits']
        assert all(o['original_logits'] == expected for o in label['occurrences'])
        error, action_match, passed = compare(expected, p['logits'], plan['max_abs_logit_error'])
        errors.append(error)
        if not passed:
            mismatches.append(dict(index=i, record_id=p['record_id'], max_abs=error,
                action_match=action_match, original_logits=expected, replay_logits=p['logits']))
    result = dict(status='PASS_NATIVE_FULL_FIT_REPLAY_ONLY' if not mismatches else 'FAIL_NATIVE_FULL_FIT_REPLAY',
        unix=time.time(), inputs=5487, original_occurrences=sum(len(x['occurrences']) for x in labels),
        maximum_absolute_logit_error=max(errors), logit_threshold=plan['max_abs_logit_error'],
        rows_above_logit_threshold=sum(e > plan['max_abs_logit_error'] for e in errors),
        different_argmax_inputs=sum(not x['action_match'] for x in mismatches),
        bitwise_equal_logit_inputs=sum(e == 0 for e in errors),
        all_original_argmax_same=all(x['action_match'] for x in mismatches),
        checkpoint_sha256=loaded['checkpoint_sha256'], trainable_fingerprint=done['trainable_fingerprint'],
        model_unchanged=True, total_forward_decisions=5519, backward_calls=0, optimizer_updates=0, simulator_actions=0,
        launch_wall_seconds=launch['wall_seconds'], replay_wall_seconds=done['wall_seconds'],
        gpu_cleanup_verified=True, old_v12_fail_preserved=True, no_stop_fit_started=True,
        navigation_sr=None, sr40_goal_achieved=False, independent_navigation_gain=False,
        source_lock_sha256=c.sha(HERE / 'SOURCE_LOCK.json'), replay_logits_sha256=c.sha(path))
    c.write(HERE / 'DISCREPANCIES.json', mismatches, True)
    c.write(HERE / 'RESULT.json', result, True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
