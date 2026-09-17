"""Independent CPU audit of saved complete pairs, including terminal RGB."""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c


def main():
    protocol = c.read(HERE / 'PROTOCOL.json')
    assert c.sha(protocol['checkpoint']) == protocol['checkpoint_sha256']
    sources = c.read(HERE / 'SOURCE_LOCK.json')['existing_sources']
    assert all(c.sha(path) == digest for path, digest in sources.items())
    assert c.sha(HERE / 'cycle_policy.py') == c.sha(c.V3 / 'cycle_policy.py')
    order = c.read(HERE / 'PAIR_ORDER.json')
    pairs = c.committed_pairs()
    assert set(pairs) == set(range(100)), 'FULL_DENOMINATOR_REQUIRED'
    assert len({p['episode_id'] for p in pairs.values()}) == 100
    decisions = prefixes = no_override = 0
    fields = {'input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values',
              'image_grid_thw', 'exec_index', 'action_index'}
    for rank, pair in sorted(pairs.items()):
        assert all(pair[k] == order[rank][k] for k in ('rank', 'index', 'episode_id', 'house', 'inference_order'))
        assert pair['protocol_sha256'] == c.sha(HERE / 'PROTOCOL.json')
        folder = (HERE / pair['artifact']).parent
        logs = {}
        traces = {}
        initials = {}
        for arm in ('A', 'B'):
            for name, digest in pair['logs'][arm].items():
                assert c.sha(folder / arm / name) == digest
            logs[arm] = c.records(folder / arm / 'POLICY_STEPS.jsonl')
            traces[arm] = c.records(folder / arm / 'STEPS_PRIVILEGED.jsonl')
            initials[arm] = c.read(folder / arm / 'INITIAL_STATE.json')
            episode = pair['episodes'][arm]
            assert len(logs[arm]) == len(traces[arm]) == episode['steps'] <= 500
            assert episode['stopped'] or episode['steps'] == 500
            assert episode['success'] == float(episode['stopped'] and episode['distances'][-1] < 3)
            counts = {}
            history = []
            images = [initials[arm]['rgb_sha256']]
            for step, (d, trace) in enumerate(zip(logs[arm], traces[arm]), 1):
                assert len(d['logits']) == 4 and all(math.isfinite(v) for v in d['logits'])
                assert set(d['processed']) == fields
                raw = d['raw']
                key = hashlib.sha256(json.dumps([raw['instruction'], raw['rgb_sha256'], raw['executed_history']],
                    ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                assert key == raw['input_key']
                assert raw['instruction'] == episode['instruction']
                assert raw['rgb_sha256'] == images[-2:] and raw['executed_history'] == history[-8:]
                native = max(range(4), key=lambda i: d['logits'][i])
                repeated = key in counts
                attempts = counts.setdefault(key, [0, 0, 0])
                choice = max(range(3), key=lambda i: (-attempts[i], d['logits'][i])) if arm == 'B' and repeated and native != 3 else native
                assert d['native_action'] == c.ACTIONS[native]
                assert d['executed_action'] == trace['action'] == c.ACTIONS[choice]
                assert d['repeated_input'] == repeated and d['override'] == (choice != native)
                assert d['step'] == trace['step'] == step
                assert trace['position'] == episode['positions'][step]
                assert trace['distance_to_goal'] == episode['distances'][step]
                if choice != 3:
                    attempts[choice] += 1
                    history.append(c.ACTIONS[choice])
                else:
                    assert step == len(logs[arm]) and episode['stopped']
                images.append(trace['rgb_sha256'])
                decisions += 1
        assert initials['A'] == initials['B']
        first = next((i for i, d in enumerate(logs['B']) if d['override']), None)
        prefix_count = len(logs['B']) if first is None else first + 1
        assert prefix_count == pair['audit']['prefix_decisions']
        assert len(logs['A']) >= prefix_count
        for a, b in zip(logs['A'][:prefix_count], logs['B'][:prefix_count]):
            audit = c.prefix_compare(a, b)
            assert audit['input_prefix_matched'] and audit['action_prefix_matched']
            assert audit['logits_bitwise_equal'], 'OBSERVED_BITWISE_CLAIM_NOT_REPRODUCED'
            prefixes += 1
        if first is None:
            assert len(logs['A']) == len(logs['B'])
            assert traces['A'] == traces['B'], 'NO_OVERRIDE_FULL_TRACE_INCLUDING_TERMINAL_RGB'
            no_override += 1
    result = dict(status='SAVED_EVIDENCE_VERIFIED', complete_pairs=len(pairs), complete_episodes=2*len(pairs),
        decisions=decisions, prefix_decision_pairs=prefixes, no_override_full_trace_pairs=no_override,
        checkpoint_unchanged=True, frozen_sources_unchanged=True, native_and_recovery_recomputed=True, executed_history_and_rgb_checked=True,
        terminal_rgb_checked=True, log_hashes_checked=True, state_seals_required=True,
        cpu_only=True, new_model_forwards=0, new_environment_decisions=0)
    c.write(HERE / 'SAVED_EVIDENCE_AUDIT.json', result, True)
    print(result)


if __name__ == '__main__':
    main()
