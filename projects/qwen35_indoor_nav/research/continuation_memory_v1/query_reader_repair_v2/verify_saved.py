"""CPU verification of saved updates and the frozen source identities."""
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
sys.path.insert(0, str(LINE / 'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c


def main():
    run = HERE / 'run_001'
    config, result = c.read(HERE / 'CONFIG.json'), c.read(run / 'RESULT.json')
    sources = {path: c.sha(LINE / path) == digest for path, digest in config['source_hashes'].items()}
    assert all(sources.values())
    initial = torch.load(HERE.parent / 'pilot/run_002/INITIAL_MEMORY.pt', map_location='cpu', weights_only=True)
    checks = {}
    for name, arm in result['arms'].items():
        path = run / (name + '_MEMORY.pt')
        state = torch.load(path, map_location='cpu', weights_only=True)
        tensors = {key: c.tensor_identity(value) for key, value in state.items()}
        digest = hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest()
        assert digest == arm['final_state_sha256']
        steps = c.records(run / (name + '_STEPS.jsonl'))
        assert [row['step'] for row in steps] == list(range(1, 101))
        assert state.keys() == initial.keys()
        changed = [key for key in state if not torch.equal(state[key], initial[key])]
        assert changed and all(bool(torch.isfinite(value).all()) for value in state.values())
        old_name = 'B2' if name == 'B2' else 'Ours'
        old = torch.load(HERE.parent / f'pilot/run_002/{old_name}_MEMORY.pt', map_location='cpu', weights_only=True)
        old_equal = all(torch.equal(state[key], old[key]) for key in state)
        assert old_equal == arm['identical_to_prior_checkpoint']
        checks[name] = dict(updates=len(steps), file_sha256=c.sha(path), file_bytes=path.stat().st_size,
                           state_sha256=digest, changed_tensors=changed, all_finite=True,
                           equal_to_prior_arm=old_equal, local_only_checkpoint=True)
    evidence = dict(status='SAVED_UPDATE_EVIDENCE_VERIFIED', gpu_used=False, source_checks=sources,
                    arms=checks, optimizer_updates=sum(row['updates'] for row in checks.values()),
                    qwen_or_navigation_gain_tested=False)
    c.write(HERE / 'SAVED_EVIDENCE_AUDIT.json', evidence, True)
    print(evidence)


if __name__ == '__main__':
    main()
