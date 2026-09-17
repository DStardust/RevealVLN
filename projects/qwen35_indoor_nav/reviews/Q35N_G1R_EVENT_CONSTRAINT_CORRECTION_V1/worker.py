import importlib.util
import json
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
s = importlib.util.spec_from_file_location('derived', OUT.parent/'Q35N_G1R_ROTATION_HISTORY_V1/worker.py')
derived = importlib.util.module_from_spec(s); s.loader.exec_module(derived)
core = derived.core; core.OUT = OUT
ORIGINAL = derived.ORIGINAL
CACHE = OUT.parent/'Q35N_G1R_COVERAGE_V1/ROUTE_PLANS.jsonl'


class Engine(core.Engine):
    def discovery(self):
        pool = self.u_pool(); u = pool[22]
        assert u['episode_id'] == 1116 and u['path_index'] == 3
        core.save('PLANNING_CACHE_PROVENANCE.json', {'path': str(CACHE), 'sha256': core.digest(CACHE.read_bytes()),
            'same_gpu': 3, 'same_engine_plan_function': True, 'fresh_complete_loop_and_family_replays_required': True})
        with CACHE.open() as f:
            for line in f:
                row = json.loads(line)
                self.route_cache[row['key']] = (row['actions'], row['trace'], row['failure'])
        core.save('CACHE_COUNT.json', {'cached_route_plans': len(self.route_cache)})
        attempts = 0
        for yaw in [0, 1]:
            if time.time()-self.started > 600: raise core.Reject('WALL_CAP')
            try:
                base = self.candidate_base(u['position'], yaw)
            except core.Reject as e:
                core.log('BASE_ATTEMPTS.jsonl', {'u_index': 22, 'yaw_bin': yaw, 'reason': e.code, 'detail': e.detail})
                continue
            for tail in core.TAILS:
                attempts += 1
                try:
                    candidate = self.complete_candidate(u, yaw, base, tail)
                    core.save('FROZEN_CANDIDATE.json', candidate)
                    return candidate
                except core.Reject as e:
                    core.log('FAMILY_ATTEMPTS.jsonl', {'attempt': attempts, 'yaw_bin': yaw,
                        'tail': tail, 'reason': e.code, 'detail': e.detail})
        raise core.Reject('CORRECTED_CONSTRAINT_DIAGNOSTIC_EXHAUSTED', base_limit=2, family_attempts=attempts)


def main():
    assert not (OUT/'result.json').exists()
    core.save('SOURCE_DERIVATION.json', {'original_sha256': core.digest(ORIGINAL.read_bytes()),
        'literal_patches': derived.PATCHES, 'derived_sha256': core.digest(derived.source.encode()),
        'loop_implementation': 'original core.Engine.loop, NOT rotation-only override'})
    eng = None; started = time.time()
    try:
        eng = Engine('event_constraint'); eng.preview(); candidate = eng.discovery()
        core.save('result.json', {'decision': 'FROZEN_CANDIDATE_READY_FOR_CERTIFICATION',
            'candidate_hash': candidate['frozen_candidate_hash'], 'family_certified': False, 'scientific_pass': False})
    except core.Reject as e:
        core.save('result.json', {'decision': e.code, 'detail': e.detail, 'family_certified': False, 'scientific_pass': False})
    finally:
        if eng: eng.close()
        core.save('WALL.json', {'seconds': time.time()-started})


if __name__ == '__main__': main()
