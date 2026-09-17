import collections
import hashlib
import importlib.util
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ORIGINAL = LINE/'reviews/Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2/engine.py'
source = ORIGINAL.read_text()
PATCHES = [('cfg.gpu_device_id=2;', 'cfg.gpu_device_id=3;'),
           ("self.loop(position,bin,'L',['D','K','B'])", "self.loop(position,bin,'L',['K','B'])")]
for a, b in PATCHES:
    assert source.count(a) == 1
    source = source.replace(a, b)
s = importlib.util.spec_from_file_location('core', ORIGINAL)
core = importlib.util.module_from_spec(s)
exec(compile(source, str(ORIGINAL), 'exec'), core.__dict__)
core.OUT = OUT


class Engine(core.Engine):
    def loop(self, position, bin, kind, forbidden):
        failures = []
        for n in range(1, 13):
            for side, back in [('L', 'R'), ('R', 'L')]:
                outbound = [side]*n
                raw, inverse = core.inverse(outbound)
                actions = outbound + inverse
                assert core.transform(raw) == core.transform(inverse)
                tr = self.trace(position, bin, actions)
                if not self.pattern(tr, [kind], forbidden):
                    failures.append('ROTATION_EVENT_PATTERN')
                    continue
                rawtr = self.trace(position, bin, outbound+raw)
                p, r = core.pose_spread(tr['observations'][0]['pose'], tr['observations'][-1]['pose'])
                core.log('ROTATION_LOOPS.jsonl', {'kind': kind, 'raw_trace': rawtr,
                    'compressed_trace': tr, 'raw_inverse': raw, 'compressed_inverse': inverse,
                    'position_spread_m': p, 'angle_spread_rad': r})
                if not rawtr['complete'] or p > 1e-4 or r > 1e-4:
                    failures.append('ROTATION_LEGALITY_OR_DRIFT'); continue
                return actions, tr, {'kind': kind, 'n': n, 'side': side}, failures
        raise core.Reject('NO_VALID_'+kind+'_ROTATION_LOOP', subfailures=dict(collections.Counter(failures)))

    def discovery(self):
        pool = self.u_pool(); bases = 0; full = 0
        for yaw in [0, 12, 6, 18, 3, 15, 9, 21]:
            for index, u in enumerate(pool):
                bases += 1; self.counts['base_configurations'] = bases
                try:
                    if u['position'] is None: raise core.Reject('NONFINITE_U')
                    base = self.candidate_base(u['position'], yaw)
                except core.Reject as e:
                    core.log('BASE_ATTEMPTS.jsonl', {'attempt': bases, 'u_index': index, 'yaw_bin': yaw,
                        'reason': e.code, 'detail': e.detail})
                    base = None
                if base is not None:
                    for tail in core.TAILS:
                        full += 1; self.counts['family_configurations'] = full
                        try:
                            result = self.complete_candidate(u, yaw, base, tail)
                            core.save('FROZEN_CANDIDATE.json', result)
                            return result
                        except core.Reject as e:
                            core.log('FAMILY_ATTEMPTS.jsonl', {'attempt': full, 'base_attempt': bases,
                                'tail': tail, 'reason': e.code, 'detail': e.detail})
                if bases % 8 == 0:
                    core.save('PROGRESS.json', {'bases': bases, 'families': full, 'counts': dict(self.counts)})
                    print(f'{bases}/256 bases; {full} full family candidates', flush=True)
        raise core.Reject('ROTATION_HISTORY_POOL_EXHAUSTED', bases=bases, family_configurations=full)


def main():
    assert not (OUT/'result.json').exists()
    core.save('SOURCE_DERIVATION.json', {'original': str(ORIGINAL), 'sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        'literal_patches': PATCHES, 'derived_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'overrides': ['Engine.loop: real rotation-only loops', 'Engine.discovery: frozen 256-base ordering']})
    eng = None; started = time.time()
    try:
        eng = Engine('rotation_history'); eng.preview(); candidate = eng.discovery()
        core.save('result.json', {'decision': 'FROZEN_CANDIDATE_READY_FOR_CERTIFICATION',
            'candidate_hash': candidate['frozen_candidate_hash'], 'family_certified': False, 'scientific_pass': False})
    except core.Reject as e:
        core.save('result.json', {'decision': e.code, 'detail': e.detail, 'family_certified': False, 'scientific_pass': False})
    finally:
        if eng: eng.close()
        core.save('WALL.json', {'seconds': time.time()-started})


if __name__ == '__main__': main()
