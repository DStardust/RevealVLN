import importlib.util
import json
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ORIGINAL = LINE/'reviews/Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2/engine.py'
source = ORIGINAL.read_text()
PATCHES = [('cfg.gpu_device_id=2;', 'cfg.gpu_device_id=3;'),
           ("self.loop(position,bin,'L',['D','K','B'])", "self.loop(position,bin,'L',['K','B'])"),
           ("if not hash_equal or detail['position_spread_m']>1e-4 or detail['rotation_spread_rad']>1e-4:",
            "if detail['position_spread_m']>1e-4 or detail['rotation_spread_rad']>1e-4:")]
for a, b in PATCHES:
    assert source.count(a) == 1
    source = source.replace(a, b)
s = importlib.util.spec_from_file_location('core', ORIGINAL)
core = importlib.util.module_from_spec(s)
exec(compile(source, str(ORIGINAL), 'exec'), core.__dict__); core.OUT = OUT


class Engine(core.Engine):
    def __init__(self, phase='numerical_join'):
        self.normalization = None; self._active = False; self._step = 0
        self._previous = None; self._normalizations = []; self._keep = False
        super().__init__(phase)

    def load_frozen(self, candidate):
        self.normalization = candidate['numerical_join']

    def record(self, obs):
        raw, rgb, sem = super().record(obs)
        if self._active and self._step == self.normalization['step']:
            n = self.normalization; target = n['target_pose']
            p, r = core.pose_spread(raw['pose'], target)
            sensor_spans = [core.pose_spread(raw['pose']['sensors'][k], target['sensors'][k]) for k in ['rgb', 'semantic']]
            if p > 1e-5 or r > 1e-5 or any(max(x) > 1e-5 for x in sensor_spans):
                raise core.Reject('NUMERICAL_JOIN_CORRECTION_TOO_LARGE', position_m=p, angle_rad=r)
            if self._keep:
                for arr, h, suffix in [(rgb, raw['rgb_hash'], 'rgb'), (sem, raw['semantic_hash'], 'semantic')]:
                    path = core.OUT/'content'/f'{h}.{suffix}.npy'; path.parent.mkdir(exist_ok=True)
                    if not path.exists(): core.np.save(path, arr, allow_pickle=False)
            state = core.hs.AgentState(); state.position = core.np.asarray(target['position'], dtype=core.np.float32)
            state.rotation = core.np.quaternion(*target['rotation'])
            self.sim.get_agent(0).set_state(state, reset_sensors=True, infer_sensor_states=True)
            after, rgb, sem = super().record(self.sim.get_sensor_observations())
            def signature(record):
                if not record['evidence_complete'] or not self._previous['evidence_complete']:
                    raise core.Reject('NUMERICAL_JOIN_UNKNOWN_EVENT')
                return {k: [idx for idx in ids if record['pixels'].get(idx, 0) >= 256
                            and self._previous['pixels'].get(idx, 0) >= 256] for k, ids in self.eligible.items()}
            before_ev, after_ev = signature(raw), signature(after)
            if before_ev != after_ev: raise core.Reject('NUMERICAL_JOIN_EVENT_CHANGED')
            event = {'step': self._step, 'kind': 'bounded_numerical_common_state_reconstruction',
                     'position_correction_m': p, 'angle_correction_rad': r,
                     'sensor_corrections': sensor_spans, 'raw_record': raw,
                     'canonical_record': after, 'raw_see2': before_ev, 'canonical_see2': after_ev,
                     'events_unchanged': True, 'threshold_m_rad': 1e-5}
            self._normalizations.append(event)
            core.log('NUMERICAL_JOIN_EVENTS.jsonl', event)
            self.counts['explicit_numerical_joins'] += 1
            raw = after
        self._step += 1; self._previous = raw
        return raw, rgb, sem

    def trace(self, position, bin, actions, seed=0, keep=False, tag='discovery'):
        acts = list(actions); n = self.normalization
        self._active = bool(n and len(acts) > n['step'] and bin == n['yaw_bin']
                            and list(map(float, position)) == n['position']
                            and acts[:n['step']] in n['registered_histories'])
        self._step = 0; self._previous = None; self._normalizations = []; self._keep = keep
        try:
            tr = super().trace(position, bin, acts, seed, keep, tag)
            if self._active:
                if len(self._normalizations) != 1 and tr['complete']:
                    raise core.Reject('NUMERICAL_JOIN_BOUNDARY_COUNT')
                tr['normalization_events'] = self._normalizations
                tr.pop('trace_hash'); tr['trace_hash'] = core.digest(tr)
            return tr
        finally:
            self._active = False

    def complete_candidate(self, u, bin, base, tail):
        first = next(iter(base.values()))
        self.normalization = {'protocol': 'bounded_numerical_join.v1', 'step': len(first['actions']),
                              'position': list(map(float, u['position'])), 'yaw_bin': bin,
                              'target_pose': first['trace']['observations'][0]['pose'],
                              'registered_histories': [x['actions'] for x in base.values()],
                              'max_correction_m_rad': 1e-5, 'raw_exact_pixel_merge_claimed': False}
        candidate = super().complete_candidate(u, bin, base, tail)
        candidate['numerical_join'] = self.normalization
        candidate.pop('frozen_candidate_hash'); candidate['frozen_candidate_hash'] = core.digest(candidate)
        return candidate

    def discovery(self):
        pool = self.u_pool(); u = pool[22]
        cache = OUT.parent/'Q35N_G1R_COVERAGE_V1/ROUTE_PLANS.jsonl'
        core.save('PLANNING_CACHE_PROVENANCE.json', {'path': str(cache), 'sha256': core.digest(cache.read_bytes()),
            'use': 'same-GPU actual-plan cache only; full trajectories freshly replayed'})
        with cache.open() as f:
            for line in f:
                row = json.loads(line); self.route_cache[row['key']] = (row['actions'], row['trace'], row['failure'])
        for yaw in [0, 1]:
            self.normalization = None
            try:
                base = self.candidate_base(u['position'], yaw)
            except core.Reject as e:
                core.log('BASE_ATTEMPTS.jsonl', {'yaw_bin': yaw, 'reason': e.code, 'detail': e.detail}); continue
            core.save('RAW_BASE_'+str(yaw)+'.json', base)
            for tail in core.TAILS:
                try:
                    candidate = self.complete_candidate(u, yaw, base, tail)
                    core.save('FROZEN_CANDIDATE.json', candidate); return candidate
                except core.Reject as e:
                    core.log('FAMILY_ATTEMPTS.jsonl', {'yaw_bin': yaw, 'tail': tail, 'reason': e.code, 'detail': e.detail})
        raise core.Reject('NUMERICAL_JOIN_DISCOVERY_EXHAUSTED')


def main():
    assert not (OUT/'result.json').exists()
    core.save('SOURCE_DERIVATION.json', {'original_sha256': core.digest(ORIGINAL.read_bytes()),
        'literal_patches': PATCHES, 'derived_source_sha256': core.digest(source.encode()),
        'actual_state_reconstruction': 'record override, once at u, explicit raw/canonical evidence'})
    eng = None; started = time.time()
    try:
        eng = Engine(); eng.preview(); candidate = eng.discovery()
        core.save('result.json', {'decision': 'NUMERICAL_JOIN_CANDIDATE_READY_FOR_CERTIFICATION',
            'candidate_hash': candidate['frozen_candidate_hash'], 'family_certified': False,
            'raw_exact_merge_pass': False, 'scientific_pass': False})
    except core.Reject as e:
        core.save('result.json', {'decision': e.code, 'detail': e.detail, 'family_certified': False, 'scientific_pass': False})
    finally:
        if eng: eng.close()
        core.save('WALL.json', {'seconds': time.time()-started})


if __name__ == '__main__': main()
