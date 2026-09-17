import collections
import heapq
import importlib.util
import json
import math
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
s = importlib.util.spec_from_file_location('numerical', OUT.parent/'Q35N_G1R_NUMERICAL_JOIN_V1/worker.py')
num = importlib.util.module_from_spec(s); s.loader.exec_module(num)
core = num.core; core.OUT = OUT
ORIGINAL = num.ORIGINAL


class Engine(num.Engine):
    def search_suffix(self, position, yaw, anchor, forbidden, name):
        obs = self.start(position, yaw)
        initial, _, _ = core.Engine.record(self, obs)
        counts = collections.Counter(); serial = 0
        def visible(record):
            return tuple(idx for k in core.KINDS for idx in self.eligible[k] if record['pixels'].get(idx, 0) >= 256)
        def key(record, seen):
            p = record['pose']['position']; y = core.yawbin(core.np.quaternion(*record['pose']['rotation']))
            return tuple(round(x/.125) for x in p)+(y, visible(record), seen)
        def heuristic(record, seen):
            kind = 'B' if anchor is None or seen else anchor
            p = core.np.asarray(record['pose']['position'])
            return min(float(core.np.linalg.norm(p-core.np.asarray(v['position']))) for v in self.views[kind])/.25
        queue = [(3*heuristic(initial, False), 0, serial, initial, [], False)]
        best = {key(initial, False): 0}; found = None
        while queue and counts['expanded'] < 12000:
            _, g, _, current, acts, seen = heapq.heappop(queue)
            if best.get(key(current, seen)) != g: continue
            counts['expanded'] += 1
            if len(acts) >= 159: continue
            for action in ['F', 'L', 'R']:
                state = core.hs.AgentState(); state.position = core.np.asarray(current['pose']['position'], dtype=core.np.float32)
                state.rotation = core.np.quaternion(*current['pose']['rotation'])
                self.sim.get_agent(0).set_state(state, reset_sensors=True, infer_sensor_states=True)
                obs = self.sim.step(core.NAMES[action]); counts['branch_actions'] += 1
                self.counts['planning_branch_primitive_actions'] += 1
                nxt, _, _ = core.Engine.record(self, obs)
                if obs['collided']:
                    counts['collision_pruned'] += 1; continue
                if not current['evidence_complete'] or not nxt['evidence_complete']:
                    counts['unknown_pruned'] += 1; continue
                events = {k: [idx for idx in ids if current['pixels'].get(idx, 0) >= 256 and nxt['pixels'].get(idx, 0) >= 256]
                          for k, ids in self.eligible.items()}
                if any(events[k] for k in forbidden):
                    counts['forbidden_event_pruned'] += 1; continue
                newacts = acts+[action]
                if events['B'] and (anchor is None or seen):
                    found = newacts+['S']; break
                newseen = seen or (bool(events[anchor]) if anchor else False)
                newkey = key(nxt, newseen); ng = len(newacts)
                if best.get(newkey, 10**9) <= ng: continue
                best[newkey] = ng; serial += 1
                heapq.heappush(queue, (ng+3*heuristic(nxt, newseen), ng, serial, nxt, newacts, newseen))
            if found: break
            if counts['expanded'] % 500 == 0:
                core.save('SEARCH_PROGRESS.json', {'continuation': name, 'counts': dict(counts), 'queue': len(queue)})
                print(name, dict(counts), flush=True)
        core.save('SEARCH_'+name+'.json', {'counts': dict(counts), 'found_actions': found, 'queue_remaining': len(queue),
            'search_branch_initializations_are_not_full_trajectories': True})
        self.counts.update({'search_'+k: v for k, v in counts.items()})
        if found is None: raise core.Reject('EVENT_AWARE_SEARCH_EXHAUSTED', continuation=name, counts=dict(counts))
        tr = self.trace(position, yaw, found)
        core.save('SUFFIX_REPLAY_'+name+'.json', tr)
        if not self.pattern(tr, ['B']+([anchor] if anchor else []), forbidden) or not tr['observations'][-1]['see2']['B']:
            raise core.Reject('SEARCH_PATH_FULL_REPLAY_FAILED', continuation=name)
        return found, tr

    def discovery(self):
        pool = self.u_pool(); u = pool[22]; yaw = 0; tail = 'LRLRLRLR'
        path = OUT.parent/'Q35N_G1R_NUMERICAL_JOIN_V1/RAW_BASE_0.json'
        base = json.loads(path.read_text())
        core.save('HISTORY_PROVENANCE.json', {'path': str(path), 'sha256': core.digest(path.read_bytes()),
            'use': 'registered action lists; every actual family freshly replayed'})
        first = next(iter(base.values()))
        self.normalization = {'protocol': 'bounded_numerical_join.v1', 'step': len(first['actions']),
            'position': list(map(float, u['position'])), 'yaw_bin': yaw, 'target_pose': first['trace']['observations'][0]['pose'],
            'registered_histories': [x['actions'] for x in base.values()], 'max_correction_m_rad': 1e-5,
            'raw_exact_pixel_merge_claimed': False}
        histories = {}; ends = []
        for h, value in base.items():
            acts = value['actions']+list(tail); tr = self.trace(u['position'], yaw, acts)
            core.save('HISTORY_'+h+'.json', tr)
            if not self.pattern(tr, ['K'] if h == 'H_K' else ['D'], ['D'] if h == 'H_K' else ['K']):
                raise core.Reject('HISTORY_PATTERN_REJECT')
            if any(o['see2'][k] for o in tr['observations'][-8:] for k in ['D', 'K', 'B']):
                raise core.Reject('TAIL_EVENT_REJECT')
            histories[h] = acts; ends.append(tr['observations'][-1])
        if len({(o['rgb_hash'], o['semantic_hash']) for o in ends}) != 1: raise core.Reject('MERGE_HASH_REJECT')
        if any(max(core.pose_spread(ends[0]['pose'], o['pose'])) > 1e-4 for o in ends): raise core.Reject('MERGE_POSE_REJECT')
        position = ends[0]['pose']['position']; syaw = core.yawbin(core.np.quaternion(*ends[0]['pose']['rotation']))
        continuations = {}; queries = {}
        for name, anchor, forbidden in [('C0', None, ['D', 'K']), ('C_D', 'D', ['K']), ('C_K', 'K', ['D'])]:
            acts, tr = self.search_suffix(position, syaw, anchor, forbidden, name)
            query = core.make_query(tr)
            if len(query['sequence']) > 160: raise core.Reject('QUERY_LENGTH_REJECT', continuation=name)
            continuations[name] = acts; queries[name] = query
        checks = []
        for h, acts in histories.items():
            for name, suffix in continuations.items():
                tr = self.trace(u['position'], yaw, acts+suffix)
                core.save('PREFLIGHT_'+h+'_'+name+'.json', tr)
                if not tr['complete']: raise core.Reject('FULL_PREFLIGHT_ILLEGAL')
                for task in ['g_D_v2', 'g_K_v2']:
                    y = core.task_eval(tr, task)
                    if y != core.expected(task, h, name): raise core.Reject('PREFLIGHT_MATRIX_MISMATCH', task=task, history=h, continuation=name, outcome=y)
                    checks.append({'task': task, 'history': h, 'continuation': name, 'outcome': y, 'trace_hash': tr['trace_hash']})
        candidate = {'family_id': 'F17_DK_B_numeric_v1', 'u': u, 'yaw_bin': yaw, 'histories': histories,
            'continuations': continuations, 'queries': queries, 'public_tail': tail, 'merge_observation': ends[0],
            'preflight_checks': checks, 'numerical_join': self.normalization,
            'suffix_planner': 'bounded_event_aware_best_first.v1'}
        candidate['frozen_candidate_hash'] = core.digest(candidate)
        core.save('FROZEN_CANDIDATE.json', candidate); return candidate


def main():
    assert not (OUT/'result.json').exists()
    core.save('SOURCE_DERIVATION.json', {'normalizer_worker_sha256': core.digest((OUT.parent/'Q35N_G1R_NUMERICAL_JOIN_V1/worker.py').read_bytes()),
        'original_sha256': core.digest(ORIGINAL.read_bytes()), 'overrides': ['event-constrained offline suffix search', 'single fixed family assembly']})
    eng = None; started = time.time()
    try:
        eng = Engine('event_aware_suffix'); eng.preview(); candidate = eng.discovery()
        core.save('result.json', {'decision': 'REAL_NUMERICAL_JOIN_CANDIDATE_READY_FOR_CERTIFICATION',
            'candidate_hash': candidate['frozen_candidate_hash'], 'family_certified': False, 'raw_exact_merge_pass': False, 'scientific_pass': False})
    except core.Reject as e:
        core.save('result.json', {'decision': e.code, 'detail': e.detail, 'family_certified': False, 'scientific_pass': False})
    finally:
        if eng: eng.close()
        core.save('WALL.json', {'seconds': time.time()-started})


if __name__ == '__main__': main()
