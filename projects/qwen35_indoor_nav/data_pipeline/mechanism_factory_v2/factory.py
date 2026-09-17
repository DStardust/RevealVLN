"""Backend-independent, budgeted crossed-history construction, not a simulator.

The backend must implement reset(position,yaw,seed), observe(), step(F/L/R),
reconstruct(pose), and proposal-only routes(position,yaw,role). observe returns
pixel counts, hashes and physical/sensor poses, never a learned prediction.
Every physical step, including planning trials, must go through TraceRunner.
No Habitat imports, old-engine monkey patches or global mutable role maps.
"""
import collections
import copy
import math

from compiler import digest, slice_continuation
from planning import cache_key


class Reject(ValueError):
    def __init__(self, code, **details):
        super().__init__(code)
        self.code, self.details = code, details


def require(ok, code, **details):
    if not ok:
        raise Reject(code, **details)


def rotations(n):
    n %= 24
    return ['L'] * n if n <= 12 else ['R'] * (24-n)


def compress(actions):
    result, turns = [], 0
    for action in actions:
        require(action in ('F', 'L', 'R'), 'ILLEGAL_MOTION')
        if action == 'F':
            result += rotations(turns) + ['F']
            turns = 0
        else:
            turns += 1 if action == 'L' else -1
    return result + rotations(turns)


def inverse(actions):
    require(all(a in ('F', 'L', 'R') for a in actions), 'ILLEGAL_MOTION')
    raw = []
    for action in reversed(actions):
        raw += ['L'] * 12 + ['F'] + ['R'] * 12 if action == 'F' else ['R' if action == 'L' else 'L']
    return raw, compress(raw)


def pose_distance(a, b):
    def norm(v):
        return math.sqrt(sum(x*x for x in v))
    require(len(a['position']) == len(b['position']) == 3, 'POSITION_SHAPE')
    require(len(a['rotation']) == len(b['rotation']) == 4, 'ROTATION_SHAPE')
    p, q = list(a['rotation']), list(b['rotation'])
    require(all(math.isfinite(x) for x in a['position'] + b['position'] + p + q), 'NONFINITE_POSE')
    pn, qn = norm(p), norm(q)
    require(pn > 0 and qn > 0, 'ZERO_QUATERNION')
    p, q = [x/pn for x in p], [x/qn for x in q]
    chord = min(norm([x-y for x, y in zip(p, q)]), norm([x+y for x, y in zip(p, q)]))
    return norm([x-y for x, y in zip(a['position'], b['position'])]), 4*math.asin(min(1., chord/2))


def yaw_bin(pose):
    w, x, y, z = pose['rotation']
    # Rotate camera forward (0,0,-1), scalar-first quaternion.
    return int(round(math.atan2(2*(x*z+w*y), 1-2*(x*x+y*y))*12/math.pi)) % 24


class TraceRunner:
    def __init__(self, backend, compiler, budget, emit):
        self.backend, self.compiler, self.budget, self.emit = backend, compiler, budget, emit
        self.counts = collections.Counter()

    def observe(self, step):
        self.budget.check_time()
        o = copy.deepcopy(self.backend.observe())
        self.counts['observation_returns'] += 1
        self.budget.check_time()
        require({'pixels', 'pose', 'rgb_hash', 'semantic_hash', 'evidence_complete'} <= set(o), 'OBSERVATION_FIELDS')
        require(type(o['evidence_complete']) is bool, 'EVIDENCE_TYPE')
        require(all(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                    for h in (o['rgb_hash'], o['semantic_hash'])), 'PIXEL_HASH_FORMAT')
        require(len({str(k) for k in o['pixels']}) == len(o['pixels']), 'PIXEL_KEY_COLLISION')
        o['pixels'] = {str(k): v for k, v in o['pixels'].items()}
        require(all(type(v) is int and v >= 0 for v in o['pixels'].values()), 'PIXEL_COUNTS')
        o.pop('see2', None)
        o['step'] = step
        return o

    def run(self, position, yaw, actions, seed=1109, join=None):
        actions = list(actions)
        require(all(a in ('F', 'L', 'R', 'S') for a in actions) and 'S' not in actions[:-1], 'ACTION_SEQUENCE')
        if join is not None:
            require(type(join['step']) is int and 0 < join['step'] < len(actions), 'JOIN_STEP')
            require(actions[:join['step']] in join['registered_histories'], 'UNREGISTERED_JOIN_HISTORY')
            require(position == join['position'] and yaw == join['yaw_bin'], 'JOIN_START_MISMATCH')
        self.budget.check_time()
        self.backend.reset(position, yaw, seed)
        self.budget.check_time()
        observations, events, collisions = [], [], 0
        stopped = False
        for t in range(len(actions)+1):
            record = self.observe(t)
            if events and events[-1]['step'] == t-1:
                event = events[-1]
                require(record['evidence_complete'], 'JOIN_NEXT_UNKNOWN')
                before_next = self.compiler.atoms([dict(event['raw_record'], step=0), dict(record, step=1)])[-1]
                after_next = self.compiler.atoms([dict(event['canonical_record'], step=0), dict(record, step=1)])[-1]
                require(before_next == after_next, 'JOIN_NEXT_EVENTS_CHANGED')
                event['next_boundary_event_unchanged'] = True
                self.emit('numerical_join_next_boundary', {'step': t, 'raw_alternative_events': before_next,
                                                         'canonical_events': after_next})
            if join is not None and t == join['step']:
                target, raw = join['target_pose'], copy.deepcopy(record)
                require(digest(target) == digest(observations[0]['pose']), 'JOIN_TARGET_NOT_INITIAL_STATE')
                corrections = {'agent': pose_distance(raw['pose'], target)}
                require(set(raw['pose']['sensors']) == set(target['sensors']) == {'rgb', 'semantic'}, 'JOIN_SENSORS')
                corrections.update({k: pose_distance(raw['pose']['sensors'][k], target['sensors'][k])
                                    for k in ('rgb', 'semantic')})
                require(all(max(v) <= 1e-5 for v in corrections.values()), 'JOIN_CORRECTION_TOO_LARGE', corrections=corrections)
                self.backend.reconstruct(copy.deepcopy(target))
                self.budget.check_time()
                record = self.observe(t)
                require(digest(record['pose']) == digest(target), 'JOIN_CANONICAL_POSE_MISMATCH')
                require(raw['evidence_complete'] and record['evidence_complete'] and observations[-1]['evidence_complete'], 'JOIN_UNKNOWN')
                previous = dict(observations[-1], step=0)
                before = self.compiler.atoms([previous, dict(raw, step=1)])[-1]
                after = self.compiler.atoms([previous, dict(record, step=1)])[-1]
                require(before == after, 'JOIN_EVENTS_CHANGED')
                event = {'step': t, 'raw_record': raw, 'canonical_record': copy.deepcopy(record),
                         'corrections': corrections, 'raw_see2': before, 'canonical_see2': after,
                         'boundary_event_unchanged': True, 'next_boundary_event_unchanged': None,
                         'protocol': 'bounded_numerical_join.v1'}
                events.append(event)
                self.emit('numerical_join', event)
            observations.append(record)
            if t == len(actions):
                break
            if actions[t] == 'S':
                self.counts['executed_stops'] += 1
                stopped = True
                break
            # Reservation is durable/irreversible even when backend raises.
            self.budget.reserve_action()
            collision = self.backend.step(actions[t])
            require(type(collision) is bool, 'BACKEND_COLLISION_TYPE')
            self.counts['confirmed_action_returns'] += 1
            self.emit('action_completed', {'step': t, 'action': actions[t], 'collision': collision,
                                          'confirmed_action_returns': self.counts['confirmed_action_returns']})
            self.budget.check_time()
            if collision:
                collisions += 1
                break
        complete = not collisions and len(observations) == len(actions)+(0 if stopped else 1)
        if complete and join is not None:
            require(len(events) == 1 and events[0]['next_boundary_event_unchanged'] is True, 'JOIN_COUNT_OR_NEXT_EVIDENCE')
        trace = {'actions': actions, 'observations': observations, 'complete': complete,
                 'collisions': collisions, 'seed': seed, 'initial_position': list(position),
                 'initial_yaw_bin': yaw, 'normalization_events': events}
        for record, atoms in zip(observations, self.compiler.atoms(observations)):
            record['see2'] = atoms
        trace['trace_hash'] = digest(trace)
        self.emit('trace', trace)
        return trace


class FamilyFactory:
    """Actual construction operations on an explicitly injected backend.

    Backend proposal order is frozen before outcomes; routes() MUST be geometry-
    only, deterministic and non-stepping. Physical probes use self.runner.
    A real adapter and replay acceptance remain separate from this CPU module.
    """
    def __init__(self, backend, compiler, budget, emit, task_a, task_b, *, context):
        self.backend, self.compiler, self.budget, self.emit = backend, compiler, budget, emit
        self.runner = TraceRunner(backend, compiler, budget, emit)
        self.tasks = (task_a, task_b)
        self.a, self.b, self.end, self.irrelevant = 'anchor_A', 'anchor_B', 'terminal', 'irrelevant'
        require(set(compiler.roles) == {self.a, self.b, self.end, self.irrelevant}, 'FACTORY_ROLE_SLOTS')
        require(task_a != task_b and set(compiler.tasks) == set(self.tasks), 'FACTORY_TASKS')
        require(all(compiler.eligible[k] for k in compiler.roles), 'ELIGIBLE_SET_EMPTY')
        require(all(i > 0 and i not in (65535, 4294967295) for ids in compiler.eligible.values() for i in ids), 'RESERVED_INSTANCE_ID')
        for task, anchor in zip(self.tasks, (self.a, self.b)):
            require(compiler.tasks[task]['anchor'] == anchor and compiler.tasks[task]['terminal'] == self.end, 'FACTORY_TASK_ROLE_MAPPING')
        require(set(context) == {'house_id', 'asset_config'}, 'CONTEXT_FIELDS')
        self.context = copy.deepcopy(context)
        roles = {k: list(v) for k, v in compiler.roles.items()}
        self.context_key = cache_key(context['house_id'], context['asset_config'], roles,
                                     extra={'eligible': dict(compiler.eligible), 'tasks': {k: dict(v) for k, v in compiler.tasks.items()}})

    def pattern(self, trace, required, forbidden):
        if not self.compiler.complete(trace):
            return False
        observed = {k for ev in self.compiler.atoms(trace['observations']) for k, ids in ev.items() if ids}
        return set(required) <= observed and not set(forbidden) & observed

    def routes(self, position, yaw, role):
        self.budget.check_time()
        # Metadata proposals do not certify reachability or task eligibility.
        for actions in self.backend.routes(position, yaw, role):
            self.budget.check_time()
            actions = list(actions)
            require(all(a in ('F', 'L', 'R') for a in actions), 'PROPOSAL_ACTIONS')
            if len(actions) <= 504:
                yield actions

    def loop(self, position, yaw, role, forbidden):
        for outbound in self.routes(position, yaw, role):
            tr = self.runner.run(position, yaw, outbound)
            if not self.pattern(tr, [role], forbidden):
                self.emit('route_rejected', {'role': role, 'reason': 'OUTBOUND_EVENT_OR_LEGALITY'})
                continue
            raw, compressed = inverse(outbound)
            if len(outbound)+len(compressed) > 504:
                self.emit('route_rejected', {'role': role, 'reason': 'HISTORY_LENGTH'})
                continue
            # Never replace physical inverse validation by ideal geometry.
            raw_trace = self.runner.run(position, yaw, outbound+raw)
            full = self.runner.run(position, yaw, outbound+compressed)
            if not self.compiler.complete(raw_trace) or not self.pattern(full, [role], forbidden):
                continue
            if max(pose_distance(full['observations'][0]['pose'], full['observations'][-1]['pose'])) <= 1e-5:
                return outbound+compressed
        raise Reject('NO_VALID_LOOP', role=role)

    def histories(self, position, yaw):
        initial = self.runner.run(position, yaw, [])['observations'][0]['pose']
        a = self.loop(position, yaw, self.a, [self.b])
        b = self.loop(position, yaw, self.b, [self.a])
        irrelevant = self.loop(position, yaw, self.irrelevant, [self.b, self.end])
        histories = {'H_A': a, 'H_B': b, 'H_A_I': a+irrelevant}
        length = max(map(len, histories.values()))
        require(length+8 <= 512, 'HISTORY_LENGTH')
        neutral = None
        for pad in (['L', 'R'], ['R', 'L'], ['L']*24, ['R']*24):
            if self.pattern(self.runner.run(position, yaw, pad), [], [self.a, self.b, self.end]):
                neutral = pad
                break
        require(neutral is not None, 'NO_NEUTRAL_PADDING')
        for name, actions in histories.items():
            need = length-len(actions)
            require(need % len(neutral) == 0, 'PADDING_PARITY')
            histories[name] = actions+neutral*(need//len(neutral))
            trace = self.runner.run(position, yaw, histories[name])
            required = [self.b] if name == 'H_B' else [self.a] + ([self.irrelevant] if name == 'H_A_I' else [])
            forbidden = [self.a] if name == 'H_B' else [self.b]
            require(self.pattern(trace, required, forbidden), 'PADDED_HISTORY_PATTERN', history=name)
        return histories, {'step': length, 'position': position, 'yaw_bin': yaw,
                           'target_pose': initial, 'registered_histories': list(histories.values())}

    def continuations(self, position, yaw):
        result = {}
        for cid, role, forbidden in [('C0', None, [self.a, self.b]), ('C_A', self.a, [self.b]), ('C_B', self.b, [self.a])]:
            found = None
            for prefix in ([[]] if role is None else self.routes(position, yaw, role)):
                first = self.runner.run(position, yaw, prefix)
                if not self.pattern(first, [role] if role else [], forbidden):
                    continue
                final = first['observations'][-1]['pose']
                for suffix in self.routes(final['position'], yaw_bin(final), self.end):
                    actions = prefix+suffix+['S']
                    if len(actions) > 160:
                        continue
                    trace = self.runner.run(position, yaw, actions)
                    if not self.pattern(trace, [self.end]+([role] if role else []), forbidden):
                        continue
                    if not self.compiler.atoms(trace['observations'])[-1][self.end]:
                        continue
                    query = self.compiler.query_from_trace(trace)
                    if len(query['sequence']) > 160:
                        continue
                    found = actions
                    break
                if found is not None:
                    break
            require(found is not None, 'NO_VALID_CONTINUATION', continuation=cid)
            result[cid] = found
        return result

    @staticmethod
    def expected(task_index, history, continuation):
        passed = history != 'H_B' or continuation == 'C_A' if task_index == 0 else history == 'H_B' or continuation == 'C_B'
        return 'pass' if passed else 'fail'

    def validate_matrix(self, candidate, seed=1109):
        require(candidate['context_key'] == self.context_key and candidate['context'] == self.context, 'CANDIDATE_CONTEXT_MISMATCH')
        histories, continuations = candidate['histories'], candidate['continuations']
        require(set(histories) == {'H_A', 'H_B', 'H_A_I'} and set(continuations) == {'C0', 'C_A', 'C_B'}, 'FAMILY_MATRIX')
        require(len({len(a) for a in histories.values()}) == 1, 'HISTORY_LENGTH_MISMATCH')
        cutoff = len(next(iter(histories.values())))
        require(8 < cutoff <= 512, 'HISTORY_LENGTH')
        require(candidate['public_tail'] in ('LRLRLRLR', 'FFFFFFFF'), 'PUBLIC_TAIL')
        require(candidate['numerical_join']['step'] == cutoff-8, 'JOIN_NOT_BEFORE_TAIL')
        rows, traces, common, queries, prefixes, contents = [], {}, None, {}, {}, {}
        for h, actions in histories.items():
            require(''.join(actions[-8:]) == candidate['public_tail'], 'TAIL_ACTION_MISMATCH')
            for c, continuation in continuations.items():
                require(0 < len(continuation) <= 160 and continuation[-1] == 'S', 'CONTINUATION_LENGTH_STOP')
                tr = self.runner.run(candidate['position'], candidate['yaw_bin'], actions+continuation,
                                     seed, candidate['numerical_join'])
                require(self.compiler.complete(tr), 'REAL_TRACE_INCOMPLETE')
                ev = self.compiler.atoms(tr['observations'])
                seen = {k for e in ev[:cutoff+1] for k, ids in e.items() if ids}
                required = {self.b} if h == 'H_B' else {self.a} | ({self.irrelevant} if h == 'H_A_I' else set())
                forbidden = {self.a} if h == 'H_B' else {self.b}
                require(required <= seen and not forbidden & seen, 'HISTORY_EVENT_PATTERN')
                require(not any(ev[t][k] for t in range(cutoff-7, cutoff+1) for k in (self.a, self.b, self.end)), 'PUBLIC_TAIL_EVENT')
                window = [{k: o[k] for k in ('rgb_hash', 'semantic_hash', 'pose')}
                          for o in tr['observations'][cutoff-1:cutoff+1]]
                signature = digest(window)
                common = signature if common is None else common
                require(signature == common, 'MERGE_SHORT_WINDOW_MISMATCH')
                ph = digest(tr['observations'][:cutoff+1])
                require(prefixes.setdefault(h, ph) == ph, 'PREFIX_CHANGED_ACROSS_CONTINUATIONS')
                q = self.compiler.query_from_trace(slice_continuation(tr, cutoff))
                require(len(q['sequence']) <= 160, 'QUERY_LENGTH')
                require(queries.setdefault(c, digest(q)) == digest(q), 'QUERY_CHANGED_ACROSS_HISTORIES')
                for i, task in enumerate(self.tasks):
                    y = self.compiler.evaluate(tr, task)
                    require(y == self.expected(i, h, c), 'MATRIX_OUTCOME_MISMATCH', task=task, history=h, continuation=c, actual=y)
                    rows.append({'task': task, 'history': h, 'continuation': c, 'outcome': y, 'seed': seed})
                traces[(h, c)] = tr
                contents[h+'/'+c] = digest({'actions': tr['actions'], 'observations': tr['observations']})
        counts = {h: dict(collections.Counter(a)) for h, a in histories.items()}
        return {'rows': rows, 'replays': len(traces), 'short_window_hash': common,
                'prefix_hashes': prefixes, 'query_hashes': queries, 'trace_content_hashes': contents, 'action_counts': counts,
                'semantic_family_preflight_pass': True, 'scientific_pass': False,
                'training_admission': False}

    def construct(self, config):
        position, yaw, tail = config['u_position'], config['yaw_bin'], config['public_tail']
        require(tail in ('LRLRLRLR', 'FFFFFFFF'), 'PUBLIC_TAIL')
        histories, join = self.histories(position, yaw)
        histories = {h: a+list(tail) for h, a in histories.items()}
        prefix = self.runner.run(position, yaw, histories['H_A'], join=join)
        require(self.compiler.complete(prefix), 'PUBLIC_TAIL_LEGALITY')
        state = prefix['observations'][-1]['pose']
        continuations = self.continuations(state['position'], yaw_bin(state))
        candidate = {'schema_version': 'q35n.factory_candidate.v4', 'position': position,
                     'context': copy.deepcopy(self.context), 'context_key': self.context_key,
                     'yaw_bin': yaw, 'public_tail': tail, 'histories': histories,
                     'continuations': continuations, 'numerical_join': join}
        candidate['preflight'] = self.validate_matrix(candidate)
        candidate['candidate_hash'] = digest(candidate)
        return candidate

    def discover(self, configurations, freeze_ledger, bundle_id):
        freeze_ledger.start(bundle_id)
        require(len(configurations) <= 64, 'CONFIGURATION_CAP')
        for config in configurations:
            self.budget.check_time()
            try:
                candidate = self.construct(config)
            except Reject as error:
                self.emit('discovery_rejected', {'bundle': bundle_id, 'config': config,
                                                'reason': error.code, 'details': error.details})
                continue
            freeze_ledger.freeze(bundle_id, candidate)
            return candidate
        raise Reject('CONFIGURATIONS_EXHAUSTED', configurations=len(configurations))

    def replay_seeds(self, candidate):
        # Caller owns certification phase budget and durable failure state.
        require(candidate.get('candidate_hash') == digest({k: v for k, v in candidate.items() if k != 'candidate_hash'}), 'FROZEN_CANDIDATE_HASH')
        results = [self.validate_matrix(candidate, seed) for seed in (1109, 2209, 3309)]
        require(len({r['short_window_hash'] for r in results}) == 1, 'SEED_MERGE_MISMATCH')
        require(len({digest(r['prefix_hashes']) for r in results}) == 1, 'SEED_PREFIX_MISMATCH')
        require(len({digest(r['query_hashes']) for r in results}) == 1, 'SEED_QUERY_MISMATCH')
        require(len({digest(r['trace_content_hashes']) for r in results}) == 1, 'SEED_TRACE_MISMATCH')
        return {'replays': sum(r['replays'] for r in results), 'evaluations': sum(len(r['rows']) for r in results),
                'seeds': results, 'scientific_pass': False, 'training_admission': False}
