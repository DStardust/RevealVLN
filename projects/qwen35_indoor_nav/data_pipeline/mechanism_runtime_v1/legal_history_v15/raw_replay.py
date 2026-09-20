"""Execute every raw action; an interior numerical join is prohibited."""
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, LINE, RUNTIME, RESEARCH, c, load, raw_window, pose_delta


def main(run):
    protocol = c.read(run/'CONFIG.json')
    for path, digest in protocol['source_hashes'].items():
        assert c.sha(LINE/path) == digest, path
    backend_module = load('v15_raw_backend', RUNTIME/'habitat_backend.py')
    from content_store import AuditedContentStore
    compiler_module = load('v15_frozen_see2', LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    decisions = 0
    results = []
    backend = None

    def progress(**items):
        c.write(run/'PROGRESS.json', dict(unix=time.time(), decisions=decisions, **items))

    class NoInteriorJoin(backend_module.HabitatBackend):
        def reconstruct(self, pose):
            raise RuntimeError('INTERIOR_STATE_ASSIGNMENT_FORBIDDEN')

    try:
        with AuditedContentStore(run/'content', protocol['output_gib']*2**30) as store:
            for family in protocol['families']:
                for path, digest in family['assets'].items():
                    assert c.sha(path) == digest, path
                folder = run/family['family_id']
                folder.mkdir()
                backend = NoInteriorJoin(family['scene'], protocol['gpu'], family['roles'], store,
                    dict(runtime_allowed=True, scene_glb=family['scene'], gpu_device=protocol['gpu']))
                assert backend.eligible == family['compiler']['eligible']
                compiler = compiler_module.Compiler(**family['compiler'])
                candidate = family['candidate']
                prefixes = {}
                grid = {}
                for history_id, history in candidate['histories'].items():
                    for continuation_id, suffix in candidate['continuations'].items():
                        actions = list(history)+list(suffix)
                        assert len(actions) <= protocol['max_episode_decisions']
                        assert actions[-1] == 'S' and 'S' not in actions[:-1]
                        backend.reset(candidate['position'], candidate['yaw_bin'], protocol['environment_seed'])
                        trace = dict(actions=[], observations=[dict(backend.observe(), step=0)],
                                     collisions=0, complete=False, interior_state_assignments=0)
                        for action in actions:
                            assert decisions < protocol['max_decisions'], 'DECISION_LIMIT'
                            if action != 'S':
                                collision = backend.step(action)
                                trace['collisions'] += int(collision)
                                trace['observations'].append(dict(backend.observe(), step=len(trace['observations'])))
                            trace['actions'].append(action)
                            decisions += 1
                            if decisions % 20 == 0:
                                progress(family=family['family_id'], history=history_id, continuation=continuation_id)
                        trace['complete'] = True
                        assert backend.counts['explicit_reconstructions'] == 0
                        name = history_id+'__'+continuation_id
                        c.write(folder/(name+'.json'), trace, True)
                        cutoff = len(history)
                        prefix = dict(raw=raw_window(trace, cutoff), pose=trace['observations'][cutoff]['pose'])
                        assert prefixes.setdefault(history_id, prefix) == prefix, 'SAME_HISTORY_REPLAY_CHANGED'
                        grid[name] = dict(labels={task:compiler.evaluate(trace, task) for task in compiler.tasks},
                            terminal_only=('unknown' if not compiler.complete(trace) else
                                'pass' if compiler.atoms(trace['observations'])[-1]['terminal'] else 'fail'),
                            collisions=trace['collisions'], decisions=len(actions))
                reference = prefixes['H_A']
                comparison = {name:dict(raw_window_equal=value['raw']==reference['raw'],
                    pose_equal=value['pose']==reference['pose'], max_pose_delta=pose_delta(value['pose'], reference['pose']))
                    for name,value in prefixes.items()}
                result = dict(family_id=family['family_id'], house=family['house'], comparison=comparison,
                    prefixes=prefixes, grid=grid, backend_counts=backend.counts,
                    no_interior_state_assignment=True, original_training_admission=False,
                    training_admission=False, reason='Interface measurement only; independent controls/data still required.')
                c.write(folder/'RAW_CERTIFICATE.json', result, True)
                results.append(result)
                backend.close()
                backend = None
                progress(completed_families=len(results))
            audit = store.full_audit()
            assert audit['audit_pass']
            c.write(run/'CONTENT_AUDIT.json', audit, True)
            c.write(run/'TIMESTAMP_CHECKS.json', store.timestamp_checks, True)
        c.write(run/'RESULT.json', dict(status='RAW_INTERFACE_MEASURED', families=results,
            actual_decisions=decisions, complete_traces=sum(len(x['grid']) for x in results),
            optimizer_updates=0, model_loads=0, scientific_pass=False, training_admission=False), True)
        print(dict(decisions=decisions, comparisons={x['family_id']:x['comparison'] for x in results}), flush=True)
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    run = Path(sys.argv[1])
    try:
        main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc()), True)
        raise
