"""Freeze a two-arm comparison sharing one loaded model and runtime."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'ordinary_cycle_recovery_v1'
BASE = HERE.parent / 'ordinary_expanded_dev_after_single_v1'


def write(name, value):
    with (HERE / name).open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not (HERE / 'run_001').exists()
    for name in ('common.py', 'executor.py', 'launch.py', 'path_metrics.py', 'official_distance.py', 'cycle_policy.py'):
        write(name, (OLD / name).read_text())
    text = (OLD / 'evaluate.py').read_text()
    text = replace(text, '            diff=(batch-single).float();relative=',
        '            repeated=torch.cat([forward([x]) for x in items])\n'
        "            assert torch.equal(single,repeated),'INPROCESS_FORWARD_NOT_REPEATABLE'\n"
        '            del repeated\n'
        '            diff=(batch-single).float();relative=')
    text = replace(text,
        '                    action,cycle_info=guards[lane].choose(window.instruction,window.images,window.executed,values)',
        "                    if schedules[lane][cursors[lane]] >= p['base_episode_count']:\n"
        '                        action,cycle_info=guards[lane].choose(window.instruction,window.images,window.executed,values)\n'
        '                    else:\n'
        '                        action=c.ACTIONS[max(range(4),key=values.__getitem__)]\n'
        '                        cycle_info=dict(native_action=action,cycle_override=False,repeated_input=False,input_key=None)')
    write('evaluate.py', text)
    text = (OLD / 'aggregate.py').read_text()
    start = text.index('    overall=stats(rows);')
    end = text.index("    fields=['index'", start)
    text = text[:start] + '''    arms={}
    for name,part in [('native',rows[:100]),('recovery',rows[100:])]:
        assert len(part)==100
        arms[name]=dict(**stats(part),by_house={h:stats([x for x in part if x['house']==h]) for h in p['houses']},
            collisions=sum(x['collisions'] for x in part),environment_actions=sum(x['steps'] for x in part),
            stopped=sum(int(x['stopped']) for x in part),
            failure_categories=dict(collections.Counter(x['failure_category'] for x in part)))
    result=dict(status='COMPLETE',unix=time.time(),checkpoint_updates=p['checkpoint_updates'],
        checkpoint_sha256=p['checkpoint_sha256'],benchmark='R2R train held-out INTERNAL_DEV paired engineering diagnostic',
        episodes_per_arm=100,executed_episodes=len(rows),missing=0,**arms,
        trace_audit_passed=True,audited_actions=audited,source_lock_verified=True,selected_batch_size=1,
        shared_model_process=True,optimizer_updates=0,scientific_gain_verified=False,
        wall_seconds=launch['wall_seconds'],run_dir=str(RUN))
''' + text[end:]
    write('aggregate.py', text)
    p = json.loads((OLD / 'PROTOCOL.json').read_text())
    p.update(id='Q35N_ORDINARY_CYCLE_PAIR_V2', episode_count=200, base_episode_count=100, wall_seconds=4200,
             checkpoint_role='shared_process_native_and_recovery',
             paired_arms=dict(native_indices=[0, 99], recovery_indices=[100, 199]),
             main_criterion='Same-process matched prefixes/logits plus original absolute and paired improvement gates')
    p['house_counts'] = {k: 2*v for k, v in p['house_counts'].items()}
    write('PROTOCOL.json', p)
    episodes = json.loads((BASE / 'EPISODES_PRIVILEGED.json').read_text())
    assert len(episodes) == 100
    write('EPISODES_PRIVILEGED.json', episodes + episodes)
    geometry = json.loads((BASE / 'GEOMETRY_PREFLIGHT.json').read_text())
    assert len(geometry['rows']) == 100
    geometry['rows'] *= 2
    geometry['paired_reuse_note'] = 'Identical original episodes; executor rechecks every reset distance'
    write('GEOMETRY_PREFLIGHT.json', geometry)
    write('PARITY_FIXTURES.json', (BASE / 'PARITY_FIXTURES.json').read_text())
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    c = runpy.run_path(str(HERE / 'common.py'))
    schedules = c['schedules'](200, 8)
    assert sorted(i for row in schedules for i in row) == list(range(200))
    for i in range(100):
        assert (episodes + episodes)[i] == (episodes + episodes)[i + 100]
    files = json.loads((OLD / 'SOURCE_LOCK.json').read_text())['files']
    for path in [*HERE.glob('*.py'), *HERE.glob('*.json'), HERE / 'SPEC_ZH.md']:
        files[str(path.resolve())] = c['sha'](path)
    write('SOURCE_LOCK.json', dict(files=files, comparison='One fixed loaded model, two copies of the original 100 tasks'))
    write('CPU_TEST_RESULT.json', dict(passed=True, paired_tasks_identical=100, schedule_complete=200,
        all_sources_parse=True, unchanged_recovery_rule_sha256=hashlib.sha256((HERE / 'cycle_policy.py').read_bytes()).hexdigest(),
        original_rule_tests='V1 three tests, including all original 100 traces; code copied byte-for-byte',
        gpu_gate='Sixteen repeat-forwards must be bitwise equal before any navigation action'))
    print('Prepared shared-process pair, 100 native + 100 recovery episodes')


if __name__ == '__main__':
    main()
