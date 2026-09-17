"""Freeze controller diagnostic before viewing the matched baseline outcome."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
BASE=HERE.parent/'ordinary_expanded_dev_after_single_v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    assert not (HERE/'PROTOCOL.json').exists(),'ALREADY_FROZEN'
    test=subprocess.run([sys.executable,'-I','-S','-B',str(HERE/'tests.py')],capture_output=True,text=True,timeout=60)
    assert test.returncode==0,test.stderr
    s=importlib.util.spec_from_file_location('guard_source_checks',HERE/'reuse.py')
    r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
    checks=['12_controller_cpu_tests']
    for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):
        text=r.source(name);ast.parse(text)
        if name not in ('evaluate.py','aggregate.py'):assert text==r.parent.source(name)
        checks.append(name+':syntax_and_exact_unchanged_or_narrow_anchor')
    assert 'batch_size=1  # Matched comparison' in r.source('evaluate.py')
    assert "executed=action)" in r.source('evaluate.py')
    checks+=['batch_one_preserved','actual_action_logged_and_entered_causal_history']
    p=read(BASE/'PROTOCOL.json')
    p.update(id='Q35N_ORDINARY_VISUAL_STALL_GUARD_V1',checkpoint_role='after_with_causal_visual_stall_guard',
        policy_controller='causal_visual_stall_guard_v1',pure_model_result=False,
        executed_policy='greedy_model_proposal_with_declared_causal_controller',
        controller=dict(identical_rgb_forward_threshold=8,per_episode_max_overrides=4,recovery_action='turn_left',
                        reads=['past_and_current_rgb','actually_executed_actions'],stop_override=False),
        prerequisite=str(BASE/'run_001/RESULT.json'),
        main_criterion='Development engineering signal: SR delta > 0 and SPL delta >= -0.02; not scientific PASS')
    save(HERE/'PROTOCOL.json',p)
    for name in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
        with (HERE/name).open('xb') as f:f.write((BASE/name).read_bytes())
        assert sha(HERE/name)==sha(BASE/name)
    files=dict(read(BASE/'SOURCE_LOCK.json')['files'])
    for path,digest in files.items():
        assert Path(path).resolve(strict=True).is_relative_to(ROOT) and sha(path)==digest,path
    checks+=['identical_inputs_checkpoint_seed_and_physics','all_inherited_source_locks_unchanged']
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,checks=checks,controller_test_output=test.stderr,optimizer_updates=0))
    for path in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[HERE/'SPEC_ZH.md']:files[str(path)]=sha(path)
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='Single causal visual stall controller; physical metrics unchanged',training_updates=0))
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_CONDITIONAL_BOUNDED_ENGINEERING_DIAGNOSTIC',
        user_request='检查并做出针对性修改',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),
        launch_only_after_matched_case_complete=True,novelty_claim=False,automatic_training=False,automatic_retry=False,
        actual_navigation_effect='UNTESTED'))
    print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False))


if __name__=='__main__':main()
