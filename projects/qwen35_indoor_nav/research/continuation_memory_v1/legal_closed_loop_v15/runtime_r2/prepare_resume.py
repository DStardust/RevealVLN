"""Retain the complete, sealed first V15 condition; schedule every other condition."""
import ast
import difflib
from pathlib import Path
import sys
CODE=Path(__file__).resolve().parent
HERE=CODE.parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c


def function(path,name):
    return next(node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,ast.FunctionDef) and node.name==name)


def main():
    source=HERE/'CONTINUATION_PROTOCOL.json';protocol=c.read(source);old=HERE/'continuation_run_003'
    assert (old/'LAUNCH_RESULT.json').exists()
    seal=c.read(old/'STATE_SEAL.json');assert seal['base_unchanged'] and seal['heads_unchanged']
    assert set(range(18))<=set(seal['completed_ranks'])
    audit=c.read(old/'CONDITION_000_AUDITS.json');assert len(audit)==12
    assert all(x['input_prefix_matched'] and x['action_prefix_matched'] for x in audit)
    for rank in range(18):assert (old/'rollouts'/f'{rank:03d}'/'ROLLOUT.json').exists()
    assert not (old/'CONDITION_001_AUDITS.json').exists(),'UNEXPECTED_EXTRA_COMPLETE_GROUP'
    for name in ('forward','prefix_audit'):
        assert ast.dump(function(HERE/'evaluate_continuations.py',name))==ast.dump(function(CODE/'evaluate_continuations.py',name)),name
    assert ast.dump(function(HERE/'continuation_service.py','initial'))==ast.dump(function(CODE/'continuation_service.py','initial'))
    original=list(protocol['rollouts'])
    protocol['rollouts']=[dict(row,rank=rank) for rank,row in enumerate(original) if row['condition']!=0]
    assert [r['rank'] for r in protocol['rollouts']]==list(range(18,216))
    assert all({r['model'] for r in protocol['rollouts'] if r['condition']==i}==set(protocol['models']) for i in range(1,12))
    protocol['orchestration_revision']=dict(version=2,original_experiment_protocol_sha256=c.sha(source),
        reason='Reuse simulator within one family; every rollout still performs exact audited initialization and complete physical prefix replay.',
        prior_run='continuation_run_003',prior_seal_sha256=c.sha(old/'STATE_SEAL.json'),
        admitted_prior_condition=0,admitted_prior_ranks=list(range(18)),
        discarded_incomplete_group_ranks=[x for x in seal['completed_ranks'] if x>=18],
        no_completed_group_retried=True,no_result_based_selection=True,total_original_rollouts=216,
        core_forward_and_prefix_audit_AST_unchanged=True,initialization_AST_unchanged=True,
        policy_input_or_action_or_STOP_or_checker_change=False)
    for path in CODE.glob('*.py'):protocol['source_hashes'][str(path.relative_to(LINE))]=c.sha(path)
    c.write(CODE/'PROTOCOL.json',protocol,True)
    patches=[]
    for name in ('launch_continuations.py','evaluate_continuations.py','continuation_service.py'):
        patches.extend(difflib.unified_diff((HERE/name).read_text().splitlines(True),(CODE/name).read_text().splitlines(True),
            fromfile=name,tofile='runtime_r2/'+name))
    (CODE/'ORCHESTRATION_DIFF.patch').write_text(''.join(patches))
    c.write(CODE/'CPU_TEST_RESULT.json',dict(passed=True,
        checks=['exact_remaining_ranks_18_through_215','all_18_models_per_remaining_condition',
        'prior_complete_group_sealed','Qwen_forward_AST_identical','prefix_audit_AST_identical','initialization_AST_identical'],
        prior_retained_rollouts=18,next_scheduled_rollouts=198,policy_updates=0,GPU_used=False),True)
    print(dict(retained=18,scheduled=198))


if __name__=='__main__':main()
