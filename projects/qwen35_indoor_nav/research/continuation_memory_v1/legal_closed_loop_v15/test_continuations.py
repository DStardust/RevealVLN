"""CPU checks for deployed STOP, paired numerical audit, and the actual budget."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_continuations as e


def main():
    decision=dict(raw={'instruction':'x','rgb':['a'],'history':[]},processed={'input_ids':'a'},
        logits=[2.,1.,0.,-1.],native_action='move_forward',executed_action='move_forward',override=False,decision=1)
    changed=deepcopy(decision);changed['logits'][0]+=1e-6
    audit=e.prefix_audit([decision],[changed])
    assert audit['input_prefix_matched'] and not audit['logits_bitwise_equal'] and not audit['argmax_flip_count']
    changed['native_action']='turn_left'
    assert 'first_divergence' in e.prefix_audit([decision],[changed])
    changed=deepcopy(decision);changed['processed']={'input_ids':'b'}
    assert not e.prefix_audit([decision],[changed])['input_prefix_matched']
    changed=deepcopy(decision);changed['executed_action']='turn_left';changed['override']=True
    assert e.prefix_audit([decision],[changed])['first_method_action_difference']==1
    assert e.runtime.selected_action([0.,1.,2.,3.],[99.,1.,2.,-3.])==('STOP','STOP')
    assert e.c.advance('STOP',499,False)==(500,True)
    assert e.c.advance('turn_left',499,False)==(500,True)
    try:e.c.advance('move_forward',500,True)
    except ValueError:pass
    else:raise AssertionError('BUDGET_NOT_ENFORCED')
    e.c.write(e.HERE/'CONTINUATION_CPU_TEST_RESULT.json',dict(passed=True,
        checks=['finite_float_difference_logged','argmax_flip_rejected','processed_mismatch_rejected',
        'intentional_method_difference_ends_prefix','native_STOP_preserved','STOP_and_motion_use_500_budget'],
        simulator_loaded=False,base_model_loaded=False,method_benefit_measured=False))


if __name__=='__main__':main()
