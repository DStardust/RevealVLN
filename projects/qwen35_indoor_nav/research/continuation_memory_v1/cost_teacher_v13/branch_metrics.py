"""Policy-action use of real old history at genuine minimum-PASS teacher forks."""
import torch


@torch.no_grad()
def evaluate(net,cache,data,admission):
    rows=[]
    for family,ledger in zip(data['families'],admission['family_rows']):
        assert family['family_id']==ledger['family_id']
        indices=torch.tensor([p['features'] for p in family['prefixes']],device=cache['features'].device)
        states,_=net.encode(cache['features'][indices])
        final=states[:,-1]
        for case in ledger['cases']:
            prefix_ids=[case['left_prefix'],case['right_prefix'],case['sham_prefix']]
            memory=final[prefix_ids]
            tail=family['cells'][case['left_cell']]['tail']
            for t in range(1,case['step']+1):
                feature=cache['features'][tail[t]['feature']].unsqueeze(0).expand(3,-1)
                memory,_=net.update(feature,memory)
            feature=cache['features'][case['common_feature']].unsqueeze(0).expand(3,-1)
            base=cache['logits'][case['common_feature']].unsqueeze(0).expand(3,-1)
            logits=net.action_logits(memory,base,feature)
            swapped=net.action_logits(memory[[1,0,2]],base,feature)
            assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(swapped).all())
            native=int(base[0].argmax())
            actions=[3]*3 if native==3 else logits.argmax(-1).tolist()
            wrong=[3]*3 if native==3 else swapped.argmax(-1).tolist()
            left,right=case['left_action'],case['right_action']
            assert left!=right
            if net.no_memory:
                assert torch.equal(logits[0],logits[1]) and actions[0]==actions[1]
            row=dict(family_id=family['family_id'],house=family['house'],split=family['split'],
                task_id=case['task_id'],suffix_step=case['step'],old_event_gap=case['old_event_gap'],
                input_feature=case['common_feature'],native_action=native,
                target_actions=[left,right],executed_policy_actions=actions[:2],
                wrong_history_actions=wrong[:2],same_task_state_sham_action=actions[2] if case['sham_valid'] else None,
                both_correct=actions[0]==left and actions[1]==right,
                both_correct_wrong=wrong[0]==left and wrong[1]==right,
                both_correct_sham=actions[2]==left and actions[1]==right if case['sham_valid'] else None,
                individual_correct=int(actions[0]==left)+int(actions[1]==right),
                native_stop_forces_same_action=native==3,
                method_logits=logits.tolist(),wrong_history_logits=swapped.tolist(),
                same_task_state_sham_available=case['sham_valid'],
                both_teachers_within_full500=case['both_teachers_within_full500'])
            assert not net.no_memory or not row['both_correct']
            rows.append(row)
    summaries={}
    for split in ('fit','check'):
        selected=[r for r in rows if r['split']==split]
        summaries[split]=dict(pairs=len(selected),both_correct=sum(r['both_correct'] for r in selected),
            both_correct_wrong=sum(r['both_correct_wrong'] for r in selected),
            sham_pairs=sum(r['same_task_state_sham_available'] for r in selected),
            both_correct_sham=sum(bool(r['both_correct_sham']) for r in selected),
            individual_correct=sum(r['individual_correct'] for r in selected),
            native_stop_blocked_pairs=sum(r['native_stop_forces_same_action'] for r in selected),
            full500_teacher_pairs=sum(r['both_teachers_within_full500'] for r in selected),
            houses=sorted({r['house'] for r in selected}))
    return dict(rows=rows,summaries=summaries,
        scope='Offline real policy action head at logged legal shared-state forks. Raw short windows and native inputs match. Future labels select/evaluate teachers only; no simulator rollout or SR measured.',
        sham_scope='One-sided H_A_I replacement for H_A, same exact SEE2 task state and current input; does not prove a uniquely disentangled task variable.',
        no_memory_pair_both_correct_upper_bound=0,source_admission_unchanged=True,original_training_admission=False)
