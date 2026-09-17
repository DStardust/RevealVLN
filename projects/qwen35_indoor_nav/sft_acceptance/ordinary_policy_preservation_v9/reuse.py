"""Single objective revision on the frozen V8R1 recipe; same model/data/batches."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'ordinary_onpolicy_fp32_master_v8r1/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='433339071012c08f55a87f10336729f59bbb6daa88fa1247a877554fc3b2846d'
s=importlib.util.spec_from_file_location('frozen_precision_recipe',PARENT);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
HASHES=parent.HASHES
def replace_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)
def source(name):
    text=parent.source(name)
    if name=='supervise_filestore.py':
        text=replace_once(text,'triton_ordinary_fp32_master_v8r1','triton_ordinary_policy_preservation_v9')
        text=replace_once(text,'inductor_ordinary_fp32_master_v8r1','inductor_ordinary_policy_preservation_v9')
    if name=='train_filestore.py':
        text=replace_once(text,"'Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1'","'Q35N_ORDINARY_POLICY_PRESERVATION_V9'")
        text=replace_once(text,"    # Old checkpoints saved rank-0's local count;", """    anchor_module=load_module('fixed_policy_anchor',HERE/'anchor.py')
    reference_config=protocol['reference_checkpoint']
    require(sha256(reference_config['path'])==reference_config['sha256'],'REFERENCE_SOURCE_CHANGED')
    reference_state=torch.load(reference_config['path'],map_location='cpu',weights_only=True)
    reference=anchor_module.Anchor(raw_policy,reference_state['trainable'])
    require(cursor['updates']==0,'V9_NO_UNREGISTERED_RESUME')
    require(protocol['reference_kl_lambda']==1.0 and protocol['reference_temperature']==1.0,'REFERENCE_OBJECTIVE_CHANGED')
    del reference_state
    # Old checkpoints saved rank-0's local count;""")
        old="""            logits = policy(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                            mm_token_type_ids=batch['mm_token_type_ids'],
                            image_grid_thw=batch['image_grid_thw'], pixel_values=batch['pixel_values'],
                            exec_index=batch['exec_index'], action_index=batch['action_index'])"""
        new="""            kwargs=dict(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                        mm_token_type_ids=batch['mm_token_type_ids'],image_grid_thw=batch['image_grid_thw'],
                        pixel_values=batch['pixel_values'],exec_index=batch['exec_index'],action_index=batch['action_index'])
            teacher_logits=reference.forward(kwargs,verify=(cursor['updates']%200==0 or cursor['updates']==999))
            logits=policy(**kwargs)
            kl=anchor_module.per_sample_kl(logits,teacher_logits)
            if cursor['updates']==0:
                parity=anchor_module.first_parity(logits,teacher_logits)
                print(json.dumps(dict(event='REFERENCE_READY',rank=rank,parity=parity,reference_fixed=True)),flush=True)"""
        text=replace_once(text,old,new)
        text=replace_once(text,"            loss = (ce * batch['weights']).sum() * world / weight_sum_global",
                              "            loss = ((ce + kl) * batch['weights']).sum() * world / weight_sum_global")
        assert text.count("stats = dict(weighted_ce=0.0, weight_sum=0.0, decisions=0,")==2
        text=text.replace("stats = dict(weighted_ce=0.0, weight_sum=0.0, decisions=0,",
                          "stats = dict(weighted_ce=0.0, weighted_kl=0.0, weight_sum=0.0, decisions=0,")
        text=replace_once(text,"            stats['weighted_ce'] +=", "            stats['weighted_kl'] += float((kl * batch['weights']).sum().detach())\n            stats['weighted_ce'] +=")
        text=replace_once(text,"[stats['weighted_ce'], stats['weight_sum'], stats['decisions'], cursor['decisions']]",
                          "[stats['weighted_ce'], stats['weight_sum'], stats['decisions'], cursor['decisions'], stats['weighted_kl']]")
        text=replace_once(text,"gce, gw, window_decisions, reduced_plan_decisions = packed.tolist()",
                          "gce, gw, window_decisions, reduced_plan_decisions, gkl = packed.tolist()")
        text=replace_once(text,"metrics=dict(mean_ce=gce / max(gw, 1e-12),",
                          "metrics=dict(mean_ce=gce / max(gw, 1e-12),mean_kl=gkl/max(gw,1e-12),mean_loss=(gce+gkl)/max(gw,1e-12),")
        text=replace_once(text,"            global_decisions += int(batch_decisions_global)",
                          "            global_decisions += int(batch_decisions_global)\n            require(reference.decisions==cursor['decisions'] and reference.calls==cursor['updates'],'TEACHER_FORWARD_COUNTER')")
        text=replace_once(text,"                                 global_plan_decisions=global_decisions,",
                          "                                 global_plan_decisions=global_decisions,teacher_forward_decisions=global_decisions,total_policy_forward_decisions=2*global_decisions,")
        text=replace_once(text,"\n                     charged_compute_decisions=charged(),",
                          "\n                     charged_compute_decisions=charged(),teacher_forward_decisions=global_decisions,total_policy_forward_decisions=2*global_decisions,")
        text=replace_once(text,"\n                      charged_compute_decisions=charged(),",
                          "\n                      charged_compute_decisions=charged(),teacher_forward_decisions=global_decisions,total_policy_forward_decisions=2*global_decisions,")
    return text
def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':fixed-reference-KL','exec'),namespace)
