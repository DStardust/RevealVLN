"""Two real causal decisions; synthetic reader target; one diagnostic update only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
MODEL=LINE/'runtime/models/Qwen3.5-2B_15852e8'
DATA=LINE/'data_pipeline/ordinary_pilot_v1'
DEVICE='cuda:0'  # Launcher maps the leased physical GPU explicitly.
ACTIONS=['move_forward','turn_left','turn_right','STOP']


def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def log(event,**kwargs):
    rec=dict(event=event,time=time.time(),**kwargs)
    with (OUT/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    print(json.dumps(rec),flush=True)


def sha_tensor(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class Policy(nn.Module):
    def __init__(self,base,processor):
        super().__init__();self.base=base;self.processor=processor;self.mm=base.model
        self.special=['<NAV_OLD_MEMORY>','<NAV_WRITE_QUERY>','<NAV_ACTION_QUERY>']+[
            f'<EXEC_{a.upper()}_{s}>' for a in ACTIONS for s in ['OK','COLLISION']]
        added=processor.tokenizer.add_special_tokens({'additional_special_tokens':self.special})
        # Preserve pretrained reserved rows; expand only if newly assigned IDs exceed them.
        rows_before=base.get_input_embeddings().weight.shape[0]
        base.resize_token_embeddings(max(rows_before,len(processor.tokenizer)),mean_resizing=False)
        for p in base.parameters():p.requires_grad_(False)
        self.mm.language_model=get_peft_model(self.mm.language_model,
            LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none'))
        self.old_slot=nn.Parameter(torch.randn(8,2048,device=DEVICE,dtype=torch.bfloat16)*.01)
        self.write_query=nn.Parameter(torch.randn(8,2048,device=DEVICE,dtype=torch.bfloat16)*.01)
        self.action_query=nn.Parameter(torch.randn(1,2048,device=DEVICE,dtype=torch.bfloat16)*.01)
        self.writer=nn.Linear(2048,2048,bias=False,device=DEVICE,dtype=torch.float32)
        self.action_head=nn.Linear(2048,4,device=DEVICE,dtype=torch.float32)
        self.query_encoder=nn.Embedding(8,128,device=DEVICE,dtype=torch.float32)
        self.reader=nn.Sequential(nn.Linear(2048+2048+128,128),nn.GELU(),nn.Linear(128,1)).to(DEVICE)
        self.sid={x:processor.tokenizer.convert_tokens_to_ids(x) for x in self.special}
        save('TOKENIZER_ADDITION.json',{'added':added,'ids':self.sid,'embedding_rows_before':rows_before,'tokenizer_length':len(processor.tokenizer),'embedding_shape':list(base.get_input_embeddings().weight.shape),
            'embedding_sha256':sha_tensor(base.get_input_embeddings().weight)})

    def step(self,instruction,images,executed,memory,feature_probe=False):
        assert len(images)<=2 and len(executed)<=8
        assert memory.shape==(1,8,2048)
        self.mm.rope_deltas=None
        text=self.processor.apply_chat_template([{'role':'user','content':[
            *[{'type':'image'} for _ in images],{'type':'text','text':instruction}]}],tokenize=False,add_generation_prompt=True)
        b=self.processor(text=[text],images=images,return_tensors='pt').to(DEVICE)
        assert 'mm_token_type_ids' in b
        base_ids=b['input_ids'];n=base_ids.shape[1]
        text_positions=(base_ids!=self.base.config.image_token_id)&(b['mm_token_type_ids']==b['mm_token_type_ids'][0,0])
        assert text_positions.any()
        text_type=b['mm_token_type_ids'][text_positions][0].item()
        suffix=[self.sid['<NAV_OLD_MEMORY>']]*8+[
            self.sid[f'<EXEC_{a.upper()}_OK>'] for a in executed]+[
            self.sid['<NAV_WRITE_QUERY>']]*8+[self.sid['<NAV_ACTION_QUERY>']]
        ids=torch.cat([base_ids,torch.tensor([suffix],device=DEVICE)],1)
        mask=torch.cat([b['attention_mask'],torch.ones((1,len(suffix)),device=DEVICE,dtype=b['attention_mask'].dtype)],1)
        types=torch.cat([b['mm_token_type_ids'],torch.full((1,len(suffix)),text_type,device=DEVICE,dtype=b['mm_token_type_ids'].dtype)],1)
        positions,deltas=self.mm.get_rope_index(input_ids=ids,mm_token_type_ids=types,
                                               image_grid_thw=b['image_grid_thw'],attention_mask=mask)
        assert positions.shape==(3,1,ids.shape[1])
        emb=self.base.get_input_embeddings()(ids)
        # All image computation frozen. The early feature leaf is a sensitivity probe, not vision training.
        with torch.no_grad():
            features=torch.cat(self.mm.get_image_features(b['pixel_values'],b['image_grid_thw'],return_dict=True).pooler_output,0)
        probe=features.detach().clone().requires_grad_(True) if feature_probe else None
        if probe is not None:features=probe
        image_mask,_=self.mm.get_placeholder_mask(ids,inputs_embeds=emb,image_features=features)
        assert int((ids==self.base.config.image_token_id).sum())==features.shape[0]
        emb=emb.masked_scatter(image_mask,features.to(emb.dtype))
        causal_text_pool=emb[:,:n][text_positions].float().mean(0,keepdim=True)
        old_start=n;write_start=n+8+len(executed)
        emb=emb.clone()
        emb[:,old_start:old_start+8]=memory+self.old_slot
        emb[:,write_start:write_start+8]=self.write_query
        emb[:,-1:]=self.action_query
        out=self.mm.language_model(inputs_embeds=emb,attention_mask=mask,position_ids=positions,
                                   past_key_values=None,use_cache=False,return_dict=True)
        assert out.last_hidden_state.shape==(1,ids.shape[1],2048)
        assert out.past_key_values is None and self.mm.rope_deltas is None
        m=self.writer(out.last_hidden_state[:,write_start:write_start+8].float()).to(torch.bfloat16)
        logits=self.action_head(out.last_hidden_state[:,-1].float())
        details={'base_tokens':n,'full_tokens':ids.shape[1],'position_shape':list(positions.shape),
            'image_grid':b['image_grid_thw'].tolist(),'packed_visual_tokens':features.shape[0],
            'text_token_type_from_processor':text_type,'memory_shape':list(m.shape),'action_shape':list(logits.shape),
            'cache_returned':False,'rope_delta_persisted':False}
        return m,logits,causal_text_pool,probe,details

    def continuation(self,m,text_pool,q):
        qe=self.query_encoder(q).mean(1)
        return self.reader(torch.cat([m.float().mean(1),text_pool,qe],1)).squeeze(-1)


def grad_record(p):
    if p.grad is None:return {'present':False,'finite':None,'norm':None}
    norm=float(p.grad.float().norm());return {'present':True,'finite':bool(torch.isfinite(p.grad).all()),'norm':norm}


def main():
    started=time.time();torch.manual_seed(1109);np.random.seed(1109);random.seed(1109)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    assert transformers.__version__=='5.15.0'
    assert torch.cuda.is_available()
    log('load_model_start')
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    base=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(DEVICE)
    paths={'root':type(base).__name__,'model':type(base.model).__name__,
        'visual':type(base.model.visual).__name__,'language_model':type(base.model.language_model).__name__,
        'processor':type(processor).__name__,'hidden_size':base.config.text_config.hidden_size}
    assert paths['model']=='Qwen3_5Model' and paths['language_model']=='Qwen3_5TextModel' and paths['hidden_size']==2048
    save('OBJECT_INTROSPECTION.json',paths)
    log('model_loaded',**paths)
    policy=Policy(base,processor);policy.eval()
    loras=[(n,p) for n,p in policy.named_parameters() if 'lora_B' in n]
    assert loras
    save('TRAINABLE_PARAMETERS.json',{'names_shapes':[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in policy.named_parameters() if p.requires_grad],
         'total':sum(p.numel() for p in policy.parameters() if p.requires_grad),'lora_B_probe':loras[0][0]})
    row=json.loads((DATA/'TRAINING_INDEX.jsonl').read_text().splitlines()[0])
    pol=json.loads((DATA/row['policy_file']).read_text());sup=json.loads((DATA/row['supervision_file']).read_text())
    assert len(sup['actions'])>=3
    images=[]
    for ref in pol['rgb_sequence'][:3]:
        p=(DATA/row['rgb_reference_root']/ref).resolve();assert p.is_relative_to(LINE)
        images.append(Image.open(p).convert('RGB'))
    save('BATCH_PROVENANCE.json',{'index_row':row,'decision_times':[1,2],'image_refs':pol['rgb_sequence'][:3],
         'actions':sup['actions'][:3],'auxiliary_target':'SYNTHETIC_INTERFACE_ONLY_ONE_NOT_REAL_TASK_LABEL',
         'public_or_heldout_samples':False})
    mzero=torch.zeros((1,8,2048),device=DEVICE,dtype=torch.bfloat16)
    log('first_causal_forward_start')
    m1,a1,tp1,leaf,shape1=policy.step(pol['instruction'],images[:2],sup['actions'][:1],mzero,True)
    m1.retain_grad();log('first_causal_forward_done',**shape1)
    m2,a2,tp2,_,shape2=policy.step(pol['instruction'],images[1:3],sup['actions'][:2],m1)
    log('second_causal_forward_done',**shape2)
    frozen_m=m2.detach().clone();frozen_a=a2.detach().clone()
    q=torch.tensor([[0,1,2]],device=DEVICE);q_alt=torch.tensor([[2,3,4]],device=DEVICE)
    logit=policy.continuation(m2,tp2,q)
    alt=policy.continuation(m2,tp2,q_alt)
    assert torch.equal(m2.detach(),frozen_m) and torch.equal(a2.detach(),frozen_a)
    bce=F.binary_cross_entropy_with_logits(logit,torch.ones_like(logit))
    ce=F.cross_entropy(a2,torch.tensor([ACTIONS.index(sup['actions'][2])],device=DEVICE))
    assert torch.isfinite(bce) and torch.isfinite(ce)
    log('backward_start',synthetic_bce=float(bce.detach()),real_action_ce=float(ce.detach()))
    bce.backward(retain_graph=True)
    probes={'early_frozen_image_feature_leaf':grad_record(leaf),'early_memory':grad_record(m1),
        'writer':grad_record(policy.writer.weight),'write_query':grad_record(policy.write_query),
        'lora_B':grad_record(loras[0][1]),'reader':grad_record(policy.reader[0].weight)}
    gradient_pass=all(x['present'] and x['finite'] and x['norm']>1e-12 for x in probes.values())
    vision_frozen=all(not p.requires_grad and p.grad is None for p in base.model.visual.parameters())
    save('GRADIENT_PROBE.json',{'probes':probes,'gradient_pass':gradient_pass,'vision_frozen':vision_frozen,
         'loss_used_for_probe':'synthetic_continuation_BCE_only','threshold':1e-12})
    assert gradient_pass and vision_frozen
    ce.backward()
    opt=torch.optim.SGD([p for p in policy.parameters() if p.requires_grad],lr=.01)
    targets={'writer':policy.writer.weight,'write_query':policy.write_query,'lora_B':loras[0][1],'reader':policy.reader[0].weight}
    before={k:p.detach().clone() for k,p in targets.items()};opt.step()
    updated={k:bool(torch.any(p.detach()!=before[k])) for k,p in targets.items()}
    save('DIAGNOSTIC_UPDATE.json',{'optimizer':'SGD','learning_rate':.01,'diagnostic_steps':1,'changed':updated,
         'training_runs':0,'saved_policy_checkpoints':0})
    assert all(updated.values())
    del m1,m2,a1,a2,tp1,tp2,leaf,logit,alt,bce,ce
    opt.zero_grad(set_to_none=True)
    # Two identical reset forwards after the same diagnostic update must agree.
    with torch.no_grad():
        reset1=policy.step(pol['instruction'],images[:2],sup['actions'][:1],mzero)
        reset2=policy.step(pol['instruction'],images[:2],sup['actions'][:1],mzero)
    reset_delta=float((reset1[0].float()-reset2[0].float()).abs().max())
    assert reset_delta<1e-5
    save('result.json',{'decision':'G2_MINIMAL_INTERFACE_PASS','interface_pass':True,'gradient_pass':True,
        'shape_checks':[shape1,shape2],'future_query_noninterference':True,'reset_max_abs_delta':reset_delta,
        'vision_frozen':vision_frozen,'diagnostic_optimizer_steps':1,'new_training_runs':0,
        'mechanism_labels_used':0,'synthetic_auxiliary_target_used':True,'scientific_pass':False,
        'navigation_gain':None,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'wall_seconds':time.time()-started})
    log('interface_pass',peak_vram_bytes=torch.cuda.max_memory_allocated())


if __name__=='__main__':main()
