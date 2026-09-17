"""Action-only extraction of sealed G2 Policy. See CODE_DIFF.patch."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

OUT=Path(__file__).resolve().parents[1]
LINE=OUT.parents[1]
ROOT=LINE.parents[1]
DATA=LINE/'data_pipeline/ordinary_pilot_v1'
MODEL=LINE/'runtime/models/Qwen3.5-2B_15852e8'
DEVICE='cuda:0'
ACTIONS=['move_forward','turn_left','turn_right','STOP']

class Policy(nn.Module):
    def __init__(self,base,processor):
        super().__init__(); self.base=base; self.processor=processor; self.mm=base.model
        self.special=['<NAV_OLD_MEMORY>','<NAV_WRITE_QUERY>','<NAV_ACTION_QUERY>']+[
            f'<EXEC_{a.upper()}_{s}>' for a in ACTIONS for s in ['OK','COLLISION']]
        processor.tokenizer.add_special_tokens({'additional_special_tokens':self.special})
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
        self.sid={x:processor.tokenizer.convert_tokens_to_ids(x) for x in self.special}
        expected=json.loads((LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())
        assert self.sid==expected['ids'] and rows_before==expected['embedding_rows_before']
        self.forward_tokens=0

    def step(self,instruction,images,executed,memory):
        assert isinstance(instruction,str) and 1<=len(images)<=2 and len(executed)<=8
        assert all(isinstance(x,Image.Image) and x.size==(224,224) for x in images)
        assert all(x in ACTIONS[:-1] for x in executed)
        assert memory.shape==(1,8,2048)
        self.mm.rope_deltas=None
        text=self.processor.apply_chat_template([{'role':'user','content':[
            *[{'type':'image'} for _ in images],{'type':'text','text':instruction}]}],tokenize=False,add_generation_prompt=True)
        b=self.processor(text=[text],images=images,return_tensors='pt').to(DEVICE)
        base_ids=b['input_ids']; n=base_ids.shape[1]
        text_positions=(base_ids!=self.base.config.image_token_id)&(b['mm_token_type_ids']==b['mm_token_type_ids'][0,0])
        assert text_positions.any()
        text_type=b['mm_token_type_ids'][text_positions][0].item()
        suffix=[self.sid['<NAV_OLD_MEMORY>']]*8+[
            self.sid[f'<EXEC_{a.upper()}_OK>'] for a in executed]+[
            self.sid['<NAV_WRITE_QUERY>']]*8+[self.sid['<NAV_ACTION_QUERY>']]
        ids=torch.cat([base_ids,torch.tensor([suffix],device=DEVICE)],1)
        assert ids.shape[1]<=512,'TOKEN_SEQUENCE_LIMIT: no truncation authorized'
        self.forward_tokens+=ids.shape[1]
        assert self.forward_tokens<=14000000,'ALL_FORWARD_TOKEN_BUDGET'
        mask=torch.cat([b['attention_mask'],torch.ones((1,len(suffix)),device=DEVICE,dtype=b['attention_mask'].dtype)],1)
        types=torch.cat([b['mm_token_type_ids'],torch.full((1,len(suffix)),text_type,device=DEVICE,dtype=b['mm_token_type_ids'].dtype)],1)
        positions,_=self.mm.get_rope_index(input_ids=ids,mm_token_type_ids=types,image_grid_thw=b['image_grid_thw'],attention_mask=mask)
        assert positions.shape==(3,1,ids.shape[1])
        emb=self.base.get_input_embeddings()(ids)
        with torch.no_grad():
            features=torch.cat(self.mm.get_image_features(b['pixel_values'],b['image_grid_thw'],return_dict=True).pooler_output,0)
        image_mask,_=self.mm.get_placeholder_mask(ids,inputs_embeds=emb,image_features=features)
        assert int((ids==self.base.config.image_token_id).sum())==features.shape[0]
        emb=emb.masked_scatter(image_mask,features.to(emb.dtype))
        old_start=n; write_start=n+8+len(executed)
        emb=emb.clone()
        emb[:,old_start:old_start+8]=memory+self.old_slot
        emb[:,write_start:write_start+8]=self.write_query
        emb[:,-1:]=self.action_query
        out=self.mm.language_model(inputs_embeds=emb,attention_mask=mask,position_ids=positions,past_key_values=None,use_cache=False,return_dict=True)
        assert out.past_key_values is None and self.mm.rope_deltas is None
        m=self.writer(out.last_hidden_state[:,write_start:write_start+8].float()).to(torch.bfloat16)
        logits=self.action_head(out.last_hidden_state[:,-1].float())
        assert torch.isfinite(m).all() and torch.isfinite(logits).all()
        return m,logits,{'tokens':ids.shape[1],'visual_tokens':features.shape[0]}

def build():
    torch.manual_seed(1109);np.random.seed(1109);random.seed(1109)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    base=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEVICE)
    return Policy(base,processor)

def zero_memory():return torch.zeros((1,8,2048),device=DEVICE,dtype=torch.bfloat16)

def load_record(row):
    p=json.loads((DATA/row['policy_file']).read_text()); s=json.loads((DATA/row['supervision_file']).read_text())
    assert set(p)=={'instruction','rgb_sequence'}
    images=[]
    for ref in p['rgb_sequence']:
        path=(DATA/row['rgb_reference_root']/ref).resolve()
        assert path.is_relative_to(DATA)
        with Image.open(path) as im:img=im.convert('RGB')
        assert hashlib.sha256(np.asarray(img).tobytes()).hexdigest()==path.stem
        images.append(img)
    assert len(images)==len(s['actions'])
    return p['instruction'],images,s['actions']

def payload(instruction,images,actions,t):
    assert 0<=t<len(images)
    return {'instruction':instruction,'images':images[max(0,t-1):t+1],'executed':actions[max(0,t-8):t]}

def policy_forward(policy,inputs,memory):
    if set(inputs)!={'instruction','images','executed'}:raise ValueError('Non-whitelisted policy input')
    return policy.step(memory=memory,**inputs)

def trainable_state(policy):return {n:p.detach().cpu().clone() for n,p in policy.named_parameters() if p.requires_grad}

def load_trainable(policy,state):
    target={n:p for n,p in policy.named_parameters() if p.requires_grad}
    assert set(target)==set(state)
    with torch.no_grad():
        for n,p in target.items():p.copy_(state[n].to(p.device))

