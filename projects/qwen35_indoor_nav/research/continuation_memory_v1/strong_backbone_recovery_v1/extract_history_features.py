"""Teacher-force only actually executed actions; save features before each chunk.

Future labels/program state never enter this module's model call. Images are
from the new physical replay, not resized old evidence or generated imagery.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u


def main(data,output,family_id):
    output.mkdir(parents=True,exist_ok=False);u.setup_imports(output)
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    import streamvln_eval as official
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    source=u.read(u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json')
    family=next(f for f in source['families'] if f['family_id']==family_id)
    audit=u.read(data/family_id/'AUDIT.json');tasks={**family['compiler']['tasks'],'task_T':family['task_terminal_only']}
    tokenizer=transformers.AutoTokenizer.from_pretrained(str(u.MODEL),model_max_length=4096,padding_side='right',local_files_only=True)
    ids=[tokenizer.encode(x,add_special_tokens=False) for x in ['STOP','↑','←','→']]
    assert all(len(x)==1 for x in ids);ids=[x[0] for x in ids]
    config=transformers.AutoConfig.from_pretrained(str(u.MODEL),local_files_only=True)
    config.mm_vision_tower=str(u.MODEL/'siglip-so400m-patch14-384');config.vision_tower=config.mm_vision_tower
    model,loading=official.StreamVLNForCausalLM.from_pretrained(str(u.MODEL),config=config,attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,low_cpu_mem_usage=False,local_files_only=True,output_loading_info=True)
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),loading
    model.model.num_history=8;model.requires_grad_(False);model.to(0);model.eval();model.reset(1)
    def state_hash():
        digest=hashlib.sha256()
        for name,tensor in model.state_dict().items():
            digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
            digest.update(memoryview(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()))
        return digest.hexdigest()
    frozen=state_hash()
    u.write(output/'RUNTIME_IDENTITY.json',dict(base_sha256=frozen,base_updates=0,dtype='bfloat16',attention='flash_attention_2',
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,transformers_version=transformers.__version__))
    args=argparse.Namespace(save_video=False,num_frames=32,num_future_steps=4,num_history=8)
    helper=official.VLNEvaluator(str(u.HERE/'runs/native_001/fixtures/000.yaml'),split='val_unseen',env_num=1,
        output_path=str(output),model=model,tokenizer=tokenizer,epoch=0,args=args)
    current={};rows=[];began=time.time();visual_cache={};visual_calls=0
    def feature_hook(module,inputs,result):
        if current.get('pending'):
            current['feature']=result.last_hidden_state[0,-1].detach().cpu().float().clone();current['pending']=False
    hook=model.model.register_forward_hook(feature_hook)
    class ExecutedActions(transformers.LogitsProcessor):
        def __init__(self,actions):self.actions=actions;self.count=0
        def __call__(self,input_ids,scores):
            if self.count==0:current['base_logits']=scores[0,ids].detach().cpu().float().clone()
            target=ids[self.actions[self.count]] if self.count<len(self.actions) else tokenizer.eos_token_id
            self.count+=1
            forced=torch.full_like(scores,float('-inf'));forced[:,target]=0
            return forced
    for item in audit['traces']:
        path=Path(item['path']);assert u.sha(path)==item['sha256'];trace=u.read(path)
        assert trace['complete'] and trace['collisions']==0 and trace['actions'][-1]=='S' and len(trace['actions'])<=500
        for task_id,task in tasks.items():
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);model.reset_for_env(0)
            images=[]
            for obs in trace['observations']:
                array=np.load(data/'content'/(obs['rgb_hash']+'.rgb.npy'),allow_pickle=False)
                assert hashlib.sha256(array.tobytes()).hexdigest()==obs['rgb_hash']
                images.append(helper.image_processor.preprocess(Image.fromarray(array),return_tensors='pt')['pixel_values'][0])
            # Continuous observation for the added memory; original action-model
            # queries retain their four-action schedule. Every feature is a
            # frozen function of one current RGB, instruction and past action.
            memory_features=[]
            with torch.inference_mode():
                instruction_ids=torch.tensor(tokenizer.encode(task['instruction'],add_special_tokens=False),device='cuda')
                language=model.get_input_embeddings()(instruction_ids).float().mean(0)
                language=torch.nn.functional.layer_norm(language,language.shape)
                for t,(obs,img) in enumerate(zip(trace['observations'],images)):
                    key=obs['rgb_hash']
                    if key not in visual_cache:
                        visual_cache[key]=model.encode_images(img[None].cuda().bfloat16()).float().mean(1)[0].cpu()
                        visual_calls+=1
                    visual=visual_cache[key];visual=torch.nn.functional.layer_norm(visual,visual.shape)
                    past_action=torch.zeros_like(language)
                    if t:
                        aid=ids[{'S':0,'F':1,'L':2,'R':3}[trace['actions'][t-1]]]
                        past_action=model.get_input_embeddings().weight[aid].float()
                        past_action=torch.nn.functional.layer_norm(past_action,past_action.shape)
                    memory_features.append((visual+language.cpu()+past_action.cpu())/3)
            features=[];base_logits=[];queries=[];input_hashes=[];previous=None;past=None;time_ids=[]
            for step in range(0,len(trace['actions']),4):
                if step%32==0:model.reset_for_env(0);previous=None;past=None;time_ids=[]
                time_ids=list(range(step-step%32,step+1))
                if previous is None:
                    sources=copy.deepcopy(helper.conversation)
                    sources[0]['value']=sources[0]['value'].replace(' Where should you go next to stay on track?', ' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                    sources[0]['value']=sources[0]['value'].replace('<video>\n','').replace('<instruction>.',task['instruction'])
                    if step:sources[0]['value']+=' These are your historical observations <memory>.'
                else:sources=[dict(**{'from':'human'},value=''),dict(**{'from':'gpt'},value='')]
                tokens,_=helper.preprocess_qwen([sources],tokenizer,True,add_system=previous is None)
                tokens=tokens.cuda()
                if previous is not None:tokens=torch.cat([previous,tokens],1)
                selected=[images[step]]
                if step and step%32==0:selected=images[slice(0,time_ids[0],time_ids[0]//8)]+selected
                visual=torch.stack(selected)[None].cuda().bfloat16()
                executed=[{'S':0,'F':1,'L':2,'R':3}[x] for x in trace['actions'][step:step+4]]
                current.clear();current['pending']=True;forced=ExecutedActions(executed)
                # Only the force processor sees action targets. The saved feature
                # and logits are taken before it selects the first target token.
                value=model.generate(inputs=tokens,images=visual,depths=None,poses=None,intrinsics=None,
                    env_id=0,time_ids=[time_ids],task_type=[0],do_sample=False,num_beams=1,max_new_tokens=len(executed)+1,
                    use_cache=True,return_dict_in_generate=True,past_key_values=past,
                    logits_processor=transformers.LogitsProcessorList([forced]))
                assert not current['pending'];actual=helper.parse_actions(tokenizer.decode(value.sequences[0]))
                assert actual==executed,'TEACHER_FORCING_CHANGED_EXECUTED_ACTIONS'
                previous=value.sequences;past=value.past_key_values
                features.append(current['feature']);base_logits.append(current['base_logits']);queries.append(step)
                input_hashes.append(dict(step=step,ids_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
                    current_rgb_hash=trace['observations'][step]['rgb_hash'],pixel_values_sha256=hashlib.sha256(visual.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()))
            name=item['history']+'__'+item['continuation']+'__'+task_id;dest=output/(name+'.pt')
            torch.save(dict(memory_features=torch.stack(memory_features),memory_input='Frozen current projected visual mean + instruction embedding mean + actual previous action embedding, separately normalized; no future input',features=torch.stack(features),base_action_logits=torch.stack(base_logits),query_steps=queries,
                input_hashes=input_hashes,instruction=task['instruction'],trace_sha256=item['sha256'],future_labels_read_by_encoder=False),dest)
            rows.append(dict(history=item['history'],continuation=item['continuation'],task=task_id,path=str(dest),sha256=u.sha(dest),queries=len(queries)))
            u.write(output/'STATUS.json',dict(status='EXTRACTING',family=family_id,complete=len(rows),planned=len(audit['traces'])*len(tasks),
                wall_seconds=time.time()-began,base_updates=0))
    hook.remove();assert frozen==state_hash(),'BASE_STATE_CHANGED'
    assert not any(p.requires_grad for p in model.parameters())
    u.write(output/'RESULT.json',dict(status='FEATURES_COMPLETE',family=family_id,rows=rows,base_updates=0,
        all_parameters_and_persistent_buffers_unchanged=True,base_state_sha256=frozen,base_state_identity_source=str(u.HERE/'runs/native_001/ASSET_VERIFY.json'),
        source_sha256=u.sha(__file__),memory_cadence='Upstream four-action query schedule; actor features before first teacher-forced action token',
        extra_visual_calls=visual_calls,memory_observation_cadence='Each physical observation; shared by BC/B2/OURS, cost additional to original StreamVLN',no_future_query_or_label_in_features=True,wall_seconds=time.time()-began))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--family',required=True)
    a=p.parse_args();main(a.data,a.output,a.family)
