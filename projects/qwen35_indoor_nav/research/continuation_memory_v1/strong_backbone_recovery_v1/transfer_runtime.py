"""The trained dense memory's causal inference path; no environment truth."""
import hashlib
import torch
import torch.nn.functional as F
from PIL import Image


def tensor_hash(tensor):
    raw=tensor.detach().contiguous().cpu()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


class DenseRuntime:
    def __init__(self, base, tokenizer, processor, head, token_ids, instruction):
        self.base=base;self.processor=processor;self.head=head;self.token_ids=token_ids
        self.memory=head.reset();self.writes=0
        ids=torch.tensor(tokenizer.encode(instruction,add_special_tokens=False),device='cuda')
        with torch.inference_mode():
            language=base.get_input_embeddings()(ids).float().mean(0)
            self.language=F.layer_norm(language,language.shape).cpu()
            self.actions=[]
            for token in token_ids:
                value=base.get_input_embeddings().weight[token].float()
                self.actions.append(F.layer_norm(value,value.shape).cpu())

    @torch.inference_mode()
    def feature(self, rgb, previous_action):
        pixel=self.processor.preprocess(Image.fromarray(rgb),return_tensors='pt')['pixel_values'][0]
        visual=self.base.encode_images(pixel[None].cuda().bfloat16()).float().mean(1)[0].cpu()
        visual=F.layer_norm(visual,visual.shape)
        past=torch.zeros_like(self.language) if previous_action is None else self.actions[previous_action]
        return (visual+self.language+past)/3

    @torch.inference_mode()
    def observe(self, rgb, previous_action):
        if previous_action == 0:raise ValueError('STOP_MUST_NOT_CREATE_OBSERVATION')
        feature=self.feature(rgb,previous_action)
        self.memory=self.head.update(feature[None].cuda(),self.memory)
        self.writes+=1
        if not torch.isfinite(self.memory).all():raise ValueError('NONFINITE_MEMORY')
        return dict(memory_input_sha256=tensor_hash(feature),memory_sha256=tensor_hash(self.memory),writes=self.writes)

    @torch.inference_mode()
    def delta(self, actor_feature):
        value=self.head.action_delta(actor_feature,self.memory)
        if not torch.isfinite(value).all():raise ValueError('NONFINITE_RESIDUAL')
        return value

