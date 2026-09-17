"""Ordinary baseline v3 model: Qwen3.5-2B + LoRA + 4-way action head, no memory module.

Structure revision versus v6 (documented in RECIPE_DECISION.md): the unvalidated
8-slot memory/writer is removed. Executed-action history uses a small trainable
embedding table; the action query is a single trainable vector; logits come from
the last position of each right-padded sample. Base and vision tower stay frozen.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
MODEL = LINE / 'runtime/models/Qwen3.5-2B_15852e8'
ACTIONS = ['move_forward', 'turn_left', 'turn_right', 'STOP']
EXEC_TOKENS = {a: '<EXEC_%s_OK>' % a.upper() for a in ACTIONS[:-1]}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class PolicyV3(nn.Module):
    def __init__(self, base, processor, lora_rank=8, lora_alpha=16):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        self.base = base
        self.processor = processor
        self.mm = base.model
        specials = ['<NAV_OLD_MEMORY>', '<NAV_WRITE_QUERY>', '<NAV_ACTION_QUERY>'] + [
            '<EXEC_%s_%s>' % (a.upper(), s) for a in ACTIONS for s in ['OK', 'COLLISION']]
        processor.tokenizer.add_special_tokens({'additional_special_tokens': specials})
        rows_before = base.get_input_embeddings().weight.shape[0]
        base.resize_token_embeddings(max(rows_before, len(processor.tokenizer)), mean_resizing=False)
        for p in base.parameters():
            p.requires_grad_(False)
        self.mm.language_model = get_peft_model(
            self.mm.language_model,
            LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.,
                       target_modules=['q_proj', 'v_proj'], bias='none'))
        self.exec_embed = nn.Embedding(len(EXEC_TOKENS), 2048, device='cuda:0', dtype=torch.bfloat16)
        self.action_query = nn.Parameter(torch.randn(1, 2048, device='cuda:0', dtype=torch.bfloat16) * .01)
        self.action_head = nn.Linear(2048, 4, device='cuda:0', dtype=torch.float32)
        self.sid = {x: processor.tokenizer.convert_tokens_to_ids(x) for x in specials}
        expected = json.loads(
            (LINE / 'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())
        require(self.sid == expected['ids'] and rows_before == expected['embedding_rows_before'],
                'TOKENIZER_BINDING')
        self.exec_sid = {a: self.sid[EXEC_TOKENS[a]] for a in ACTIONS[:-1]}
        self.query_sid = self.sid['<NAV_ACTION_QUERY>']
        self.forward_tokens = 0

    def forward(self, **kwargs):
        return self.forward_batch(**kwargs)

    def forward_batch(self, input_ids, attention_mask, mm_token_type_ids, image_grid_thw,
                      pixel_values, exec_index, action_index):
        """One batched forward. exec_index: LongTensor [N_exec, 3] = (row, pos, action_id);
        action_index: LongTensor [B, 2] = (row, pos) of the action-query position."""
        p = self
        p.mm.rope_deltas = None
        pos, _ = p.mm.get_rope_index(input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
                                     image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        emb = p.base.get_input_embeddings()(input_ids)
        with torch.no_grad():
            features = p.mm.get_image_features(
                pixel_values, image_grid_thw, return_dict=True).pooler_output
            if isinstance(features, (list, tuple)):
                features = torch.cat(list(features), 0)
        image_mask, _ = p.mm.get_placeholder_mask(input_ids, inputs_embeds=emb, image_features=features)
        require(int((input_ids == p.base.config.image_token_id).sum()) == features.shape[0],
                'VISUAL_TOKEN_COUNT')
        emb = emb.masked_scatter(image_mask, features.to(emb.dtype)).clone()
        if exec_index.numel():
            emb[exec_index[:, 0], exec_index[:, 1]] = p.exec_embed(exec_index[:, 2]).to(emb.dtype)
        emb[action_index[:, 0], action_index[:, 1]] = p.action_query.to(emb.dtype)
        out = p.mm.language_model(inputs_embeds=emb, attention_mask=attention_mask,
                                  position_ids=pos, past_key_values=None, use_cache=False,
                                  return_dict=True)
        require(out.past_key_values is None and p.mm.rope_deltas is None, 'PERSISTENT_KV_FORBIDDEN')
        last = out.last_hidden_state[action_index[:, 0], action_index[:, 1]]
        logits = p.action_head(last.float())
        # exec_embed is unused on batches with no executed history (all t=0); the
        # zero term keeps it in the graph so DDP all-reduces it as an explicit zero
        # instead of dropping the other ranks' contributions (caught by parity gate).
        logits = logits + 0.0 * p.exec_embed.weight.sum()
        require(bool(torch.isfinite(logits).all()), 'NONFINITE_FORWARD')
        return logits


def build_policy(seed=1109):
    import random
    import numpy as np
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
    return PolicyV3(base, processor)


def broadcast_params(policy, src=0):
    """One-time parameter broadcast at startup (params are seed-identical; this
    only guards against any init drift)."""
    import torch.distributed as dist
    for p in policy.parameters():
        dist.broadcast(p.data, src)


def sync_grads(policy, world):
    """Manual gradient averaging: all-reduce after backward is complete (no
    hook/stream interleaving). Missing grads are contributed as explicit zeros
    so conditionally-used params (exec_embed on all-t0 batches) stay correct."""
    if world == 1:
        return
    import torch.distributed as dist
    import torch
    for p in policy.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)


def trainable_state(policy):
    return {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}


def load_trainable(policy, state):
    target = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    require(set(target) == set(state), 'TRAINABLE_STATE_MISMATCH')
    with torch.no_grad():
        for n, p in target.items():
            p.copy_(state[n].to(p.device))


class DecisionDataset(torch.utils.data.Dataset):
    """Yields per-sample processor output; all CPU cost stays in DataLoader workers."""

    def __init__(self, samples, store, processor):
        self.samples = samples
        self.store = store
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        item = self.store.get(sample['record_idx'], sample['t'])
        text = self.processor.apply_chat_template([{'role': 'user', 'content': [
            *[{'type': 'image'} for _ in item['images']],
            {'type': 'text', 'text': item['instruction']}]}],
            tokenize=False, add_generation_prompt=True)
        encoded = self.processor(text=[text], images=item['images'], return_tensors='pt')
        ids = encoded['input_ids'][0]
        types = encoded['mm_token_type_ids'][0]
        return dict(ids=ids, types=types, pixel_values=encoded['pixel_values'],
                    image_grid_thw=encoded['image_grid_thw'],
                    executed=list(item['executed']),
                    target=sample['target'], weight=sample['weight'])


def make_collate(pad_id, exec_sid, query_sid, image_token_id):
    """Pads per-sample encodings, appends executed-action + action-query suffix ids,
    and emits exec/action embedding assignment indices. Suffix positions use real
    vocab ids as placeholders; their embeddings are overwritten by trainable params.
    Carries only plain values so DataLoader workers never inherit CUDA state.
    """
    require(pad_id is not None, 'PAD_TOKEN_REQUIRED')

    def collate(items):
        batch = len(items)
        ids_list, types_list = [], []
        exec_entries = []
        action_positions = []
        for row, item in enumerate(items):
            ids, types = item['ids'], item['types']
            text_type = int(types[ids != image_token_id][0])
            require(all(a in exec_sid for a in item['executed']), 'EXECUTED_INTERFACE')
            exec_start = int(ids.shape[0])  # right padding keeps absolute indices
            for offset, action_name in enumerate(item['executed']):
                exec_entries.append((row, exec_start + offset, ACTIONS.index(action_name)))
            action_pos = exec_start + len(item['executed'])
            suffix_ids = [exec_sid[a] for a in item['executed']] + [query_sid]
            ids_list.append(torch.cat([ids, torch.tensor(suffix_ids, dtype=ids.dtype)]))
            types_list.append(torch.cat([types, torch.full((len(suffix_ids),), text_type,
                                                           dtype=types.dtype)]))
            action_positions.append((row, action_pos))
        max_len = max(x.shape[0] for x in ids_list)
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
        attention = torch.zeros((batch, max_len), dtype=torch.long)
        types = torch.zeros((batch, max_len), dtype=torch.long)
        for row, (ids, typ) in enumerate(zip(ids_list, types_list)):
            input_ids[row, :ids.shape[0]] = ids
            attention[row, :ids.shape[0]] = 1
            types[row, :ids.shape[0]] = typ
            if ids.shape[0] < max_len:  # pad positions carry this sample's text type
                types[row, ids.shape[0]:] = int(typ[ids != image_token_id][0])
        exec_index = torch.tensor(exec_entries, dtype=torch.long).reshape(-1, 3)
        action_index = torch.tensor(action_positions, dtype=torch.long)
        pixels = torch.cat([item['pixel_values'] for item in items], 0)
        grids = torch.cat([item['image_grid_thw'] for item in items], 0)
        targets = torch.tensor([item['target'] for item in items], dtype=torch.long)
        weights = torch.tensor([item['weight'] for item in items], dtype=torch.float32)
        return dict(input_ids=input_ids, attention_mask=attention, mm_token_type_ids=types,
                    pixel_values=pixels, image_grid_thw=grids, exec_index=exec_index,
                    action_index=action_index, targets=targets, weights=weights)

    return collate
