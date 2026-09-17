"""CPU tokenizer and lazy-RGB interface audit. Never loads model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
MODEL = LINE/'runtime/models/Qwen3.5-2B_15852e8'


def main(snapshot):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import sys
    sys.path.insert(0, str(HERE))
    import data
    from PIL import Image
    from transformers import AutoProcessor
    started = time.monotonic()
    snapshot = Path(snapshot).resolve()
    if snapshot.parent != HERE:
        raise ValueError('SNAPSHOT_SCOPE')
    output = HERE/'TOKEN_AUDIT.json'
    if output.exists():
        raise ValueError('TOKEN_AUDIT_EXISTS_USE_NEW_VERSION')
    rows = [json.loads(line) for line in (snapshot/'TRAINING_INDEX.jsonl').read_text().splitlines()]
    prior = json.loads((LINE/'sft_acceptance/v1/SOURCE_LOCK.json').read_text())
    metadata_hashes = {}
    for name, expected_sha in prior.items():
        path = data.ROOT / name
        if path.parent == MODEL and not path.name.endswith('.safetensors'):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
            metadata_hashes[name] = expected_sha
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    actions = ['move_forward', 'turn_left', 'turn_right', 'STOP']
    specials = ['<NAV_OLD_MEMORY>', '<NAV_WRITE_QUERY>', '<NAV_ACTION_QUERY>'] + [
        f'<EXEC_{a.upper()}_{state}>' for a in actions for state in ['OK','COLLISION']]
    processor.tokenizer.add_special_tokens({'additional_special_tokens': specials})
    expected = json.loads((LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())
    assert {s: processor.tokenizer.convert_tokens_to_ids(s) for s in specials} == expected['ids']
    # Fixed-size placeholders measure only token layout, never a model response.
    blank = Image.new('RGB', (224, 224))
    def prompt(instruction, n=2):
        return processor.apply_chat_template([{'role':'user','content':[
            *[{'type':'image'} for _ in range(n)], {'type':'text','text':instruction}]}],
            tokenize=False, add_generation_prompt=True)
    text = prompt('CPU layout check.')
    raw_length = len(processor.tokenizer(text)['input_ids'])
    encoded = processor(text=[text], images=[blank, blank], return_tensors='pt')
    expansion = encoded['input_ids'].shape[1] - raw_length
    texts, ids = [], []
    for i, row in enumerate(rows):
        path = data.ROOT/row['sourceRoot']/row['policy_file']
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == row['policy_sha256']
        instruction = json.loads(blob)['instruction']
        texts.append(prompt(instruction))
        ids.append(row['record_id'])
        if i % 2000 == 0:
            print(json.dumps({'metadata_instructions_read': i}), flush=True)
    encoded_text = processor.tokenizer(texts, padding=False, truncation=False)['input_ids']
    # 8 old-memory + up to 8 executed-action + 8 write-query + 1 action-query.
    upper = [len(x) + expansion + 25 for x in encoded_text]
    order = sorted(range(len(rows)), key=lambda i: upper[i])
    chosen = sorted(set(order[-10:] + order[:5] + order[::max(1,len(order)//10)]))
    layout_checks = []
    for i in chosen:
        actual = processor(text=[texts[i]], images=[blank,blank], return_tensors='pt')
        assert actual['input_ids'].shape[1]+25 == upper[i]
        layout_checks.append(ids[i])
    # Decode actual production pixels through the adapter for one instruction
    # per source/house. No synthetic observations used in this interface check.
    seen, rgb_checked = set(), 0
    for row in rows:
        key = (row['source'], row['scene_group'])
        if key in seen:
            continue
        seen.add(key)
        record = data.OrdinaryRecord(row)
        for t in sorted({0, row['decisions']-1}):
            payload = record.decision(t, decode_rgb=True)
            assert set(payload['policy']) == {'instruction','images','executed_actions'}
            assert payload['supervision']['target_action'] in actions
            rgb_checked += len(payload['policy']['images'])
    result = dict(status='CPU_TOKENIZER_AND_LAZY_RGB_CHECKED_NO_MODEL_FORWARD',
                  snapshot_result_sha256=hashlib.sha256((snapshot/'RESULT.json').read_bytes()).hexdigest(),
                  records=len(rows), max_step_token_upper_bound=max(upper),
                  p50=sorted(upper)[len(upper)//2], p95=sorted(upper)[int(len(upper)*.95)],
                  records_exceeding_old_512=sum(x>512 for x in upper),
                  records_exceeding_1024=sum(x>1024 for x in upper),
                  fixed_two_frame_expansion=expansion, exact_processor_layout_checks=layout_checks,
                  longest_records=[{'record_id': ids[i], 'tokens_upper_bound':upper[i]} for i in reversed(order[-10:])],
                  actual_rgb_references_decoded_and_pixel_hash_verified=rgb_checked,
                  source_house_pairs_checked=len(seen), model_weights_loaded=False,
                  frozen_model_metadata_sha256=metadata_hashes,
                  gpu_operations=0, training_started=False, wall_seconds=time.monotonic()-started)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot', type=Path, default=HERE/'snapshot_v1')
    main(p.parse_args().snapshot)
