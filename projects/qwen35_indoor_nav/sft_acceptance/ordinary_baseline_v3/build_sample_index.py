"""One-time CPU build of SAMPLE_INDEX.jsonl for ordinary baseline v3.

Reads the frozen snapshot via the sealed preflight (hash-bound), tokenizes all
instructions once, and writes per-decision samples with inflection weights and
token-length estimates. Deterministic; output is sealed into PROTOCOL.json.
"""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

sys.path.insert(0, str(HERE))
import data  # noqa: E402

MODEL = LINE / 'runtime/models/Qwen3.5-2B_15852e8'
COEF = 3.2  # VLN-CE R2R inflection weighting coefficient


def main():
    started = time.monotonic()
    require = data.require
    out = HERE / 'SAMPLE_INDEX.jsonl'
    rows, report = data.load_rows()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    specials = ['<NAV_OLD_MEMORY>', '<NAV_WRITE_QUERY>', '<NAV_ACTION_QUERY>'] + [
        '<EXEC_%s_%s>' % (a.upper(), s) for a in data.ACTIONS for s in ['OK', 'COLLISION']]
    processor.tokenizer.add_special_tokens({'additional_special_tokens': specials})
    expected = json.loads(
        (LINE / 'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())
    require({s: processor.tokenizer.convert_tokens_to_ids(s) for s in specials} == expected['ids'],
            'TOKENIZER_BINDING')
    result = data.build_sample_index(rows, processor, COEF, out)
    receipt = dict(status='SAMPLE_INDEX_BUILT', unix=time.time(), coef=COEF,
                   snapshot=report, **result, wall_seconds=time.monotonic() - started)
    with (HERE / 'SAMPLE_INDEX_BUILD.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(dict(samples=result['samples'], sha256=result['sha256'],
                          wall=round(receipt['wall_seconds'], 1))))


if __name__ == '__main__':
    main()
