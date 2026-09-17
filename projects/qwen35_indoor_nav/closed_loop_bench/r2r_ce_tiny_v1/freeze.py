"""CPU-only one-shot source/input freeze before the sole GPU attempt."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def main():
    assert not (HERE/'SOURCE_LOCK.json').exists() and not (HERE/'run_001').exists()
    sources=sorted(HERE.glob('*.py'))
    for path in sources:compile(path.read_text(),str(path),'exec')
    tests=subprocess.run([str(c.LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',str(HERE/'test_interface.py')],
                         cwd=c.ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    receipt=dict(unix=time.time(),returncode=tests.returncode,passed=tests.returncode==0,
                 output=tests.stdout,scope='CPU interface/official metrics; not navigation evidence',
                 syntax_files=[str(x) for x in sources])
    c.write(HERE/'CPU_TEST_RESULT.json',receipt,True)
    assert receipt['passed'],tests.stdout
    files=json.loads((HERE/'INPUT_BINDINGS.json').read_text())
    for path,expected in files.items():assert c.sha(path)==expected,path
    model=c.LINE/'runtime/models/Qwen3.5-2B_15852e8'
    additional=list(model.glob('*'))+[c.LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json']
    for path in additional:
        if path.is_file():files[str(path)]=c.sha(path)
    approval=dict(unix=time.time(),decision='APPROVED_FOR_TINY_BASELINE_EVALUATION_ONLY',
        scientific_novelty_claim=False,performance_pass_threshold=None,episode_count=8,max_actions=4000,
        optimizer_updates=0,source_review='Main agent sequential protocol/implementation/measurement review',
        no_independent_reviewer_claim=True,cpu_passed=True,existing_training_and_production='read-only',
        privileged_input_contract='only original instruction, last two RGB, eight executed action names',
        startup_gpu_admission='GPU1 UUID frozen, no contexts, <128 MiB; no holder signals',
        automatic_retry=False)
    c.write(HERE/'MAIN_AGENT_APPROVAL.json',approval,True)
    for path in HERE.iterdir():
        if path.is_file():files[str(path)]=c.sha(path)
    auth=c.LINE/'authorizations/ORDINARY_NAVBENCH_TINY_V1_20260911.json';files[str(auth)]=c.sha(auth)
    c.write(HERE/'SOURCE_LOCK.json',dict(unix=time.time(),files=files),True)
    print(json.dumps(dict(cpu_passed=True,locked_files=len(files),protocol=c.sha(HERE/'PROTOCOL.json'))))


if __name__=='__main__':main()
