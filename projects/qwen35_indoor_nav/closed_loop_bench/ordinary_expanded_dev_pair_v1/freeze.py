"""Record CPU checks and immutable code before any new development episode."""
import importlib.util
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('pair_prepare_freeze',HERE/'prepare_cases.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
test=subprocess.run([str(p.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'test_pair.py')],
                    capture_output=True,text=True,timeout=60)
p.save(HERE/'CPU_TEST_RESULT.json',dict(passed=test.returncode==0,output=test.stdout+test.stderr,unix=time.time()))
assert test.returncode==0
paths=list(HERE.glob('*.py'))+[HERE/'SPEC_ZH.md',HERE/'SELECTION_SEAL.json',HERE/'SELECTION.json',HERE/'EPISODES_PRIVILEGED.json',HERE/'GEOMETRY_PREFLIGHT.json']
for role in ('before','after'):paths+=list((HERE.parent/f'ordinary_expanded_dev_{role}_v1').glob('*.py'))
p.save(HERE/'CODE_FREEZE.json',{str(path):p.sha(path) for path in paths})
p.save(HERE/'MAIN_AGENT_APPROVAL.json',dict(user_request='研究下一步并且执行',scope='bounded standard ordinary model development comparison',
     code_freeze_sha256=p.sha(HERE/'CODE_FREEZE.json'),episode_count_per_model=100,models=2,
     gpu=1,requires_empty_gpu=True,holders_may_be_released=False,automatic_training=False,unix=time.time()))
print('PAIRED_DEV_CPU_FREEZE_PASS')
