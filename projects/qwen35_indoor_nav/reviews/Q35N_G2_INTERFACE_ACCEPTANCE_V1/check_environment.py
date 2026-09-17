import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ROOT=LINE.parents[1]


def main():
    import torch,transformers,torchvision,peft
    from transformers.models.qwen3_5 import modeling_qwen3_5
    for m in [torch,transformers,torchvision,peft,modeling_qwen3_5]:assert Path(m.__file__).resolve().is_relative_to(LINE)
    assert torch.__version__.split('+')[0]=='2.8.0' and transformers.__version__=='5.15.0'
    local=OUT/'official_source/modeling_qwen3_5.py'
    installed=Path(modeling_qwen3_5.__file__)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [local,installed]}
    r=subprocess.run([str(ROOT/'.tools/uv/uv'),'pip','check','--python',sys.executable],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    report={'python':sys.version,'prefix':sys.prefix,'torch':torch.__version__,'torch_built_cuda':torch.version.cuda,
        'transformers':transformers.__version__,'torchvision':torchvision.__version__,'peft':peft.__version__,
        'module_paths_inside_line':True,'dependency_check_pass':True,'dependency_check_output':r.stdout+r.stderr,
        'source_hashes':hashes,'installed_modeling_matches_tag_source':len(set(hashes.values()))==1,
        'gpu_kernel_tested':False,'distributions':[{'name':d.metadata['Name'],'version':d.version} for d in importlib.metadata.distributions()]}
    with (OUT/'ENVIRONMENT_ACCEPTANCE.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('distributions','dependency_check_output')},indent=2))


if __name__=='__main__':main()
