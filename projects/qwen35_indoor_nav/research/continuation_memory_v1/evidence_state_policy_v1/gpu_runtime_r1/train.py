"""Run only assigned heads, through the CPU-verified optimizer and loss implementation."""
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run):
    cfg=config(run)
    for tag in os.environ['B2_TRAIN_TAGS'].split(','):
        mode,seed=tag.rsplit('_',1);target=run/'train'/tag
        if (target/'RESULT.json').exists():
            result=read(target/'RESULT.json')
            if result['updates']!=cfg['steps'] or sha(target/'FINAL.pt')!=result['checkpoint_sha256']:raise ValueError('FINAL_HEAD_CHANGED')
            continue
        args=[cfg['torch_python'],'-I','-B',str(PARENT/'train.py'),'--run',str(run),'--output',str(target),
              '--mode',mode,'--seed',seed,'--device','cuda:0','--updates',str(cfg['steps'])]
        if target.exists():args.append('--resume')
        subprocess.run(args,cwd=ROOT,stdin=subprocess.DEVNULL,check=True)

if __name__=='__main__':main(Path(sys.argv[1]))
