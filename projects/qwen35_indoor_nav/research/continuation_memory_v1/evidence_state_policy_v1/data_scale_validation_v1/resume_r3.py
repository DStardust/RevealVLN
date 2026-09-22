"""Independent recovery bootstrap, then the unchanged registered experiment."""
import argparse,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():raise ValueError('RUN_ID')
    cfg=read(HERE/'PROTOCOL.json');run=HERE/'runs'/a.run_id
    if not (run/'PREPARED.json').exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        subprocess.run([cfg['torch_python'],'-I','-B',str(HERE/'recover_preparation_r3.py'),'--source',str(HERE/'runs/scale_002'),'--run',str(run)],env=env,stdin=subprocess.DEVNULL,check=True)
    cmd=[cfg['standalone_python'],'-I','-B',str(HERE/'pipeline.py'),'--config',str(HERE/'PROTOCOL.json'),'--run-id',a.run_id,'--resume']
    os.execv(cmd[0],cmd)
if __name__=='__main__':main()
