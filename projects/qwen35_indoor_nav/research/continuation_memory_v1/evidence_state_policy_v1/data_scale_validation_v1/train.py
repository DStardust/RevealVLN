import os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
def main(run):
 cfg=config(run)
 for tag in os.environ['B2_TRAIN_TAGS'].split(','):
  arm,seed=tag.rsplit('_',1);folder=run/'train'/tag
  if (folder/'RESULT.json').exists():
   r=read(folder/'RESULT.json')
   if r['updates']!=cfg['steps'] or sha(folder/'FINAL.pt')!=r['checkpoint_sha256']:raise ValueError('FINAL_CHANGED')
   continue
  cmd=[cfg['torch_python'],'-I','-B',str(HERE/'train_one.py'),'--run',str(run),'--output',str(folder),'--mode','MONOTONIC','--arm',arm,'--seed',seed,'--device','cuda','--updates',str(cfg['steps'])]
  if folder.exists():cmd+=['--resume']
  subprocess.run(cmd,stdin=subprocess.DEVNULL,check=True)
if __name__=='__main__':main(Path(sys.argv[1]))
