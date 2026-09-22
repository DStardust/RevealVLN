import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
def main(run):
 cfg=config(run);data=read(run/'TRAIN_DATA.json');out=run/'features'
 if (out/'FEATURE_RESULT.json').exists():
  if sha(out/'FEATURES.pt')!=read(out/'FEATURE_RESULT.json')['file_sha256']:raise ValueError('CACHE_CHANGED')
  return
 x=load('scale_assembler',V16/'extract_features.py');x.assemble(out,'FEATURES',len(data['features']))
 seals=[read(p) for p in out.glob('session_*/STATE_SEAL.json')]
 if not seals or any(not s['base_unchanged'] or s['base_state_sha256']!=seals[0]['base_state_sha256'] for s in seals):raise ValueError('FEATURE_STATE_SEALS')
 old=read(CPU/'BINDING.json')['feature_result'];ordinary=Path(old['ordinary_path'])
 if sha(ordinary)!=old['ordinary_sha256']:raise ValueError('ORDINARY_CACHE_IDENTITY')
 result=dict(parameters_unchanged=True,base_state_sha256=seals[0]['base_state_sha256'],file_sha256=sha(out/'FEATURES.pt'),windows=len(data['features']),ordinary_path=str(ordinary),ordinary_sha256=old['ordinary_sha256'],ordinary_shared_old_cache=True,base_updates=0,data_sha256=sha(run/'TRAIN_DATA.json'),real_qwen_forwards=sum(sum(1 for _ in p.open()) for p in out.glob('session_*/FEATURES_INPUTS_*.jsonl'))+sum(len(read(p)) for p in out.glob('session_*/WARMUP.json'))+2*sum(len(read(p)) for p in out.glob('session_*/GOLDEN_INPUTS.json')))
 immutable(out/'FEATURE_RESULT.json',result)
 files={str(run/p):h for p,h in read(run/'PREPARED.json')['files'].items()}
 files.update({str(LINE/p):h for p,h in read(run/'SOURCE_LOCK.json')['files'].items()})
 for path in (run/'PROTOCOL.json',out/'FEATURES.pt',ordinary,PARENT.parent/'natural_transfer_v9/DATA.json'):files[str(path)]=sha(path)
 immutable(run/'BINDING.json',dict(files=files,feature_result=result,source_commit=cfg['source_commit']))
if __name__=='__main__':main(Path(sys.argv[1]))
