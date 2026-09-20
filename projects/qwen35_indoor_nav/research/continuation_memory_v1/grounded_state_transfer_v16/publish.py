"""Snapshot the V16 handoff, commit only its paths, push and verify the branch."""
import argparse
import io
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *

BRANCH='codex/q35n-grounded-state-v16-20260920'

def snapshot():
    import package_evidence
    package_evidence.main()
    out=HERE/'handoff'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    root_files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.txt')]
    paths=[]
    for parent in ('runs','standalone_jobs'):
        for directory,dirs,files in os.walk(HERE/parent):
            dirs[:]=[d for d in dirs if d not in ('content','cache','tmp','__pycache__')]
            for name in files:
                p=Path(directory)/name
                if p.suffix in ('.py','.json','.jsonl','.md','.csv','.log','.txt'):paths.append(p)
    entries=[];archive=out/'EVIDENCE_LOGS.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=5) as bundle:
        for p in sorted(paths):
            try:blob=p.read_bytes()
            except FileNotFoundError:continue # An atomic temporary file disappeared.
            if '.tmp' in p.name:continue
            relative=str(p.relative_to(HERE));info=tarfile.TarInfo(relative);info.size=len(blob);info.mtime=int(time.time())
            bundle.addfile(info,io.BytesIO(blob))
            entries.append(dict(path=relative,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    pieces=[]
    if archive.stat().st_size>40*2**20:
        with archive.open('rb') as source:
            for number in range(10000):
                blob=source.read(40*2**20)
                if not blob:break
                path=out/f'EVIDENCE_LOGS.tar.gz.part{number:03d}';path.write_bytes(blob);pieces.append(path)
        archive.unlink() # Only the temporary archive just produced by this function.
    else:pieces=[archive]
    artifact_refs=[]
    for p in (HERE/'runs').glob('*/train/*/FINAL.pt'):
        root_files.append(p)
    for pattern in ('*/train/*/attempt_*/CHECKPOINT_*.json','*/features/FEATURE_RESULT.json','*/RAW_DATA_AUDIT.json'):
        artifact_refs.extend(str(p.relative_to(HERE)) for p in (HERE/'runs').glob(pattern))
    index=dict(snapshot_unix=time.time(),reviewable_files=entries,
        archives=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in pieces],
        server_asset_catalog_references=artifact_refs,
        omitted_binary_assets='Raw RGB/semantic arrays, base model, feature caches and optimizer/RNG checkpoint binaries remain on the project server. Their recorded paths/hashes are in the archived asset and checkpoint metadata. Final V16 heads, when present, are committed.',
        active_run_semantics='Point-in-time snapshot; active logs may end at a partial JSONL line. Only sealed complete groups count as evaluated.')
    write(out/'INDEX.json',index,True)
    # Frequently reviewed summaries remain directly accessible to a fresh web session.
    names=('REPORT_ZH.md','REVIEW.json','COUNTS.json','SPLIT_AUDIT.json','RAW_DATA_AUDIT.json','BASE_TRAINING_SCENE_AUDIT.json',
           'EVALUATION_REGISTRY.json','ROLLOUTS.csv','CACHE_LIVE_PARITY.json','FAILURE_LOCALIZATION.json','RESULT.json','STATUS.json')
    for run in sorted((HERE/'runs').iterdir()):
        if not run.is_dir():continue
        for name in names:
            p=run/name
            if p.is_file():
                dest=out/run.name/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    write(HERE/'GITHUB_HANDOFF.json',dict(snapshot=str(out.relative_to(HERE)),index_sha256=sha(out/'INDEX.json'),
        branch=BRANCH,report='REPORT_ZH.md',code='README.md'))
    return root_files+list(out.rglob('*'))+[HERE/'GITHUB_HANDOFF.json']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--message',default='Implement V16 shared-data experiment and publish measured progress')
    parser.add_argument('--snapshot-only',action='store_true');args=parser.parse_args()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!=BRANCH:raise ValueError('UNEXPECTED_CHECKOUT')
    paths=[str(p.relative_to(ROOT)) for p in snapshot() if p.is_file()]
    if args.snapshot_only:return
    staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).split(b'\0')
    prefix=str(HERE.relative_to(ROOT))+'/'
    if any(p and not p.decode().startswith(prefix) for p in staged):raise ValueError('UNRELATED_STAGED_CHANGES_PRESERVED; refusing mixed commit')
    for offset in range(0,len(paths),100):subprocess.run(['git','add','-f','--',*paths[offset:offset+100]],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m',args.message],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    python=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
    helper=LINE/'reviews/Q35N_V15_HANDOFF_20260920/proxy_connect.py'
    proxy=shlex.join([str(python),'-I','-S','-B',str(helper),'%h','%p'])
    env=dict(os.environ,GIT_SSH_COMMAND=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ProxyCommand='+proxy]))
    def remote(args,capture=False):
        return subprocess.run(['bash','-lc','proxyon >/dev/null 2>&1 && exec '+shlex.join(args)],cwd=ROOT,env=env,text=True,
            stdout=subprocess.PIPE if capture else None,check=True)
    remote(['git','push','-u','origin',BRANCH])
    remote_commit=remote(['git','ls-remote','--heads','origin','refs/heads/'+BRANCH],True).stdout.split()[0]
    if remote_commit!=commit:raise ValueError('REMOTE_COMMIT_MISMATCH')
    result=dict(status='PUSHED_AND_REMOTE_VERIFIED',branch=BRANCH,commit=commit,unix=time.time(),credentials_written=False)
    append(HERE/'PUSH_RECEIPTS.jsonl',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
