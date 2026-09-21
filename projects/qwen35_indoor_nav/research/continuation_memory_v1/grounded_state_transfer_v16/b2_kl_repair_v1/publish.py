"""Publish bounded evidence snapshots and exact sources; never touch unrelated staged work."""
import fcntl,hashlib,io,json,os,re,shlex,subprocess,sys,tarfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

REVIEW=LINE/'reviews/Q35N_B2_KL_REPAIR_20260921'
TEXT={'.py','.json','.jsonl','.md','.csv','.log','.txt','.html'}
BLOCK=re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)')


def blob(path):
    data=path.read_bytes()
    if BLOCK.search(data):raise ValueError('CREDENTIAL_PATTERN_IN_PUBLICATION:'+str(path))
    return data


def snapshot(run):
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out=REVIEW/'snapshots'/stamp;out.mkdir(parents=True)
    code=[];evidence=[];heads=[]
    modules=['identifiable_data_v1','identifiable_pilot_v1','b2_state_coverage_v1','b2_matched_control_v1','b2_localization_v1','b2_kl_repair_v1']
    for module in modules:
        folder=V16/module
        code.extend(p for p in folder.glob('*') if p.is_file() and p.suffix in TEXT)
        for sub in ('monitor_v1','parallel_eval_v1'):
            code.extend(p for p in (folder/sub).glob('*') if p.is_file() and p.suffix in TEXT and p.name not in ('MONITOR.jsonl',))
        for dirname,dirs,files in os.walk(folder):
            dirs[:]=[d for d in dirs if d not in ('content','cache','__pycache__','tmp','snapshots')]
            for name in files:
                p=Path(dirname)/name
                if '.tmp' in name:continue
                if p.suffix in TEXT and p not in code:evidence.append(p)
                if name=='FINAL.pt':heads.append(p)
    lock=read(HERE/'SOURCE_LOCK.json')
    code.extend(LINE/p for p in lock['files'])
    archive=out/'LOGS.tar.gz';entries=[]
    with tarfile.open(archive,'w:gz',compresslevel=4) as bundle:
        for p in sorted(set(evidence)):
            try:data=blob(p)
            except FileNotFoundError:continue
            name=str(p.relative_to(LINE));info=tarfile.TarInfo(name);info.size=len(data);info.mtime=int(time.time())
            bundle.addfile(info,io.BytesIO(data));entries.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    parts=[]
    if archive.stat().st_size>40*2**20:
        with archive.open('rb') as stream:
            for i in range(10000):
                data=stream.read(40*2**20)
                if not data:break
                part=out/f'LOGS.tar.gz.part{i:03d}';part.write_bytes(data);parts.append(part)
        archive.unlink() # Only this invocation's temporary aggregate; the parts preserve its bytes.
    else:parts=[archive]
    direct=[]
    for folder in (CONTROL,V16/'b2_localization_v1/run_002',run):
        for name in ('REPORT_ZH.md','RESULT.json','STATUS.json','COUNTS.json','DECISION.json','GRADIENT_SCALE_ADDENDUM.json','ROLLOUTS.json','EVALUATION_REGISTRY.json','SOURCE_LOCK.json','PROTOCOL.json','REUSE_BINDING.json','FEATURE_RESULT.json'):
            p=folder/name
            if p.exists():
                dest=out/folder.parent.name/folder.name/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob(p));direct.append(dest)
    # Preserve lightweight final heads. Base weights, raw arrays, feature caches and optimizer binaries stay on server.
    for p in set(code):blob(p)
    index=dict(timestamp_utc=stamp,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        evidence_files=entries,archive_parts=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in parts],
        final_heads=[dict(path=str(p.relative_to(LINE)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(heads))],
        omitted='Base weights, licensed scene/RGB/semantic arrays, feature cache binaries, optimizer/RNG checkpoint binaries and Python environments remain on server. Paths and fingerprints are preserved in bound manifests. A GitHub checkout alone cannot run Habitat.',
        snapshot_semantics='Active-run logs are point-in-time. Only full sealed model groups count. Completion snapshots are published by the independent pipeline.')
    write(out/'INDEX.json',index,True)
    status=read(run/'STATUS.json') if (run/'STATUS.json').exists() else dict(status='NOT_STARTED')
    result=read(run/'RESULT.json') if (run/'RESULT.json').exists() else None
    body=f'''# Q35N：B2 保持损失的可控修复与交接

最新证据快照：[{stamp}](snapshots/{stamp}/INDEX.json)。当前实验状态：**{result['status'] if result else status['status']}**。没有完成结果时，修复收益为 UNKNOWN。

审阅先读：

1. [本轮修复协议](../../research/continuation_memory_v1/grounded_state_transfer_v16/b2_kl_repair_v1/PROTOCOL.json)
2. [实现和运行命令](../../research/continuation_memory_v1/grounded_state_transfer_v16/b2_kl_repair_v1/README.md)
3. [上一轮完整1152次结果](snapshots/{stamp}/runs/matched_001/REPORT_ZH.md)
4. [冻结特征定位结果](snapshots/{stamp}/b2_localization_v1/run_002/REPORT_ZH.md)
5. [梯度幅度补充](snapshots/{stamp}/b2_localization_v1/run_002/GRADIENT_SCALE_ADDENDUM.json)

上一轮主任务：B1 59/192，B2 61/192，Terminal 41/192；控制任务：114/192、74/192、80/192。没有采用B2。当前事件特征探针FIT约97–100%，DEV约62–71%；B2记忆的历史状态线性探针FIT约89–90%，DEV约56%。这不支持单纯扩大记忆容量。

本轮唯一训练变化：B2Fix 在有合法教师监督、原生argmax与该条教师动作不同的位置取消KL分子项，原分母不变。不是删除全部KL，不改变数据、动作标签、普通动作CE、底模、8×64架构或STOP定义。用相同初始化、schedule训练三个B2Fix final1200；复用原B1/B2六个封存权重，不复用旧导航轨迹。

三臂×三种子，同条件九模型在一个底模进程比较；128条件、1152次新续接，每臂主任务192、控制任务192，八卡分组。主要差值B2Fix−B2，B1为参考。仍是单个已暴露DEV房屋的受控任务，不是普通VLN、盲测、论文创新或部署结论。24/24局部梯度反向不证明KL为根因；其幅度中位数约为动作梯度的24%。

代码、报告、逐步日志包、最终轻量权重及SHA可在此分支复核。原始场景、底模、特征缓存和优化器二进制未随GitHub发布，见INDEX边界；不要声称网页审阅等于模型复现。

最终结果和最新快照由独立运行器完成后自动提交推送，无需Codex保持在线。资源/正确性失败会保留所有attempt；不会按分数重试。
'''
    if result:body+=f"\n本轮结果：[完整报告](snapshots/{stamp}/runs/{run.name}/REPORT_ZH.md)。\n"
    (REVIEW/'README_ZH.md').write_text(body)
    return sorted(set(code+heads+direct+parts+[out/'INDEX.json',REVIEW/'README_ZH.md']))


def main(run):
    branch=read(HERE/'PROTOCOL.json')['publication_branch']
    with (HERE/'PUBLISH.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=branch:raise ValueError('PUBLICATION_BRANCH_CHANGED')
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        if staged:raise ValueError('STAGED_CHANGES_PRESENT; refusing to mix another commit')
        paths=[str(p.relative_to(ROOT)) for p in snapshot(run)]
        for i in range(0,len(paths),80):subprocess.run(['git','add','-f','--',*paths[i:i+80]],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Record controlled B2 KL repair and measured evidence'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        cfg=read(HERE/'PROTOCOL.json');helper=LINE/'reviews/Q35N_V15_HANDOFF_20260920/proxy_connect.py'
        proxy=shlex.join([cfg['standalone_python'],'-I','-S','-B',str(helper),'%h','%p'])
        ssh=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','HostName=ssh.github.com','-p','443','-o','ProxyCommand='+proxy])
        env=dict(os.environ,GIT_SSH_COMMAND=ssh)
        def remote(args):
            return subprocess.run(['bash','-lc','proxyon >/dev/null 2>&1 && exec '+shlex.join(args)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=600)
        pushed=remote(['git','push','-u','origin',branch])
        if pushed.returncode:raise RuntimeError('GITHUB_PUSH_FAILED: '+pushed.stderr[-1500:])
        checked=remote(['git','ls-remote','--heads','origin','refs/heads/'+branch])
        if checked.returncode or checked.stdout.split()[0]!=commit:raise RuntimeError('REMOTE_COMMIT_VERIFICATION')
        receipt=dict(status='PUSHED_AND_VERIFIED',branch=branch,commit=commit,unix=time.time())
        append(HERE/'PUSH_RECEIPTS.jsonl',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
