"""Publish the versioned study and bounded evidence; retain server-only licensed assets."""
import fcntl,hashlib,io,json,os,re,shlex,subprocess,sys,tarfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

REVIEW=LINE/'reviews/Q35N_EVIDENCE_STATE_GPU_20260921'
TEXT={'.py','.json','.jsonl','.md','.csv','.log','.txt','.html'}
BLOCK=re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)')

def blob(path):
    value=path.read_bytes()
    if BLOCK.search(value):raise ValueError('CREDENTIAL_PATTERN:'+str(path))
    return value

def snapshot(run):
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out=REVIEW/'snapshots'/stamp;out.mkdir(parents=True)
    code=[LINE/p for p in read(HERE/'SOURCE_LOCK.json')['files']]
    for folder in (PARENT,HERE,HERE/'monitor_v1'):
        code.extend(p for p in folder.glob('*') if p.is_file() and p.suffix in TEXT)
    sources=[CPU,Path(read(HERE/'PROTOCOL.json')['training_reused_from']),run,HERE/'standalone_jobs']
    old=V16/'b2_kl_repair_v1/runs/repair_001'
    entries=[];direct=[];heads=sorted((run/'train').glob('*/FINAL.pt'))
    archive=out/'LOGS.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=4) as bundle:
        for folder in sources:
            for parent,dirs,files in os.walk(folder):
                dirs[:]=[d for d in dirs if d not in ('content','cache','__pycache__','tmp')]
                for name in sorted(files):
                    p=Path(parent)/name
                    if p.suffix not in TEXT or '.tmp' in name:continue
                    value=blob(p);rel=str(p.relative_to(LINE));info=tarfile.TarInfo(rel);info.size=len(value)
                    bundle.addfile(info,io.BytesIO(value));entries.append(dict(path=rel,bytes=len(value),sha256=hashlib.sha256(value).hexdigest()))
    parts=[]
    if archive.stat().st_size>40*2**20:
        with archive.open('rb') as stream:
            i=0
            while value:=stream.read(40*2**20):
                p=out/f'LOGS.tar.gz.part{i:03d}';p.write_bytes(value);parts.append(p);i+=1
        archive.unlink()
    else:parts=[archive]
    for folder in (old,CPU,run):
        for name in ('REPORT_ZH.md','RESULT.json','STATUS.json','USER_STOP_REQUEST.json','COUNTS.json','CPU_TEST_RESULT.json','PROTOCOL.json','SOURCE_LOCK.json','DATA_BINDING.json','CALIBRATION.json','EVALUATION_REGISTRY.json'):
            source=folder/name
            if source.exists():
                target=out/folder.name/name;target.parent.mkdir(exist_ok=True);target.write_bytes(blob(source));direct.append(target)
    for p in set(code):blob(p)
    write(out/'INDEX.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          evidence_files=entries,archive_parts=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in parts],
          heads=[dict(path=str(p.relative_to(LINE)),sha256=sha(p)) for p in heads],
          boundary='Raw RGB/semantic arrays, scene assets, Qwen base/checkpoint, feature caches, Python environments and optimizer binaries remain on server. Bound paths/hashes are preserved. Partial logs are point-in-time; only complete sealed groups count.'),True)
    state=read(run/'STATUS.json');result=read(run/'RESULT.json') if (run/'RESULT.json').exists() else None
    base='../../research/continuation_memory_v1/evidence_state_policy_v1'
    (REVIEW/'README_ZH.md').write_text(f'''# Q35N：证据状态策略 GPU 开发实验

最新快照：[{stamp}](snapshots/{stamp}/INDEX.json)。状态：**{result['status'] if result else state['status']}**。尚无完整结果时，方法收益为 UNKNOWN。

入口：[协议]({base}/gpu_runtime_r1/PROTOCOL.json)、[运行说明]({base}/gpu_runtime_r1/README.md)、[模型]({base}/model.py)、[损失]({base}/objective.py)、[Jev 只读调研]({base}/JEV_RESEARCH_NOTE_ZH.md)。

旧 B2Fix 实验按用户要求停止：1143/1152 续接、127/128 完整组。历史任务 B2Fix 40/192、B2 61/192，两臂各3条未完成；不重写为完整或有效修复。见[停止证据](snapshots/{stamp}/repair_001/USER_STOP_REQUEST.json)和[部分结果](snapshots/{stamp}/repair_001/REPORT_ZH.md)。

新实验共享现有 CPU 验收数据、初始化与监督，对照直接预测状态 DIRECT、单调累积 MONOTONIC、可修订历史 REVISE。每种三种子，共九个全新轻量模型、各1200次更新；冻结 Qwen best4k，原始全量保持 KL 不变。计划128条件×九模型=1152次真实续接。主任务和历史无关控制每臂各192，使用七卡1–7并行完整配对组。

当前数据为已暴露的 FIT四屋/32父族、DEV一屋/8父族，不是独立TEST。新模块的贡献是假设，不宣称普通VLN收益或论文新颖性已经通过。现有B2不是本轮原样匹配臂，不把新增事件监督、动作状态输入的共同变化归于REVISE。

本次运行器修复CONTENT_PATH错误，复用gpu_v1已训练完成的九份权重（新增训练更新0）；模型、损失、动作和评测均未变，旧失败日志保留。

Jev暂按TypeSafe于2026-09-15发布的项目理解。仅增加真实自主输出的只读Brier/ECE统计；不调用其API、不改变动作、不选择阈值，不把概率输出本身作为新发明。

独立systemd流水线完成后复核、恢复GPU2–7占位、发布最终权重及日志，不需要Codex在线。监控为服务器127.0.0.1:18770。当前快照不代替最终封存报告。
''')
    return sorted(set(code+heads+direct+parts+[out/'INDEX.json',REVIEW/'README_ZH.md']))

def main(run):
    cfg=read(HERE/'PROTOCOL.json');branch=cfg['publication_branch']
    with (V16/'b2_kl_repair_v1/PUBLISH.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=branch:raise ValueError('BRANCH_CHANGED')
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('OTHER_STAGED_CHANGES')
        paths=[str(p.relative_to(ROOT)) for p in snapshot(run)]
        for i in range(0,len(paths),80):subprocess.run(['git','add','-f','--',*paths[i:i+80]],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Record evidence-state GPU study and preserved stopped repair'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        helper=LINE/'reviews/Q35N_V15_HANDOFF_20260920/proxy_connect.py'
        proxy=shlex.join([cfg['standalone_python'],'-I','-S','-B',str(helper),'%h','%p'])
        ssh=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','HostName=ssh.github.com','-p','443','-o','ProxyCommand='+proxy])
        env=dict(os.environ,GIT_SSH_COMMAND=ssh)
        def remote(args):
            return subprocess.run(['bash','-lc','proxyon >/dev/null 2>&1 && exec '+shlex.join(args)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=600)
        pushed=remote(['git','push','-u','origin',branch])
        if pushed.returncode:raise RuntimeError('GITHUB_PUSH_FAILED:'+pushed.stderr[-1500:])
        check=remote(['git','ls-remote','--heads','origin','refs/heads/'+branch])
        if check.returncode or not check.stdout.split() or check.stdout.split()[0]!=commit:raise ValueError('REMOTE_SHA_NOT_VERIFIED')
        receipt=dict(status='PUSHED_AND_VERIFIED',commit=commit,branch=branch,unix=time.time())
        append(HERE/'PUSH_RECEIPTS.jsonl',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
