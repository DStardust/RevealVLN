const el=id=>document.getElementById(id),pct=x=>x==null?'未测':(100*x).toFixed(2)+'%',num=x=>Number(x).toLocaleString('zh-CN');
function rows(id,values){el(id).replaceChildren(...values.map(row=>{const tr=document.createElement('tr');row.forEach(x=>{const td=document.createElement('td');td.textContent=x;tr.appendChild(td);});return tr;}));}
function plot(arms){
 const svg=el('loss-chart'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
 const colors=['#87d8ff','#ffb566'],data=Object.values(arms).map(a=>a.history);
 const max=Math.max(2,...data.flat().map(p=>p.ce)),rounded=Math.ceil(max*10)/10;
 function node(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!=null)n.textContent=text;svg.appendChild(n);return n;}
 for(let i=0;i<=4;i++){const y=205-i*45;node('line',{x1:50,y1:y,x2:880,y2:y,stroke:'#344158'});node('text',{x:3,y:y+5,fill:'#abbad0','font-size':13},(rounded*i/4).toFixed(2));}
 node('text',{x:50,y:230,fill:'#abbad0','font-size':13},'0');node('text',{x:840,y:230,fill:'#abbad0','font-size':13},'4000');
 data.forEach((points,i)=>{if(points.length)node('polyline',{points:points.map(p=>(50+830*p.updates/4000)+','+(205-180*p.ce/rounded)).join(' '),fill:'none',stroke:colors[i],'stroke-width':2});});
 if(!data.some(a=>a.length))node('text',{x:290,y:105,fill:'#abbad0','font-size':18},'尚无训练日志，不绘制预期曲线');
}
async function refresh(){try{
 const res=await fetch('/api/status',{cache:'no-store'});if(!res.ok)throw Error('HTTP '+res.status);
 const d=await res.json();if(d.monitor_version!=='ordinary_history8_pair_r1')throw Error('监控版本不匹配');
 const p=d.paired_history,arms=p.arms,labels={control_recent2:'2帧对照',treatment_prefix8:'8帧历史'};
 let phase='训练准备中';if(p.diagnostic_process_live)phase='固定微批4 GPU累积核验中';
 if(p.diagnostic_result?.status==='PASS_INTERFACE_ONLY')phase='固定微批4已通过，等待训练准入';
 if(p.approval_present)phase='训练已准入，等待实际进程';
 for(const [key,a]of Object.entries(arms))if(a.live_rank_pids.length)phase=labels[key]+'运行中';
 if(Object.values(arms).some(a=>a.launch?.status==='FAILED'))phase='训练退出/失败，查看保留回执';
 if(Object.values(arms).every(a=>a.acceptance))phase='两分支训练完成 · 待闭环评测';
 el('phase').textContent=phase;
 el('diagnostic').textContent='固定微批4核验：'+(p.diagnostic_result?.status||'尚未完成')+'；资源回执：'+(p.diagnostic_launch?.status||'尚未闭合')+'。这些不是SR结果。';
 rows('training',Object.entries(arms).map(([key,a])=>{const r=a.progress;return [labels[key],(a.result||r)?num((a.result||r).updates)+' / 4000':'未开始',(a.result||r)?num(a.result?a.result.global_decisions:r.decisions)+' / 384,000':'未开始',r?r.metrics.mean_ce.toFixed(4):'—',a.live_rank_pids.join(', ')||'无',a.acceptance?'完整训练已验收':a.launch?.status||(a.live_rank_pids.length?'运行中':'等待')];}));
 el('prior-memory').textContent='前版在'+(p.prior_memory_training?.updates??'未知')+'步因CPU RSS上限停止，失败与占位恢复记录保留。R1采用单线程有界预取，两组各100个真实样本的输入张量一致；64 GiB上限不变。';
 el('training-time').textContent=Object.entries(arms).map(([key,a])=>labels[key]+'最近训练日志：'+(a.progress?new Date(a.progress.unix*1000).toLocaleString():'无')).join('；');
 plot(arms);
 rows('navigation',[['完整1,839条','历史41800步',pct(d.sr40.historical_full.sr),pct(d.sr40.historical_full.spl)],['内部100条','保留最佳4k',pct(d.sr40.best_internal.sr),pct(d.sr40.best_internal.spl)],['新配对实验','两分支尚未接入闭环结果','未测','未测'],['完整1,839条','阶段A目标','≥40%','完整报告']]);
 el('resource').textContent=d.gpu?.error?'资源读取失败：'+d.gpu.error:(d.gpu?.devices||[]).map(x=>'GPU'+x.index+' '+num(x.used_mib)+' MiB').join('；');
 el('updated').textContent='网页与资源读取：'+new Date().toLocaleString()+'；每10秒刷新。';
 el('error').textContent='';
}catch(err){el('error').textContent='状态读取失败：'+err.message+'；下方可能为旧显示，请核对日志时间。'}}refresh();setInterval(refresh,10000);


