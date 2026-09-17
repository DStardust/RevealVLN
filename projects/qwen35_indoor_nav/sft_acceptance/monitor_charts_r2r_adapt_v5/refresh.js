'use strict';
const el=id=>document.getElementById(id), fmt=x=>Number.isFinite(x)?x.toLocaleString('zh-CN'):'—', pct=x=>Number.isFinite(x)?(100*x).toFixed(2)+'%':'—';
function draw(points){
 const c=el('loss'),box=c.getBoundingClientRect(),ratio=window.devicePixelRatio||1;c.width=box.width*ratio;c.height=220*ratio;
 const ctx=c.getContext('2d');ctx.scale(ratio,ratio);const w=box.width,h=220,p=34;
 const rows=points.filter(x=>Number.isFinite(x.rolling_ce)&&Number.isFinite(x.updates));
 if(!rows.length){ctx.fillStyle='#bac9da';ctx.fillText('等待实际训练日志',p,50);return;}
 const top=Math.max(...rows.map(x=>x.rolling_ce))*1.08, xmax=8000;
 ctx.strokeStyle='#40516a';ctx.beginPath();ctx.moveTo(p,10);ctx.lineTo(p,h-p);ctx.lineTo(w-10,h-p);ctx.stroke();
 ctx.fillStyle='#bac9da';ctx.font='12px system-ui';ctx.fillText(top.toFixed(2),0,18);ctx.fillText('0',12,h-p);ctx.fillText('0',p,h-10);ctx.fillText('8,000 更新',w-85,h-10);
 let last=null;for(const x of rows){const px=p+x.updates/xmax*(w-p-10),py=h-p-x.rolling_ce/top*(h-p-10);
 ctx.strokeStyle=x.segment==='expanded_stage1'?'#8796aa':'#65b8ff';
 if(last&&last.segment===x.segment){ctx.beginPath();ctx.moveTo(last.x,last.y);ctx.lineTo(px,py);ctx.stroke();}last={segment:x.segment,x:px,y:py};}
}
async function refresh(){
 try{
 const res=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(15000)});if(!res.ok)throw Error('HTTP '+res.status);const d=await res.json();
 el('warning').textContent='';el('state').textContent=d.display_state;el('bar').value=d.current_segment_new_updates;
 el('updates').textContent=fmt(d.current_segment_new_updates);el('decisions').textContent=fmt(d.current_segment_new_decisions);
 el('eta').textContent=Number.isFinite(d.estimated_segment_remaining_seconds)?Math.ceil(d.estimated_segment_remaining_seconds/60)+' 分钟':'—';
 const v=d.navigation.continued;el('eval').textContent=v.completed+' / 100';
 el('workflow').textContent='流程：'+(d.continuation_workflow.status||'准备中');
 el('restoration').textContent='本段资源恢复：'+(d.training_completion.holder_restoration_verified?'已核实':'运行中或等待核实');
 el('fresh').textContent='本次读取：'+new Date().toLocaleString()+'；接口版本：'+d.monitor_version;
 const labels=[['before','旧基座'],['matched','保留的较好 4,000 步'],['high_lr','混合来源 8,000 步'],['low_lr','低学习率 8,000 步'],['stop_calibrated','停止校准（未采用）'],['full_epoch','完整一轮（未采用）'],['continued','本次 R2R 8,000 步']];
 el('nav').replaceChildren();for(const [key,label] of labels){const s=d.navigation[key],r=s&&s.status==='COMPLETE'?s.result:null;const tr=document.createElement('tr');
 for(const val of [label,pct(r&&r.sr),pct(r&&r.spl),pct(r&&r.ndtw),s?s.status:'未开始']){const td=document.createElement('td');td.textContent=val;tr.appendChild(td);}el('nav').appendChild(tr);}
 el('gate').textContent=d.main_gate+(d.continuation_result?' 本轮门槛：'+(d.continuation_result.positive_development_signal?'满足（仅开发信号）':'不满足'):' 本轮尚未得出结果。');
 el('budget').textContent=d.eta_note;draw(d.points);
 }catch(e){el('warning').textContent='状态读取失败：'+String(e)+'。显示的旧值不可当成实时状态；页面会继续重试。';}
}
refresh();setInterval(refresh,10000);window.addEventListener('resize',refresh);
