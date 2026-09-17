const el=id=>document.getElementById(id);
const pct=x=>x==null?'—':(100*x).toFixed(2)+'%';
function chart(rows){
 const c=el('chart'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
 if(!rows.length){ctx.fillStyle='#abbad0';ctx.font='22px sans-serif';ctx.fillText('等待首批训练指标',25,70);return;}
 const ys=rows.map(x=>x.ce),lo=Math.min(...ys),hi=Math.max(...ys),span=hi-lo||1;
 ctx.strokeStyle='#344158';ctx.strokeRect(60,20,810,230);ctx.fillStyle='#abbad0';ctx.font='18px sans-serif';
 ctx.fillText(hi.toFixed(3),0,35);ctx.fillText(lo.toFixed(3),0,248);ctx.fillText('0',60,285);ctx.fillText('本段更新步数 → 1000',645,285);
 ctx.beginPath();ctx.strokeStyle='#87d8ff';ctx.lineWidth=3;
 rows.forEach((r,i)=>{const x=60+810*r.updates/1000,y=250-230*(r.ce-lo)/span;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
}
async function refresh(){
 try{
  const res=await fetch('/api/status',{cache:'no-store'});if(!res.ok)throw Error('HTTP '+res.status);
  const d=await res.json(),t=d.onpolicy_training,p=t.progress||{},r=t.training_result||{};
  const n=(r.cursor||p.cursor||{}).updates||0,w=t.workflow||{},fin=t.result;
  el('transport').textContent='V10：只将旧图从前一步改为八步窗口起点；固定输出约束版SR17%未采用。当前训练 '+(n?('已完成 '+n+' 次更新'):'等待首批更新')+'。';
  el('status').textContent=fin?(fin.positive_development_signal?'完整复测达到事前正向门槛':'完整复测未达到正向门槛'):(w.status||'训练准备中');
  el('summary').textContent=n<1000?'正在检验视觉时间跨度与八步执行动作对齐能否改善导航；尚无新导航结论。':'固定训练段已结束，查看下方完整复测结论；未出齐时不判定收益。';
  el('progress').value=n;el('counters').textContent='本段 '+n+' / 1000步；优化器累计 '+(4000+n)+'步；本段已处理 '+(r.global_decisions||p.global_plan_decisions||0)+' / 99047次决策';
  chart(t.history);el('ce').textContent=t.history.length?'最近加权CE：'+t.history.at(-1).ce.toFixed(4)+'（仅训练指标）':'尚无指标';
  el('acceptance').textContent='首200步验收：'+(t.first_acceptance?.status||'等待')+'；最终权重验收：'+(t.final_acceptance?.status||'等待');
  el('anchor').textContent='仍最多编码两张图；在线缓存最多九张过去帧。固定99,047次样本读取无新视图标签冲突；实际输入RGB哈希将在完整评测中逐动作核对。无额外教师前向。';
  const rows=[['旧51301步',.14,.126726765,.367404866],['当前最佳：扩产4k',.21,.18016983923442284,.3658797007353029],['继续到8k（未采用）',.14,.1232098103,.332880856],['低LR8k（未采用）',.13,.1143599416,.380144997],['完整一轮（未采用）',.14,.1298740061,.3537311656],['R2R适配8k（未采用）',.15,.1339120518,.4041111783],['原精度纠错1k（未采用）',.19,.17527383450573364,.3933786870665479],['FP32主权重1k（未采用）',.15,.1342932342183672,.3143712169898085],['固定输出约束1k（未采用）',.17,.15405530911805246,.37448192402776925]];
  if(fin){const v=fin.after.result;rows.push(['本次：4k＋两帧动作对齐1k',v.sr,v.spl,v.ndtw]);}else rows.push(['本次：结果未完成',null,null,null]);
  const body=el('results');body.replaceChildren();for(const row of rows){const tr=document.createElement('tr');row.forEach((v,i)=>{const td=document.createElement('td');td.textContent=i?pct(v):v;tr.appendChild(td);});body.appendChild(tr);}
  el('verdict').textContent=fin?'配对新增成功 '+fin.paired.wins+' 条，失去 '+fin.paired.losses+' 条。'+(fin.positive_development_signal?'这是内部开发集的工程正向信号，尚无独立确认。':'不采用为更优模型，继续保留最佳4k。'):'等待完整100条结果和轨迹审计，不以部分路线宣称正向。';
  const ep=t.eval_progress||{},er=t.eval_result||{};
  el('evaluation').textContent='闭环评测：'+(ep.completed??er.episodes??0)+' / 100条；'+(ep.status||er.status||'等待最终权重')+'。仅完整轨迹审计后给出结论。';
  el('resources').textContent='训练占位恢复：'+(t.lease_result?.holders_restored?'已核实':'训练中 / 等待恢复验收')+'；评测状态：'+(t.eval_result?.status||t.eval_progress?.status||'等待权重验收');
  el('error').textContent='';el('updated').textContent='更新于 '+new Date().toLocaleString()+' · 每10秒刷新 · 只读接口 /api/status';
 }catch(e){el('error').textContent='状态读取失败：'+e+'。保留上次内容，10秒后重试。';}
}
refresh();setInterval(refresh,10000);
