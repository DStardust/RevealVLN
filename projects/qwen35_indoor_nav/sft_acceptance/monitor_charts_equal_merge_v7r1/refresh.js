const e=id=>document.getElementById(id),pct=x=>x==null?'—':(100*x).toFixed(2)+'%';
async function refresh(){try{
 const q=await fetch('/api/status',{cache:'no-store'});if(!q.ok)throw Error('HTTP '+q.status);const d=await q.json(),m=d.equal_merge;
 const f=m.fit_progress||{},v=m.dev_progress||{},g=m.fit_gate,r=m.result,done=r?.status==='COMPLETE'&&r.after;
 const names={EVALUATING_FIT:'正在做64条训练侧诊断',CHECKING_FIT_GATE:'核验训练侧门槛',ACTIVATING_PREDECLARED_DEV:'FIT通过，正在准入固定DEV',EVALUATING_DEV:'正在做100条开发集复测',COMPLETE_FIT_GATE_FAILED:'FIT门槛未通过：不启动DEV',COMPLETE:'完整试验已结束',FAILED_OR_BLOCKED:'运行异常，查看记录'};
 e('status').textContent=names[m.workflow?.status]||m.workflow?.status||'准备中';
 e('fitbar').value=f.completed||0;e('devbar').value=v.completed||0;
 e('fitcount').textContent=(f.completed||0)+' / 64条；'+(g?'新增成功'+g.wins+'条，失去'+g.losses+'条。':'等待完整结果与独立轨迹审计。');
 e('devcount').textContent=m.dev_admitted?(v.completed||0)+' / 100条；等待完整动作审计。':'尚未准入，不会因局部成绩自动跳过FIT门槛。';
 e('verdict').textContent=done?(r.positive_development_signal?'达到事前内部开发集正向门槛；尚非独立泛化证明。':'未达到正向门槛，继续保留最佳4k。'):(g&&!g.pass_gate?'本候选关闭，不换比例扫参；尚无新导航正向结果。':'当前不把FIT或局部路线结果当作正向。');
 const rows=[['DEV · 最佳4k',.21,.18016983923442284,.3658797007353029],['DEV · 纠错1000（未采用）',.19,.17527383450573364,.3933786870665479],['FIT · 最佳4k',.25,.21716978871779347,.456868587871994]];
 if(g)rows.push(['FIT · 固定0.5合并',g.after.sr,g.after.spl,g.after.ndtw]);else rows.push(['FIT · 合并待完成',null,null,null]);
 if(done)rows.push(['DEV · 固定0.5合并',r.after.result.sr,r.after.result.spl,r.after.result.ndtw]);else rows.push(['DEV · 合并未完成/未运行',null,null,null]);
 const body=e('results');body.replaceChildren();for(const row of rows){const tr=document.createElement('tr');row.forEach((val,i)=>{const td=document.createElement('td');td.textContent=i?pct(val):val;tr.appendChild(td);});body.appendChild(tr);}
 e('updated').textContent='更新于 '+new Date().toLocaleString()+' · 每10秒刷新 · 只读 /api/status';e('error').textContent='';
}catch(x){e('error').textContent='状态读取失败：'+x+'；保留上次内容并自动重试。';}}
refresh();setInterval(refresh,10000);
