const el=id=>document.getElementById(id);
const pct=x=>x==null?'未测':(100*x).toFixed(2)+'%';
const num=x=>Number(x).toLocaleString('zh-CN');
function rows(id, values){
  el(id).replaceChildren(...values.map(row=>{
    const tr=document.createElement('tr');
    row.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.appendChild(td);});
    return tr;
  }));
}
async function refresh(){
  try{
    const response=await fetch('/api/status',{cache:'no-store'});
    if(!response.ok)throw Error('HTTP '+response.status);
    const d=await response.json(),g=d.sr40;
    if(d.monitor_version!=='ordinary_sr40_v1'||!g)throw Error('SR40状态版本不匹配');
    el('phase').textContent=g.stage==='BASELINE_RESET_PREPARATION'?'基座配方准备中 · 新训练尚未启动':g.stage;
    el('count').textContent='现有普通数据：'+num(g.snapshot.strict_routes)+'条物理路线 / '+num(g.snapshot.instruction_records)+'条指令 / '+num(g.snapshot.instruction_conditioned_decisions)+'次指令条件决策；'+g.fit_houses+'个FIT房屋。';
    el('next').textContent='下一步：固定普通历史输入/适配训练对照，先通过执行一致性和资源退出测试，再启动有明确预算的训练。';
    el('snapshot').textContent='计划审核快照：'+new Date(g.unix*1000).toLocaleString()+'；本节点GPU启动 '+g.new_gpu_launches+'、训练更新 '+g.new_training_updates+'。这不是后台训练进度。';
    rows('official',[
      ['历史完整评测：41800步',pct(g.historical_full.sr),pct(g.historical_full.spl),'旧权重；batch '+g.historical_full.selected_batch_size],
      ['SR40新阶段',pct(d.sr40_full_result?.sr),pct(d.sr40_full_result?.spl),'未开始完整评测'],
      ['阶段A验收目标','≥40%','完整报告','至少 '+g.minimum_successes_for_40+' / '+g.official_episode_count+' 成功'],
      ['阶段C研究目标','约60%','完整报告','尚未执行，不承诺必达']
    ]);
    el('gate').textContent=g.pass_goal?'达到阶段A登记门槛，仍须查看完整审核。':'阶段A尚未达标。内部100条或局部诊断的提升，不计为完整基准达标。';
    const b=g.best_internal,f=g.full_epoch_internal,v=d.onpolicy_training.result.after.result;
    rows('internal',[
      ['保留最佳：扩产4k',pct(b.sr),pct(b.spl),pct(b.ndtw)],
      ['完整一轮数据（未采用）',pct(f.sr),pct(f.spl),pct(f.ndtw)],
      ['最近路点教师V11（未采用）',pct(v.sr),pct(v.spl),pct(v.ndtw)]
    ]);
    const devices=d.gpu?.devices||[];
    el('resource').textContent=d.gpu?.error?'资源读取失败：'+d.gpu.error:devices.map(x=>'GPU'+x.index+'：'+num(x.used_mib)+' MiB / 利用率 '+x.utilization_percent+'%').join('；');
    el('updated').textContent='网页与资源读取：'+new Date().toLocaleString()+'；每10秒刷新。';
    el('error').textContent='';
  }catch(error){
    el('error').textContent='状态读取失败：'+error.message+'。下方保留上次显示，不代表实时状态；请核对更新时间。';
  }
}
refresh();setInterval(refresh,10000);

