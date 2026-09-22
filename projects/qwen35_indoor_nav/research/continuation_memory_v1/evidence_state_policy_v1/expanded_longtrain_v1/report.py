"""Export every registered checkpoint, never select a best result for reporting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
def main(run):
    lines=['EXPANDED 100000步开发学习曲线','','种子1209，从原1200步完整优化器/RNG断点续训98800步；Qwen best4k更新0。每5000步固定评测；旧对照不被本任务覆盖。','',
        '|步数|实际完成/计划|主任务PASS/128|主任务识别界|task_T PASS/128|','|---:|---:|---:|---|---:|']
    for step in read(run/'PROTOCOL.json')['checkpoint_steps']:
        p=run/'evaluations'/f'STEP_{step:06d}'/'RESULT.json'
        if not p.exists():lines.append(f'|{step}|待完成|—|—|—|');continue
        r=read(p);m=next(x for x in r['metrics'] if x['endpoint']=='main' and x['house'] is None);t=next(x for x in r['metrics'] if x['endpoint']=='control' and x['house'] is None)
        lines.append(f"|{step}|{r['complete']}/{r['planned']}|{m['passed']}/128|[{m['lower']:.2%},{m['upper']:.2%}]|{t['passed']}/128|")
    lines+=['','缺失槽位不减分母；识别界不是置信区间。20个checkpoint反复评测在同一已暴露四屋，只作为开发学习曲线。','单种子长训不构成同预算OLD对照、盲测泛化、普通VLN-CE收益或新算法贡献；不自动采用。']
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main(Path(sys.argv[1]))
