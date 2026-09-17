"""Read-only source inspection and lossless RGB export for report illustrations."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
IDS = [
    'WF_MULTI_f031322d1102e6b14ca31fa4',
    'WF_MULTI_ca389d2f76b686bd48756322',
    'WF_MULTI_d33c53f44d0119fe44eb54f6',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    old = load(LINE/'reports/opencode_navbench_fusion_20260911/LIVE_READONLY_SNAPSHOT.json')
    sources = {v['attempt_id']: v for v in old['special']['export_manifests']}
    out = HERE/'image_review'
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for aid in IDS:
        source = sources[aid]
        mp = Path(source['path'])
        assert sha(mp) == source['sha256']
        folder = mp.parent
        m = load(mp)
        content = load(folder/'CONTENT_INDEX.json')
        traces = {(h,c): load(folder/f'traces/t{hi}_{ci}.json')
                  for hi,h in enumerate(('H_A','H_B','H_A_I'))
                  for ci,c in enumerate(('C0','C_A','C_B'))}
        k = len(m['candidate']['histories']['H_A'])
        assert all(len(v)==k for v in m['candidate']['histories'].values())
        cert = load(folder.parent/'CERTIFICATE.json')
        assert cert['replays']==27 and cert['evaluations']==54
        assert all(t['complete'] and t['collisions']==0 for t in traces.values())
        # Same legal recent observations, physical pose and executed actions at boundary.
        ref = traces['H_A','C0']
        for h in ('H_B','H_A_I'):
            t = traces[h,'C0']
            for step in (k-1,k):
                assert t['observations'][step]['rgb_hash']==ref['observations'][step]['rgb_hash']
                assert t['observations'][step]['pose']==ref['observations'][step]['pose']
            assert t['actions'][k-8:k]==ref['actions'][k-8:k]
        # Observed event contrast before common boundary, not inferred from one image.
        assert any(o['see2']['anchor_A'] for o in ref['observations'][:k+1])
        assert not any(o['see2']['anchor_A'] for o in traces['H_B','C0']['observations'][:k+1])
        assert any(o['see2']['anchor_B'] for o in traces['H_B','C0']['observations'][:k+1])
        cells=[json.loads(s) for s in (folder/'SUPERVISION_ONLY.jsonl').read_text().splitlines()]
        selected_y={(r['task_id'],r['history_id'],r['continuation_id']):r['y'] for r in cells}
        assert selected_y['task_A','H_A','C0']==1
        assert selected_y['task_A','H_B','C0']==0
        assert selected_y['task_A','H_B','C_A']==1

        def array(o, kind):
            key=o[kind+'_hash']+'.'+kind
            info=content[key]
            path=LINE/info['line_relative_path']
            assert path.resolve()==path and path.is_relative_to(LINE)
            assert sha(path)==info['file_sha256']
            a=np.load(path,allow_pickle=False)
            assert hashlib.sha256(a.tobytes()).hexdigest()==info['raw_pixel_sha256']
            return a, path

        def best_pair(h,c,role,start,end):
            t=traces[h,c]; scores=[]
            for step in range(max(1,start),min(end,len(t['observations']))):
                o=t['observations'][step]; ids=o['see2'][role]
                if not ids: continue
                prev=t['observations'][step-1]
                score=max(min(int(prev['pixels'].get(str(i),0)),int(o['pixels'].get(str(i),0))) for i in ids)
                if score>=256:scores.append((score,step))
            assert scores,(aid,h,c,role)
            return max(scores)[1]

        a_step=best_pair('H_A','C0','anchor_A',1,k+1)
        b_step=best_pair('H_B','C0','anchor_B',1,k+1)
        goal_step=best_pair('H_A','C0','terminal',k+1,len(ref['observations']))
        frames=[]
        for label,h,c,step,role in [
            ('history_A_prev','H_A','C0',a_step-1,'anchor_A'),
            ('history_A_event','H_A','C0',a_step,'anchor_A'),
            ('history_B_prev','H_B','C0',b_step-1,'anchor_B'),
            ('history_B_event','H_B','C0',b_step,'anchor_B'),
            ('join_A','H_A','C0',k,None),
            ('join_B','H_B','C0',k,None),
            ('goal_prev','H_A','C0',goal_step-1,'terminal'),
            ('goal_event','H_A','C0',goal_step,'terminal'),
        ]:
            o=traces[h,c]['observations'][step]; rgb,path=array(o,'rgb');sem,spath=array(o,'semantic')
            assert rgb.shape==(224,224,3) and rgb.dtype==np.uint8
            vals,counts=np.unique(sem,return_counts=True)
            assert {str(int(v)):int(n) for v,n in zip(vals,counts)}==o['pixels']
            target_ids=m['compiler_config']['eligible'][role] if role else []
            visible=int(np.isin(sem,target_ids).sum()) if role else None
            if role: assert visible>=256
            png=out/f'{aid}_{label}.png'
            Image.fromarray(rgb).save(png)
            assert np.array_equal(np.asarray(Image.open(png)),rgb)
            frames.append(dict(label=label,history=h,continuation=c,step=step,role=role,
                png=str(png),png_sha256=sha(png),rgb_source=str(path),rgb_file_sha256=sha(path),
                raw_rgb_sha256=o['rgb_hash'],semantic_source=str(spath),semantic_file_sha256=sha(spath),
                target_ids=target_ids,target_pixels=visible,see2=o['see2'],pose=o['pose']))
        canvas=Image.new('RGB',(960,620),'white');draw=ImageDraw.Draw(canvas)
        draw.text((10,8),aid,fill='black')
        for i,f in enumerate(frames):
            x=10+(i%4)*238;y=35+(i//4)*285
            canvas.paste(Image.open(f['png']),(x,y))
            draw.text((x,y+228),f['label']+' step='+str(f['step']),fill='black')
            draw.text((x,y+242),'target pixels='+str(f['target_pixels']),fill='black')
        preview=out/f'{aid}_contact.png';canvas.save(preview)
        records.append(dict(attempt_id=aid,source=str(folder),manifest_sha256=sha(mp),
            certificate_sha256=sha(folder.parent/'CERTIFICATE.json'),house=source['house'],
            tasks=m['compiler_config']['tasks'],roles=m['compiler_config']['roles'],cutoff=k,
            short_window_exact=True,event_contrast_verified=True,frames=frames,
            preview=str(preview),selected_outcomes={str(k):v for k,v in selected_y.items()},
            note='image selection for explanation only, not representative performance sampling'))
    (out/'FRAME_AUDIT.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(json.dumps([{'id':r['attempt_id'],'preview':r['preview']} for r in records],indent=2))


if __name__=='__main__':
    main()
