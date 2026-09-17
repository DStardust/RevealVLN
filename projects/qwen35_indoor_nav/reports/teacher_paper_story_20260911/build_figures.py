"""Typeset verified simulator PNGs without synthesizing or retouching scene pixels."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import hashlib
import json

HERE=Path(__file__).resolve().parent
FONT=HERE/'assets/NotoSansCJKsc-Regular.otf'
INK='#183a56';MUTED='#526779';BLUE='#e9f0f6'
def font(n):return ImageFont.truetype(str(FONT),n)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def wrapped(draw,text,xy,width,size=24,fill=INK,spacing=6):
    x,y=xy;line=''
    for ch in text:
        if ch=='\n' or (line and draw.textlength(line+ch,font=font(size))>width):
            draw.text((x,y),line,font=font(size),fill=fill);y+=size+spacing;line=''
        if ch!='\n':line+=ch
    if line:draw.text((x,y),line,font=font(size),fill=fill);y+=size+spacing
    return y

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',required=True)
    args=parser.parse_args()
    counts=read(HERE/'DIVERSITY_COUNTS.json');assert counts['small_batch_acceptance']
    records=read(HERE/'new_image_review/FRAME_AUDIT.json')
    row=next(r for r in records if r['attempt_id']==args.family)
    approval=read(HERE/'SOURCE_IMAGE_REVIEW.json')
    assert approval['selected_family']==args.family and approval['selected_source_frames_personally_viewed']
    assert approval['semantic_visual_identity_consistent'] and approval['fresh_physical_data_only']
    figs=HERE/'figures';figs.mkdir(exist_ok=False)
    pic=Image.new('RGB',(1000,734),'white');draw=ImageDraw.Draw(pic)
    draw.text((22,10),'相同近期画面，不同任务历史',font=font(32),fill=INK)
    draw.text((22,55),'新生成并审核的真实仿真回放 · 非模型自主导航结果',font=font(21),fill=MUTED)
    labels=['前置椅子（1/2）','前置椅子（2/2）','对照桌子（1/2）','对照桌子（2/2）',
            '共同决策点：甲','共同决策点：乙','终点事件：前一帧','终点事件：当前帧']
    for i,(frame,label) in enumerate(zip(row['frames'],labels)):
        x=22+(i%4)*246;y=100+(i//4)*300
        path=Path(frame['png']);assert sha(path)==frame['png_sha256']
        rgb=Image.open(path).convert('RGB');assert rgb.size==(224,224)
        # Native RGB underlay is unchanged; only explicitly labelled semantic boxes overlay it.
        pic.paste(rgb,(x,y))
        if frame['target_bbox']:
            a,b,c,e=frame['target_bbox']
            draw.rectangle((x+max(0,a-2),y+max(0,b-2),x+min(223,c+2),y+min(223,e+2)),
                           outline='#EF8C25',width=2)
        draw.text((x,y+230),label,font=font(21),fill=INK)
        draw.text((x,y+258),'回放第 '+str(frame['step'])+' 步',font=font(18),fill=MUTED)
    draw.text((22,704),'橙框定位语义目标（非模型预测）；共同点两幅图像相同',font=font(21),fill=MUTED)
    pic.save(figs/'new_family_case.png')
    chart=Image.new('RGB',(1200,550),'white');draw=ImageDraw.Draw(chart)
    draw.text((28,12),'本轮特殊数据调整：进展与覆盖范围',font=font(34),fill=INK)
    draw.text((28,64),'对预先固定的6个候选逐一核验；保留所有拒绝与中断记录',font=font(24),fill=MUTED)
    cards=[
      ('可核验的语言表达','5 种已实现',str(counts['observed_new_language_variants'])+'种已用于本轮通过样本；改写不增加物理样本数'),
      ('真实移动公共路段',str(counts['accepted_variants'])+' / 6 组通过','八次前进约两米；每组27次重复回放认证'),
      ('房屋覆盖',str(counts['accepted_houses'])+' 个既有房屋','本轮新增房屋为0；扩展的是旧房屋中的路径'),
      ('任务程序','1 种已物理核验','保留有顺序的连续可见事件\n自然到访任务尚待验证'),
    ]
    for i,(title,value,note) in enumerate(cards):
        x=28+(i%2)*584;y=113+(i//2)*205
        draw.rounded_rectangle((x,y,x+560,y+186),radius=15,fill=BLUE)
        draw.text((x+22,y+12),title,font=font(25),fill=MUTED)
        draw.text((x+22,y+49),value,font=font(39),fill=INK)
        wrapped(draw,note,(x+22,y+111),514,23,MUTED)
    chart.save(figs/'diversity_progress.png')
    data=dict(source_family=args.family,source_frames_audit_sha256=sha(HERE/'new_image_review/FRAME_AUDIT.json'),
              counts_sha256=sha(HERE/'DIVERSITY_COUNTS.json'),
              original_sensor_png_files_unchanged=True,
              scene_synthesis_or_retouching=False,semantic_bbox_annotation_overlay=True,
              images={p.name:dict(path=str(p),sha256=sha(p)) for p in figs.iterdir()})
    with (HERE/'FIGURE_BUILD.json').open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print(json.dumps(data,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
