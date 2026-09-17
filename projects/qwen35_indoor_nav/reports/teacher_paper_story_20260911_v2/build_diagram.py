"""Code-native information-flow schematic. No scene generation or photo editing."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = HERE.parent / 'teacher_paper_story_20260911/assets/NotoSansCJKsc-Regular.otf'
OUT = HERE / 'figures/memory_training_plan_v2.png'
INK, MUTED, BLUE = '#183a56', '#526779', '#e9f0f6'
img = Image.new('RGB', (1320, 590), 'white')
d = ImageDraw.Draw(img)


def txt(x, y, s, size=26, fill=INK):
    d.text((x, y), s, font=ImageFont.truetype(str(FONT), size), fill=fill)


def box(coords, title, lines, fill=BLUE):
    x, y, r, b = coords
    d.rounded_rectangle(coords, radius=14, fill=fill)
    txt(x+20, y+18, title, 30)
    for i, line in enumerate(lines):
        txt(x+20, y+68+i*37, line, 24, MUTED)


def arrow(points):
    d.line(points, fill=INK, width=4)
    x, y = points[-1]
    px, py = points[-2]
    if x == px:
        d.polygon([(x,y),(x-9,y-14),(x+9,y-14)], fill=INK)
    else:
        d.polygon([(x,y),(x-14,y-9),(x-14,y+9)], fill=INK)


txt(25, 12, '同一份导航记忆，接受动作与执行结果两种监督', 34)
txt(25, 64, '拟实现的联合训练方案 · 运行期决策与训练期监督分开设计', 24, MUTED)
box((25,120,355,292), '因果输入', ['指令、当前及过去的画面', '已执行动作、上一步记忆'])
box((455,120,785,292), '更新运行期记忆', ['保留与当前任务有关的经历', '这份记忆用于实际动作'])
box((885,120,1295,292), '动作读出', ['结合当前观察选择导航动作', '训练时使用合法动作示范'])
arrow([(355,205),(455,205)])
arrow([(785,205),(885,205)])
d.rounded_rectangle((425,327,1295,528), radius=14, outline='#bf9571', width=3)
txt(442,333,'仅训练：查询与标签不输入上排',22,'#966139')
box((455,382,785,499), '后续路线查询', ['例：看终点椅子后停止'], '#faf1e8')
box((885,382,1265,499), '任务结果读出', ['真实交叉执行结果监督'], '#faf1e8')
arrow([(785,443),(885,443)])
arrow([(820,205),(820,365),(1075,365),(1075,382)])
txt(25,546,'部署时保留上排；移除结果读出器。下排标签与未来查询不作为导航输入。',26)
OUT.parent.mkdir(exist_ok=True)
assert not OUT.exists()
img.save(OUT)
print(OUT)
