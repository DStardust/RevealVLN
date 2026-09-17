"""Build an editable, dependency-free Word report from the accompanying source."""
from pathlib import Path
from xml.sax.saxutils import escape
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED
import re
import json
import hashlib

HERE = Path(__file__).resolve().parent
SOURCE = HERE / '论文思路_教师沟通版.md'
OUTPUT = HERE / '居家医疗养老机器人_导航论文思路_教师沟通版.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
links = []
images = []

def picture(alt, relative):
    path=(HERE/relative).resolve()
    assert path.is_relative_to(HERE/'figures') and path.suffix=='.png'
    data=path.read_bytes()
    assert data[:8]==b'\x89PNG\r\n\x1a\n'
    import struct
    width,height=struct.unpack('!II',data[16:24])
    cx=6121400;cy=round(cx*height/width)
    assert cy<7600000
    index=len(images)+1;rid=f'image{index}'
    images.append((rid,path))
    return ('<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/></w:pPr><w:r><w:drawing>'
       f'<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
       f'<wp:docPr id="{index}" name="{escape(path.name)}" descr="{escape(alt)}"/>'
       '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
       '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
       f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{index}" name="{escape(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
       f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
       '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')



def run(text, bold=False):
    prop = '<w:rPr><w:b/></w:rPr>' if bold else ''
    return f'<w:r>{prop}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def rich(text):
    result = []
    for piece in re.split(r'(https://[^\s]+)', text):
        if piece.startswith('https://'):
            rid = f'link{len(links)+1}'
            links.append((rid, piece))
            result.append(f'<w:hyperlink r:id="{rid}"><w:r><w:rPr><w:rStyle w:val="Hyperlink"/></w:rPr><w:t>{escape(piece)}</w:t></w:r></w:hyperlink>')
        else:
            result.append(run(piece))
    return ''.join(result)


def p(text, style='Normal', props=''):
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{props}</w:pPr>{rich(text)}</w:p>'


def table(rows):
    cols = len(rows[0])
    widths = [2800, 3420, 3420] if cols == 3 else [2650, 6990]
    assert sum(widths) == 9640 and cols == len(widths)
    border = ''.join(f'<w:{edge} w:val="single" w:sz="4" w:color="CEDAE4"/>' for edge in ['top','left','bottom','right','insideH','insideV'])
    out = ['<w:tbl><w:tblPr><w:tblW w:w="9640" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders>'+border+'</w:tblBorders><w:tblCellMar><w:top w:w="85" w:type="dxa"/><w:left w:w="110" w:type="dxa"/><w:bottom w:w="85" w:type="dxa"/><w:right w:w="110" w:type="dxa"/></w:tblCellMar></w:tblPr>']
    out.append('<w:tblGrid>'+''.join(f'<w:gridCol w:w="{x}"/>' for x in widths)+'</w:tblGrid>')
    for i, row in enumerate(rows):
        assert len(row) == cols
        out.append('<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i == 0 else '')+'</w:trPr>')
        for width, text in zip(widths, row):
            shade = 'E6EEF5' if i == 0 else ('F5F8FA' if i % 2 == 0 else 'FFFFFF')
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:shd w:fill="{shade}"/><w:vAlign w:val="center"/></w:tcPr>'+p(text,'TableHead' if i == 0 else 'TableText')+'</w:tc>')
        out.append('</w:tr>')
    out.append('</w:tbl>')
    return ''.join(out)


def build_body(source):
    lines = source.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- PAGEBREAK -->':
            out.append('<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr><w:r><w:br w:type="page"/></w:r></w:p>')
        elif line.startswith('!['):
            match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line);assert match
            out.append(picture(*match.groups()))
        elif line.startswith('图') and '：' in line[:12]:
            out.append(p(line,'Caption'))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', x) for x in row):
                    rows.append(row)
                i += 1
            out.append(table(rows))
            out.append(p('', 'TableGap'))
            continue
        elif line.startswith('### '):
            out.append(p(line[4:], 'Heading2'))
        elif line.startswith('## '):
            out.append(p(line[3:], 'Heading1'))
        elif line.startswith('# '):
            out.append(p(line[2:], 'Title'))
        elif line.startswith(('核心主张：', '研究问题：')):
            out.append(p(line,'Callout'))
        elif line.startswith('教师沟通稿'):
            out.append(p(line,'Subtitle'))
        elif line.startswith('[') or line.startswith('项目依据：'):
            out.append(p(line,'Reference'))
        else:
            out.append(p(line))
        i += 1
    return ''.join(out)


def style(sid, name, size, color='253444', bold=False, before=0, after=100, line=300, keep=False):
    return f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'+('<w:keepNext/>' if keep else '')+f'<w:widowControl/></w:pPr><w:rPr><w:color w:val="{color}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'+('<w:b/>' if bold else '')+'</w:rPr></w:style>'


def main():
    source = SOURCE.read_text(encoding='utf-8')
    body = build_body(source)
    styles = XML+f'<w:styles xmlns:w="{W}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="宋体"/><w:sz w:val="22"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="300" w:lineRule="auto"/><w:widowControl/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
    for args in [
        ('Title','Title',36,'183A56',True,0,150,280,True),
        ('Subtitle','Subtitle',20,'687C8D',False,0,140,280,True),
        ('Heading1','heading 1',29,'183A56',True,0,150,280,True),
        ('Heading2','heading 2',24,'245573',True,130,90,280,True),
        ('Callout','Callout',23,'183A56',True,40,160,300,False),
        ('TableText','Table Text',20,'253444',False,0,20,270,False),
        ('TableHead','Table Head',20,'183A56',True,0,20,270,False),
        ('TableGap','Table Gap',4,'253444',False,0,20,240,False),
        ('Reference','Reference',17,'687C8D',False,0,60,260,False),
        ('Caption','Caption',18,'687C8D',False,20,90,260,False),
        ('Header','Header',17,'687C8D',False,0,0,240,False),
    ]:
        styles += style(*args)
    styles += '<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/><w:rPr><w:color w:val="245573"/><w:u w:val="single"/></w:rPr></w:style></w:styles>'
    section = '<w:sectPr><w:headerReference w:type="default" r:id="header1"/><w:footerReference w:type="default" r:id="footer1"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1020" w:right="1133" w:bottom="1020" w:left="1133" w:header="450" w:footer="450"/><w:cols w:space="720"/></w:sectPr>'
    document = XML+f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{body}{section}</w:body></w:document>'
    header = XML+f'<w:hdr xmlns:w="{W}" xmlns:r="{R}">'+p('居家医疗养老机器人  |  导航论文研究思路','Header','<w:pBdr><w:bottom w:val="single" w:sz="4" w:color="CEDAE4"/></w:pBdr>')+'</w:hdr>'
    footer = XML+f'<w:ftr xmlns:w="{W}"><w:p><w:pPr><w:pStyle w:val="Header"/><w:jc w:val="right"/></w:pPr>'+run('教师沟通稿  ·  2026.09.11    |    ')+ '<w:fldSimple w:instr=" PAGE "><w:r><w:t>1</w:t></w:r></w:fldSimple></w:p></w:ftr>'
    relns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    rels = XML+f'<Relationships xmlns="{relns}">'
    for rid, kind, target in [('styles','styles','styles.xml'),('settings','settings','settings.xml'),('header1','header','header1.xml'),('footer1','footer','footer1.xml')]:
        rels += f'<Relationship Id="{rid}" Type="{R}/{kind}" Target="{target}"/>'
    for rid,path in images:
        rels += f'<Relationship Id="{rid}" Type="{R}/image" Target="media/{path.name}"/>'
    for rid, target in links:
        rels += f'<Relationship Id="{rid}" Type="{R}/hyperlink" Target="{escape(target)}" TargetMode="External"/>'
    rels += '</Relationships>'
    parts = {
        'word/document.xml': document,
        'word/styles.xml': styles,
        'word/settings.xml': XML+f'<w:settings xmlns:w="{W}"><w:zoom w:percent="100"/><w:defaultTabStop w:val="420"/><w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat></w:settings>',
        'word/header1.xml': header,
        'word/footer1.xml': footer,
        'word/_rels/document.xml.rels': rels,
        '_rels/.rels': XML+f'<Relationships xmlns="{relns}"><Relationship Id="rId1" Type="{R}/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>',
        'docProps/core.xml': XML+'<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>让导航记忆服务于任务执行</dc:title><dc:subject>面向居家医疗养老机器人的论文研究思路</dc:subject><dc:description>教师沟通稿；拟贡献与实际进展分别说明。</dc:description></cp:coreProperties>',
    }
    ct = XML+'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
    types = {'document':'document.main','styles':'styles','settings':'settings','header1':'header','footer1':'footer'}
    for name, kind in types.items():
        ct += f'<Override PartName="/word/{name}.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml"/>'
    ct += '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>'
    parts['[Content_Types].xml'] = ct
    for name, data in parts.items():
        ET.fromstring(data)
    with ZipFile(OUTPUT, 'x', ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data.encode('utf-8'))
        for _,path in images: z.writestr('word/media/'+path.name,path.read_bytes())
    with ZipFile(OUTPUT) as z:
        assert z.testzip() is None
        root = ET.fromstring(z.read('word/document.xml'))
        ns = {'w': W}
        text = ''.join(root.itertext())
        assert 'EgoCoT-Bench' in text and '核心主张' in text
        assert text.count('UAD') == 0
        qa=json.loads((HERE/'FIGURE_REVIEW.json').read_text())
        assert qa['all_inserted_figures_personally_viewed'] is True
        assert len(images)==2 and len(root.findall('.//w:drawing',ns))==2
        for _,path in images:
            assert qa['figures'][path.name]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert not any(x in text for x in ('待填','TODO','_PENDING','老师关心','故事包装','不能声称','不能靠'))
        assert len(root.findall('.//w:br[@w:type="page"]',ns)) == 5
        assert len(root.findall('.//w:tbl',ns)) == 6
        assert len(links) == 7
    report = {
        'file': str(OUTPUT), 'bytes': OUTPUT.stat().st_size,
        'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'zip_integrity': 'PASS', 'all_xml_parse': 'PASS',
        'explicit_page_sections': 6, 'editable_tables': 6, 'external_reference_links': 7,
        'embedded_figures':len(images),'figures_personally_viewed':True,
        'word_or_libreoffice_visual_render_performed': False,
        'note': 'No office renderer available; pagination designed with explicit breaks, not visually certified.',
    }
    (HERE/'BUILD_QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
