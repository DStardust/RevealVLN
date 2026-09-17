"""Reuse the previous editable OOXML layout with versioned content and QA expectations.

The previous report/builder/images are read-only. Only this v2 directory receives
the new DOCX, copied approved figure assets and build receipts. This is not an
Office rendering engine; XML/ZIP checks do not certify actual Word pagination.
"""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'teacher_paper_story_20260911'
REVIEW = json.loads((HERE / 'FIGURE_REVIEW.json').read_text())
assert REVIEW['all_inserted_figures_personally_viewed']
assert (HERE.parent / 'TEACHER_INNOVATION_ADJUDICATION_20260911_V2_ZH.md').exists()
for name in ('new_family_case.png', 'diversity_progress.png'):
    source = OLD / 'figures' / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == REVIEW['figures'][name]['sha256']
    dest = HERE / 'figures' / name
    if dest.exists():
        assert dest.read_bytes() == source.read_bytes()
    else:
        shutil.copyfile(source, dest)

code = (OLD / 'build_docx.py').read_text()
changes = {
    "OUTPUT = HERE / '居家医疗养老机器人_导航论文思路_教师沟通版.docx'":
    "OUTPUT = HERE / '居家医疗养老机器人_论文思路_教师沟通修订版.docx'",
    "assert len(images)==2 and len(root.findall('.//w:drawing',ns))==2":
    "assert len(images)==3 and len(root.findall('.//w:drawing',ns))==3",
    "assert len(root.findall('.//w:br[@w:type=\"page\"]',ns)) == 5":
    "assert len(root.findall('.//w:br[@w:type=\"page\"]',ns)) == 7",
    "assert len(links) == 7": "assert len(links) == 11",
    "'explicit_page_sections': 6, 'editable_tables': 6, 'external_reference_links': 7":
    "'explicit_page_sections': 8, 'editable_tables': 6, 'external_reference_links': 11",
}
for before, after in changes.items():
    assert code.count(before) == 1, f'PREVIOUS_BUILDER_CHANGED: {before}'
    code = code.replace(before, after)

evidence = {
    'layout_builder': str(OLD / 'build_docx.py'),
    'layout_builder_sha256': hashlib.sha256((OLD / 'build_docx.py').read_bytes()).hexdigest(),
    'old_docx_sha256': hashlib.sha256((OLD / '居家医疗养老机器人_导航论文思路_教师沟通版.docx').read_bytes()).hexdigest(),
    'innovation_decision_sha256': hashlib.sha256((HERE.parent / 'TEACHER_INNOVATION_ADJUDICATION_20260911_V2_ZH.md').read_bytes()).hexdigest(),
    'source_image_review': str(OLD / 'SOURCE_IMAGE_REVIEW.json'),
    'draft_diagram_not_embedded': 'figures/memory_training_plan.png',
    'source_docx_untouched': True,
}
(HERE / 'BUILD_PROVENANCE.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
exec(compile(code, str(HERE / 'reused_ooxml_layout.py'), 'exec'),
     {'__file__': str(HERE / 'reused_ooxml_layout.py'), '__name__': '__main__'})
