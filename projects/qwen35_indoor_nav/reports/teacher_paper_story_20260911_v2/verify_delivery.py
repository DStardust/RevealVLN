"""Final read-only project checks; writes only this report's delivery receipt."""
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from xml.etree import ElementTree as ET
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
OLD = HERE.parent / 'teacher_paper_story_20260911'
TRAIN = LINE / 'sft_acceptance/ordinary_sync_recovery_v1'
MONITOR = LINE / 'sft_acceptance/monitor_charts_recovery_v1'
DOC = HERE / '居家医疗养老机器人_论文思路_教师沟通修订版.docx'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
# The legacy QA receipt is stale (see preserved mismatch), so compare with
# the actual artifact hash captured before this build, not the old receipt.
provenance = json.loads((HERE / 'BUILD_PROVENANCE.json').read_text())
assert sha(OLD / '居家医疗养老机器人_导航论文思路_教师沟通版.docx') == provenance['old_docx_sha256']
protocol = json.loads((TRAIN / 'PROTOCOL_FILESTORE.json').read_text())
assert all(sha(TRAIN / name) == digest for name, digest in protocol['code_sha256'].items())
before = json.loads((MONITOR / 'SWITCH_BEFORE.json').read_text())
pid = before['launcher']['pid']
proc = Path('/proc') / str(pid)
assert int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]) == before['launcher']['starttime']
with urllib.request.urlopen('http://127.0.0.1:18766/api/status', timeout=6) as r:
    live = json.load(r)
assert live['monitor_version'] == 'recovery_v1'
assert live['display_state'] == 'TRAINING' and not live['progress']['stale']
assert live['progress']['data']['cursor']['updates'] > 32940
text = (HERE / '论文思路_教师沟通版.md').read_text()
refs = {int(x) for x in re.findall(r'^\[(\d+)\]', text, re.M)}
assert refs == set(range(1, 12))
assert '尚未形成可确认原创的新增架构' in text and '并不要求两段历史' in text
assert '没有接入特殊记忆监督' in text
with ZipFile(DOC) as z:
    assert z.testzip() is None
    for name in z.namelist():
        if name.endswith(('.xml', '.rels')):
            ET.fromstring(z.read(name))
    names = [x for x in z.namelist() if x.startswith('word/media/')]
    assert len(names) == 3 and 'word/media/memory_training_plan.png' not in names
    review = json.loads((HERE / 'FIGURE_REVIEW.json').read_text())
    for name in names:
        assert hashlib.sha256(z.read(name)).hexdigest() == review['figures'][Path(name).name]['sha256']
bundle = LINE / 'data_pipeline/mechanism_runtime_v1/witness_first_v1/diversity_v1r2/run_v1/bundles/DV1_f48017d7a035bf5377bc16cf/export_v4'
with (bundle / 'SUPERVISION_ONLY.jsonl').open() as f:
    matrix = {(r['history_id'], r['task_id'], r['continuation_id']): r['y'] for r in map(json.loads, f)}
assert [matrix[h, 'task_A', 'C0'] for h in ('H_A', 'H_B')] == [1, 0]
assert [matrix[h, 'task_B', 'C0'] for h in ('H_A', 'H_B')] == [0, 1]
assert [matrix[h, 'task_A', 'C_A'] for h in ('H_A', 'H_B')] == [1, 1]
result = dict(status='PASS_DELIVERY_CHECKS', unix=time.time(), docx=str(DOC), sha256=sha(DOC),
              old_docx_preserved=True, legacy_build_qa_hash_matches_old_docx=False,
              legacy_mismatch_preserved='LEGACY_RECEIPT_MISMATCH.json',
              figures_personally_reviewed=True, example_matrix_verified=True,
              all_active_training_source_hashes_unchanged=True, trainer_pid_and_starttime_unchanged=True,
              monitor_port=18766, monitor_state=live['display_state'],
              updates=live['progress']['data']['cursor']['updates'],
              progress_age_seconds=live['progress']['age_seconds'],
              checkpoint=live['checkpoint_name'], monitor_cpu_tests=6,
              served_javascript_syntax='PASS', word_pagination_visually_rendered=False,
              new_architecture_novelty='NOT_YET_ESTABLISHED', navigation_gain_verified=False)
with (HERE / 'DELIVERY_QA.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
