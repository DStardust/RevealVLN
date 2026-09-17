"""Exact V6 transport, prospective GPU6 never-attempted 426+998 subset."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_V6=HERE.parent/'runtime_v6'
assert hashlib.sha256((SOURCE_V6/'INPUT_LOCK.json').read_bytes()).hexdigest()=='764abaf82d6b05876fe1abca9569a1d488e632252882e4564213332a2118abcd'
locked=json.loads((SOURCE_V6/'INPUT_LOCK.json').read_text())
raw=(SOURCE_V6/'transport.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()==locked[str((SOURCE_V6/'transport.py').relative_to(HERE.parents[4]))]
adapted=raw.decode()


def change(old,new,count=1):
    global adapted
    assert adapted.count(old)==count,('EXACT_AUTO_TRANSPORT_COUNT',old,adapted.count(old),count)
    adapted=adapted.replace(old,new)


change('GPU4 prospective rescue of 953','GPU6 prospective never-attempted continuation of 1424')
change("RESCUE=HERE.parent/'salvage_rescue_v1/run_v1'","RESCUE=HERE/'source_v1'")
change('d68f813bf7d21f3d013acc81116f21f0774d1d15b1bba09f012639e3f0ef554a','b025336fd264aed02606a6cd5b9adfa9a250b3231ba51572365f6efc2edb8af7')
change("{'4':[0,1]}","{'6':[2,4]}",2)
change('GPU4_RESCUE_0_1_ONLY','GPU6_UNATTEMPTED_2_4_ONLY')
change('len(rows)==953','len(rows)==1424')
change('((0,279),(1,674))','((2,426),(4,998))')
change("ledger=(root/'LEDGER.jsonl').read_bytes();assert ledger.endswith(b'\\n')",
       "ledger=(root/'LEDGER.jsonl').read_bytes() if (root/'LEDGER.jsonl').exists() else b'';assert not ledger or ledger.endswith(b'\\n')")
change("directories={p.name for p in (root/'routes').iterdir()}",
       "directories={p.name for p in (root/'routes').iterdir()} if (root/'routes').exists() else set()")
change('for s in (0,1) for j in result[s]','for s in (2,4) for j in result[s]')
change("return PARALLEL/'rescue_production'/f'shard_{shard:04d}'","return HERE/'production'/f'shard_{shard:04d}'")
change('for gpu in (3,4):','for gpu in (6,):',2)
change('runtime_v2/lanes','runtime_v3/lanes',3)
change("'choices=[4]'","'choices=[6]'")
change('default=\'{\\"4\\":[0,1]}\'','default=\'{\\"6\\":[2,4]}\'')
change('Q35N_ORDINARY_UNATTEMPTED_RESCUE_RUNTIME_V6','Q35N_ORDINARY_AUTO_GENERATION_UNATTEMPTED_V1')
change('lanes=={4:(0,1)}','lanes=={6:(2,4)}')
change("HERE.parent/'rescue_production'","HERE/'production'",2)
change("{'0':279,'1':674}","{'2':426,'4':998}")
change(" '953')"," '1424')")
change("('RESCUE_JOBS.json','RESULT.json','INPUT_HASHES.json','INTERRUPTED.json','QUARANTINE.json','RESTORATION_EVIDENCE.json','TRAINING_INDEX.jsonl')",
       "('RESCUE_JOBS.json','PLAN.json','INPUT_LOCK.json')")
change('((0,3),(1,4))','((2,6),(4,6))')
change("for name in ('JOBS.json','LEDGER.jsonl','INPUT_LOCK.json')]",
       "for name in ('JOBS.json','LEDGER.jsonl','INPUT_LOCK.json') if (c.PARALLEL/'production'/f'shard_{oldshard:04d}'/name).exists()]")
exec(compile(adapted,str(__file__),'exec'),globals())
_rescued_prepare_source=prepare_source


def prepare_source():
    source=_rescued_prepare_source()
    source=exact(source,'    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})',
        "    paths += [transport.SOURCE_V6/'transport.py',transport.SOURCE_V6/'INPUT_LOCK.json']\n    for path,h in c.read(transport.RESCUE/'INPUT_LOCK.json').items():assert c.sha(c.ROOT/path)==h,'FROZEN_AUTO_SOURCE_CHANGED'\n    lock.update(c.read(transport.RESCUE/'INPUT_LOCK.json'))\n    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})")
    return source
