"""Hash-bound V2 plan/optimizer audit with one registered LR intervention."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'ordinary_expanded_continue_v2/prepare.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='9869aa202905a5789bc9a16bedb574a6e71669086cc9fced073e54905d8e3920'
text=PARENT.read_text()
changes={
 "id='Q35N_ORDINARY_EXPANDED_CONTINUE_V2',created='2026-09-12'":"id='Q35N_ORDINARY_EXPANDED_LOW_LR_V3',created='2026-09-13'",
 "protocol['budget'].update(wall_seconds=4200,max_decisions=700712,max_updates=8000)":"protocol['optimizer']['learning_rate']=5e-6\n    protocol['matched_high_lr_protocol']=str(HERE.parent/'ordinary_expanded_continue_v2/PROTOCOL_FILESTORE.json')\n    protocol['budget'].update(wall_seconds=4200,max_decisions=700712,max_updates=8000)",
 "segment_note='Continue positions [4000,8000); no controller, special data, optimizer reset or automatic retry'":"segment_note='Same positions [4000,8000) as V2, only peak LR reduced tenfold; no controller or reset'",
 "    assert sum(next_counts)==len(after)":"    assert sum(next_counts)==len(after)\n    paired=read(HERE.parent/'ordinary_expanded_continue_v2/RESUME_AUDIT.json')\n    assert next_counts==paired['next_rank_decisions'] and exposure==paired['next_segment_by_source']\n    assert read(HERE.parent.parent.parent/'reviews/Q35N_ORDINARY_CONTINUE_V2/WORKFLOW_RESULT.json')['positive_development_signal'] is False",
 "HERE/'RESUME_AUDIT.json']":"HERE/'RESUME_AUDIT.json',PARENT,HERE.parent/'ordinary_expanded_continue_v2/PROTOCOL_FILESTORE.json']",
 "/cache/ordinary_expanded_continue_v2/":"/cache/ordinary_expanded_low_lr_v3/",
 "runtime/cache/ordinary_expanded_continue_v2":"runtime/cache/ordinary_expanded_low_lr_v3",
 "'expanded_continuation_train'":"'low_lr_matched_train'",
 "USER_20260912_CONTINUE_TOWARD_POSITIVE_OUTCOME_BOUNDED_STANDARD_BC":"USER_CONTINUE_TOWARD_POSITIVE_OUTCOME_SINGLE_FACTOR_LOW_LR",
}
for old,new in changes.items():
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
exec(compile(text,str(HERE/'prepare.py')+':frozen-parent','exec'),globals())
