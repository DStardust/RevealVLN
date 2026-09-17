"""New preworker-only bridge: exact approval object, original runtime unchanged.

The first queue failed before lane creation: metadata fields in the main review
object violated the original exact-dictionary schema. Preserve that file and
failure. Only approval-file location changes; all source/resource checks remain.
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parent
ROOT=RUNTIME.parents[4]
FIRST=ROOT/'projects/qwen35_indoor_nav/data_pipeline/auto_production_v1/ordinary_queue_v1/lane_gpu_6'


def validate_failure(result,log,lane_exists,expected,approval,old_approval):
    assert result['error']=="RuntimeError('PRODUCTION_NONZERO_NO_RETRY:1')"
    assert 'AssertionError: MAIN_APPROVAL_REQUIRED' in log
    assert not lane_exists,'FIRST_ATTEMPT_MUST_BE_PREWORKER_ONLY'
    assert approval==expected,'EXACT_NEW_APPROVAL_REQUIRED'
    assert all(old_approval[k]==v for k,v in expected.items())
    assert set(old_approval)-set(expected)=={'main_agent_review','training_allowed','scientific_pass'}
    assert old_approval['training_allowed'] is False and old_approval['scientific_pass'] is False


def execute():
    sys.path.insert(0,str(RUNTIME))
    import common as c
    old=c.read(RUNTIME/'MAIN_AGENT_APPROVAL_GPU6.json')
    approval=c.read(HERE/'APPROVAL.json')
    validate_failure(c.read(FIRST/'RESULT.json'),
        (FIRST/'ordinary_gpu6_unattempted_1424_v1_production.log').read_text(),
        (RUNTIME/'lanes').exists(),c.approval_value(6),approval,old)
    c.immutable_verify()
    def approved(gpu):
        assert gpu==6
        lock=c.immutable_verify()
        assert c.read(HERE/'APPROVAL.json')==c.approval_value(gpu)
        return lock
    c.approved=approved
    spec=importlib.util.spec_from_file_location('original_ordinary_approved_run',RUNTIME/'run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.execute(6)


if __name__=='__main__':execute()
