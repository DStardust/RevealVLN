"""Small invariant tests for the v1r1 selection contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/collect_mf3zl_rcsp_v1r1.py"
    spec = importlib.util.spec_from_file_location("v1r1_selection", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_selection_is_sealed_and_train_only():
    module = _module()
    selection = module.build_selection()
    assert selection["status"] == "SEALED_COMPLETE_R2R_INSTRUCTION_VARIANT_POPULATION"
    assert selection["outcome_fields_used_for_selection"] == []
    assert selection["adaptive_stopping_allowed"] is False
    assert all(row["split"] == "train" for row in selection["routes"])


def test_selection_excludes_parent_canonical_episodes():
    module = _module()
    selection = module.build_selection()
    parent = module.json.loads(module.PARENT_SELECTION.read_text())
    parent_ids = {
        str(row["episode_id"])
        for row in parent["routes"]
        if row["dataset"] == "R2R"
    }
    assert not parent_ids & {str(row["episode_id"]) for row in selection["routes"]}


def test_selection_has_unique_episode_identity():
    module = _module()
    selection = module.build_selection()
    ids = [str(row["episode_id"]) for row in selection["routes"]]
    assert len(ids) == len(set(ids))
