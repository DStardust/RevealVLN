import unittest

from revealnav_mf3.mf3zp_reference import (
    ReferenceContractError,
    balanced_agreement_audit_sample,
    branch_aliases,
    build_annotation_contract,
    derive_semantic_state,
    derive_uad,
    disagreement_event_ids,
    resolvable,
    reveal_interval,
    validate_annotation_response,
)


def response(event_id="event", step=1, aliases=("B01", "B02")):
    return {
        "schema_version": "revealnav-mf3zp-semantic-reference/1",
        "event_id": event_id,
        "prefix_step": step,
        "visible_candidate_aliases": list(aliases),
        "indistinguishable_alias_groups": [],
        "candidates_visually_distinguishable": True,
        "instruction_uniquely_selects_one": True,
        "selected_candidate_alias": aliases[1],
        "decisive_instruction_spans": ["turn left at the table"],
        "decisive_frame_steps": [step],
        "future_evidence_required": False,
        "rationale": "Both exits are visible and the instruction selects B02.",
    }


class MF3ZPReferenceTest(unittest.TestCase):
    def test_aliases_do_not_encode_input_order(self):
        forward = branch_aliases("event", ("native", "runner", "other"))
        reverse = branch_aliases("event", ("other", "runner", "native"))
        self.assertEqual(forward, reverse)
        self.assertEqual(set(forward.values()), {"B01", "B02", "B03"})

    def test_causal_contract_rejects_future(self):
        with self.assertRaises(ReferenceContractError):
            build_annotation_contract(
                event_id="event",
                prefix_step=1,
                instruction="go left",
                chronological_frames=[
                    {"step": 0, "frame_id": "P0"},
                    {"step": 2, "frame_id": "P2"},
                ],
                current_candidates=[{"alias": "B01"}],
            )

    def test_causal_contract_rejects_outcome(self):
        with self.assertRaises(ReferenceContractError):
            build_annotation_contract(
                event_id="event",
                prefix_step=0,
                instruction="go left",
                chronological_frames=[{
                    "step": 0,
                    "frame_id": "P0",
                    "delta_utility": 1.0,
                }],
                current_candidates=[{"alias": "B01"}],
            )

    def test_response_validation_and_projection(self):
        value = response()
        self.assertEqual(
            validate_annotation_response(
                value,
                event_id="event",
                prefix_step=1,
                allowed_aliases=("B01", "B02"),
            ),
            (),
        )
        state = derive_semantic_state(
            value,
            target_alias="B02",
            native_alias="B01",
        )
        self.assertEqual(state, {
            "target_in_set": True,
            "candidate_separated": True,
            "evidence_closed": True,
        })

    def test_projection_requires_target_and_native_visibility(self):
        value = response()
        self.assertFalse(derive_semantic_state(
            value,
            target_alias=None,
            native_alias="B01",
        )["evidence_closed"])
        value["visible_candidate_aliases"] = ["B02"]
        self.assertFalse(derive_semantic_state(
            value,
            target_alias="B02",
            native_alias="B01",
        )["candidate_separated"])

    def test_projection_uses_deterministic_target_presence(self):
        value = response()
        value["visible_candidate_aliases"] = ["B01"]
        self.assertEqual(
            derive_semantic_state(
                value,
                target_alias="B02",
                native_alias="B01",
                target_present=False,
            ),
            {
                "target_in_set": False,
                "candidate_separated": False,
                "evidence_closed": False,
            },
        )

    def test_uad_is_fixed_k3_derivation(self):
        states = derive_uad(
            [False, True, True, True, True],
            [False, True, True, True, False],
            [False, True, True, True, False],
        )
        self.assertEqual(states, ("U", "A", "A", "D", "A"))
        self.assertEqual(reveal_interval(states), (1, 3))
        self.assertTrue(resolvable((1, 3), 3))
        self.assertFalse(resolvable((1, 3), 2))

    def test_disagreement_is_event_level(self):
        left = {"e": [{
            "prefix_step": 0,
            "candidate_separated": True,
            "evidence_closed": False,
        }]}
        right = {"e": [{
            "prefix_step": 0,
            "candidate_separated": False,
            "evidence_closed": False,
        }]}
        self.assertEqual(disagreement_event_ids(left, right), ("e",))

    def test_balanced_audit_sample_excludes_disagreements(self):
        rows = [
            {
                "event_id": f"{domain}-{scene}-{index}",
                "dataset": domain,
                "scene_id": scene,
            }
            for domain in ("R2R", "RxR")
            for scene in ("s1", "s2", "s3")
            for index in range(10)
        ]
        first = balanced_agreement_audit_sample(
            rows,
            {"R2R-s1-0"},
            sample_events=40,
        )
        second = balanced_agreement_audit_sample(
            list(reversed(rows)),
            {"R2R-s1-0"},
            sample_events=40,
        )
        self.assertEqual(first, second)
        self.assertNotIn("R2R-s1-0", first)


if __name__ == "__main__":
    unittest.main()
