"""Fail-closed invariants for the MF3ZN temporal exact action lattice."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.temporal_exact_lattice import (
    CausalActionSnapshot,
    LatticeArmFold,
    LatticeArmIdentity,
    canonical_prefix_sha256,
    seal_action_lattice,
    validate_exact_lattice_treatment,
    validate_lattice_fold_integrity,
)


def physical_trace(target_action: str = "native") -> list[dict]:
    return [
        {
            "act": 4,
            "ghost_vp": "prefix-vp",
            "cur_vp": "start",
            "front_vp": "front-0",
            "back_path_len": 0,
            "reward": 99.0,
        },
        {
            "act": 4,
            "ghost_vp": target_action,
            "cur_vp": "prefix-vp",
            "front_vp": "front-1",
            "back_path_len": 1,
            "reward": -99.0,
        },
        {
            "act": 4,
            "ghost_vp": "future-vp",
            "cur_vp": target_action,
            "front_vp": "front-2",
            "back_path_len": 2,
        },
    ]


def snapshot(
    *,
    dataset: str = "RxR",
    scene: str = "scene-a",
    episode: str = "episode-a",
    scores: tuple[float, ...] = (0.0, 0.95, 0.8, 0.8, 100.0),
    executable: tuple[int, ...] = (1, 2, 3),
) -> CausalActionSnapshot:
    return CausalActionSnapshot(
        dataset=dataset,
        scene_id=scene,
        episode_id=episode,
        decision_step=1,
        native_action_id="native",
        global_action_ids=("STOP", "native", "alt-b", "alt-a", "not-live"),
        executable_action_indices=executable,
        policy_scores=scores,
        native_prefix_sha256=canonical_prefix_sha256(physical_trace(), 1),
    )


def arm(event, action_id: str) -> LatticeArmIdentity:
    return LatticeArmIdentity(
        dataset=event.dataset,
        scene_id=event.scene_id,
        episode_id=event.episode_id,
        decision_step=event.decision_step,
        native_prefix_sha256=event.native_prefix_sha256,
        action_id=action_id,
    )


def decision_trace(adapted: str, *, changed: bool) -> list[dict]:
    index = {"native": 1, "alt-b": 2, "alt-a": 3}[adapted]
    return [
        {
            "step": 0,
            "native_action_index": 1,
            "adapted_action_index": 1,
            "native_action_id": "native",
            "adapted_action_id": "native",
            "action_changed": False,
        },
        {
            "step": 1,
            "native_action_index": 1,
            "adapted_action_index": index,
            "native_action_id": "native",
            "adapted_action_id": adapted,
            "action_changed": changed,
        },
        {
            "step": 2,
            "native_action_index": 1,
            "adapted_action_index": 1,
            "native_action_id": "native",
            "adapted_action_id": "native",
            "action_changed": False,
        },
    ]


class ExactActionLatticeTest(unittest.TestCase):
    def test_seals_fixed_top_two_executable_non_native_with_id_tiebreak(self):
        event = seal_action_lattice([snapshot()]).events[0]
        self.assertEqual(event.ranked_non_native_action_ids, ("alt-a", "alt-b"))
        self.assertEqual(event.alternative_action_ids, ("alt-a", "alt-b"))
        self.assertNotIn("not-live", event.frozen_candidate_action_ids)
        self.assertEqual(event.action_ids, ("native", "alt-a", "alt-b"))

        one = snapshot(executable=(1, 2))
        self.assertEqual(
            seal_action_lattice([one]).events[0].alternative_action_ids,
            ("alt-b",),
        )

    def test_seal_is_outcome_blind_canonical_and_content_committed(self):
        first = snapshot(episode="episode-a")
        second = snapshot(episode="episode-b")
        forward = seal_action_lattice([first, second])
        reverse = seal_action_lattice([second, first])
        self.assertEqual(
            forward.action_list_commitment_sha256,
            reverse.action_list_commitment_sha256,
        )
        manifest = forward.as_manifest()
        self.assertEqual(manifest["status"], "SEALED_BEFORE_TREATMENT_OUTCOMES")
        self.assertEqual(manifest["outcome_fields_used_for_selection"], [])
        self.assertIs(manifest["treatment_results_read"], False)
        self.assertIs(manifest["adaptive_collection_allowed"], False)

        # A causal score change is committed even when it leaves the top two
        # action IDs unchanged.
        changed = snapshot(
            episode="episode-a",
            scores=(0.0, 0.95, 0.8, 0.8, -10.0),
        )
        self.assertNotEqual(
            seal_action_lattice([first]).action_list_commitment_sha256,
            seal_action_lattice([changed]).action_list_commitment_sha256,
        )

    def test_strict_snapshot_mapping_refuses_outcomes_or_unknown_fields(self):
        base = {
            "dataset": "RxR",
            "scene_id": "scene-a",
            "episode_id": "episode-a",
            "decision_step": 1,
            "native_action_id": "native",
            "global_action_ids": ["STOP", "native", "alt-b"],
            "executable_action_indices": [1, 2],
            "policy_scores": [0.0, 0.9, 0.8],
            "native_prefix_sha256": canonical_prefix_sha256(
                physical_trace(), 1
            ),
        }
        seal_action_lattice([base])
        for key, value in (
            ("delta_utility", 0.2),
            ("catastrophic", False),
            ("future_candidate_ids", ["leak"]),
            ("oracle_uad", "D"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                seal_action_lattice([{**base, key: value}])
        with self.assertRaises(ValueError):
            seal_action_lattice([{**base, "unregistered_metadata": "x"}])

    def test_snapshot_rejects_bad_action_identity_or_support(self):
        with self.assertRaises(ValueError):
            replace(snapshot(), global_action_ids=(
                "STOP", "native", "alt-b", "alt-b", "not-live"
            ))
        with self.assertRaises(ValueError):
            replace(snapshot(), executable_action_indices=(0, 2, 3))
        with self.assertRaises(ValueError):
            replace(snapshot(), executable_action_indices=(1,))
        with self.assertRaises(ValueError):
            replace(snapshot(), native_action_id="not-present")

    def test_prefix_hash_uses_only_physical_history_before_target(self):
        original = physical_trace()
        changed_outcome = [dict(row) for row in original]
        changed_outcome[0]["reward"] = -123456.0
        changed_future = [dict(row) for row in original]
        changed_future[2]["ghost_vp"] = "arbitrary-future"
        changed_target = [dict(row) for row in original]
        changed_target[1]["ghost_vp"] = "another-target"
        self.assertEqual(
            canonical_prefix_sha256(original, 1),
            canonical_prefix_sha256(changed_outcome, 1),
        )
        self.assertEqual(
            canonical_prefix_sha256(original, 1),
            canonical_prefix_sha256(changed_future, 1),
        )
        self.assertEqual(
            canonical_prefix_sha256(original, 1),
            canonical_prefix_sha256(changed_target, 1),
        )
        changed_prefix = [dict(row) for row in original]
        changed_prefix[0]["cur_vp"] = "wrong-prefix"
        self.assertNotEqual(
            canonical_prefix_sha256(original, 1),
            canonical_prefix_sha256(changed_prefix, 1),
        )

    def test_validates_exact_prefix_one_switch_execution_and_roundtrip(self):
        event = seal_action_lattice([snapshot()]).events[0]
        result = validate_exact_lattice_treatment(
            event,
            arm(event, "native"),
            arm(event, "alt-a"),
            physical_trace("native"),
            physical_trace("alt-a"),
            decision_trace("native", changed=False),
            decision_trace("alt-a", changed=True),
        )
        self.assertTrue(result["exact_prefix_verified"])
        self.assertTrue(result["exact_one_switch_verified"])
        self.assertTrue(result["candidate_executability_verified"])
        self.assertTrue(result["action_identity_roundtrip_verified"])
        self.assertTrue(result["complete_decision_traces_verified"])
        self.assertTrue(result["non_target_action_consistency_verified"])

    def test_treatment_rejects_prefix_episode_switch_or_action_drift(self):
        event = seal_action_lattice([snapshot()]).events[0]
        native_arm = arm(event, "native")
        treatment_arm = arm(event, "alt-a")
        arguments = [
            event,
            native_arm,
            treatment_arm,
            physical_trace("native"),
            physical_trace("alt-a"),
            decision_trace("native", changed=False),
            decision_trace("alt-a", changed=True),
        ]

        wrong_prefix = physical_trace("alt-a")
        wrong_prefix[0]["front_vp"] = "wrong"
        with self.assertRaises(ValueError):
            validate_exact_lattice_treatment(*[
                *arguments[:4], wrong_prefix, *arguments[5:]
            ])

        wrong_episode = replace(treatment_arm, episode_id="other-episode")
        with self.assertRaises(ValueError):
            validate_exact_lattice_treatment(*[
                event, native_arm, wrong_episode, *arguments[3:]
            ])

        unsupported = replace(treatment_arm, action_id="not-live")
        with self.assertRaises(ValueError):
            validate_exact_lattice_treatment(*[
                event, native_arm, unsupported, *arguments[3:]
            ])

        two_switches = decision_trace("alt-a", changed=True)
        two_switches[2] = {
            **two_switches[2],
            "adapted_action_index": 3,
            "adapted_action_id": "alt-a",
            "action_changed": True,
        }
        with self.assertRaises(ValueError):
            validate_exact_lattice_treatment(*[
                *arguments[:6], two_switches
            ])

        wrong_roundtrip = decision_trace("alt-a", changed=True)
        wrong_roundtrip[1]["adapted_action_index"] = 2
        with self.assertRaises(ValueError):
            validate_exact_lattice_treatment(*[
                *arguments[:6], wrong_roundtrip
            ])

        wrong_execution = physical_trace("alt-b")
        with self.assertRaises(ValueError):
            validate_exact_lattice_treatment(*[
                *arguments[:4], wrong_execution, *arguments[5:]
            ])

    def test_non_target_change_cannot_hide_behind_false_flag(self):
        event = seal_action_lattice([snapshot()]).events[0]
        arguments = [
            event,
            arm(event, "native"),
            arm(event, "alt-a"),
            physical_trace("native"),
            physical_trace("alt-a"),
            decision_trace("native", changed=False),
        ]

        hidden_index_change = decision_trace("alt-a", changed=True)
        hidden_index_change[2]["adapted_action_index"] = 3
        with self.assertRaisesRegex(ValueError, "roundtrip|hides"):
            validate_exact_lattice_treatment(*arguments, hidden_index_change)

        hidden_identity_change = decision_trace("alt-a", changed=True)
        hidden_identity_change[2]["adapted_action_id"] = "alt-a"
        with self.assertRaisesRegex(ValueError, "roundtrip|hides"):
            validate_exact_lattice_treatment(*arguments, hidden_identity_change)

        false_changed_flag = decision_trace("alt-a", changed=True)
        false_changed_flag[2].update({
            "adapted_action_index": 3,
            "adapted_action_id": "alt-a",
            "action_changed": False,
        })
        with self.assertRaisesRegex(ValueError, "hides"):
            validate_exact_lattice_treatment(*arguments, false_changed_flag)

        native_hidden_change = decision_trace("native", changed=False)
        native_hidden_change[0].update({
            "adapted_action_index": 2,
            "adapted_action_id": "alt-b",
            "action_changed": False,
        })
        with self.assertRaisesRegex(ValueError, "hides"):
            validate_exact_lattice_treatment(*[
                *arguments[:5], native_hidden_change,
                decision_trace("alt-a", changed=True),
            ])

    def test_decision_trace_must_be_complete_ordered_and_unique(self):
        event = seal_action_lattice([snapshot()]).events[0]
        arguments = [
            event,
            arm(event, "native"),
            arm(event, "alt-a"),
            physical_trace("native"),
            physical_trace("alt-a"),
            decision_trace("native", changed=False),
        ]

        missing = decision_trace("alt-a", changed=True)[:-1]
        with self.assertRaisesRegex(ValueError, "complete physical trace"):
            validate_exact_lattice_treatment(*arguments, missing)

        repeated = decision_trace("alt-a", changed=True)
        repeated[2]["step"] = 1
        with self.assertRaisesRegex(ValueError, "repeats step"):
            validate_exact_lattice_treatment(*arguments, repeated)

        reordered = decision_trace("alt-a", changed=True)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(ValueError, "ordered step sequence"):
            validate_exact_lattice_treatment(*arguments, reordered)

        extra = decision_trace("alt-a", changed=True)
        extra.append({**extra[-1], "step": 3})
        with self.assertRaisesRegex(ValueError, "complete physical trace"):
            validate_exact_lattice_treatment(*arguments, extra)

    def test_native_and_treatment_decisions_match_before_target(self):
        event = seal_action_lattice([snapshot()]).events[0]
        treatment = decision_trace("alt-a", changed=True)
        treatment[0].update({
            "native_action_index": 2,
            "adapted_action_index": 2,
            "native_action_id": "alt-b",
            "adapted_action_id": "alt-b",
        })
        with self.assertRaisesRegex(ValueError, "differ before the target"):
            validate_exact_lattice_treatment(
                event,
                arm(event, "native"),
                arm(event, "alt-a"),
                physical_trace("native"),
                physical_trace("alt-a"),
                decision_trace("native", changed=False),
                treatment,
            )

    def test_scene_episode_and_lattice_arms_share_one_fold(self):
        first = snapshot(dataset="RxR", scene="shared", episode="rxr-1")
        second = snapshot(dataset="R2R", scene="shared", episode="r2r-1")
        seal = seal_action_lattice([first, second])
        assignments = [
            LatticeArmFold(arm(event, action_id), 2)
            for event in seal.events
            for action_id in event.action_ids
        ]
        result = validate_lattice_fold_integrity(seal, assignments)
        self.assertEqual(result["scenes"], 1)
        self.assertEqual(result["lattices"], 2)
        self.assertEqual(result["arms"], 6)

        split_scene = list(assignments)
        split_scene[-1] = replace(split_scene[-1], fold=3)
        with self.assertRaises(ValueError):
            validate_lattice_fold_integrity(seal, split_scene)

        with self.assertRaises(ValueError):
            validate_lattice_fold_integrity(seal, assignments[:-1])

        duplicate = [*assignments, assignments[0]]
        with self.assertRaises(ValueError):
            validate_lattice_fold_integrity(seal, duplicate)


if __name__ == "__main__":
    unittest.main()
