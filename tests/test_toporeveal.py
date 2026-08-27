import unittest
import json
import tempfile
from pathlib import Path

from toporeveal.screening import (
    is_reveal_candidate,
    pilot_sample,
    screening_summary,
    screen_vlnce,
)
from toporeveal.evidence import load_phase0_snapshot

from toporeveal import (
    BranchCandidate,
    BranchPolicy,
    BranchStatus,
    CandidateProvenance,
    CheckpointGate,
    CheckpointProposal,
    ConstraintStatus,
    DecisionContext,
    DecisionKind,
    NoSafeOptionCertificate,
    Phase0Evidence,
    Resolvability,
    RevealBelief,
    RevealEvent,
    RevealPrefix,
    RevealState,
    SafeControlWitness,
    SafeDestinationKind,
    TopologicalMemory,
    validate_event_collection,
)


def branch(
    branch_id: str,
    target: float,
    information: float = 0.0,
    coverage: float = 0.0,
    risk: float = 0.0,
) -> BranchCandidate:
    return BranchCandidate(
        branch_id=branch_id,
        target_probability=target,
        information_gain=information,
        constraint_coverage=coverage,
        travel_cost=1.0,
        return_cost=1.0,
        irreversible_risk=risk,
    )


class CheckpointGateTest(unittest.TestCase):
    def test_requires_stable_branching_and_positive_return_value(self) -> None:
        gate = CheckpointGate()
        useful = CheckpointProposal(
            "cp0", 2, 2, 0.6, 0.7, 1.0, 0.4, memory_cost=0.1
        )
        unstable = CheckpointProposal(
            "cp1", 2, 1, 0.6, 0.7, 1.0, 0.4, memory_cost=0.1
        )
        corridor = CheckpointProposal(
            "cp2", 1, 3, 0.9, 0.9, 1.0, 0.9, memory_cost=0.0
        )
        self.assertTrue(gate.should_create(useful))
        self.assertFalse(gate.should_create(unstable))
        self.assertFalse(gate.should_create(corridor))


class BranchPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = BranchPolicy()
        self.memory = TopologicalMemory()
        self.memory.add_checkpoint(
            "cp0",
            [
                branch("first_right", 0.80, information=0.10, risk=0.05),
                branch("straight", 0.20, information=0.70, coverage=0.8),
            ],
        )

    def test_low_entropy_does_not_commit_when_target_is_unobserved(self) -> None:
        context = DecisionContext(
            "cp0", RevealBelief(0.85, 0.10, 0.05), 0.10, 3.0, True
        )
        decision = self.policy.decide(context, self.memory)
        self.assertNotEqual(decision.kind, DecisionKind.COMMIT)

    def test_target_present_without_referential_evidence_does_not_commit(self) -> None:
        context = DecisionContext(
            "cp0", RevealBelief(0.05, 0.85, 0.10), 0.20, 3.0, True
        )
        decision = self.policy.decide(context, self.memory)
        self.assertNotEqual(decision.kind, DecisionKind.COMMIT)

    def test_discriminable_target_commits(self) -> None:
        context = DecisionContext(
            "cp0", RevealBelief(0.05, 0.10, 0.85), 0.90, 1.0, False
        )
        decision = self.policy.decide(context, self.memory)
        self.assertEqual(decision.kind, DecisionKind.COMMIT)
        self.assertEqual(decision.branch_id, "first_right")

    def test_unsafe_local_options_backtrack_to_previous_candidate(self) -> None:
        self.memory.add_checkpoint(
            "cp1", [branch("dead_end", 0.1, risk=0.9)], "cp0", travel_cost=2.0
        )
        self.memory.set_branch_status(
            "cp0", "first_right", BranchStatus.EXHAUSTED
        )
        context = DecisionContext(
            "cp1", RevealBelief(0.1, 0.8, 0.1), 0.1, 0.0, False
        )
        decision = self.policy.decide(context, self.memory)
        self.assertEqual(decision.kind, DecisionKind.BACKTRACK)
        self.assertEqual(decision.checkpoint_id, "cp0")
        self.assertEqual(decision.branch_id, "straight")
        self.assertEqual(decision.path, ("cp1", "cp0"))

    def test_goal_detection_stops(self) -> None:
        context = DecisionContext(
            "cp0", RevealBelief(0.1, 0.1, 0.8), 0.8, 1.0, False, goal_found=True
        )
        self.assertEqual(
            self.policy.decide(context, self.memory).kind, DecisionKind.STOP
        )

    def test_top2_retains_tail_and_promotes_after_exhaustion(self) -> None:
        memory = TopologicalMemory()
        memory.add_checkpoint(
            "junction",
            [
                branch("b1", 0.90),
                branch("b2", 0.70),
                branch("b3", 0.50),
                branch("b4", 0.30),
            ],
        )
        self.assertEqual(
            tuple(option.branch_id for option in self.policy.active_options(
                memory, "junction")),
            ("b1", "b2"),
        )
        self.assertEqual(len(memory.checkpoint("junction").branches), 4)

        memory.set_branch_status("junction", "b1", BranchStatus.EXHAUSTED)
        self.assertEqual(
            tuple(option.branch_id for option in self.policy.active_options(
                memory, "junction")),
            ("b2", "b3"),
        )

    def test_top2_reranks_after_causal_score_update(self) -> None:
        memory = TopologicalMemory()
        memory.add_checkpoint(
            "junction",
            [branch("b1", 0.80), branch("b2", 0.60), branch("b3", 0.20)],
        )
        memory.checkpoint("junction").branches["b3"].target_probability = 0.95
        self.assertEqual(
            tuple(option.branch_id for option in self.policy.active_options(
                memory, "junction")),
            ("b3", "b1"),
        )

    def test_checkpoint_rejects_duplicate_branch_ids(self) -> None:
        memory = TopologicalMemory()
        with self.assertRaisesRegex(ValueError, "branch ids must be unique"):
            memory.add_checkpoint(
                "junction", [branch("same", 0.8), branch("same", 0.2)]
            )


class BenchmarkSchemaTest(unittest.TestCase):
    @staticmethod
    def prefix(
        index: int,
        candidates: tuple[str, ...],
        separable: bool,
        ordinal: ConstraintStatus,
        safe: bool = True,
    ) -> RevealPrefix:
        witnesses = (
            SafeControlWitness(
                witness_id=f"safe-{index}",
                destination_kind=SafeDestinationKind.TARGET_BRANCH,
                destination_id="second_right",
                action_ids=("move_forward", "turn_right"),
                replay_ref=f"replays/safe-{index}.json",
                replay_sha256=f"{index + 101:064x}",
                path_cost=2.0,
            ),
        ) if safe else ()
        certificate = None if safe else NoSafeOptionCertificate(
            certificate_id=f"unsafe-{index}",
            search_ref=f"replays/unsafe-{index}.json",
            search_sha256=f"{index + 201:064x}",
        )
        return RevealPrefix(
            prefix_index=index,
            history_end_step=index * 2,
            observation_ref=f"scene-a/episode-1/prefix-{index}",
            observation_sha256=f"{index + 1:064x}",
            parent_observation_sha256=(
                None if index == 0 else f"{index:064x}"
            ),
            candidate_ids=candidates,
            candidate_separable=separable,
            language_constraints={"ordinal_second": ordinal},
            safe_option_witnesses=witnesses,
            no_safe_option_certificate=certificate,
        )

    def event(self, **overrides: object) -> RevealEvent:
        fields: dict[str, object] = {
            "dataset": "rxr-ce",
            "scene_id": "scene-a",
            "split": "val_seen",
            "episode_id": "1",
            "event_id": "1-second-right",
            "counterfactual_group_id": "1-second-right-cf",
            "annotation_ref": "data/phase0/raw/rxr_vlnce_v0/val_seen/val_seen_guide.json.gz",
            "annotation_sha256": "081e3fa5c2c4a120640c914b9428de268dea4464f0a246ab471c37783e9ba816",
            "candidate_frontend_id": "oracle-current-v1",
            "sensor_protocol_id": "rgb-fov90-v1",
            "return_controller_id": "oracle-return-v1",
            "candidate_provenance": CandidateProvenance.ORACLE,
            "target_branch_id": "second_right",
            "prefixes": (
                self.prefix(0, ("straight",), False, ConstraintStatus.UNRESOLVED),
                self.prefix(
                    1,
                    ("straight", "second_right"),
                    False,
                    ConstraintStatus.UNRESOLVED,
                ),
                self.prefix(
                    2,
                    ("straight", "second_right"),
                    True,
                    ConstraintStatus.RESOLVED,
                ),
                self.prefix(
                    3,
                    ("straight", "second_right"),
                    True,
                    ConstraintStatus.RESOLVED,
                ),
                self.prefix(
                    4,
                    ("straight", "second_right"),
                    True,
                    ConstraintStatus.RESOLVED,
                ),
                self.prefix(
                    5,
                    ("straight", "second_right"),
                    True,
                    ConstraintStatus.RESOLVED,
                    safe=False,
                ),
            ),
            "reveal_interval": (2, 2),
            "resolvability": Resolvability.RESOLVABLE_BEFORE_SPLIT,
            "counterfactual_action_costs": {"commit_second_right": 1.0},
        }
        fields.update(overrides)
        return RevealEvent(**fields)

    def test_uad_is_derived_and_stable_onset_is_validated(self) -> None:
        event = self.event()
        self.assertEqual(
            event.reveal_states,
            (
                RevealState.UNOBSERVED,
                RevealState.AMBIGUOUS,
                RevealState.DISCRIMINABLE,
                RevealState.DISCRIMINABLE,
                RevealState.DISCRIMINABLE,
                RevealState.DISCRIMINABLE,
            ),
        )
        self.assertEqual(event.stable_reveal_onset, 2)
        self.assertEqual(event.expiry_index, 4)

        with self.assertRaisesRegex(ValueError, "contain the stable D onset"):
            self.event(reveal_interval=(1, 1))

    def test_event_rejects_string_enums_and_noncontiguous_prefixes(self) -> None:
        with self.assertRaisesRegex(TypeError, "CandidateProvenance"):
            self.event(candidate_provenance="oracle")

        prefixes = self.event().prefixes
        skipped = prefixes[:2] + (
            self.prefix(
                5,
                ("straight", "second_right"),
                True,
                ConstraintStatus.RESOLVED,
            ),
        )
        with self.assertRaisesRegex(ValueError, "sorted, and contiguous"):
            self.event(prefixes=skipped, reveal_interval=None)

    def test_expiry_is_derived_from_replayable_safe_options(self) -> None:
        prefixes = self.event().prefixes
        with self.assertRaisesRegex(ValueError, "right-censored"):
            self.event(prefixes=prefixes[:-1])

        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            self.event(expiry_index=3)

    def test_unstable_instantaneous_d_remains_ambiguous(self) -> None:
        prefixes = (
            self.prefix(0, ("second_right",), True, ConstraintStatus.RESOLVED),
            self.prefix(
                1,
                ("second_right",),
                False,
                ConstraintStatus.UNRESOLVED,
            ),
            self.prefix(
                2,
                ("second_right",),
                False,
                ConstraintStatus.UNRESOLVED,
                safe=False,
            ),
        )
        event = self.event(
            prefixes=prefixes,
            reveal_interval=None,
            resolvability=Resolvability.UNRESOLVABLE_BEFORE_SPLIT,
        )
        self.assertEqual(event.instantaneous_reveal_states[0], RevealState.DISCRIMINABLE)
        self.assertEqual(event.reveal_states[0], RevealState.AMBIGUOUS)

    def test_prefix_copies_constraint_mapping(self) -> None:
        constraints = {"ordinal_second": ConstraintStatus.RESOLVED}
        prefix = RevealPrefix(
            prefix_index=0,
            history_end_step=0,
            observation_ref="scene-a/episode-1/prefix-0",
            observation_sha256="1" * 64,
            parent_observation_sha256=None,
            candidate_ids=("second_right",),
            candidate_separable=True,
            language_constraints=constraints,
            safe_option_witnesses=(),
            no_safe_option_certificate=NoSafeOptionCertificate(
                certificate_id="unsafe-0",
                search_ref="replays/unsafe-0.json",
                search_sha256="2" * 64,
            ),
        )
        constraints["ordinal_second"] = ConstraintStatus.UNRESOLVED
        self.assertTrue(prefix.evidence_complete)

    def test_collection_rejects_scene_split_leakage(self) -> None:
        first = self.event()
        second = self.event(
            split="train",
            event_id="other",
            counterfactual_group_id="other-cf",
        )
        with self.assertRaisesRegex(ValueError, "multiple event splits"):
            validate_event_collection((first, second), Path.cwd())

    def test_phase0_rejects_val_unseen_and_missing_prefix_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "train and val_seen"):
            self.event(split="val_unseen")

        event = self.event(scene_id="2n8kARJN3HM", episode_id="1")
        with self.assertRaisesRegex(ValueError, "regular project-local file"):
            validate_event_collection((event,), Path.cwd())


class TopologicalMemoryTest(unittest.TestCase):
    def test_shortest_return_path_uses_graph_cost(self) -> None:
        memory = TopologicalMemory()
        memory.add_checkpoint("a", [branch("a0", 0.2)])
        memory.add_checkpoint("b", [branch("b0", 0.2)], "a", 5.0)
        memory.add_checkpoint("c", [branch("c0", 0.2)], "a", 1.0)
        memory.add_checkpoint("d", [branch("d0", 0.2)], "c", 1.0)
        memory.connect("d", "b", 1.0)
        self.assertEqual(memory.shortest_path("a", "b"), ("a", "c", "d", "b"))


class InstructionScreeningTest(unittest.TestCase):
    def test_requires_a_branch_and_relational_trigger(self) -> None:
        self.assertTrue(is_reveal_candidate(("branch", "ordinal")))
        self.assertFalse(is_reveal_candidate(("branch",)))
        self.assertFalse(is_reveal_candidate(("temporal",)))

    def test_rejects_val_unseen_screening(self) -> None:
        with self.assertRaisesRegex(ValueError, "train and val_seen"):
            list(
                screen_vlnce(
                    [], dataset="rxr-ce", split="val_unseen", languages={"en-US"}
                )
            )

    def test_screens_official_vlnce_fields_and_summarizes(self) -> None:
        episodes = [
            {
                "episode_id": "1",
                "trajectory_id": "10",
                "scene_id": "data/scene_datasets/mp3d/scene-a/scene-a.glb",
                "instruction": {
                    "instruction_id": "1",
                    "language": "en-US",
                    "instruction_text": "Pass the first door and take the second right.",
                },
            },
            {
                "episode_id": "2",
                "trajectory_id": "11",
                "scene_id": "data/scene_datasets/mp3d/scene-b/scene-b.glb",
                "instruction": {
                    "instruction_id": "2",
                    "language": "hi-IN",
                    "instruction_text": "unparsed non-English instruction",
                },
            },
        ]
        candidates = list(
            screen_vlnce(
                episodes,
                dataset="rxr-ce",
                split="val_seen",
                languages={"en-US"},
            )
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].triggers, ("ordinal", "exclusion", "branch"))
        self.assertEqual(
            screening_summary(candidates),
            {
                "candidate_instructions": 1,
                "unique_trajectories": 1,
                "unique_scenes": 1,
                "languages": {"en-US": 1},
                "triggers": {"branch": 1, "exclusion": 1, "ordinal": 1},
            },
        )

    def test_pilot_sample_is_seeded_and_path_unique(self) -> None:
        candidates = [
            screen_vlnce(
                [
                    {
                        "episode_id": str(index),
                        "trajectory_id": str(index // 2),
                        "scene_id": f"scene-{index % 3}.glb",
                        "instruction": {
                            "instruction_id": str(index),
                            "language": "en-US",
                            "instruction_text": (
                                "Pass the first door and turn right."
                                if index % 2
                                else "After the door, turn left."
                            ),
                        },
                    }
                ],
                dataset="rxr-ce",
                split="val_seen",
                languages={"en-US"},
            )
            for index in range(12)
        ]
        flattened = [candidate for group in candidates for candidate in group]
        first = pilot_sample(flattened, 6, seed=7)
        second = pilot_sample(reversed(flattened), 6, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(
            len(
                {
                    (candidate.scene_id, candidate.trajectory_id)
                    for candidate in first
                }
            ),
            len(first),
        )
        self.assertGreater(len({candidate.triggers for candidate in first}), 1)

        with self.assertRaisesRegex(ValueError, "only 12 available"):
            pilot_sample(flattened, 13, seed=7)

    def test_rejects_malformed_official_fields(self) -> None:
        malformed = {
            "episode_id": None,
            "trajectory_id": "1",
            "scene_id": "mp3d/scene-a/scene-a.glb",
            "instruction": {
                "instruction_id": "1",
                "language": "en-US",
                "instruction_text": ["After the door, turn left."],
            },
        }
        with self.assertRaisesRegex(ValueError, "episode_id"):
            list(
                screen_vlnce(
                    [malformed],
                    dataset="rxr-ce",
                    split="train",
                    languages={"en-US"},
                )
            )


class Phase0EvidenceTest(unittest.TestCase):
    def test_go_requires_every_frozen_gate(self) -> None:
        evidence = Phase0Evidence(
            project_self_contained=True,
            mp3d_scene_count=90,
            mp3d_access_authorized=True,
            official_metadata_verified=True,
            habitat_ready=True,
            waypoint_frontend_reproduced=True,
            etpr1_reproduced=True,
            screened_instructions=2000,
            candidate_trajectories=2000,
            reviewed_candidates=50,
            valid_candidates=15,
            validated_events=15,
            unique_expiry_events=15,
        )
        self.assertTrue(evidence.go)
        self.assertEqual(evidence.blockers, ())

    def test_no_go_reports_all_observed_blockers(self) -> None:
        evidence = Phase0Evidence(
            project_self_contained=False,
            mp3d_scene_count=0,
            mp3d_access_authorized=False,
            official_metadata_verified=False,
            habitat_ready=False,
            waypoint_frontend_reproduced=False,
            etpr1_reproduced=False,
            screened_instructions=100,
            candidate_trajectories=100,
            reviewed_candidates=20,
            valid_candidates=2,
            validated_events=2,
            unique_expiry_events=1,
        )
        self.assertFalse(evidence.go)
        self.assertEqual(len(evidence.blockers), 11)

    def test_zero_validated_events_can_never_go(self) -> None:
        evidence = Phase0Evidence(
            project_self_contained=True,
            mp3d_scene_count=90,
            mp3d_access_authorized=True,
            official_metadata_verified=True,
            habitat_ready=True,
            waypoint_frontend_reproduced=True,
            etpr1_reproduced=True,
            screened_instructions=6219,
            candidate_trajectories=6219,
            reviewed_candidates=50,
            valid_candidates=13,
            validated_events=0,
            unique_expiry_events=0,
        )
        self.assertFalse(evidence.go)
        self.assertIn(
            "no Reveal Event has passed full artifact validation",
            evidence.blockers,
        )

    def test_rejects_truthy_string_gate(self) -> None:
        with self.assertRaisesRegex(TypeError, "project_self_contained"):
            Phase0Evidence(
                project_self_contained="false",
                mp3d_scene_count=90,
                mp3d_access_authorized=True,
                official_metadata_verified=True,
                habitat_ready=True,
                waypoint_frontend_reproduced=True,
                etpr1_reproduced=True,
                screened_instructions=50,
                candidate_trajectories=50,
                reviewed_candidates=50,
                valid_candidates=15,
                validated_events=15,
                unique_expiry_events=15,
            )


class EvidenceSnapshotTest(unittest.TestCase):
    def test_claim_cannot_diverge_from_semantic_screening_artifact(self) -> None:
        project_root = Path.cwd()
        source = project_root / "artifacts/phase0/evidence_current.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["claims"]["screened_instructions"] += 1
        with tempfile.TemporaryDirectory(dir=project_root) as directory:
            snapshot = Path(directory) / "spoof.json"
            snapshot.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "screened_instructions is not derived"
            ):
                load_phase0_snapshot(snapshot, project_root)


if __name__ == "__main__":
    unittest.main()
