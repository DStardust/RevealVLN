# RevealNav Method-Freeze-2 Correctness Revision 2

**Revision ID:** `MF2-CR2`  
**Date:** 2026-08-24  
**Status:** Phase-0C protocol correction only; training remains forbidden.  
**Supersedes:** the review-authorization sentence in `MF2-CR1`; every other
definition, threshold, dataset boundary, module, non-claim, and gate in
`METHOD_FREEZE_2_CORRECTNESS_REVISION_1.md` remains unchanged.  
**Canonical document:** `FROZEN_SPEC.md` remains byte-for-byte unchanged.

## Protocol deadlock correction

MF2-CR1 says that no semantic review packet is authorized until all Phase-0C
gates pass, while Gate 6 itself requires human language/branch review. Taken
literally, those two requirements create a circular gate that cannot be
satisfied. This revision makes the narrow distinction below.

After Gates 1--5, the machine-geometric part of Gate 6, and the automatic
candidate-to-semantic-track subgate pass on the frozen queue, the project may
construct a **private, unannotated human-review packet** from their fixed-set
intersection. Packet construction is evidence packaging, not semantic
annotation and not a passed Gate 6. It must satisfy all of the following:

1. selection is a deterministic intersection of already frozen machine
   results; no resampling or threshold change is allowed;
2. every human judgment field is null, `reviewed=false`, and the packet states
   that zero Reveal Events have been human validated;
3. Matterport-derived media and instruction text remain private and are not
   publication artifacts;
4. a row is admitted only after an authorized reviewer fills every required
   field and signs the row; uncertain rows are rejected fail-closed;
5. machine code may validate completeness, hashes, and logical consistency,
   but must never infer or fabricate the human fields; and
6. Gate 6, overall Phase-0C, canonical-freeze replacement, feature generation,
   and training all remain `NO_GO` until the reviewed derivative passes a
   separate, versioned acceptance gate.

Before any human label is entered, the review acceptance rule is frozen as
follows. All 35 rows must be judged independently by two authorized reviewers;
reviewers cannot see one another's labels while making the first judgment.
Every disagreement or either-reviewer uncertainty is rejected unless a third
authorized reviewer records an explicit adjudication. The accepted subset
must contain at least 15 unique events across at least 10 scenes, matching the
pre-existing Gate-2 feasibility floor, and must contain zero unresolved
semantic ambiguity. The reviewed deliverable must retain both original label
tables, the adjudication table, reviewer pseudonyms, timestamps, completeness
checks, and an inter-rater agreement report. These rules apply to the full
fixed packet; reviewing only a favorable subset is forbidden.

The pending packet may include private causal frames and an explicitly marked
offline geometry panel solely as review aids. Future frames and offline
geometry cannot be treated as evaluated-model inputs. Reviewers must judge
causal reveal from the ordered causal frames and use the offline panel only to
verify the proposed exit-region identity.

## Non-claims

This correction does not validate any language-dependent Reveal Event, does
not authorize a learned `INSPECT` policy, does not establish a benchmark or
CVPR claim, and does not alter the high re-entry-risk finding for the
resource-conditioned last-passage label.
