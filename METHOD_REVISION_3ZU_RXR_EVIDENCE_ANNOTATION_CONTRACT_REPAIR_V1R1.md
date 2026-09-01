# MF3ZU v1r1 — RxR evidence-annotation contract repair

## 1. Purpose

Identifier:

`mf3zu_rxr_evidence_annotation_recovery_v1r1`

This is a bounded, outcome-blind annotation-contract repair for the pre-result
MF3ZU RxR feasibility run. It is not a new candidate population, model arm,
training result, or public evaluation. The parent instruction annotation stays
immutable at 142/154 PASS and its technical failure is recorded in a separate
ledger rather than rewritten.

No MF3ZU candidate-ranking model has been trained and no exact candidate target
or performance value has been opened by this repair.

## 2. Observed correctness issue

The parent instruction stage produced 142 valid graphs and 12 responses rejected
at the top-level `instruction_atoms` list check, before atom content was
validated. The failed instructions are long or instruction-dense (627–1469
characters), accepted graphs reach the implementation's 32-atom maximum, and
one earlier failure passed an identical retry with 30 atoms. The parent runner
did not retain invalid provider JSON, so the exact failed list shapes cannot be
reconstructed.

The fixed request and output schema never stated a 32-atom bound. Atom IDs use
the sealed two-digit `aNN` representation, whose sequential positive range is
`a01` through `a99`. Rejecting otherwise valid lists at 33 was therefore an
undisclosed implementation support restriction rather than part of the method.

## 3. Fixed instruction repair

V1r1 changes only the parser's list bound from 32 to the natural `aNN` support
bound of 99. Atom schema, ontology, ordering, dependency rules, model, prompt,
and decoding remain unchanged:

```text
model = qwen3.8-max
temperature = 0
thinking = false
max_tokens = 8000
1 <= instruction_atoms <= 99
```

The 142 parent responses that already satisfy the stricter parser remain
read-only inputs. V1r1 retries exactly the 12 failed request IDs recorded in the
parent manifest, once each. There is no client-side truncation, merging,
fabrication, renumbering, schema relaxation beyond the corrected bound, prompt
change, model search, or episode-specific instruction.

Provider-level transport retries remain the parent's fixed transport behavior.
Every new logical response, including invalid raw JSON, is stored append-only.
An invalid response is never merged and is not retried semantically inside this
revision.

Before each logical provider call, the runner atomically writes a request-intent
record and holds an exclusive stage lock. If a process dies after the provider
may have received the request but before a response is committed, a later run
stops on the ambiguous intent instead of issuing a duplicate request. All raw
message bodies encountered during transport/JSON retries are retained in the
response attempt ledger.

## 4. Independent merged annotation view

If all 12 repairs pass the corrected parser, v1r1 creates a separate view that
contains the 142 immutable parent PASS responses and 12 new valid responses.
Parent files are never overwritten. The merged index must cover exactly 154
episodes and must not contain or copy the exact-target artifact.

The sanitized 1,428-decision population, 154 episodes, 59 raw scenes, raw-scene
folds, replay observations, ETP checkpoint, ontology, retrieval rule, model
arms, optimizer, and scientific PASS/FAIL gates remain unchanged.

## 5. Evidence annotation

V1r1 evidence requests and provider payloads are byte-equivalent to the parent
implementation given the corrected instruction graph. Evidence uses the same
model, system prompt, causal visual inputs, output contract, and decoding
settings. There is one logical provider response per decision. The runner is
changed only to retain invalid raw JSON before strict validation, so future
contract failures can be classified instead of erased.

Only responses passing the unchanged evidence validator may enter evidence
memory. There is no response-dependent prompt, semantic retry, label repair,
threshold search, or human adjudication in v1r1.

Immediately before evidence-memory materialization, all 1,428 responses are
revalidated against their sealed request payload, event identity, model,
protocol, graph, and request intent, and the response-bundle SHA is recomputed.
Deterministic request, memory, and manifest writes are resumable only when an
existing final or partial artifact is byte-identical to the recomputed value.

## 6. Boundaries and stop condition

This repair may read instructions, sanitized decision identities, causal replay
observations, candidate aliases/headings, and provider responses. It may not
read the separate exact-target file, correct-candidate values, model performance,
route outcomes, future observations, or any public split.

Manual review remains skipped, so all annotations are provisional,
`human_verified=false`, and `gold=false`.

If instruction annotation cannot reach 154/154, or evidence annotation cannot
reach 1,428/1,428, v1r1 stops as an annotation-readiness failure. Any subsequent
format repair requires another versioned, pre-response protocol based only on
the retained contract errors. V1r1 does not train a model. Complete evidence
annotation authorizes only deterministic evidence-memory materialization and
audit; training remains a separate subsequent action.
