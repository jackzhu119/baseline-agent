# TidyRoom Real-Run Optimization

## 1. Real-run problem

The supplied official-environment run connected successfully and most RPCs returned success, but the TidyRoom subject did not select `finish_task`, ended near the server's roughly 400-second boundary with `local_finished=False`, repeatedly targeted low-value/Unknown objects, and used about 19 VLM calls and 237k total tokens. The earlier delayed-evaluation lifecycle symptom did not recur.

## 2. Root causes

- Raw visible objects, including duplicate IDs, static geometry, and semantically Unknown objects, reached planning context.
- A successful action RPC was treated too optimistically; there was no enforced single-object pick/put/observation-verification loop.
- Exact JSON hashing missed semantically identical movement actions.
- Placement geometry was not separated clearly enough from an object-to-target semantic assignment.
- Unknown scene coverage had no bounded final-scan policy, so the local agent could fail to finish.
- Too much world state, history, and action history was repeatedly sent to the VLM.
- Configuration objects could be logged with credential-bearing fields.

## 3. Modified files

- `arenaagent/utils/redaction.py`, `arenaagent/utils/config.py`, `arenaagent/agent_base.py`: recursive credential redaction at logging boundaries.
- `arenaagent/competition/solvers/tidyroom_policy.py`: visible-object deduplication, deterministic clutter classification, compact projection, and semantic target assignment.
- `arenaagent/competition/solvers/tidyroom.py`: pick/place attempt registry, strict pending/current-object state, verification, bounded coverage scans, and compact prompt state.
- `arenaagent/competition/runtime.py`: TidyRoom legality guards, semantic action fingerprints, action-value telemetry, elapsed-time phase, compact context, and finish guard integration.
- `arenaagent/competition/task_router.py`: deterministic TidyRoom pick/put/final-scan/finish actions when public evidence is sufficient.
- `arenaagent/vlm_agent/vlm_agent.py`: pre-prompt dedup/filtering, compact TidyRoom history, and deterministic-strategy routing.
- TidyRoom prompt policy files and focused regression tests under `tests/`.

Counting, NPC, Raven, Jigsaw, protobuf/gRPC, official server timeout, and the existing `submit_answer -> evaluate_subject` lifecycle fix were not redesigned.

## 4. State machine

The local control loop now represents these phases explicitly:

`DISCOVER -> SELECT_OBJECT -> PICK -> VERIFY_PICK -> SELECT_TARGET -> PUT -> VERIFY_PUT -> MARK_COMPLETED`

Only one object may be current. A new pickup is rejected while an object is held, a pickup is awaiting observation, or a placement is awaiting observation. RPC `success` starts a pending transition; it does not mark semantic completion.

The public hand API currently supplies a boolean hand state rather than a reliable held-object identity in this client path. Pick verification therefore binds that hand transition to the one pending object. Placement verification requires an empty hand plus either official completion evidence, an atomic-container action, or a fresh observed object position within tolerance of the selected target. `UNKNOWN` remains unverified.

## 5. Object filtering

`TidyObjectClassifier` admits only recognized clutter (shoe/boot/slipper, cup/mug/bottle, food, trash, pillow/cushion) or explicit official required IDs. Unknown/unlabelled classes enter `uncertain_items`; furniture, architecture, room bounds, static geometry, and placement markers enter `rejected_items`. Completed and bounded-failure objects are excluded. Duplicate same-frame IDs are merged before registry and prompt use.

## 6. Target selection

`TidyTargetAssignment` keeps `source_location`, candidate target surfaces, selected target, confidence, and reason separate. The clutter object's own `place_location` is deliberately not used as its destination. Semantic mappings cover shoes to shoe storage, trash to a bin, cushions to sofa/bed, drinkware to its area/surface, and food to storage. Automatic placement requires an explicit surface `place_location`, confidence at least 0.85, and a unique margin of at least 0.08. AABB-top evidence is capped below automatic-selection confidence because its height/open-area precision is insufficient.

## 7. Duplicate and loop protection

The attempt registry records per-object pick/put attempts, last action/result, completion/failure, and cooldown. A pickup is limited to one normal attempt plus one recovery attempt. The semantic fingerprint collapses changed `think` text and maps `move_to_object(id)` and `move_to_location(location_of_id)` to the same physical target. Existing `repeated_action_limit=2` now operates on that semantic signature.

## 8. Finish policy and time budget

`finish_task` is selected locally only when no held/pending object exists, no high-confidence candidate remains, no failed object remains unresolved, and either official/verified required-object evidence exists or two bounded final scans found no new high-confidence clutter. Uncertain objects may trigger the bounded scans but do not become pickup candidates by default. A new candidate resets coverage.

Local elapsed phases are EARLY `<120s`, MID `<250s`, LATE `<340s`, and CRITICAL `>=340s`, combined conservatively with the existing step budget. CRITICAL blocks low-value exploration and permits only direct completion actions plus the bounded final scan.

## 9. Prompt compression and VLM-call reduction

- Deduplicate by object ID before registry/prompt construction.
- Send at most 18 compact relevant objects; omit raw AABB and rotation fields.
- Model-facing TidyRoom state includes bounded samples/counts instead of every rejected/uncertain ID.
- Placement candidates are limited to semantic target surfaces.
- TidyRoom text history is capped at two messages and action history at three entries; stale base64 images were already stripped.
- High-confidence object/target pairs execute pick and put through Python without another VLM call. The VLM remains available for missing semantic evidence or bounded recovery.

## 10. API-key redaction

Recursive redaction covers field names containing API key, token, secret, credential, authorization, password, access key, or private key, plus common `sk-...`, Bearer, and Basic credential strings. Tests capture Loguru output and verify that the original value is absent. No real key was added to Git.

## 11. Tests

Focused regressions cover Unknown filtering, ID deduplication, held-object pickup blocking, bounded repeated pickup, completed-object blocking, post-put verification, source/destination separation, semantic repeat detection, automatic finish after bounded scans, no early finish with unresolved clutter, prompt deduplication/limit, token budget, and credential redaction.

Final commands and results are recorded in the final handoff. The official requested commands are:

```text
uv run --python 3.12.14 python -m pytest -q
uv run --python 3.12.14 python -m compileall -q arenaagent scripts tests
```

## 12. BEFORE token estimate

- Supplied real run: about 235k prompt tokens and 1.6k completion tokens, about 237k total. This is user-supplied historical evidence, not reproduced in this checkout.
- Reproducible synthetic 70-object prompt before filtering: 16,260 serialized characters, approximately 4,065 tokens using the explicitly approximate `characters / 4` offline estimator.

## 13. AFTER offline token estimate

The same synthetic scene (10 duplicate records for one shoe, 50 Unknown objects, 10 wall objects) projects to one relevant object. The complete rendered message is 5,638 serialized characters, approximately 1,410 tokens by the same estimator: an estimated 65.3% reduction. This test includes the fixed prompt instructions, not only the object list. Provider tokenization and a real episode may differ.

No real post-change VLM-call count, prompt-token total, score, or success-rate improvement is claimed.

## 14. Not yet verified in TongSIM

`REAL_TONGSIM_AFTER_NOT_VERIFIED`

The official release/service is not part of the automated unit-test environment. A new real run must verify: (1) whether public object semantics expose the expected target surfaces, (2) whether the post-put frame reports the moved object close enough for verification, and (3) whether two final scans provide adequate room coverage before active finish.
