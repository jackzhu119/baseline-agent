from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arenaagent.competition.evaluation.postconditions import (
    POSTCONDITION_FAILURE,
    POSTCONDITION_SUCCESS,
    POSTCONDITION_UNKNOWN,
    ActionExpectation,
    PhysicalPostconditionTracker,
)
from arenaagent.competition.solvers.npc import NPCMemory
from arenaagent.competition.solvers.tidyroom import TidyRoomTracker

TASK_TYPES = {"tidyroom", "counting", "npc", "raven", "jigsaw", "unknown"}
VECTOR_DIMENSIONS = 3
EARLY_PHASE_RATIO = 0.65
MID_PHASE_RATIO = 0.3
CRITICAL_REMAINING_STEPS = 5
MAX_RAVEN_ATTEMPTS = 3
DIRECT_PLACEMENT_CONFIDENCE = 0.85
DIRECT_PLACEMENT_MARGIN = 0.08
FAILURE_CATEGORIES = {
    "PERCEPTION",
    "OBJECT_ID",
    "COUNTING_QUERY",
    "COUNT_DUPLICATION",
    "PLANNING",
    "INVALID_ACTION",
    "NAVIGATION",
    "PICK",
    "PLACEMENT",
    "PHYSICAL_POSTCONDITION",
    "NPC_MISSING_FACT",
    "NPC_BAD_QUESTION",
    "RAVEN_REASONING",
    "JIGSAW_GRID",
    "JIGSAW_ROTATION",
    "JSON",
    "MODEL_API",
    "LOOP",
    "PREMATURE_FINISH",
    "TIME_BUDGET",
    "UNKNOWN",
}
OBJECT_ACTIONS = {
    "look_at_object",
    "point_at_object",
    "move_and_take_object",
    "move_to_object",
    "pour_water",
    "sit_down_to_object",
    "slice_food",
    "wash_hands",
    "wash_object_in_hand",
    "mop_floor",
}
TERMINAL_ACTIONS = {"finish_task", "submit_answer", "submit_puzzle_answer", "solve_raven"}
SUPPORTED_ACTIONS = {
    "finish_task",
    "submit_answer",
    "submit_puzzle_answer",
    "solve_raven",
    "look_at_location",
    "look_at_object",
    "point_at_object",
    "move_and_take_object",
    "move_forward",
    "move_backward",
    "put_down_sth",
    "turn_in_degree",
    "turn_around_to_degree",
    "move_to_object",
    "move_to_npc",
    "move_to_location",
    "pour_water",
    "sit_down_to_object",
    "slice_food",
    "wash_hands",
    "wash_object_in_hand",
    "mop_floor",
    "rest",
    "speak_to_npc",
    "move_and_put_down",
    "move_and_put_down_object_in_container",
}


def route_task(subject: Any) -> str:
    """Return one of the five official preliminary task types or ``unknown``."""
    if isinstance(subject, dict):
        direct = str(subject.get("task_type") or "").strip().lower()
        if direct in TASK_TYPES:
            return direct
        stage = str(subject.get("stage") or "").strip().lower()
        stage_aliases = {
            "raven_room": "raven",
            "npc_room": "npc",
            "counting_room": "counting",
            "tidy_room": "tidyroom",
            "jigsaw_room": "jigsaw",
        }
        if stage in stage_aliases:
            return stage_aliases[stage]
        text = " ".join(str(subject.get(k) or "") for k in ("subject", "goal", "task_prompt"))
    else:
        text = str(subject or "")
    lowered = text.lower()
    keyword_routes = (
        ("raven", ("raven", "瑞文")),
        ("jigsaw", ("jigsaw", "拼图")),
        ("npc", ("npc", "对话", "询问", "问问")),
        ("counting", ("count", "多少", "几种", "数量")),
        ("tidyroom", ("tidy", "整理", "收拾", "摆放")),
    )
    for task_type, keywords in keyword_routes:
        if any(keyword in lowered for keyword in keywords):
            return task_type
    return "unknown"


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return round(value, 3)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _hash(value: Any) -> str:
    encoded = json.dumps(_canonical(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _object_id(item: dict[str, Any]) -> str:
    for key in ("object_id", "id", "objectId"):
        value = item.get(key)
        if value is not None and not isinstance(value, bool):
            return str(value)
    return ""


def _object_position(item: dict[str, Any]) -> list[float] | None:
    bounds = item.get("world_aabb")
    if isinstance(bounds, dict) and isinstance(bounds.get("min"), dict) and isinstance(bounds.get("max"), dict):
        lower = bounds["min"]
        upper = bounds["max"]
        values: list[float] = []
        for axis in ("X", "Y", "Z"):
            low = _as_number(lower.get(axis, lower.get(axis.lower())))
            high = _as_number(upper.get(axis, upper.get(axis.lower())))
            if low is None or high is None:
                return None
            values.append(round((low + high) / 2.0, 3))
        return values
    for key in ("position", "location", "place_location"):
        value = item.get(key)
        if _is_location(value):
            if isinstance(value, dict):
                lowered = {str(name).lower(): raw for name, raw in value.items()}
                return [float(lowered[axis]) for axis in ("x", "y", "z")]
            return [float(raw) for raw in value[:VECTOR_DIMENSIONS]]
    return None


def _semantic_identity(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item.get(key) or "").strip().lower()
        for key in ("name", "semantic_type", "category", "type", "color", "shape")
    )


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_location(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return len(value) >= VECTOR_DIMENSIONS and all(
            _as_number(item) is not None for item in value[:VECTOR_DIMENSIONS]
        )
    if isinstance(value, dict):
        lowered = {str(key).lower(): item for key, item in value.items()}
        return all(axis in lowered and _as_number(lowered[axis]) is not None for axis in ("x", "y", "z"))
    return False


def _result_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    status = str(result.get("result") or result.get("status") or "").strip().lower()
    return status in {"failed", "failure", "error", "false"} or result.get("success") is False


def _result_success(result: Any) -> bool:
    if not isinstance(result, dict):
        return result is not None
    status = str(result.get("result") or result.get("status") or "").strip().lower()
    return status in {"success", "succeeded", "ok", "completed", "true"} or result.get("success") is True


def _extract_numeric(value: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                number = _as_number(value[key])
                if number is not None:
                    return number
        for child in value.values():
            found = _extract_numeric(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _extract_numeric(child, keys)
            if found is not None:
                return found
    return None


def classify_failure(  # noqa: PLR0911, PLR0912
    failure_class: str, error: str, action: Any, task_type: str
) -> str:
    raw = str(failure_class or "").upper()
    message = str(error or "").lower()
    name = str(action.get("action") or "").lower() if isinstance(action, dict) else ""
    if raw in FAILURE_CATEGORIES:
        return raw
    if raw == "PERCEPTION_ERROR":
        return "OBJECT_ID" if "object_id" in message else "PERCEPTION"
    if raw == "MODEL_ERROR":
        return "MODEL_API"
    if raw == "LOOP_ERROR":
        return "LOOP"
    if raw in {"TERMINATION_ERROR", "PREMATURE_FINISH"}:
        return "PREMATURE_FINISH"
    if raw == "MEMORY_ERROR":
        return "NPC_BAD_QUESTION" if task_type == "npc" else "PLANNING"
    if raw in {"NPC_BAD_QUESTION", "NPC_MISSING_FACT", "PHYSICAL_POSTCONDITION", "TIME_BUDGET"}:
        return raw
    if raw == "REASONING_ERROR":
        return {
            "raven": "RAVEN_REASONING",
            "jigsaw": "JIGSAW_ROTATION" if "rotation" in message else "JIGSAW_GRID",
            "npc": "NPC_MISSING_FACT",
        }.get(task_type, "PLANNING")
    if raw == "ACTION_ERROR":
        if name in {"put_down_sth", "move_and_put_down", "move_and_put_down_object_in_container"}:
            return "PLACEMENT"
        if name == "move_and_take_object":
            return "PICK"
        if name.startswith("move_") or name in {"turn_in_degree", "turn_around_to_degree"}:
            return "NAVIGATION"
        if "malformed" in message or "missing action" in message or "empty" in message:
            return "JSON"
        return "INVALID_ACTION"
    return "UNKNOWN"


@dataclass(slots=True)
class ActionValidation:
    valid: bool
    action: dict[str, Any]
    error: str = ""
    failure_class: str = ""


@dataclass(slots=True)
class EpisodeMetrics:
    episode_id: str
    task_type: str
    task_text: str
    success: bool | None = None
    score: float | None = None
    steps: int = 0
    invalid_actions: int = 0
    replans: int = 0
    retries: int = 0
    llm_calls: int = 0
    vision_calls: int = 0
    tokens: int = 0
    latency: float = 0.0
    stuck_count: int = 0
    model_errors: int = 0
    model_failure_count: int = 0
    perception_errors: int = 0
    repeated_actions: int = 0
    finish_guard_blocks: int = 0
    postcondition_successes: int = 0
    postcondition_failures: int = 0
    postcondition_unknowns: int = 0
    prompt_chars: int = 0
    max_prompt_chars: int = 0
    termination_reason: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    verified: bool = False


class CompetitionRuntime:
    """Compact world state, action guard, metrics, and failure classification.

    The runtime uses only the public subject, perception and action results.  It
    never queries evaluator internals or hidden task state.
    """

    def __init__(
        self,
        *,
        log_dir: str = "logs",
        repeated_action_limit: int = 2,
        stagnant_observation_limit: int = 5,
        default_max_steps: int = 60,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.repeated_action_limit = max(int(repeated_action_limit), 1)
        self.stagnant_observation_limit = max(int(stagnant_observation_limit), 2)
        self.default_max_steps = max(int(default_max_steps), 1)
        self.metrics: EpisodeMetrics | None = None
        self.subject: dict[str, Any] = {}
        self.max_steps = self.default_max_steps
        self.objects: dict[str, dict[str, Any]] = {}
        self.object_aliases: dict[str, str] = {}
        self.visible_object_ids: set[str] = set()
        self.visible_canonical_ids: set[str] = set()
        self.observation_hashes: list[str] = []
        self.action_records: list[dict[str, Any]] = []
        self.failed_signatures: Counter[str] = Counter()
        self.questions_asked: set[tuple[str, str]] = set()
        self.npc_facts: list[dict[str, Any]] = []
        self.first_failure: dict[str, Any] | None = None
        self.last_observation_diff: dict[str, list[str]] = {
            "appeared": [],
            "disappeared": [],
            "changed": [],
        }
        self.run_metadata: dict[str, Any] = {}
        self.strategy_context: dict[str, Any] = {}
        self.progress = TidyRoomTracker(retry_limit=self.repeated_action_limit)
        self.npc_memory = NPCMemory()
        self.postconditions = PhysicalPostconditionTracker()
        self._started_monotonic = 0.0
        self._last_subject_key = ""

    @property
    def task_type(self) -> str:
        return self.metrics.task_type if self.metrics else "unknown"

    def ensure_episode(self, subject: Any) -> None:
        subject_dict = dict(subject) if isinstance(subject, dict) else {"subject": str(subject or "")}
        identity_payload = {
            key: subject_dict.get(key)
            for key in ("task_id", "subject_id", "id", "task_type", "stage", "subject", "goal")
            if subject_dict.get(key) is not None
        }
        subject_key = _hash(identity_payload)
        if self.metrics is not None and subject_key == self._last_subject_key:
            return
        if self.metrics is not None:
            self.finish(termination_reason="subject_changed_without_evaluation")
        self.subject = subject_dict
        task_text = str(
            subject_dict.get("subject") or subject_dict.get("goal") or subject_dict.get("task_prompt") or ""
        )
        episode_hint = subject_dict.get("task_id") or subject_dict.get("subject_id") or subject_dict.get("id")
        episode_id = str(episode_hint or f"{int(time.time() * 1000)}-{subject_key[:8]}")
        self.metrics = EpisodeMetrics(episode_id=episode_id, task_type=route_task(subject_dict), task_text=task_text)
        self.max_steps = self._resolve_max_steps(subject_dict)
        self.objects = {}
        self.object_aliases = {}
        self.visible_object_ids = set()
        self.visible_canonical_ids = set()
        self.observation_hashes = []
        self.action_records = []
        self.failed_signatures = Counter()
        self.questions_asked = set()
        self.npc_facts = []
        self.first_failure = None
        self.last_observation_diff = {"appeared": [], "disappeared": [], "changed": []}
        self.run_metadata = {}
        self.strategy_context = {}
        self.progress.reset(subject_dict)
        self.npc_memory.reset(subject_dict)
        self.postconditions.reset()
        self._started_monotonic = time.perf_counter()
        self._last_subject_key = subject_key

    def _resolve_max_steps(self, source: Any) -> int:
        value = _extract_numeric(source, ("max_steps", "step_limit", "max_step", "steps_limit"))
        return max(int(value), 1) if value is not None else self.default_max_steps

    def observe(self, visible_objects: Any, task_response: Any = None) -> None:
        if self.metrics is None:
            return
        objects = visible_objects if isinstance(visible_objects, list) else []
        previous_visible_ids = set(self.visible_canonical_ids)
        previous_objects = {
            object_id: {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "first_seen_step",
                    "last_seen_step",
                    "seen_count",
                    "source_ids",
                    "counted",
                    "confidence",
                    "current_object_id",
                    "position_seen_step",
                }
            }
            for object_id, item in self.objects.items()
        }
        self.visible_object_ids = set()
        self.visible_canonical_ids = set()
        current_objects: dict[str, dict[str, Any]] = {}
        for raw in objects:
            if not isinstance(raw, dict):
                continue
            object_id = _object_id(raw)
            if not object_id:
                continue
            self.visible_object_ids.add(object_id)
            canonical_id = self.object_aliases.get(object_id)
            if canonical_id is None:
                canonical_id = object_id if object_id in self.objects else self._secondary_object_match(raw)
                canonical_id = canonical_id or object_id
                self.object_aliases[object_id] = canonical_id
            self.visible_canonical_ids.add(canonical_id)
            canonical_raw = _canonical(raw)
            canonical_raw["object_id"] = canonical_id
            canonical_raw["current_object_id"] = object_id
            position = _object_position(raw)
            if position is not None:
                canonical_raw["position"] = position
            current_objects[canonical_id] = {
                key: value for key, value in canonical_raw.items() if key != "current_object_id"
            }
            existing = dict(self.objects.get(canonical_id, {}))
            source_ids = set(existing.get("source_ids", []))
            source_ids.add(object_id)
            existing.update(canonical_raw)
            existing["source_ids"] = sorted(source_ids)
            existing.setdefault("counted", False)
            existing["confidence"] = 1.0 if len(source_ids) == 1 else 0.85
            existing.setdefault("first_seen_step", self.metrics.steps)
            existing["last_seen_step"] = self.metrics.steps
            if position is not None:
                existing["position_seen_step"] = self.metrics.steps
            existing["seen_count"] = int(existing.get("seen_count", 0)) + 1
            self.objects[canonical_id] = existing
        self.last_observation_diff = {
            "appeared": sorted(self.visible_canonical_ids - previous_visible_ids),
            "disappeared": sorted(previous_visible_ids - self.visible_canonical_ids),
            "changed": sorted(
                object_id
                for object_id in self.visible_canonical_ids & previous_visible_ids
                if current_objects.get(object_id) != previous_objects.get(object_id)
            ),
        }
        observation_hash = _hash(objects)
        self.observation_hashes.append(observation_hash)
        self.observation_hashes = self.observation_hashes[-self.stagnant_observation_limit :]
        if (
            len(self.observation_hashes) == self.stagnant_observation_limit
            and len(set(self.observation_hashes)) == 1
            and self.action_records
        ):
            self.metrics.stuck_count += 1
        self._capture_npc_facts(task_response)
        self._record_postcondition_events(
            self.postconditions.observe(
                self.objects,
                self.visible_canonical_ids,
                official_completion=self._has_completion_evidence(task_response),
            )
        )
        self.progress.observe(self.objects, self._has_completion_evidence(task_response))

    @staticmethod
    def _has_completion_evidence(value: Any) -> bool:
        if isinstance(value, dict):
            for key in ("completed", "is_complete", "task_completed", "goal_completed"):
                if value.get(key) is True:
                    return True
            return any(CompetitionRuntime._has_completion_evidence(child) for child in value.values())
        if isinstance(value, list):
            return any(CompetitionRuntime._has_completion_evidence(child) for child in value)
        return False

    def update_hand_state(self, has_object: bool) -> None:
        observation_step = self.metrics.steps if self.metrics is not None else None
        self.progress.update_hand_state(bool(has_object), self.objects, observation_step)
        self._record_postcondition_events(
            self.postconditions.update_hand_state(
                bool(has_object),
                self.objects,
                observation_step=observation_step,
            )
        )

    def _secondary_object_match(self, raw: dict[str, Any], tolerance: float = 5.0) -> str | None:
        """Match a remapped perception ID only with strong public-position evidence."""
        position = _object_position(raw)
        identity = _semantic_identity(raw)
        if position is None or not any(identity):
            return None
        candidates: list[tuple[float, str]] = []
        for canonical_id, existing in self.objects.items():
            if _semantic_identity(existing) != identity:
                continue
            existing_position = _object_position(existing)
            if existing_position is None:
                continue
            distance = math.dist(position, existing_position)
            if distance <= tolerance:
                candidates.append((distance, canonical_id))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _capture_npc_facts(self, task_response: Any) -> None:
        if not isinstance(task_response, dict):
            return
        for key in ("npc_reply", "reply", "hints", "facts"):
            value = task_response.get(key)
            if value not in (None, "", {}, []):
                fact = {"step": self.metrics.steps if self.metrics else 0, "source": key, "value": _canonical(value)}
                if fact not in self.npc_facts:
                    self.npc_facts.append(fact)
                self.npc_memory.ingest_fact(value)

    def record_npc_exchange(self, target: str, question: str, reply: Any, hints: Any = None) -> None:
        if self.metrics is None:
            return
        self.questions_asked.add((str(target), str(question)))
        fact = {
            "step": self.metrics.steps + 1,
            "source": str(target),
            "question": str(question),
            "reply": _canonical(reply),
            "hints": _canonical(hints),
        }
        if fact not in self.npc_facts:
            self.npc_facts.append(fact)
        self.npc_memory.record_exchange(str(target), str(question), reply, hints)

    def record_llm_call(self, token_usage: Any = None) -> None:
        if self.metrics is None:
            return
        self.metrics.llm_calls += 1
        if isinstance(token_usage, dict):
            total = token_usage.get("total_tokens")
            if _as_number(total) is not None:
                self.metrics.tokens += int(float(total))

    def record_model_failure(self) -> None:
        if self.metrics is not None:
            self.metrics.model_errors += 1
            self.metrics.model_failure_count += 1

    def record_prompt_context(self, messages: Any) -> None:
        if self.metrics is None:
            return
        try:
            serialized = json.dumps(messages, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            serialized = str(messages)
        size = len(serialized)
        self.metrics.prompt_chars += size
        self.metrics.max_prompt_chars = max(self.metrics.max_prompt_chars, size)

    def record_vision_call(self) -> None:
        if self.metrics is not None:
            self.metrics.vision_calls += 1

    def set_run_metadata(self, metadata: dict[str, Any]) -> None:
        """Attach reproducibility data without ever persisting credentials."""
        allowed = {
            "agent_build",
            "agent_class",
            "client_type",
            "model",
            "prompt_fingerprint",
            "runtime_config",
        }
        self.run_metadata.update({key: _canonical(value) for key, value in metadata.items() if key in allowed})

    def set_strategy_context(self, context: dict[str, Any]) -> None:
        self.strategy_context = _canonical(context)

    def _state_snapshot(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "step": self.metrics.steps if self.metrics else 0,
            "max_steps": self.max_steps,
            "visible_object_ids": sorted(self.visible_object_ids),
            "visible_registry_ids": sorted(self.visible_canonical_ids),
            "observation_diff": self.last_observation_diff,
            "object_counts": self._count_summary(),
            "npc_facts": self.npc_facts[-8:],
        }

    def record_step_failure(self, failure_class: str, error: str, *, stage: str) -> None:
        """Record a recoverable infrastructure/model failure as one attempted step."""
        if self.metrics is None:
            return
        self._capture_failure(failure_class, error, {})
        self.metrics.steps += 1
        self.metrics.retries += 1
        if failure_class == "MODEL_ERROR" and self.metrics.model_failure_count == 0:
            self.record_model_failure()
        if failure_class == "PERCEPTION_ERROR":
            self.metrics.perception_errors += 1
        self.action_records.append(
            {
                "step": self.metrics.steps,
                "signature": "",
                "state": self._state_snapshot(),
                "action": {},
                "result": {"result": "failed", "error": str(error)},
                "failed": True,
                "failure_class": failure_class,
                "stage": stage,
            }
        )
        self.action_records = self.action_records[-50:]

    def validate_action(  # noqa: PLR0911, PLR0912, PLR0915
        self, action: Any, *, object_in_hand: bool = False
    ) -> ActionValidation:
        if self.metrics is None:
            raise RuntimeError("ensure_episode must be called before validate_action")
        if not isinstance(action, dict) or not action:
            return self._invalid({}, "empty or malformed model action", "ACTION_ERROR")
        normalized = dict(action)
        name = str(normalized.get("action") or "").strip().lower()
        if not name:
            return self._invalid(normalized, "missing action name", "ACTION_ERROR")
        if name not in SUPPORTED_ACTIONS:
            return self._invalid(normalized, f"unsupported action type: {name}", "ACTION_ERROR")
        params = normalized.get("parameters")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return self._invalid(normalized, "parameters must be an object", "ACTION_ERROR")
        normalized["action"] = name
        normalized["parameters"] = params
        if self.metrics.steps >= self.max_steps:
            return self._invalid(normalized, "episode step budget is exhausted", "TIME_BUDGET")

        required: dict[str, tuple[str, ...]] = {
            "look_at_location": ("target_location", "location"),
            "look_at_object": ("object_id", "object"),
            "point_at_object": ("object_id", "object"),
            "move_and_take_object": ("object_id", "object"),
            "put_down_sth": ("target_location",),
            "move_to_object": ("object_id", "object"),
            "move_to_npc": ("npc_name", "npc", "target", "name"),
            "move_to_location": ("target_location", "location"),
            "speak_to_npc": ("npc_name", "npc", "npc_id", "target", "name"),
            "sit_down_to_object": ("object_id", "object"),
            "wash_hands": ("faucet_object_id", "object_id", "object"),
            "wash_object_in_hand": ("faucet_object_id", "object_id", "object"),
            "mop_floor": ("dirt_id", "object_id", "object"),
        }
        aliases = required.get(name)
        if aliases and not any(params.get(key) not in (None, "") for key in aliases):
            return self._invalid(normalized, f"missing required parameter: {' or '.join(aliases)}", "ACTION_ERROR")

        if name in {"pour_water", "slice_food"}:
            if not any(params.get(key) not in (None, "") for key in ("object_id", "object")):
                return self._invalid(normalized, "missing required parameter: object_id or object", "ACTION_ERROR")
            if not any(params.get(key) is not None for key in ("location", "target_location")):
                return self._invalid(
                    normalized, "missing required parameter: location or target_location", "ACTION_ERROR"
                )

        if name == "move_and_put_down":
            if not any(params.get(key) is not None for key in ("move_target_location", "move_location")):
                return self._invalid(normalized, "missing move_target_location", "ACTION_ERROR")
            if not any(
                params.get(key) is not None for key in ("put_target_location", "put_location", "target_location")
            ):
                return self._invalid(normalized, "missing put_target_location", "ACTION_ERROR")

        if name in OBJECT_ACTIONS:
            object_keys = ("object_id", "object", "faucet_object_id", "dirt_id")
            object_id = next((str(params[key]) for key in object_keys if params.get(key) not in (None, "")), "")
            if object_id and object_id not in self.visible_object_ids:
                return self._invalid(
                    normalized,
                    f"object_id {object_id!r} is not in the current visible-object mapping",
                    "PERCEPTION_ERROR",
                )

        location_keys = (
            "target_location",
            "location",
            "move_target_location",
            "move_location",
            "put_target_location",
            "put_location",
        )
        for key in location_keys:
            if key in params and params[key] is not None and not _is_location(params[key]):
                return self._invalid(normalized, f"{key} must be a finite 3D location", "ACTION_ERROR")

        if name in {"move_forward", "move_backward"}:
            distance = _as_number(params.get("distance", params.get("step")))
            if distance is None or distance <= 0:
                return self._invalid(normalized, "distance must be a positive finite number", "ACTION_ERROR")
        if "which_hand" in params and (
            isinstance(params["which_hand"], bool)
            or not isinstance(params["which_hand"], int)
            or params["which_hand"] < 0
        ):
            return self._invalid(normalized, "which_hand must be a non-negative integer", "ACTION_ERROR")
        if "stop_distance" in params:
            stop_distance = _as_number(params["stop_distance"])
            if stop_distance is None or stop_distance < 0:
                return self._invalid(normalized, "stop_distance must be a non-negative finite number", "ACTION_ERROR")
        for rotation_key in ("target_rotation", "put_rotation", "rotation"):
            if rotation_key not in params or params[rotation_key] is None:
                continue
            rotation = params[rotation_key]
            if not isinstance(rotation, dict) or not all(
                _as_number(rotation.get(axis)) is not None for axis in ("roll", "yaw", "pitch")
            ):
                return self._invalid(
                    normalized,
                    f"{rotation_key} must contain finite roll, yaw, and pitch values",
                    "ACTION_ERROR",
                )
        for boolean_key in ("auto_rotate", "force_locate", "is_cancel", "execute_immediately"):
            if boolean_key in params and not isinstance(params[boolean_key], bool):
                return self._invalid(normalized, f"{boolean_key} must be a boolean", "ACTION_ERROR")
        if name in {"turn_in_degree", "turn_around_to_degree"}:
            if _as_number(params.get("degree")) is None:
                return self._invalid(normalized, "degree must be a finite number", "ACTION_ERROR")
        if (
            name in {"put_down_sth", "move_and_put_down", "move_and_put_down_object_in_container"}
            and not object_in_hand
        ):
            return self._invalid(normalized, "cannot place an object while both hands are empty", "ACTION_ERROR")
        if name in {"submit_answer", "submit_puzzle_answer"} and normalized.get("output") in (None, ""):
            return self._invalid(normalized, "submit_answer requires a non-empty output", "TERMINATION_ERROR")
        if name == "finish_task" and self.task_type in {"counting", "npc", "raven"}:
            return self._invalid(
                normalized, f"{self.task_type} requires an answer submission, not finish_task", "TERMINATION_ERROR"
            )
        if name == "finish_task":
            allowed, reason = self.progress.can_finish(self.task_type, object_in_hand)
            if not allowed:
                return self._invalid(normalized, f"finish blocked: {reason}", "PREMATURE_FINISH")

        if self.task_type == "tidyroom" and self._elapsed_seconds() >= 340:
            critical_allowed = {
                "move_and_take_object",
                "put_down_sth",
                "move_and_put_down",
                "move_and_put_down_object_in_container",
                "finish_task",
            }
            final_scan_allowed = name == "turn_in_degree" and self.progress.final_scan_count <= 2
            if name not in critical_allowed and not final_scan_allowed:
                return self._invalid(
                    normalized,
                    "critical time phase blocks low-value exploration",
                    "TIME_BUDGET",
                )

        if name == "move_and_take_object":
            if self.task_type == "tidyroom" and (
                object_in_hand or self.progress.current_object or self.progress.pending_pick or self.progress.pending_place
            ):
                return self._invalid(
                    normalized,
                    "strict tidy-room loop blocks a new pickup while another object is held or awaiting verification",
                    "PLANNING_ERROR",
                )
            object_id = next(
                (str(params[key]) for key in ("object_id", "object") if params.get(key) not in (None, "")), ""
            )
            canonical_id = self.object_aliases.get(object_id, object_id)
            if canonical_id in self.progress.completed_objects:
                return self._invalid(
                    normalized,
                    f"object {object_id!r} is already verified complete",
                    "PLANNING_ERROR",
                )
            if self.task_type == "tidyroom":
                allowed_ids = self.progress.candidate_items | self.progress.initial_expected_objects
                if canonical_id not in allowed_ids:
                    return self._invalid(
                        normalized,
                        f"object {object_id!r} is not approved as high-confidence clutter",
                        "PLANNING_ERROR",
                    )
                can_attempt, reason = self.progress.can_attempt_pick(canonical_id, self.metrics.steps)
                if not can_attempt:
                    return self._invalid(normalized, reason, "LOOP_ERROR")

        if name in {"move_to_npc", "speak_to_npc"}:
            target = str(next((params[key] for key in ("npc_name", "npc", "target", "name") if params.get(key)), ""))
            if not self.npc_memory.is_allowed(target):
                return self._invalid(
                    normalized,
                    f"npc_name {target!r} is not in the official allowed mapping",
                    "NPC_REASONING",
                )
        if name == "speak_to_npc":
            message = str(next((params[key] for key in ("message", "content", "text") if params.get(key)), ""))
            if not message.strip():
                return self._invalid(
                    normalized,
                    "speak_to_npc requires a non-empty task-relevant question",
                    "NPC_REASONING",
                )
            if self.task_type == "npc" and not self.npc_memory.is_question_relevant(message):
                return self._invalid(
                    normalized,
                    "NPC question does not target any currently missing required fact",
                    "NPC_BAD_QUESTION",
                )

        signature = self.semantic_action_signature(normalized)
        # solve_raven advances through a cached ranked candidate list internally,
        # so a few identical public actions are distinct attempts, but retries
        # are still bounded to avoid an endless candidate loop.
        if name == "solve_raven":
            raven_attempts = sum(record["action"].get("action") == "solve_raven" for record in self.action_records)
            if raven_attempts >= MAX_RAVEN_ATTEMPTS:
                return self._invalid(normalized, "maximum Raven candidate attempts reached", "REASONING_ERROR")
        else:
            recent = [record["signature"] for record in self.action_records[-self.repeated_action_limit :]]
            if len(recent) == self.repeated_action_limit and all(item == signature for item in recent):
                return self._invalid(
                    normalized, "repeated-action loop blocked; choose a different observation or plan", "LOOP_ERROR"
                )
            if self.failed_signatures[signature] >= self.repeated_action_limit:
                return self._invalid(
                    normalized, "action is temporarily blacklisted after repeated failures", "LOOP_ERROR"
                )
        if name == "speak_to_npc":
            target = str(next((params[key] for key in ("npc_name", "npc", "target", "name") if params.get(key)), ""))
            message = str(next((params[key] for key in ("message", "content", "text") if params.get(key)), ""))
            if self.npc_memory.was_asked(target, message):
                return self._invalid(normalized, "duplicate NPC question blocked", "MEMORY_ERROR")
        return ActionValidation(valid=True, action=normalized)

    def _invalid(self, action: dict[str, Any], error: str, failure_class: str) -> ActionValidation:
        if self.metrics is not None:
            self.metrics.invalid_actions += 1
            self.metrics.replans += 1
            if failure_class == "LOOP_ERROR":
                self.metrics.repeated_actions += 1
            if failure_class in {"TERMINATION_ERROR", "PREMATURE_FINISH"}:
                self.metrics.finish_guard_blocks += 1
        self._capture_failure(failure_class, error, action)
        return ActionValidation(valid=False, action=action, error=error, failure_class=failure_class)

    @staticmethod
    def action_signature(action: dict[str, Any]) -> str:
        payload = {
            "action": action.get("action"),
            "parameters": action.get("parameters") or {},
            "output": action.get("output"),
        }
        return _hash(payload)

    def semantic_action_signature(self, action: dict[str, Any]) -> str:
        """Collapse JSON variations that pursue the same physical target."""
        name = str(action.get("action") or "").lower()
        params = action.get("parameters") or {}
        if name in {"move_and_take_object", "move_to_object", "look_at_object"}:
            raw_id = str(params.get("object_id") or params.get("object") or "")
            canonical_id = self.object_aliases.get(raw_id, raw_id)
            verb = "pick" if name == "move_and_take_object" else "move_target" if name == "move_to_object" else "look"
            return _hash({"semantic_action": verb, "target": canonical_id})
        if name == "move_to_location":
            location = params.get("target_location", params.get("location"))
            if _is_location(location):
                point = _object_position({"position": location})
                nearby: list[tuple[float, str]] = []
                if point is not None:
                    for canonical_id in self.visible_canonical_ids:
                        object_position = _object_position(self.objects.get(canonical_id, {}))
                        if object_position is not None and math.dist(point, object_position) <= 5.0:
                            nearby.append((math.dist(point, object_position), canonical_id))
                if nearby:
                    nearby.sort()
                    return _hash({"semantic_action": "move_target", "target": nearby[0][1]})
        if name in {"put_down_sth", "move_and_put_down"}:
            location = next(
                (
                    params.get(key)
                    for key in ("put_target_location", "put_location", "target_location")
                    if params.get(key) is not None
                ),
                None,
            )
            point = _object_position({"position": location}) if _is_location(location) else None
            if point is not None:
                return _hash({"semantic_action": "put", "target_cell": [round(value / 5) for value in point]})
        return self.action_signature(action)

    def action_value_score(self, action: dict[str, Any]) -> float:
        """Score whether an action advances the current TidyRoom subgoal."""
        name = str(action.get("action") or "").lower()
        if name in {"finish_task", "move_and_take_object", "put_down_sth", "move_and_put_down"}:
            return 1.0
        if name == "move_to_location" and self.progress.current_object:
            return 0.8
        if name in {"look_at_object", "look_at_location"}:
            return 0.6
        if name == "turn_in_degree" and self.progress.final_scan_count <= 2:
            return 0.5
        if name == "move_to_object":
            return 0.4
        return 0.0

    def _elapsed_seconds(self) -> float:
        return max(time.perf_counter() - self._started_monotonic, 0.0)

    def record_action(self, action: dict[str, Any], result: Any, *, validation: ActionValidation | None = None) -> None:
        if self.metrics is None:
            return
        self.metrics.steps += 1
        signature = self.semantic_action_signature(action)
        failed = validation is not None and not validation.valid or _result_failed(result)
        if failed:
            self.failed_signatures[signature] += 1
            if validation is None or validation.valid:
                self.metrics.retries += 1
                self._capture_failure("ACTION_ERROR", str(result), action, step=self.metrics.steps)
        name = str(action.get("action") or "").lower()
        params = action.get("parameters") or {}
        raw_object_id = next(
            (
                str(params[key])
                for key in ("object_id", "object", "faucet_object_id", "dirt_id")
                if params.get(key) not in (None, "")
            ),
            "",
        )
        canonical_object_id = (
            self.object_aliases.get(raw_object_id, raw_object_id) or self.progress.current_object or ""
        )
        self.progress.record_action(action, result, canonical_object_id, self.metrics.steps)
        if not failed:
            self._record_postcondition_events(
                self.postconditions.start(
                    action,
                    object_id=canonical_object_id,
                    objects=self.objects,
                    visible_ids=self.visible_canonical_ids,
                    step=self.metrics.steps,
                )
            )
        if name == "speak_to_npc" and not failed:
            params = action.get("parameters") or {}
            target = str(next((params[key] for key in ("npc_name", "npc", "target", "name") if params.get(key)), ""))
            message = str(next((params[key] for key in ("message", "content", "text") if params.get(key)), ""))
            self.questions_asked.add((target, message))
        self.action_records.append(
            {
                "step": self.metrics.steps,
                "signature": signature,
                "state": self._state_snapshot(),
                "action": _canonical(action),
                "action_value_score": self.action_value_score(action) if self.task_type == "tidyroom" else None,
                "result": _canonical(result),
                "failed": bool(failed),
                "postcondition": (
                    self.postconditions.pending.context() if self.postconditions.pending is not None else None
                ),
            }
        )
        self.action_records = self.action_records[-50:]

    def _record_postcondition_events(self, events: list[ActionExpectation]) -> None:
        if self.metrics is None:
            return
        for event in events:
            if event.status == POSTCONDITION_SUCCESS:
                self.metrics.postcondition_successes += 1
            elif event.status == POSTCONDITION_FAILURE:
                self.metrics.postcondition_failures += 1
                self.metrics.retries += 1
                self.metrics.replans += 1
                self._capture_failure(
                    "PHYSICAL_POSTCONDITION",
                    event.reason,
                    {"action": event.action},
                    step=event.created_step,
                )
            elif event.status == POSTCONDITION_UNKNOWN:
                self.metrics.postcondition_unknowns += 1

    def _capture_failure(self, failure_class: str, error: str, action: Any, *, step: int | None = None) -> None:
        if self.first_failure is None:
            self.first_failure = {
                "step": int(step) if step is not None else (self.metrics.steps + 1) if self.metrics else 0,
                "error_type": failure_class,
                "category": classify_failure(failure_class, error, action, self.task_type),
                "message": str(error),
                "action": _canonical(action),
            }

    def prompt_context(self) -> dict[str, Any]:
        if self.metrics is None:
            return {}
        remaining = max(self.max_steps - self.metrics.steps, 0)
        ratio = remaining / self.max_steps
        step_phase = (
            "EARLY"
            if ratio > EARLY_PHASE_RATIO
            else "MID"
            if ratio > MID_PHASE_RATIO
            else "LATE"
            if remaining > CRITICAL_REMAINING_STEPS
            else "CRITICAL"
        )
        elapsed_seconds = self._elapsed_seconds()
        time_phase = (
            "EARLY"
            if elapsed_seconds < 120
            else "MID"
            if elapsed_seconds < 250
            else "LATE"
            if elapsed_seconds < 340
            else "CRITICAL"
        )
        phase_order = {"EARLY": 0, "MID": 1, "LATE": 2, "CRITICAL": 3}
        phase = max((step_phase, time_phase), key=phase_order.__getitem__)
        blocked_failed_actions = sum(
            1 for count in self.failed_signatures.values() if count >= self.repeated_action_limit
        )
        recent_failure_class = ""
        if self.action_records and self.action_records[-1].get("failed"):
            recent_failure_class = str(self.action_records[-1].get("failure_class") or "")
        recovery_active = (
            self.metrics.stuck_count > 0
            or self.metrics.repeated_actions > 0
            or blocked_failed_actions > 0
            or recent_failure_class in {"MODEL_ERROR", "PERCEPTION_ERROR"}
            or bool(
                self.postconditions.history
                and self.postconditions.history[-1].status in {POSTCONDITION_FAILURE, POSTCONDITION_UNKNOWN}
            )
        )
        placement_candidates = self._placement_candidates()
        visible_ids = sorted(self.visible_object_ids)
        task_progress = self.progress.context()
        if self.task_type == "tidyroom":
            relevant_ids = (
                self.progress.remaining_candidates()
                | self.progress.target_surfaces
                | ({self.progress.current_object} if self.progress.current_object else set())
            )
            visible_ids = [object_id for object_id in visible_ids if object_id in relevant_ids][:24]
            if self.progress.target_surfaces:
                placement_candidates = [
                    item for item in placement_candidates if str(item.get("object_id")) in self.progress.target_surfaces
                ][:18]
            task_progress = self.progress.prompt_context()
        return {
            "task_type": self.task_type,
            "step_budget": {
                "current_step": self.metrics.steps,
                "max_steps": self.max_steps,
                "remaining_steps": remaining,
                "elapsed_seconds": round(elapsed_seconds, 1),
                "phase": phase,
                "policy": {
                    "exploration_allowed": phase in {"EARLY", "MID"},
                    "prefer_direct_progress": phase in {"LATE", "CRITICAL"},
                    "retry_limit": 0 if phase == "CRITICAL" else 1 if phase == "LATE" else self.repeated_action_limit,
                },
            },
            "visible_object_ids": visible_ids,
            "observation_diff": self.last_observation_diff,
            "object_registry": {
                "unique_objects_seen": len(self.objects),
                "counts": self._count_summary(),
            },
            "npc": {
                "facts": self.npc_facts[-8:],
                "questions_asked": [list(item) for item in sorted(self.questions_asked)],
                "memory": self.npc_memory.context(),
            },
            "recovery": {
                "state": "RECOVERY" if recovery_active else "NORMAL",
                "stuck": self.metrics.stuck_count > 0,
                "blocked_failed_actions": blocked_failed_actions,
                "recent_failure_class": recent_failure_class,
                "required_change": (
                    "preserve world state, refresh observation, and choose a different legal subgoal/action"
                    if recovery_active
                    else "none"
                ),
            },
            "strategy": self.strategy_context,
            "task_progress": task_progress,
            "action_postconditions": self.postconditions.context(),
            "placement_candidates": placement_candidates,
        }

    def placement_candidates(self) -> list[dict[str, Any]]:
        return self._placement_candidates()

    def _placement_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        held_object_id = self.progress.current_object
        held_half_height = self._object_half_height(self.objects.get(held_object_id or "", {}))
        excluded_ids = set(self.progress.expected_objects) | set(self.progress.completed_objects)
        if held_object_id:
            excluded_ids.add(held_object_id)
        for canonical_id in sorted(self.visible_canonical_ids):
            if canonical_id in excluded_ids:
                continue
            item = self.objects.get(canonical_id, {})
            current_id = str(item.get("current_object_id") or canonical_id)
            place_location = item.get("place_location")
            if _is_location(place_location):
                confidence, evidence = self._placement_confidence(item, "place_location")
                candidates.append(
                    {
                        "object_id": current_id,
                        "source": "place_location",
                        "location": place_location,
                        "confidence": confidence,
                        "evidence": evidence,
                    }
                )
                continue
            bounds = item.get("world_aabb")
            if isinstance(bounds, dict) and isinstance(bounds.get("min"), dict) and isinstance(bounds.get("max"), dict):
                lower = bounds["min"]
                upper = bounds["max"]
                values = []
                for axis in ("X", "Y"):
                    low = _as_number(lower.get(axis, lower.get(axis.lower())))
                    high = _as_number(upper.get(axis, upper.get(axis.lower())))
                    if low is None or high is None:
                        values = []
                        break
                    values.append(round((low + high) / 2.0, 3))
                top = _as_number(upper.get("Z", upper.get("z")))
                if len(values) == VECTOR_DIMENSIONS - 1 and top is not None:
                    height_adjustment = round((held_half_height or 0.0) + 1.0, 3)
                    confidence, evidence = self._placement_confidence(item, "world_aabb_top")
                    candidates.append(
                        {
                            "object_id": current_id,
                            "source": "world_aabb_top",
                            "location": [*values, round(top + height_adjustment, 3)],
                            "height_adjustment": height_adjustment,
                            "confidence": confidence,
                            "evidence": evidence,
                        }
                    )
        candidates.sort(key=lambda item: (-float(item["confidence"]), str(item["object_id"])))
        for index, candidate in enumerate(candidates):
            next_confidence = float(candidates[index + 1]["confidence"]) if index + 1 < len(candidates) else 0.0
            margin = max(float(candidate["confidence"]) - next_confidence, 0.0)
            candidate["selection_margin"] = round(margin, 3)
            candidate["direct_safe"] = bool(
                index == 0
                and float(candidate["confidence"]) >= DIRECT_PLACEMENT_CONFIDENCE
                and (len(candidates) == 1 or margin >= DIRECT_PLACEMENT_MARGIN)
            )
            if index == 0 and not candidate["direct_safe"]:
                candidate["evidence"].append("direct placement withheld because target selection is ambiguous")
        return candidates

    @staticmethod
    def _placement_confidence(item: dict[str, Any], source: str) -> tuple[float, list[str]]:
        if source == "place_location":
            confidence = 0.92
            evidence = ["explicit public place_location"]
        else:
            confidence = 0.72
            evidence = ["coordinate estimated from public world_aabb top surface"]
        seen_count = max(int(item.get("seen_count", 0)), 0)
        stability_bonus = min(max(seen_count - 1, 0) * 0.02, 0.06)
        if stability_bonus:
            confidence += stability_bonus
            evidence.append(f"surface observed {seen_count} times")
        identity_confidence = _as_number(item.get("confidence"))
        if identity_confidence is not None and identity_confidence < 1.0:
            confidence -= min((1.0 - identity_confidence) * 0.2, 0.12)
            evidence.append("object identity was merged across perception IDs")
        return round(min(max(confidence, 0.0), 1.0), 3), evidence

    @staticmethod
    def _object_half_height(item: dict[str, Any]) -> float | None:
        bounds = item.get("world_aabb")
        if (
            not isinstance(bounds, dict)
            or not isinstance(bounds.get("min"), dict)
            or not isinstance(bounds.get("max"), dict)
        ):
            return None
        lower = bounds["min"]
        upper = bounds["max"]
        low = _as_number(lower.get("Z", lower.get("z")))
        high = _as_number(upper.get("Z", upper.get("z")))
        if low is None or high is None or high < low:
            return None
        return (high - low) / 2.0

    def _count_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for field_name in ("name", "semantic_type", "category", "type", "color", "shape"):
            counts: Counter[str] = Counter()
            for item in self.objects.values():
                value = item.get(field_name)
                if value not in (None, "", "Unknown", "unknown"):
                    counts[str(value)] += 1
            if counts:
                summary[field_name] = dict(sorted(counts.items()))
        return summary

    def finish(self, evaluation: Any = None, *, termination_reason: str = "evaluated") -> dict[str, Any] | None:
        if self.metrics is None:
            return None
        if isinstance(evaluation, dict):
            success_value = self._extract_success(evaluation)
            if success_value is not None:
                self.metrics.success = success_value
                self.metrics.verified = True
            score = _extract_numeric(evaluation, ("score", "total_score", "subject_score"))
            if score is not None:
                self.metrics.score = score
                self.metrics.verified = True
        self.metrics.termination_reason = termination_reason
        self.metrics.finished_at = datetime.now(timezone.utc).isoformat()
        self.metrics.latency = round(time.perf_counter() - self._started_monotonic, 6)
        payload = asdict(self.metrics)
        payload["run_metadata"] = self.run_metadata
        payload["evaluation"] = _canonical(evaluation)
        payload["world_state"] = {
            "known_objects": len(self.objects),
            "npc_facts": self.npc_facts,
            "recent_actions": self.action_records,
            "task_progress": self.progress.context(),
            "action_postconditions": self.postconditions.context(),
        }
        self._write_artifacts(payload)
        self.metrics = None
        return payload

    @staticmethod
    def _extract_success(value: Any) -> bool | None:
        if isinstance(value, dict):
            for key in ("success", "passed", "is_success", "completed"):
                if isinstance(value.get(key), bool):
                    return value[key]
            status = str(value.get("result") or value.get("status") or "").strip().lower()
            if status in {"success", "succeeded", "passed", "completed"}:
                return True
            if status in {"failed", "failure", "error"}:
                return False
            for child in value.values():
                found = CompetitionRuntime._extract_success(child)
                if found is not None:
                    return found
        if isinstance(value, list):
            for child in value:
                found = CompetitionRuntime._extract_success(child)
                if found is not None:
                    return found
        return None

    def _write_artifacts(self, payload: dict[str, Any]) -> None:
        metrics_dir = self.log_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        episode_id = str(payload["episode_id"]).replace("/", "_").replace("\\", "_")
        self._atomic_json(metrics_dir / f"episode_{episode_id}.json", payload)
        if payload.get("success") is False or self.first_failure is not None:
            report = {
                "episode_id": payload["episode_id"],
                "task_type": payload["task_type"],
                "first_critical_error": self.first_failure,
                "cascade": "Subsequent actions may be degraded after the first critical error.",
                "could_succeed_if_fixed": self.first_failure is not None,
                "termination_reason": payload["termination_reason"],
            }
            self._atomic_json(metrics_dir / f"failure_{episode_id}.json", report)
            aggregate_path = metrics_dir / "failure_report.json"
            existing: list[Any] = []
            if aggregate_path.exists():
                try:
                    loaded = json.loads(aggregate_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        existing = loaded
                except (OSError, json.JSONDecodeError):
                    existing = []
            existing = [item for item in existing if item.get("episode_id") != report["episode_id"]]
            existing.append(report)
            self._atomic_json(aggregate_path, existing)

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
