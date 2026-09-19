from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

PLACEMENT_TOLERANCE = 15.0
VECTOR_DIMENSIONS = 3
MAX_PICK_ATTEMPTS = 2
MAX_FINAL_SCANS = 2


def _as_ids(value: Any) -> set[str]:
    if value in (None, "", []):
        return set()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {str(item) for item in values if item not in (None, "")}


def _failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    status = str(result.get("result") or result.get("status") or "").strip().lower()
    return status in {"failed", "failure", "error", "false"} or result.get("success") is False


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _location(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        lowered = {str(key).lower(): raw for key, raw in value.items()}
        numbers = [_number(lowered.get(axis)) for axis in ("x", "y", "z")]
    elif isinstance(value, (list, tuple)) and len(value) >= VECTOR_DIMENSIONS:
        numbers = [_number(raw) for raw in value[:VECTOR_DIMENSIONS]]
    else:
        return None
    return [float(number) for number in numbers] if all(number is not None for number in numbers) else None


def _object_position(item: dict[str, Any]) -> list[float] | None:
    direct = _location(item.get("position") or item.get("place_location") or item.get("location"))
    if direct is not None:
        return direct
    bounds = item.get("world_aabb")
    if (
        not isinstance(bounds, dict)
        or not isinstance(bounds.get("min"), dict)
        or not isinstance(bounds.get("max"), dict)
    ):
        return None
    lower = {str(key).lower(): raw for key, raw in bounds["min"].items()}
    upper = {str(key).lower(): raw for key, raw in bounds["max"].items()}
    values: list[float] = []
    for axis in ("x", "y", "z"):
        low = _number(lower.get(axis))
        high = _number(upper.get(axis))
        if low is None or high is None:
            return None
        values.append((low + high) / 2.0)
    return values


@dataclass(slots=True)
class ObjectAttemptState:
    object_id: str
    pick_attempts: int = 0
    put_attempts: int = 0
    last_action: str = ""
    last_result: str = ""
    completed: bool = False
    failed: bool = False
    cooldown_until_step: int = 0

    def context(self) -> dict[str, Any]:
        return {
            "pick_attempts": self.pick_attempts,
            "put_attempts": self.put_attempts,
            "last_action": self.last_action,
            "last_result": self.last_result,
            "completed": self.completed,
            "failed": self.failed,
            "cooldown_until_step": self.cooldown_until_step,
        }


@dataclass(slots=True)
class TidyRoomTracker:
    retry_limit: int = 2
    initial_expected_objects: set[str] = field(default_factory=set)
    expected_objects: set[str] = field(default_factory=set)
    completed_objects: set[str] = field(default_factory=set)
    failed_objects: set[str] = field(default_factory=set)
    states: dict[str, str] = field(default_factory=dict)
    retries: Counter[str] = field(default_factory=Counter)
    current_object: str | None = None
    pending_pick: str | None = None
    pending_place: str | None = None
    pending_place_target: list[float] | None = None
    pending_place_mode: str = ""
    completed_goal_actions: int = 0
    completion_evidence: bool = False
    candidate_items: set[str] = field(default_factory=set)
    rejected_items: set[str] = field(default_factory=set)
    uncertain_items: set[str] = field(default_factory=set)
    target_surfaces: set[str] = field(default_factory=set)
    target_assignments: dict[str, dict[str, Any]] = field(default_factory=dict)
    object_attempt_registry: dict[str, ObjectAttemptState] = field(default_factory=dict)
    vlm_reviews: dict[str, dict[str, Any]] = field(default_factory=dict)
    review_attempts: Counter[str] = field(default_factory=Counter)
    final_scan_count: int = 0
    coverage_verified: bool = False

    def reset(self, subject: dict[str, Any]) -> None:
        self.expected_objects = _as_ids(subject.get("movable_object_id")) | _as_ids(subject.get("piece_object_id"))
        self.initial_expected_objects = set(self.expected_objects)
        self.completed_objects = set()
        self.failed_objects = set()
        self.states = {object_id: "DISCOVERED" for object_id in self.expected_objects}
        self.retries = Counter()
        self.current_object = None
        self.pending_pick = None
        self.pending_place = None
        self.pending_place_target = None
        self.pending_place_mode = ""
        self.completed_goal_actions = 0
        self.completion_evidence = False
        self.candidate_items = set()
        self.rejected_items = set()
        self.uncertain_items = set()
        self.target_surfaces = set()
        self.target_assignments = {}
        self.object_attempt_registry = {}
        self.vlm_reviews = {}
        self.review_attempts = Counter()
        self.final_scan_count = 0
        self.coverage_verified = False

    def update_classification(
        self,
        *,
        candidate_items: set[str],
        rejected_items: set[str],
        uncertain_items: set[str],
        target_surfaces: set[str] | None = None,
    ) -> None:
        newly_discovered = candidate_items - self.candidate_items - self.completed_objects
        self.candidate_items.update(candidate_items - self.completed_objects)
        self.rejected_items.update(rejected_items)
        # Uncertainty is view-local: keep only currently unresolved visible objects.
        # This allows the VLM to review the current camera view without permanently
        # blocking completion because of stale Unknown objects seen many turns ago.
        self.uncertain_items = set(uncertain_items)
        self.uncertain_items.difference_update(self.candidate_items | self.rejected_items | self.completed_objects)
        self.target_surfaces.update(target_surfaces or set())
        for object_id in self.candidate_items:
            self.states.setdefault(object_id, "DISCOVERED")
            self.object_attempt_registry.setdefault(object_id, ObjectAttemptState(object_id))
        if newly_discovered:
            self.record_final_scan(discovered_new_candidate=True)

    def apply_vlm_reviews(self, reviews: list[dict[str, Any]], confidence_threshold: float = 0.6) -> None:
        """Persist batch visual classifications so reviewed IDs are not re-reviewed forever."""
        for review in reviews:
            if not isinstance(review, dict):
                continue
            object_id = str(review.get("object_id") or "")
            if not object_id or object_id not in self.uncertain_items:
                continue
            is_clutter = review.get("is_clutter")
            category = str(review.get("category") or "other").strip().lower()
            try:
                confidence = float(review.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(confidence, 1.0))
            self.review_attempts[object_id] += 1
            stored = {
                "object_id": object_id,
                "is_clutter": is_clutter if isinstance(is_clutter, bool) else None,
                "category": category,
                "confidence": round(confidence, 3),
                "target_id": (
                    str(review.get("target_id"))
                    if review.get("target_id") not in (None, "", "null")
                    else None
                ),
            }
            self.vlm_reviews[object_id] = stored
            if not isinstance(is_clutter, bool) or confidence < confidence_threshold:
                continue
            # "clutter + other" is not actionable enough to pick safely; keep it
            # unresolved so another viewpoint/model pass can name a task category.
            if is_clutter and category == "other":
                continue
            self.uncertain_items.discard(object_id)
            if is_clutter:
                self.rejected_items.discard(object_id)
                self.candidate_items.add(object_id)
                self.states.setdefault(object_id, "DISCOVERED")
                self.object_attempt_registry.setdefault(object_id, ObjectAttemptState(object_id))
                self.record_final_scan(discovered_new_candidate=True)
            else:
                self.candidate_items.discard(object_id)
                self.rejected_items.add(object_id)

    def set_target_assignment(self, object_id: str, assignment: dict[str, Any]) -> None:
        self.target_assignments[str(object_id)] = assignment

    def selected_target(self, object_id: str) -> dict[str, Any] | None:
        assignment = self.target_assignments.get(str(object_id), {})
        selected = assignment.get("selected_target")
        return selected if isinstance(selected, dict) else None

    def remaining_candidates(self) -> set[str]:
        return self.candidate_items - self.completed_objects - self.failed_objects

    def can_attempt_pick(self, object_id: str, step: int = 0) -> tuple[bool, str]:
        attempt = self.object_attempt_registry.setdefault(str(object_id), ObjectAttemptState(str(object_id)))
        if attempt.completed:
            return False, "object is already verified complete"
        if attempt.failed or attempt.pick_attempts >= MAX_PICK_ATTEMPTS:
            return False, "object exhausted its normal and recovery pickup attempts"
        if step < attempt.cooldown_until_step:
            return False, f"object is in cooldown until step {attempt.cooldown_until_step}"
        return True, "pickup attempt is available"

    def record_final_scan(self, discovered_new_candidate: bool = False) -> None:
        if discovered_new_candidate:
            self.final_scan_count = 0
            self.coverage_verified = False
            return
        self.final_scan_count += 1
        if self.final_scan_count >= MAX_FINAL_SCANS:
            self.coverage_verified = True

    def observe(self, known_objects: dict[str, dict[str, Any]], completion_evidence: bool = False) -> None:
        self.completion_evidence = self.completion_evidence or bool(completion_evidence)
        for object_id in known_objects:
            if object_id in self.expected_objects:
                if self.states.get(object_id, "DISCOVERED") == "DISCOVERED":
                    self.states[object_id] = "TARGET_IDENTIFIED"

    def update_hand_state(
        self,
        has_object: bool,
        known_objects: dict[str, dict[str, Any]] | None = None,
        observation_step: int | None = None,
    ) -> None:
        if self.pending_pick is not None:
            object_id = self.pending_pick
            self.pending_pick = None
            if has_object:
                self.current_object = object_id
                self.states[object_id] = "PICKED"
            else:
                self.current_object = None
                self._record_failure(object_id, observation_step or 0)
        if self.pending_place is not None:
            object_id = self.pending_place
            if has_object:
                self._clear_pending_place()
                self.states[object_id] = "PICKED"
                self._record_failure(object_id, observation_step or 0)
                return

            self.current_object = None
            self.states[object_id] = "PLACED"
            if self.completion_evidence or self.pending_place_mode == "atomic_container":
                self._verify_placement(object_id)
                return

            item = (known_objects or {}).get(object_id, {})
            is_fresh = observation_step is not None and item.get("position_seen_step") == observation_step
            position = _object_position(item) if is_fresh else None
            if position is None or self.pending_place_target is None:
                return
            if math.dist(position, self.pending_place_target) <= PLACEMENT_TOLERANCE:
                self._verify_placement(object_id)
            else:
                self._clear_pending_place()
                self._record_failure(object_id, observation_step or 0)

    def record_action(
        self,
        action: dict[str, Any],
        result: Any,
        canonical_object_id: str = "",
        step: int = 0,
    ) -> None:
        name = str(action.get("action") or "").lower()
        object_id = canonical_object_id or self.current_object or ""
        attempt = (
            self.object_attempt_registry.setdefault(object_id, ObjectAttemptState(object_id)) if object_id else None
        )
        if attempt is not None:
            attempt.last_action = name
            attempt.last_result = "failure" if _failed(result) else "success"
            if name == "move_and_take_object":
                attempt.pick_attempts += 1
            elif name in {"put_down_sth", "move_and_put_down", "move_and_put_down_object_in_container"}:
                attempt.put_attempts += 1
        if _failed(result):
            if object_id and name == "move_and_take_object":
                error_text = (
                    str(result.get("error") or result.get("message") or "").lower()
                    if isinstance(result, dict)
                    else ""
                )
                if "not pickup" in error_text or "cannot take" in error_text or "can not take" in error_text:
                    # Real TongSIM uses this error for scene objects that are not
                    # physically pickable.  Treat it as negative object evidence
                    # instead of poisoning the episode with an unresolved failure.
                    self.candidate_items.discard(object_id)
                    self.uncertain_items.discard(object_id)
                    self.failed_objects.discard(object_id)
                    self.rejected_items.add(object_id)
                    self.states[object_id] = "REJECTED_NON_PICKABLE"
                    if attempt is not None:
                        attempt.failed = True
                    return
            if object_id:
                self._record_failure(object_id, step)
            return
        if name in {"move_to_object", "look_at_object"} and object_id:
            self.states[object_id] = "APPROACHING"
        elif name == "move_and_take_object" and object_id:
            self.current_object = object_id
            self.pending_pick = object_id
            self.states[object_id] = "APPROACHING"
        elif name in {"move_to_location", "move_forward", "move_backward"} and self.current_object:
            self.states[self.current_object] = "MOVING"
        elif name in {"put_down_sth", "move_and_put_down", "move_and_put_down_object_in_container"}:
            if self.current_object:
                self.pending_place = self.current_object
                self.pending_place_target = self._placement_target(action)
                self.pending_place_mode = (
                    "atomic_container" if name == "move_and_put_down_object_in_container" else "coordinate"
                )
                self.states[self.current_object] = "PLACED"
        elif name in {
            "pour_water",
            "slice_food",
            "wash_hands",
            "wash_object_in_hand",
            "mop_floor",
            "sit_down_to_object",
            "rest",
        }:
            self.completed_goal_actions += 1

    def _record_failure(self, object_id: str, step: int = 0) -> None:
        self.retries[object_id] += 1
        self.states[object_id] = "DISCOVERED"
        attempt = self.object_attempt_registry.setdefault(object_id, ObjectAttemptState(object_id))
        attempt.cooldown_until_step = max(attempt.cooldown_until_step, step + 2)
        if self.retries[object_id] >= self.retry_limit or attempt.pick_attempts >= MAX_PICK_ATTEMPTS:
            self.failed_objects.add(object_id)
            attempt.failed = True

    @staticmethod
    def _placement_target(action: dict[str, Any]) -> list[float] | None:
        params = action.get("parameters") or {}
        for key in ("put_target_location", "put_location", "target_location"):
            location = _location(params.get(key))
            if location is not None:
                return location
        return None

    def _verify_placement(self, object_id: str) -> None:
        self.completed_objects.add(object_id)
        self.expected_objects.discard(object_id)
        self.states[object_id] = "VERIFIED"
        attempt = self.object_attempt_registry.setdefault(object_id, ObjectAttemptState(object_id))
        attempt.completed = True
        attempt.failed = False
        self._clear_pending_place()

    def _clear_pending_place(self) -> None:
        self.pending_place = None
        self.pending_place_target = None
        self.pending_place_mode = ""

    def can_finish(self, task_type: str, object_in_hand: bool) -> tuple[bool, str]:
        if task_type not in {"tidyroom", "jigsaw"}:
            return True, "not a behavior task"
        if object_in_hand or self.current_object or self.pending_pick or self.pending_place:
            return False, "an object is held or awaiting pick/place verification"
        if self.completion_evidence:
            return True, "official task response reports completion"
        unverified = self.initial_expected_objects - self.completed_objects
        if unverified:
            return False, f"official objects lack verified completion: {sorted(unverified)}"
        if self.initial_expected_objects:
            return True, "all official objects have observation-verified completion"
        unresolved = self.remaining_candidates()
        if unresolved:
            return False, f"high-confidence clutter remains unresolved: {sorted(unresolved)}"
        if self.uncertain_items:
            return False, f"current view still contains objects requiring VLM review: {sorted(self.uncertain_items)}"
        if self.failed_objects:
            return False, f"failed clutter remains unresolved: {sorted(self.failed_objects)}"
        if self.coverage_verified:
            return True, "bounded final scans found no high-confidence clutter"
        return False, "task coverage is unknown and no official completion evidence exists"

    def context(self) -> dict[str, Any]:
        return {
            "states": dict(sorted(self.states.items())),
            "initial_objects": sorted(self.initial_expected_objects),
            "completed_objects": sorted(self.completed_objects),
            "failed_objects": sorted(self.failed_objects),
            "current_object": self.current_object,
            "pending_pick": self.pending_pick,
            "pending_place": self.pending_place,
            "pending_place_target": self.pending_place_target,
            "pending_place_mode": self.pending_place_mode,
            "retry_count": dict(sorted(self.retries.items())),
            "remaining_objects": sorted(self.expected_objects),
            "completed_goal_actions": self.completed_goal_actions,
            "completion_evidence": self.completion_evidence,
            "candidate_items": sorted(self.candidate_items - self.completed_objects),
            "rejected_items": sorted(self.rejected_items),
            "uncertain_items": sorted(self.uncertain_items),
            "target_surfaces": sorted(self.target_surfaces),
            "target_assignments": dict(sorted(self.target_assignments.items())),
            "vlm_reviews": dict(sorted(self.vlm_reviews.items())),
            "review_attempts": dict(sorted(self.review_attempts.items())),
            "object_attempt_registry": {
                object_id: attempt.context()
                for object_id, attempt in sorted(self.object_attempt_registry.items())
            },
            "final_scan_count": self.final_scan_count,
            "coverage_verified": self.coverage_verified,
        }

    def prompt_context(self, limit: int = 18) -> dict[str, Any]:
        """Bound the model-facing state while retaining all control facts."""
        context = self.context()
        context["candidate_items"] = context["candidate_items"][:limit]
        context["rejected_count"] = len(self.rejected_items)
        context["uncertain_count"] = len(self.uncertain_items)
        context["rejected_items"] = context["rejected_items"][:5]
        context["uncertain_items"] = context["uncertain_items"][:5]
        context["target_surfaces"] = context["target_surfaces"][:limit]
        context["vlm_reviews"] = dict(list(context["vlm_reviews"].items())[-limit:])
        relevant_attempts = set(context["candidate_items"]) | self.completed_objects
        context["object_attempt_registry"] = {
            key: value
            for key, value in context["object_attempt_registry"].items()
            if key in relevant_attempts
        }
        context["target_assignments"] = {
            key: value
            for key, value in context["target_assignments"].items()
            if key in relevant_attempts
        }
        return context
