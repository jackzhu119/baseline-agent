from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

PLACEMENT_TOLERANCE = 15.0
VECTOR_DIMENSIONS = 3


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

    def update_classification(
        self,
        *,
        candidate_items: set[str],
        rejected_items: set[str],
        uncertain_items: set[str],
    ) -> None:
        self.candidate_items.update(candidate_items - self.completed_objects)
        self.rejected_items.update(rejected_items)
        self.uncertain_items.update(uncertain_items)
        self.uncertain_items.difference_update(self.candidate_items | self.rejected_items | self.completed_objects)
        for object_id in self.candidate_items:
            self.states.setdefault(object_id, "DISCOVERED")

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
                self._record_failure(object_id)
        if self.pending_place is not None:
            object_id = self.pending_place
            if has_object:
                self._clear_pending_place()
                self.states[object_id] = "PICKED"
                self._record_failure(object_id)
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
                self._record_failure(object_id)

    def record_action(self, action: dict[str, Any], result: Any, canonical_object_id: str = "") -> None:
        name = str(action.get("action") or "").lower()
        object_id = canonical_object_id or self.current_object or ""
        if _failed(result):
            if object_id:
                self._record_failure(object_id)
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

    def _record_failure(self, object_id: str) -> None:
        self.retries[object_id] += 1
        self.states[object_id] = "DISCOVERED"
        if self.retries[object_id] > self.retry_limit:
            self.failed_objects.add(object_id)

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
        }
