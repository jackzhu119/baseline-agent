from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_PROMPT_OBJECTS = 18
_UNKNOWN_VALUES = {"", "unknown", "none", "null", "unlabeled", "unlabelled", "未知"}
_CLUTTER_TERMS = {
    "shoe",
    "shoes",
    "sneaker",
    "boot",
    "boots",
    "slipper",
    "cup",
    "mug",
    "bottle",
    "food",
    "apple",
    "banana",
    "bread",
    "trash",
    "garbage",
    "rubbish",
    "litter",
    "pillow",
    "cushion",
    "鞋",
    "靴",
    "拖鞋",
    "杯",
    "瓶",
    "食物",
    "食品",
    "垃圾",
    "枕头",
    "抱枕",
}
_STATIC_TERMS = {
    "wall",
    "floor",
    "ceiling",
    "room",
    "bounds",
    "furniture",
    "chair",
    "table",
    "desk",
    "sofa",
    "cabinet",
    "shelf",
    "rack",
    "bed",
    "door",
    "window",
    "counter",
    "architecture",
    "marker",
    "geometry",
    "墙",
    "地板",
    "天花板",
    "房间",
    "家具",
    "椅",
    "桌",
    "沙发",
    "柜",
    "架",
    "床",
    "门",
    "窗",
}
_TARGET_TERMS = {
    "table",
    "desk",
    "sofa",
    "cabinet",
    "shelf",
    "rack",
    "bin",
    "basket",
    "storage",
    "area",
    "zone",
    "bed",
    "fridge",
    "refrigerator",
    "桌",
    "沙发",
    "柜",
    "架",
    "垃圾桶",
    "篮",
    "收纳",
    "区域",
}
_SEMANTIC_FIELDS = ("name", "semantic_type", "category", "type", "class", "label", "description")
_COMPACT_FIELDS = ("name", "semantic_type", "category", "type", "color", "shape", "position")


def object_id(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("object_id", "id", "objectId", "objectID"):
        if item.get(key) not in (None, ""):
            return str(item[key])
    return ""


def deduplicate_visible_objects(raw_objects: Any) -> list[dict[str, Any]]:
    """Merge same-frame duplicate IDs without mutating the perception payload."""
    merged: dict[str, dict[str, Any]] = {}
    for raw in raw_objects if isinstance(raw_objects, list) else []:
        current_id = object_id(raw)
        if not current_id:
            continue
        incoming = dict(raw)
        incoming["object_id"] = current_id
        existing = merged.setdefault(current_id, {})
        for key, value in incoming.items():
            if key not in existing or existing[key] in (None, "", "Unknown", "unknown", [], {}):
                existing[key] = value
        existing["object_id"] = current_id
    return list(merged.values())


def _semantic_text(item: dict[str, Any]) -> str:
    values = [str(item.get(field_name) or "").strip().lower() for field_name in _SEMANTIC_FIELDS]
    return " ".join(value for value in values if value)


def _terms(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text.lower()) if part}


def _contains_term(text: str, vocabulary: set[str]) -> bool:
    words = _terms(text)
    return bool(words & vocabulary) or any(term in text for term in vocabulary if not term.isascii())


@dataclass(slots=True)
class TidyObjectClassification:
    candidate_items: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected_items: dict[str, str] = field(default_factory=dict)
    uncertain_items: dict[str, str] = field(default_factory=dict)
    target_surfaces: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_count: int = 0
    deduplicated_count: int = 0

    def context(self) -> dict[str, Any]:
        return {
            "candidate_items": sorted(self.candidate_items),
            "rejected_items": dict(sorted(self.rejected_items.items())),
            "uncertain_items": dict(sorted(self.uncertain_items.items())),
            "target_surfaces": sorted(self.target_surfaces),
            "raw_count": self.raw_count,
            "deduplicated_count": self.deduplicated_count,
        }


@dataclass(slots=True)
class TidyTargetAssignment:
    object_id: str
    object_type: str
    source_location: list[float] | None = None
    candidate_targets: list[dict[str, Any]] = field(default_factory=list)
    selected_target: dict[str, Any] | None = None
    confidence: float = 0.0
    reason: str = "no semantically compatible placement target"

    def context(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "source_location": self.source_location,
            "candidate_targets": self.candidate_targets,
            "selected_target": self.selected_target,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


def _location(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        lowered = {str(key).lower(): raw for key, raw in value.items()}
        values = [lowered.get(axis) for axis in ("x", "y", "z")]
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        values = list(value[:3])
    else:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


class TidyTargetPlanner:
    """Create conservative object-to-target assignments from public semantics."""

    AUTO_SELECT_CONFIDENCE = 0.85
    UNIQUE_MARGIN = 0.08

    def assign(
        self,
        object_item: dict[str, Any],
        target_items: dict[str, dict[str, Any]],
        placement_candidates: list[dict[str, Any]],
    ) -> TidyTargetAssignment:
        current_id = object_id(object_item)
        object_text = _semantic_text(object_item)
        assignment = TidyTargetAssignment(
            object_id=current_id,
            object_type=object_text or "unknown",
            # Deliberately excludes place_location: for a clutter object it is
            # an interaction point, not evidence of the desired destination.
            source_location=_location(object_item.get("position") or object_item.get("location")),
        )
        placements = {str(item.get("object_id")): item for item in placement_candidates}
        ranked: list[dict[str, Any]] = []
        for target_id, target_item in target_items.items():
            semantic_score, reason = self._semantic_compatibility(object_text, _semantic_text(target_item))
            placement = placements.get(str(target_id))
            if semantic_score <= 0 or placement is None or _location(placement.get("location")) is None:
                continue
            source = str(placement.get("source") or "")
            location_confidence = float(placement.get("confidence") or 0)
            confidence = 0.65 * semantic_score + 0.35 * location_confidence
            # AABB-top is useful evidence but not precise enough for automatic
            # placement because held-object height and open surface area vary.
            if source != "place_location":
                confidence = min(confidence, 0.82)
            ranked.append(
                {
                    "object_id": str(target_id),
                    "location": _location(placement.get("location")),
                    "source": source,
                    "confidence": round(confidence, 3),
                    "reason": reason,
                }
            )
        ranked.sort(key=lambda item: (-float(item["confidence"]), str(item["object_id"])))
        assignment.candidate_targets = ranked
        if not ranked:
            return assignment
        best = ranked[0]
        runner_up = float(ranked[1]["confidence"]) if len(ranked) > 1 else 0.0
        assignment.confidence = float(best["confidence"])
        if assignment.confidence >= self.AUTO_SELECT_CONFIDENCE and assignment.confidence - runner_up >= self.UNIQUE_MARGIN:
            assignment.selected_target = best
            assignment.reason = str(best["reason"])
        else:
            assignment.reason = "target evidence is ambiguous or below the automatic-placement threshold"
        return assignment

    @staticmethod
    def _semantic_compatibility(object_text: str, target_text: str) -> tuple[float, str]:
        mappings = (
            ({"shoe", "shoes", "sneaker", "boot", "boots", "slipper", "鞋", "靴", "拖鞋"},
             {"shoe", "rack", "shelf", "area", "zone", "鞋", "架", "区域"}, 0.98, "shoe storage"),
            ({"trash", "garbage", "rubbish", "litter", "垃圾"},
             {"trash", "garbage", "bin", "垃圾桶", "垃圾"}, 1.0, "trash receptacle"),
            ({"pillow", "cushion", "枕头", "抱枕"},
             {"sofa", "bed", "pillow", "cushion", "沙发", "床", "枕头", "抱枕"}, 0.96, "soft-furnishing area"),
            ({"cup", "mug", "bottle", "杯", "瓶"},
             {"cup", "mug", "bottle", "table", "shelf", "area", "杯", "瓶", "桌", "架", "区域"}, 0.91, "drinkware area"),
            ({"food", "apple", "banana", "bread", "食物", "食品"},
             {"food", "storage", "cabinet", "fridge", "refrigerator", "食品", "食物", "收纳", "柜"}, 0.95, "food storage"),
        )
        for object_terms, target_terms, score, reason in mappings:
            if _contains_term(object_text, object_terms) and _contains_term(target_text, target_terms):
                return score, reason
        return 0.0, "no semantic object-to-target match"


class TidyObjectClassifier:
    """Classify only public perception fields; Unknown is never clutter by default."""

    def __init__(self, max_prompt_objects: int = DEFAULT_MAX_PROMPT_OBJECTS) -> None:
        self.max_prompt_objects = max(int(max_prompt_objects), 1)

    def classify(
        self,
        raw_objects: Any,
        *,
        required_ids: set[str] | None = None,
        completed_ids: set[str] | None = None,
        blacklisted_ids: set[str] | None = None,
    ) -> TidyObjectClassification:
        raw_count = len(raw_objects) if isinstance(raw_objects, list) else 0
        objects = deduplicate_visible_objects(raw_objects)
        required = {str(value) for value in (required_ids or set())}
        completed = {str(value) for value in (completed_ids or set())}
        blacklisted = {str(value) for value in (blacklisted_ids or set())}
        result = TidyObjectClassification(raw_count=raw_count, deduplicated_count=len(objects))
        for item in objects:
            current_id = object_id(item)
            semantic_text = _semantic_text(item)
            if current_id in completed:
                result.rejected_items[current_id] = "already observation-verified complete"
                continue
            if current_id in blacklisted:
                result.rejected_items[current_id] = "temporarily blacklisted after bounded failures"
                continue
            if current_id in required:
                result.candidate_items[current_id] = self._compact(item, "official task object")
                continue
            if _contains_term(semantic_text, _TARGET_TERMS):
                result.target_surfaces[current_id] = self._compact(item, "semantic target surface")
            if _contains_term(semantic_text, _STATIC_TERMS):
                result.rejected_items[current_id] = "static geometry or furniture"
                continue
            normalized = semantic_text.strip().lower()
            if normalized in _UNKNOWN_VALUES or not normalized or "unknown" in _terms(normalized):
                result.uncertain_items[current_id] = "missing reliable semantic class"
                continue
            if _contains_term(semantic_text, _CLUTTER_TERMS):
                result.candidate_items[current_id] = self._compact(item, "recognized clutter class")
                continue
            result.uncertain_items[current_id] = "class is not an approved high-confidence clutter type"
        return result

    def relevant_objects(
        self,
        classification: TidyObjectClassification,
        *,
        current_object_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        current_id = str(current_object_id or "")
        if current_id and current_id in classification.candidate_items:
            ordered.append(classification.candidate_items[current_id])
        for current_id_key in sorted(classification.candidate_items):
            if current_id_key != current_id:
                ordered.append(classification.candidate_items[current_id_key])
        ordered.extend(classification.target_surfaces[key] for key in sorted(classification.target_surfaces))
        return ordered[: self.max_prompt_objects]

    @staticmethod
    def _compact(item: dict[str, Any], evidence: str) -> dict[str, Any]:
        compact = {"object_id": object_id(item), "classification_evidence": evidence}
        for field_name in _COMPACT_FIELDS:
            value = item.get(field_name)
            if value not in (None, "", "Unknown", "unknown", [], {}):
                compact[field_name] = value
        return compact
