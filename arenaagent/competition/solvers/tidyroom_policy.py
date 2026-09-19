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
