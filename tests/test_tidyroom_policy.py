from __future__ import annotations

from arenaagent.competition.solvers.tidyroom_policy import (
    DEFAULT_MAX_PROMPT_OBJECTS,
    TidyObjectClassifier,
    deduplicate_visible_objects,
)


def test_duplicate_object_ids_are_merged_once() -> None:
    objects = [
        {"object_id": "5", "name": "Unknown", "color": "Blue"},
        {"object_id": 5, "name": "shoe", "world_aabb": {"min": {}, "max": {}}},
    ]

    deduplicated = deduplicate_visible_objects(objects)

    assert len(deduplicated) == 1
    assert deduplicated[0]["object_id"] == "5"
    assert deduplicated[0]["name"] == "shoe"
    assert deduplicated[0]["color"] == "Blue"


def test_unknown_is_uncertain_and_static_geometry_is_rejected() -> None:
    result = TidyObjectClassifier().classify(
        [
            {"object_id": "20", "shape": "Unknown"},
            {"object_id": "30", "name": "wall"},
            {"object_id": "34", "name": "blue shoe"},
        ]
    )

    assert set(result.candidate_items) == {"34"}
    assert set(result.uncertain_items) == {"20"}
    assert result.rejected_items == {"30": "static geometry or furniture"}


def test_official_required_unknown_object_is_allowed_but_completed_is_not() -> None:
    classifier = TidyObjectClassifier()
    objects = [
        {"object_id": "11", "shape": "Unknown"},
        {"object_id": "12", "name": "red boot"},
    ]

    result = classifier.classify(objects, required_ids={"11"}, completed_ids={"12"})

    assert set(result.candidate_items) == {"11"}
    assert set(result.rejected_items) == {"12"}


def test_prompt_projection_excludes_unknown_aabb_and_is_bounded() -> None:
    objects = [
        {"object_id": f"unknown-{index}", "shape": "Unknown", "world_aabb": {"min": {}, "max": {}}}
        for index in range(50)
    ]
    objects.extend(
        {"object_id": f"shoe-{index}", "name": "shoe", "world_aabb": {"min": {}, "max": {}}}
        for index in range(25)
    )
    classifier = TidyObjectClassifier(max_prompt_objects=DEFAULT_MAX_PROMPT_OBJECTS)

    result = classifier.classify(objects)
    relevant = classifier.relevant_objects(result)

    assert len(relevant) == DEFAULT_MAX_PROMPT_OBJECTS
    assert all("world_aabb" not in item for item in relevant)
    assert all(item["object_id"].startswith("shoe-") for item in relevant)
