from __future__ import annotations

import json

from arenaagent.competition.solvers.tidyroom_policy import TidyObjectClassifier
from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import (
    PreliminaryBaselineAgent,
    PreliminaryBaselineAgentCfg,
)
from arenaagent.vlm_agent.prompt import PromptGenerator


def _large_scene() -> list[dict[str, object]]:
    bounds = {"min": {"X": 0, "Y": 0, "Z": 0}, "max": {"X": 1, "Y": 1, "Z": 1}}
    duplicate_shoe = [
        {"object_id": "5", "name": "shoe", "color": "blue", "world_aabb": bounds, "rotation": [0, 0, 0]}
        for _ in range(10)
    ]
    unknown = [
        {"object_id": f"unknown-{index}", "shape": "Unknown", "world_aabb": bounds, "rotation": [0, 0, 0]}
        for index in range(50)
    ]
    static = [
        {"object_id": f"wall-{index}", "name": "wall", "world_aabb": bounds, "rotation": [0, 0, 0]}
        for index in range(10)
    ]
    return [*duplicate_shoe, *unknown, *static]


def test_tidyroom_full_prompt_deduplicates_and_stays_within_offline_budget(tmp_path) -> None:
    agent = PreliminaryBaselineAgent(None, None, PreliminaryBaselineAgentCfg(log_dir=str(tmp_path)))
    agent.prompt_generator = PromptGenerator(agent.cfg.vlm_config.prompt_config)
    subject = {"task_type": "tidyroom", "subject": "整理房间"}
    agent._competition.ensure_episode(subject)
    raw = _large_scene()
    classifier = TidyObjectClassifier()
    relevant = classifier.relevant_objects(classifier.classify(raw))

    def render(objects: list[dict[str, object]]) -> str:
        variables = agent._build_prompt_variables(subject, {}, {}, objects, False)
        messages = agent.prompt_generator.Generate(variables=variables)
        return json.dumps(messages, ensure_ascii=False)

    raw_prompt = render(raw)
    compact_prompt = render(relevant)
    ids = [item["object_id"] for item in relevant]

    assert ids == ["5"]
    assert len(relevant) <= classifier.max_prompt_objects
    assert "world_aabb" not in compact_prompt
    assert len(compact_prompt) <= len(raw_prompt) * 0.6
    # This is deliberately an offline estimate, not provider-reported tokens.
    assert len(compact_prompt) / 4 <= (len(raw_prompt) / 4) * 0.6
