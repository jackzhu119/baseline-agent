from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arenaagent.preliminary_baseline_agent.preliminary_baseline_agent import (
    PreliminaryBaselineAgent,
    PreliminaryBaselineAgentCfg,
)
from arenaagent.vlm_agent.client import ClientResponse
from arenaagent.vlm_agent.prompt import PromptGenerator


class FakeTongSim:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def acquire_first_person_perception(self, character_id, width=None, height=None):
        return {"image": None, "objects": [{"object_id": "1", "name": "cup", "color": "Red"}]}

    def has_object_in_hand(self, character_id):
        return False, None

    def move_and_take_object(self, character_id, object_id, **kwargs):
        self.calls.append(("move_and_take_object", object_id))
        return {"result": "success"}

    def turn_in_degree(self, character_id, degree):
        self.calls.append(("turn_in_degree", degree))
        return {"result": "success"}

    def look_at_object(self, character_id, object_id, is_cancel=False):
        self.calls.append(("look_at_object", object_id))
        return {"result": "success"}


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return ClientResponse(
            text=self.text,
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ClientResponse(text=response, token_usage={"total_tokens": 1})


class RaisingClient:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise TimeoutError("model timeout")


class ErrorResponseClient:
    def __init__(self, error: str) -> None:
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return ClientResponse(text="", error=self.error)


class RaisingTongSim(FakeTongSim):
    def acquire_first_person_perception(self, character_id, width=None, height=None):
        raise ConnectionError("perception unavailable")


class JigsawFailureTongSim(FakeTongSim):
    def acquire_first_person_perception(self, character_id, width=None, height=None):
        return {
            "image": None,
            "objects": [
                {"object_id": "1", "name": "piece", "position": [99, 25, 25]},
                {"object_id": "placed", "name": "reference", "position": [10, 25, 25]},
            ],
        }


class JigsawHoldingTongSim(FakeTongSim):
    def acquire_first_person_perception(self, character_id, width=None, height=None):
        return {
            "image": None,
            "objects": [
                {"object_id": "piece-a", "name": "piece", "position": [99, 25, 25]},
                {"object_id": "reference-1", "name": "reference", "position": [10, 25, 25]},
                {"object_id": "reference-2", "name": "reference", "position": [10, 75, 25]},
                {"object_id": "reference-3", "name": "reference", "position": [10, 25, 75]},
            ],
        }

    def has_object_in_hand(self, character_id):
        return True, "piece-a"

    def put_down_sth(
        self,
        character_id,
        target_location,
        target_rotation=None,
        auto_rotate=False,
        force_locate=False,
    ):
        self.calls.append(("put_down_sth", list(target_location)))
        return {"result": "success"}


class AgentIntegrationTests(unittest.TestCase):
    def make_agent(self, response_text: str) -> tuple[PreliminaryBaselineAgent, FakeTongSim, FakeClient]:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        cfg = PreliminaryBaselineAgentCfg(log_dir=self.temp_dir.name)
        agent = PreliminaryBaselineAgent(stub=None, channel=None, cfg=cfg)
        tongsim = FakeTongSim()
        client = FakeClient(response_text)
        agent._initialized = True
        agent.tongsim = tongsim
        agent.character_id = "character-1"
        agent.vlm_client = client
        agent.prompt_generator = PromptGenerator(agent.cfg.vlm_config.prompt_config)
        agent.action_space = {"key": "answer"}
        return agent, tongsim, client

    def test_hallucinated_id_never_reaches_object_api_and_uses_safe_fallback(self) -> None:
        agent, tongsim, client = self.make_agent(
            '[{"think":"guess","action":"move_and_take_object","parameters":{"object_id":"999"},"output":0}]'
        )
        result = agent.run_step({"task_type": "unknown", "subject": "测试动作校验"}, {})
        self.assertEqual(client.calls, 2)
        self.assertEqual(tongsim.calls, [("turn_in_degree", 45.0)])
        self.assertEqual(result["result"], "success")
        self.assertEqual(agent._competition.metrics.invalid_actions, 2)
        self.assertEqual(agent._competition.metrics.model_failure_count, 2)

    def test_prompt_renders_action_schema_and_world_state(self) -> None:
        agent, _, _ = self.make_agent("not used")
        agent._competition.ensure_episode({"task_type": "counting", "subject": "计数"})
        agent._competition.observe([{"object_id": "1", "color": "Red"}])
        variables = agent._build_prompt_variables(
            {"task_type": "counting", "subject": "计数"},
            {},
            {"submit_answer": {"sig": "(output:any)"}},
            [{"object_id": "1", "color": "Red"}],
            False,
        )
        messages = agent.prompt_generator.Generate(variables=variables)
        self.assertIn("submit_answer", messages[0]["content"])
        self.assertNotIn("{self.api_info}", messages[0]["content"])
        user_text = messages[-1]["content"][1]["text"]
        self.assertIn("unique_objects_seen", user_text)
        self.assertNotIn("{self.competition_state}", user_text)

    def test_api_schema_exposes_all_primary_runtime_actions(self) -> None:
        agent, _, _ = self.make_agent("not used")
        api_info = agent._load_api_info()
        primary_actions = {
            "finish_task",
            "look_at_location",
            "look_at_object",
            "point_at_object",
            "pour_water",
            "sit_down_to_object",
            "slice_food",
            "wash_hands",
            "wash_object_in_hand",
            "mop_floor",
            "rest",
        }
        self.assertEqual(primary_actions - set(api_info), set())

    def test_history_drops_stale_base64_images_but_keeps_text(self) -> None:
        agent, _, _ = self.make_agent("not used")
        agent._append_history_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                        {"type": "text", "text": "current evidence"},
                    ],
                },
                {"role": "assistant", "content": "one action"},
            ]
        )
        serialized = str(agent.history_messages)
        self.assertNotIn("base64", serialized)
        self.assertNotIn("image_url", serialized)
        self.assertIn("current evidence", serialized)

    def test_visible_id_reaches_tongsim(self) -> None:
        agent, tongsim, _ = self.make_agent(
            '[{"think":"visible","action":"move_and_take_object","parameters":{"object_id":"1"},"output":0}]'
        )
        result = agent.run_step({"task_type": "unknown", "subject": "测试可见物体动作"}, {})
        self.assertEqual(result["result"], "success")
        self.assertEqual(tongsim.calls, [("move_and_take_object", "1")])

    def test_raven_routes_to_local_solver_without_llm_call(self) -> None:
        agent, _, client = self.make_agent("not used")
        with patch("arenaagent.vlm_agent.vlm_agent.handle_raven_skill", return_value={"answer": [1, 2, 3]}):
            result = agent.run_step({"task_type": "raven", "subject": "瑞文", "task_data": {}}, {})
        self.assertEqual(client.calls, 0)
        self.assertEqual(result, {"answer": [1, 2, 3]})

    def test_official_evaluation_writes_verified_episode(self) -> None:
        agent, _, _ = self.make_agent(
            '[{"think":"visible","action":"move_and_take_object","parameters":{"object_id":"1"},"output":0}]'
        )
        agent.run_step({"task_id": "verified-1", "task_type": "tidyroom", "subject": "整理房间"}, {})
        agent._on_subject_evaluated({"success": True, "score": 1})
        path = Path(self.temp_dir.name) / "metrics" / "episode_verified-1.json"
        self.assertTrue(path.exists())
        payload = path.read_text(encoding="utf-8")
        self.assertIn('"verified": true', payload)
        self.assertIn('"prompt_fingerprint"', payload)
        self.assertNotIn('"api_key"', payload)

    def test_model_timeout_uses_short_context_then_safe_fallback(self) -> None:
        agent, tongsim, _ = self.make_agent("not used")
        client = RaisingClient()
        agent.vlm_client = client
        result = agent.run_step({"task_id": "timeout-1", "task_type": "unknown", "subject": "测试恢复"}, {})
        self.assertEqual(client.calls, 2)
        self.assertEqual(tongsim.calls, [("turn_in_degree", 45.0)])
        self.assertEqual(result["result"], "success")
        self.assertEqual(agent._competition.metrics.model_errors, 2)
        self.assertEqual(agent._competition.metrics.model_failure_count, 2)
        self.assertEqual(agent._competition.metrics.steps, 1)

    def test_perception_failure_is_recoverable(self) -> None:
        agent, _, client = self.make_agent("not used")
        agent.tongsim = RaisingTongSim()
        result = agent.run_step({"task_id": "vision-1", "task_type": "counting", "subject": "计数"}, {})
        self.assertEqual(client.calls, 0)
        self.assertTrue(result["recoverable"])
        self.assertEqual(result["failure_class"], "PERCEPTION_ERROR")
        self.assertEqual(agent._competition.metrics.perception_errors, 1)
        self.assertEqual(agent._competition.metrics.vision_calls, 1)
        self.assertEqual(agent._competition.first_failure["step"], 1)

    def test_rate_limit_uses_bounded_recovery_and_safe_fallback(self) -> None:
        agent, tongsim, _ = self.make_agent("not used")
        client = ErrorResponseClient("rate limit")
        agent.vlm_client = client
        result = agent.run_step({"task_id": "rate-1", "task_type": "unknown", "subject": "测试恢复"}, {})
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["result"], "success")
        self.assertEqual(tongsim.calls, [("turn_in_degree", 45.0)])
        self.assertEqual(agent._competition.metrics.model_failure_count, 2)

    def test_safe_fallback_does_not_place_on_low_confidence_surface(self) -> None:
        agent, _, _ = self.make_agent("not used")
        runtime = agent._competition
        runtime.ensure_episode({"task_id": "placement-confidence", "task_type": "tidyroom", "movable_object_id": ["1"]})
        runtime.observe(
            [
                {"object_id": "1", "world_aabb": {"min": {"X": 0, "Y": 0, "Z": 0}, "max": {"X": 2, "Y": 2, "Z": 2}}},
                {
                    "object_id": "table",
                    "world_aabb": {
                        "min": {"X": 0, "Y": 0, "Z": 0},
                        "max": {"X": 20, "Y": 20, "Z": 10},
                    },
                },
            ]
        )
        runtime.record_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "1"}}, {"result": "success"}
        )
        runtime.update_hand_state(True)
        action = agent._safe_fallback_action(object_in_hand=True)
        self.assertIsNotNone(action)
        self.assertEqual(action["action"], "turn_in_degree")

    def test_counting_uses_bounded_scan_then_deterministic_submit(self) -> None:
        agent, tongsim, client = self.make_agent("model must not be called")
        subject = {"task_id": "count-1", "task_type": "counting", "subject": "有多少红色物体"}
        for _ in range(3):
            result = agent.run_step(subject, {})
            self.assertEqual(result["result"], "success")
        result = agent.run_step(subject, {})
        self.assertEqual(result, {"answer": "1"})
        self.assertEqual(client.calls, 0)
        self.assertEqual(
            tongsim.calls,
            [("turn_in_degree", 90.0), ("turn_in_degree", 180.0), ("turn_in_degree", 270.0)],
        )

    def test_jigsaw_failure_advances_rotation_before_same_turn_prompt(self) -> None:
        agent, _, _ = self.make_agent('[{"action":"turn_in_degree","parameters":{"degree":45},"output":0}]')
        agent.tongsim = JigsawFailureTongSim()
        subject = {
            "task_id": "jigsaw-retry-1",
            "task_type": "jigsaw",
            "subject": "完成拼图",
            "reference_bounding": [0, 100, 100, 0],
            "rows": 2,
            "columns": 2,
            "piece_object_id": ["1"],
        }
        agent._competition.ensure_episode(subject)
        agent._competition.observe([{"object_id": "1", "position": [99, 25, 25]}])
        agent._competition.record_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "1"}},
            {"result": "success"},
        )
        agent._competition.update_hand_state(True)
        agent._competition.record_action(
            {"action": "put_down_sth", "parameters": {"target_location": [10, 75, 25]}},
            {"result": "success"},
        )

        result = agent.run_step(subject, {})

        self.assertEqual(result["result"], "success")
        piece = agent._competition.strategy_context["pieces"][0]
        self.assertEqual(piece["rotation_attempt"], 1)
        self.assertEqual(piece["candidate_rotation"]["yaw"], 90.0)

    def test_high_confidence_jigsaw_placement_bypasses_vlm(self) -> None:
        agent, _, client = self.make_agent("not used")
        tongsim = JigsawHoldingTongSim()
        agent.tongsim = tongsim
        subject = {
            "task_id": "jigsaw-direct-1",
            "task_type": "jigsaw",
            "subject": "完成拼图",
            "reference_bounding": [0, 100, 100, 0],
            "rows": 2,
            "columns": 2,
            "piece_object_id": ["piece-a"],
        }
        agent._competition.ensure_episode(subject)
        agent._competition.observe([{"object_id": "piece-a", "position": [99, 25, 25]}])
        agent._competition.record_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "piece-a"}},
            {"result": "success"},
        )

        result = agent.run_step(subject, {})

        self.assertEqual(result["result"], "success")
        self.assertEqual(client.calls, 0)
        self.assertEqual(tongsim.calls, [("put_down_sth", [10.0, 75.0, 75.0])])

    def test_invalid_model_action_gets_one_bounded_repair(self) -> None:
        agent, tongsim, _ = self.make_agent("not used")
        agent.vlm_client = SequenceClient(
            [
                '[{"action":"move_and_take_object","parameters":{"object_id":"999"},"output":0}]',
                '[{"action":"move_and_take_object","params":{"object_id":"1"},"output":0}]',
            ]
        )
        result = agent.run_step({"task_id": "repair-1", "task_type": "unknown", "subject": "测试修复"}, {})
        self.assertEqual(result["result"], "success")
        self.assertEqual(agent.vlm_client.calls, 2)
        self.assertEqual(tongsim.calls, [("move_and_take_object", "1")])

    def test_new_episode_clears_previous_action_memory(self) -> None:
        agent, _, _ = self.make_agent('[{"action":"move_and_take_object","parameters":{"object_id":"1"},"output":0}]')
        agent.run_step({"task_id": "episode-a", "task_type": "tidyroom", "subject": "整理房间"}, {})
        self.assertTrue(agent._action_histories)
        agent.run_step({"task_id": "episode-b", "task_type": "tidyroom", "subject": "整理房间"}, {})
        self.assertEqual(len(agent._action_histories), 1)

    def test_prompt_schema_is_compressed_to_task_relevant_actions(self) -> None:
        agent, _, _ = self.make_agent("not used")
        api_info = agent._relevant_api_info(agent._load_api_info(), "npc")
        self.assertEqual(set(api_info), {"move_to_npc", "speak_to_npc", "submit_answer"})
        self.assertNotIn("slice_food", api_info)

    def test_prompt_context_size_is_recorded_without_image_payload(self) -> None:
        agent, _, _ = self.make_agent('[{"action":"move_and_take_object","parameters":{"object_id":"1"},"output":0}]')
        agent.run_step({"task_id": "prompt-size", "task_type": "unknown", "subject": "测试提示"}, {})
        self.assertGreater(agent._competition.metrics.prompt_chars, 0)
        self.assertEqual(agent._competition.metrics.prompt_chars, agent._competition.metrics.max_prompt_chars)

    def test_tidyroom_unknown_is_focused_before_visual_review(self) -> None:
        response = (
            '{"objects":[{"object_id":"1","role":"clutter","category":"cup","confidence":0.96,"target_id":null}]}'
        )
        agent, tongsim, client = self.make_agent(response)
        tongsim.acquire_first_person_perception = lambda *args, **kwargs: {
            "image": "QUJD",
            "objects": [{"object_id": "1", "name": "Unknown", "color": "Red"}],
        }
        subject = {"task_id": "focused-review", "task_type": "tidyroom", "subject": "整理房间"}

        first = agent.run_step(subject, {})
        second = agent.run_step(subject, {})

        self.assertEqual(first["result"], "success")
        self.assertEqual(second["result"], "success")
        self.assertEqual(tongsim.calls[0], ("look_at_object", "1"))
        self.assertEqual(client.calls, 1)
        self.assertEqual(agent._competition.objects["1"]["vlm_semantic_type"], "cup")


if __name__ == "__main__":
    unittest.main()
