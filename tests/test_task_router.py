from __future__ import annotations

import unittest

from arenaagent.competition.runtime import CompetitionRuntime
from arenaagent.competition.task_router import TaskStrategyRouter


class TaskStrategyRouterTests(unittest.TestCase):
    def test_high_confidence_held_jigsaw_piece_is_placed_without_vlm(self) -> None:
        subject = {
            "task_id": "jigsaw-router-1",
            "task_type": "jigsaw",
            "subject": "完成拼图",
            "reference_bounding": [0, 100, 100, 0],
            "rows": 2,
            "columns": 2,
            "piece_object_id": ["piece-a"],
        }
        runtime = CompetitionRuntime()
        runtime.ensure_episode(subject)
        runtime.observe(
            [
                {"object_id": "reference-1", "position": [10, 25, 25]},
                {"object_id": "reference-2", "position": [10, 75, 25]},
                {"object_id": "reference-3", "position": [10, 25, 75]},
                {"object_id": "piece-a", "position": [99, 25, 25]},
            ]
        )
        runtime.record_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "piece-a"}},
            {"result": "success"},
        )
        runtime.update_hand_state(True)
        router = TaskStrategyRouter()
        router.observe(runtime, subject)

        action = router.propose_action(subject, runtime)

        self.assertIsNotNone(action)
        self.assertEqual(action["action"], "put_down_sth")
        self.assertEqual(action["parameters"]["target_location"], [10.0, 75.0, 75.0])
        self.assertEqual(
            action["parameters"]["target_rotation"],
            {"roll": 0.0, "yaw": 0.0, "pitch": 0.0},
        )
        self.assertTrue(runtime.validate_action(action, object_in_hand=True).valid)

    def test_incomplete_jigsaw_evidence_falls_back_to_vlm(self) -> None:
        subject = {
            "task_id": "jigsaw-router-2",
            "task_type": "jigsaw",
            "subject": "完成拼图",
            "reference_bounding": [0, 100, 100, 0],
            "rows": 2,
            "columns": 2,
            "piece_object_id": ["piece-a"],
        }
        runtime = CompetitionRuntime()
        runtime.ensure_episode(subject)
        runtime.observe([{"object_id": "piece-a", "position": [99, 25, 25]}])
        runtime.record_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "piece-a"}},
            {"result": "success"},
        )
        runtime.update_hand_state(True)
        router = TaskStrategyRouter()
        router.observe(runtime, subject)

        self.assertIsNone(router.propose_action(subject, runtime))

    def test_npc_router_asks_highest_value_missing_fact_without_vlm(self) -> None:
        subject = {
            "task_id": "npc-router-1",
            "task_type": "npc",
            "subject": "钥匙在哪里？",
            "npc_asset_name": {"刘伟东": "npc_liu", "赵爷爷": "npc_zhao"},
        }
        runtime = CompetitionRuntime()
        runtime.ensure_episode(subject)
        runtime.npc_memory.ingest_fact("钥匙是赵爷爷的。")
        router = TaskStrategyRouter()
        router.observe(runtime, subject)
        action = router.propose_action(subject, runtime)
        self.assertEqual(action["action"], "speak_to_npc")
        self.assertEqual(action["parameters"], {"npc_name": "赵爷爷", "message": "钥匙在哪里？"})
        self.assertTrue(runtime.validate_action(action).valid)

    def test_npc_router_submits_only_after_required_fact_is_known(self) -> None:
        subject = {
            "task_id": "npc-router-2",
            "task_type": "npc",
            "subject": "钥匙在哪里？",
            "npc_asset_name": {"赵爷爷": "npc_zhao"},
        }
        runtime = CompetitionRuntime()
        runtime.ensure_episode(subject)
        runtime.npc_memory.record_exchange("赵爷爷", "钥匙在哪里？", "钥匙在餐桌上。")
        router = TaskStrategyRouter()
        router.observe(runtime, subject)
        action = router.propose_action(subject, runtime)
        self.assertEqual(action["action"], "submit_answer")
        self.assertEqual(action["output"], "餐桌上")

    def test_tidyroom_runs_pick_and_put_from_semantic_assignment_without_vlm(self) -> None:
        runtime = CompetitionRuntime()
        subject = {"task_type": "tidyroom", "subject": "整理房间"}
        runtime.ensure_episode(subject)
        runtime.observe(
            [
                {"object_id": "shoe-1", "name": "blue shoe", "position": [1, 2, 3]},
                {"object_id": "rack-1", "name": "shoe rack", "place_location": [10, 20, 3]},
            ]
        )
        router = TaskStrategyRouter()
        router.observe(runtime, subject)

        pick = router.propose_action(subject, runtime)
        self.assertEqual(pick["action"], "move_and_take_object")
        runtime.record_action(pick, {"result": "success"})
        runtime.update_hand_state(True)
        router.observe(runtime, subject)
        put = router.propose_action(subject, runtime)

        self.assertEqual(put["action"], "put_down_sth")
        self.assertEqual(put["parameters"]["target_location"], [10.0, 20.0, 3.0])

    def test_tidyroom_finishes_after_bounded_empty_coverage_scans(self) -> None:
        runtime = CompetitionRuntime()
        subject = {"task_type": "tidyroom", "subject": "整理房间"}
        runtime.ensure_episode(subject)
        runtime.observe([])
        router = TaskStrategyRouter()
        router.observe(runtime, subject)

        self.assertEqual(router.propose_action(subject, runtime)["action"], "turn_in_degree")
        self.assertEqual(router.propose_action(subject, runtime)["action"], "turn_in_degree")
        finish = router.propose_action(subject, runtime)

        self.assertEqual(finish["action"], "finish_task")
        self.assertTrue(runtime.validate_action(finish, object_in_hand=False).valid)

    def test_tidyroom_does_not_finish_with_unresolved_high_confidence_clutter(self) -> None:
        runtime = CompetitionRuntime()
        subject = {"task_type": "tidyroom", "subject": "整理房间"}
        runtime.ensure_episode(subject)
        runtime.observe([{"object_id": "shoe-1", "name": "shoe", "position": [1, 2, 3]}])
        router = TaskStrategyRouter()
        router.observe(runtime, subject)

        self.assertIsNone(router.propose_action(subject, runtime))
        finish = runtime.validate_action({"action": "finish_task", "parameters": {}, "output": 0})
        self.assertFalse(finish.valid)
        self.assertIn("unresolved", finish.error)


if __name__ == "__main__":
    unittest.main()
