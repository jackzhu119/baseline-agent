from __future__ import annotations

import unittest

from arenaagent.competition.runtime import CompetitionRuntime


class TidyRoomStateTests(unittest.TestCase):
    def make_runtime(self) -> CompetitionRuntime:
        runtime = CompetitionRuntime(repeated_action_limit=2)
        runtime.ensure_episode({"task_type": "tidyroom", "subject": "整理房间", "movable_object_id": ["7"]})
        runtime.observe([{"object_id": "7", "name": "cup"}])
        return runtime

    def test_pick_and_place_require_hand_state_verification(self) -> None:
        runtime = self.make_runtime()
        self.assertEqual(runtime.progress.context()["states"]["7"], "TARGET_IDENTIFIED")
        take = {"action": "move_and_take_object", "parameters": {"object_id": "7"}, "output": 0}
        runtime.record_action(take, {"result": "success"})
        self.assertEqual(runtime.progress.context()["pending_pick"], "7")
        runtime.update_hand_state(True)
        self.assertEqual(runtime.progress.context()["states"]["7"], "PICKED")

        put = {"action": "put_down_sth", "parameters": {"target_location": [1, 2, 3]}, "output": 0}
        runtime.record_action(put, {"result": "success"})
        self.assertEqual(runtime.progress.context()["states"]["7"], "PLACED")
        runtime.update_hand_state(False)
        self.assertEqual(runtime.progress.context()["states"]["7"], "PLACED")
        self.assertEqual(runtime.progress.context()["remaining_objects"], ["7"])
        runtime.observe([{"object_id": "7", "name": "cup", "position": [1, 2, 3]}])
        runtime.update_hand_state(False)
        self.assertEqual(runtime.progress.context()["states"]["7"], "VERIFIED")
        self.assertEqual(runtime.progress.context()["remaining_objects"], [])

    def test_wrong_placement_position_is_not_verified(self) -> None:
        runtime = self.make_runtime()
        take = {"action": "move_and_take_object", "parameters": {"object_id": "7"}, "output": 0}
        runtime.record_action(take, {"result": "success"})
        runtime.update_hand_state(True)
        put = {"action": "put_down_sth", "parameters": {"target_location": [1, 2, 3]}, "output": 0}
        runtime.record_action(put, {"result": "success"})
        runtime.observe([{"object_id": "7", "name": "cup", "position": [100, 100, 100]}])
        runtime.update_hand_state(False)
        context = runtime.progress.context()
        self.assertEqual(context["states"]["7"], "DISCOVERED")
        self.assertEqual(context["remaining_objects"], ["7"])
        self.assertEqual(context["retry_count"], {"7": 1})

    def test_stale_pre_pick_position_is_not_placement_evidence(self) -> None:
        runtime = self.make_runtime()
        runtime.observe([{"object_id": "7", "name": "cup", "position": [1, 2, 3]}])
        runtime.record_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "7"}, "output": 0},
            {"result": "success"},
        )
        runtime.update_hand_state(True)
        runtime.record_action(
            {"action": "put_down_sth", "parameters": {"target_location": [1, 2, 3]}, "output": 0},
            {"result": "success"},
        )
        runtime.observe([{"object_id": "7", "name": "cup"}])
        runtime.update_hand_state(False)
        context = runtime.progress.context()
        self.assertEqual(context["states"]["7"], "PLACED")
        self.assertEqual(context["remaining_objects"], ["7"])
        runtime.observe([{"object_id": "8", "name": "book"}])
        next_pick = runtime.validate_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "8"}, "output": 0}
        )
        self.assertFalse(next_pick.valid)
        self.assertEqual(next_pick.failure_class, "PLANNING_ERROR")

    def test_surface_candidate_excludes_movable_objects_and_adjusts_height(self) -> None:
        runtime = self.make_runtime()
        runtime.observe(
            [
                {
                    "object_id": "7",
                    "name": "cup",
                    "world_aabb": {"min": {"X": 0, "Y": 0, "Z": 0}, "max": {"X": 2, "Y": 2, "Z": 2}},
                },
                {
                    "object_id": "table",
                    "name": "table",
                    "world_aabb": {"min": {"X": 0, "Y": 0, "Z": 0}, "max": {"X": 20, "Y": 20, "Z": 10}},
                },
            ]
        )
        runtime.record_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "7"}, "output": 0},
            {"result": "success"},
        )
        runtime.update_hand_state(True)
        candidates = runtime.prompt_context()["placement_candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["object_id"], "table")
        self.assertEqual(candidates[0]["location"], [10.0, 10.0, 12.0])
        self.assertEqual(candidates[0]["height_adjustment"], 2.0)
        self.assertEqual(candidates[0]["confidence"], 0.72)
        self.assertFalse(candidates[0]["direct_safe"])
        self.assertIn("world_aabb", candidates[0]["evidence"][0])

    def test_unique_explicit_place_location_is_direct_safe(self) -> None:
        runtime = self.make_runtime()
        runtime.observe([{"object_id": "shelf", "name": "shelf", "place_location": [4, 5, 6]}])
        candidates = runtime.prompt_context()["placement_candidates"]
        self.assertEqual(candidates[0]["object_id"], "shelf")
        self.assertEqual(candidates[0]["confidence"], 0.92)
        self.assertTrue(candidates[0]["direct_safe"])

    def test_equal_explicit_targets_are_not_arbitrarily_selected(self) -> None:
        runtime = self.make_runtime()
        runtime.observe(
            [
                {"object_id": "shelf-a", "place_location": [1, 2, 3]},
                {"object_id": "shelf-b", "place_location": [4, 5, 6]},
            ]
        )
        candidates = runtime.prompt_context()["placement_candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertFalse(candidates[0]["direct_safe"])

    def test_held_object_motion_is_explicit(self) -> None:
        runtime = self.make_runtime()
        take = {"action": "move_and_take_object", "parameters": {"object_id": "7"}, "output": 0}
        runtime.record_action(take, {"result": "success"})
        runtime.update_hand_state(True)
        runtime.record_action(
            {"action": "move_to_location", "parameters": {"target_location": [1, 2, 3]}},
            {"result": "success"},
        )
        self.assertEqual(runtime.progress.context()["states"]["7"], "MOVING")

    def test_finish_guard_blocks_premature_finish(self) -> None:
        runtime = self.make_runtime()
        decision = runtime.validate_action(
            {"action": "finish_task", "parameters": {}, "output": 0}, object_in_hand=False
        )
        self.assertFalse(decision.valid)
        self.assertEqual(decision.failure_class, "PREMATURE_FINISH")

    def test_finish_guard_allows_verified_completion(self) -> None:
        runtime = self.make_runtime()
        runtime.progress.completed_objects.add("7")
        runtime.progress.expected_objects.clear()
        runtime.progress.states["7"] = "VERIFIED"
        decision = runtime.validate_action(
            {"action": "finish_task", "parameters": {}, "output": 0}, object_in_hand=False
        )
        self.assertTrue(decision.valid)

    def test_finish_guard_rejects_cleared_but_unverified_object(self) -> None:
        runtime = self.make_runtime()
        runtime.progress.expected_objects.clear()
        decision = runtime.validate_action(
            {"action": "finish_task", "parameters": {}, "output": 0}, object_in_hand=False
        )
        self.assertFalse(decision.valid)
        self.assertIn("lack verified completion", decision.error)

    def test_finish_guard_requires_official_evidence_when_coverage_unknown(self) -> None:
        runtime = CompetitionRuntime()
        runtime.ensure_episode({"task_type": "tidyroom", "subject": "收拾好房间"})
        runtime.progress.completed_goal_actions = 1
        decision = runtime.validate_action(
            {"action": "finish_task", "parameters": {}, "output": 0}, object_in_hand=False
        )
        self.assertFalse(decision.valid)
        runtime.observe([], {"task_completed": True})
        decision = runtime.validate_action(
            {"action": "finish_task", "parameters": {}, "output": 0}, object_in_hand=False
        )
        self.assertTrue(decision.valid)

    def test_completed_object_cannot_be_picked_again(self) -> None:
        runtime = self.make_runtime()
        runtime.progress.completed_objects.add("7")
        runtime.progress.expected_objects.clear()
        decision = runtime.validate_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "7"}, "output": 0}
        )
        self.assertFalse(decision.valid)
        self.assertEqual(decision.failure_class, "PLANNING_ERROR")

    def test_holding_object_blocks_pick_of_new_object(self) -> None:
        runtime = CompetitionRuntime(repeated_action_limit=2)
        runtime.ensure_episode(
            {"task_type": "tidyroom", "subject": "整理房间", "movable_object_id": ["7", "8"]}
        )
        runtime.observe([{"object_id": "7", "name": "cup"}, {"object_id": "8", "name": "shoe"}])
        runtime.record_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "7"}},
            {"result": "success"},
        )
        runtime.update_hand_state(True)

        decision = runtime.validate_action(
            {"action": "move_and_take_object", "parameters": {"object_id": "8"}}, object_in_hand=True
        )

        self.assertFalse(decision.valid)
        self.assertEqual(decision.failure_class, "PLANNING_ERROR")

    def test_same_object_pick_is_limited_to_normal_and_recovery_attempt(self) -> None:
        runtime = self.make_runtime()
        pick = {"action": "move_and_take_object", "parameters": {"object_id": "7"}}
        first = runtime.validate_action(pick)
        self.assertTrue(first.valid)
        runtime.record_action(pick, {"result": "success"}, validation=first)
        runtime.update_hand_state(False)
        runtime.metrics.steps = 3
        second = runtime.validate_action(pick)
        self.assertTrue(second.valid)
        runtime.record_action(pick, {"result": "success"}, validation=second)
        runtime.update_hand_state(False)

        third = runtime.validate_action(pick)

        self.assertFalse(third.valid)
        self.assertEqual(third.failure_class, "LOOP_ERROR")

    def test_semantically_equivalent_move_targets_trigger_repeat_guard(self) -> None:
        runtime = self.make_runtime()
        runtime.observe([{"object_id": "7", "name": "cup", "position": [1, 2, 3]}])
        first = {"action": "move_to_object", "parameters": {"object_id": "7"}, "think": "a"}
        second = {"action": "move_to_location", "parameters": {"target_location": [1, 2, 3]}, "think": "b"}
        runtime.record_action(first, {"result": "success"})
        runtime.record_action(second, {"result": "success"})

        decision = runtime.validate_action(first)

        self.assertFalse(decision.valid)
        self.assertEqual(decision.failure_class, "LOOP_ERROR")


if __name__ == "__main__":
    unittest.main()
