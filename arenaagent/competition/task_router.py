from __future__ import annotations

from typing import Any

from arenaagent.competition.runtime import CompetitionRuntime
from arenaagent.competition.solvers.counting import CountingSolver
from arenaagent.competition.solvers.jigsaw import JigsawSpatialSolver
from arenaagent.competition.solvers.tidyroom import MAX_FINAL_SCANS
from arenaagent.competition.solvers.tidyroom_policy import TidyObjectClassifier, TidyTargetPlanner

JIGSAW_AUTO_PLACEMENT_CONFIDENCE = 0.8
VECTOR_DIMENSIONS = 3


def _task_text(subject: Any) -> str:
    if isinstance(subject, dict):
        return str(subject.get("subject") or subject.get("goal") or subject.get("task_prompt") or "")
    return str(subject or "")


class TaskStrategyRouter:
    """Route public task observations to bounded task-specific strategies."""

    def __init__(self, *, counting_scan_degrees: list[float] | None = None) -> None:
        self.counting = CountingSolver(counting_scan_degrees)
        self.jigsaw = JigsawSpatialSolver()
        self.tidy_classifier = TidyObjectClassifier()
        self.tidy_target_planner = TidyTargetPlanner()

    def observe(self, runtime: CompetitionRuntime, subject: Any = None) -> None:
        if runtime.metrics is None:
            return
        if runtime.task_type == "counting":
            self.counting.observe(runtime.metrics.episode_id)
        elif runtime.task_type == "jigsaw" and isinstance(subject, dict):
            runtime.set_strategy_context(
                self.jigsaw.infer(
                    subject,
                    runtime.objects,
                    dict(runtime.progress.retries),
                    set(runtime.progress.completed_objects),
                ).context()
            )
        elif runtime.task_type == "npc":
            runtime.set_strategy_context({"name": "npc_information_gain", **runtime.npc_memory.context()})
        elif runtime.task_type == "tidyroom":
            self._observe_tidyroom(runtime)

    def propose_action(self, subject: Any, runtime: CompetitionRuntime) -> dict[str, Any] | None:  # noqa: PLR0911
        if runtime.metrics is None:
            return None
        if runtime.task_type == "jigsaw":
            return self._propose_jigsaw_placement(runtime)
        if runtime.task_type == "npc":
            return self._propose_npc_action(runtime)
        if runtime.task_type == "tidyroom":
            return self._propose_tidyroom_action(runtime)
        if runtime.task_type != "counting":
            return None
        scan_action = self.counting.next_scan_action()
        if scan_action is not None:
            runtime.set_strategy_context(
                {
                    "name": "counting",
                    "mode": "systematic_scan",
                    "observation_count": self.counting.observation_count,
                    "attempted_degrees": self.counting.attempted_degrees,
                    "coverage_complete": False,
                }
            )
            return scan_action

        result = self.counting.solve(_task_text(subject), runtime.objects)
        runtime.set_strategy_context(
            {
                "name": "counting",
                "mode": "deterministic_count",
                "observation_count": self.counting.observation_count,
                "attempted_degrees": self.counting.attempted_degrees,
                "coverage_complete": True,
                "result": result.context(),
            }
        )
        if not result.confident or result.answer is None:
            return None
        return {
            "think": "deterministic deduplicated count",
            "action": "submit_answer",
            "parameters": {},
            "output": result.answer,
        }

    @staticmethod
    def _propose_jigsaw_placement(runtime: CompetitionRuntime) -> dict[str, Any] | None:
        """Use a complete high-confidence public grid plan without another VLM transcription."""
        held_piece = runtime.progress.current_object
        if not held_piece or runtime.progress.pending_pick or runtime.progress.pending_place:
            return None
        strategy = runtime.strategy_context
        if (
            strategy.get("solver") != "jigsaw_spatial_grid"
            or float(strategy.get("confidence") or 0) < JIGSAW_AUTO_PLACEMENT_CONFIDENCE
        ):
            return None
        pieces = strategy.get("pieces")
        if not isinstance(pieces, list):
            return None
        piece = next(
            (
                item
                for item in pieces
                if isinstance(item, dict)
                and str(item.get("object_id")) == held_piece
                and float(item.get("confidence") or 0) >= JIGSAW_AUTO_PLACEMENT_CONFIDENCE
            ),
            None,
        )
        if piece is None:
            return None
        location = piece.get("candidate_position")
        rotation = piece.get("candidate_rotation")
        if not isinstance(location, list) or len(location) < VECTOR_DIMENSIONS or not isinstance(rotation, dict):
            return None
        return {
            "think": "high-confidence public jigsaw grid placement",
            "action": "put_down_sth",
            "parameters": {
                "target_location": location,
                "target_rotation": rotation,
                "auto_rotate": False,
            },
            "output": 0,
            "expected_change": f"piece {held_piece} leaves the hand at the inferred empty cell",
        }

    @staticmethod
    def _propose_npc_action(runtime: CompetitionRuntime) -> dict[str, Any] | None:
        """Ask the highest-value evidence question, or submit a verified single answer."""
        memory = runtime.npc_memory
        answer = memory.final_answer()
        if answer is not None:
            runtime.set_strategy_context({"name": "npc_information_gain", **memory.context()})
            return {
                "think": "all required NPC facts are evidenced",
                "action": "submit_answer",
                "parameters": {},
                "output": answer,
                "expected_change": "official task service evaluates the evidence-derived answer",
            }
        candidate = memory.best_question()
        if candidate is None:
            return None
        runtime.set_strategy_context(
            {"name": "npc_information_gain", **memory.context(), "selected_question": candidate.context()}
        )
        return {
            "think": f"request missing fact {candidate.fact_key}",
            "action": "speak_to_npc",
            "parameters": {"npc_name": candidate.npc, "message": candidate.question},
            "output": 0,
            "expected_change": f"NPC reply resolves or redirects {candidate.fact_key}",
        }

    def _observe_tidyroom(self, runtime: CompetitionRuntime) -> None:
        visible_items = [
            runtime.objects[object_id]
            for object_id in sorted(runtime.visible_canonical_ids)
            if object_id in runtime.objects
        ]
        classification = self.tidy_classifier.classify(
            visible_items,
            required_ids=runtime.progress.initial_expected_objects,
            completed_ids=runtime.progress.completed_objects,
            blacklisted_ids=runtime.progress.failed_objects,
        )
        runtime.progress.update_classification(
            candidate_items=set(classification.candidate_items),
            rejected_items=set(classification.rejected_items),
            uncertain_items=set(classification.uncertain_items),
            target_surfaces=set(classification.target_surfaces),
        )
        placements = runtime.placement_candidates()
        target_items = {
            target_id: runtime.objects[target_id]
            for target_id in classification.target_surfaces
            if target_id in runtime.objects
        }
        for candidate_id in sorted(runtime.progress.remaining_candidates()):
            item = runtime.objects.get(candidate_id)
            if item is None:
                continue
            assignment = self.tidy_target_planner.assign(item, target_items, placements).context()
            runtime.progress.set_target_assignment(candidate_id, assignment)
        runtime.set_strategy_context(
            {
                "name": "tidyroom_state_machine",
                "phase": self._tidy_phase(runtime),
                "candidate_count": len(runtime.progress.remaining_candidates()),
                "target_assignment_count": len(runtime.progress.target_assignments),
                "final_scan_count": runtime.progress.final_scan_count,
            }
        )

    @staticmethod
    def _tidy_phase(runtime: CompetitionRuntime) -> str:
        progress = runtime.progress
        if progress.pending_pick:
            return "VERIFY_PICK"
        if progress.pending_place:
            return "VERIFY_PUT"
        if progress.current_object:
            return "SELECT_TARGET" if progress.selected_target(progress.current_object) is None else "PUT"
        if progress.remaining_candidates():
            return "SELECT_OBJECT"
        if progress.coverage_verified:
            return "MARK_COMPLETED"
        return "DISCOVER"

    @staticmethod
    def _propose_tidyroom_action(runtime: CompetitionRuntime) -> dict[str, Any] | None:
        progress = runtime.progress
        if progress.pending_pick or progress.pending_place:
            return None
        if progress.current_object:
            selected = progress.selected_target(progress.current_object)
            if selected is None or not isinstance(selected.get("location"), list):
                return None
            return {
                "think": "state machine places the verified held object at its semantic target",
                "action": "put_down_sth",
                "parameters": {"target_location": selected["location"], "auto_rotate": True},
                "output": 0,
                "expected_change": f"object {progress.current_object} leaves the hand near {selected['object_id']}",
            }

        for object_id in sorted(progress.remaining_candidates() & runtime.visible_canonical_ids):
            selected = progress.selected_target(object_id)
            can_attempt, _ = progress.can_attempt_pick(object_id, runtime.metrics.steps)
            if selected is None or not can_attempt:
                continue
            item = runtime.objects.get(object_id, {})
            current_id = str(item.get("current_object_id") or object_id)
            return {
                "think": "state machine selects one high-confidence clutter object with a verified semantic target",
                "action": "move_and_take_object",
                "parameters": {"object_id": current_id, "which_hand": 0},
                "output": 0,
                "expected_change": f"hand state confirms pickup of {object_id}",
            }

        if progress.remaining_candidates() or progress.failed_objects:
            return None
        if progress.initial_expected_objects and progress.initial_expected_objects <= progress.completed_objects:
            return {
                "think": "all official tidy objects have verified placement",
                "action": "finish_task",
                "parameters": {},
                "output": 0,
                "expected_change": "official task evaluation begins",
            }
        if not progress.coverage_verified:
            progress.record_final_scan()
            return {
                "think": f"bounded final coverage scan {progress.final_scan_count}/{MAX_FINAL_SCANS}",
                "action": "turn_in_degree",
                "parameters": {"degree": 90.0},
                "output": 0,
                "expected_change": "one final view confirms whether high-confidence clutter remains",
            }
        return {
            "think": "bounded scans found no unresolved high-confidence clutter",
            "action": "finish_task",
            "parameters": {},
            "output": 0,
            "expected_change": "official task evaluation begins",
        }
