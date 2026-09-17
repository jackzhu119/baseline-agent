from __future__ import annotations

from google.protobuf import struct_pb2

from arenaagent.agent_base import AgentBase
from arenaagent.vlm_agent.vlm_agent import VLMAgent


def test_submit_answer_is_a_terminal_subject_action() -> None:
    agent = object.__new__(VLMAgent)
    agent.agent_id = "test-agent"
    agent.action_space = {"key": "answer"}
    agent.subject_finished = False

    result = agent._handle_submit_answer({}, {"output": 4})

    assert result == {"answer": "4"}
    assert agent.subject_finished is True


def test_submit_answer_reaches_evaluation_without_waiting_for_server_poll() -> None:
    agent = object.__new__(VLMAgent)
    agent.agent_id = "test-agent"
    agent.action_space = {"key": "answer"}
    agent.subject_finished = False
    agent.sleep_between_steps = 0
    events: list[str] = []
    action_space = struct_pb2.Struct()
    action_space.update({"key": "answer"})

    def current_subject_finished() -> bool:
        events.append("poll")
        return False

    def run_step(subject: dict, task_response: dict) -> dict[str, str]:
        events.append("run_step")
        return agent._handle_submit_answer({}, {"output": 4})

    def apply_action(action: dict[str, str]) -> dict:
        events.append("update_action")
        assert action == {"answer": "4"}
        return {}

    def evaluate_subject() -> dict:
        events.append("evaluate_subject")
        return {"score": 1}

    agent._current_subject_finished = current_subject_finished
    agent._get_subject_from_task = lambda: {"task_type": "counting", "subject": "count"}
    agent._get_response_from_task = lambda: {}
    agent._call_struct = lambda method, payload, deserializer: action_space
    agent.run_step = run_step
    agent._apply_action = apply_action
    agent._evaluate_subject = evaluate_subject
    agent._on_subject_evaluated = lambda evaluation: events.append("evaluated")

    AgentBase._run_subject(agent)

    assert events == ["poll", "run_step", "update_action", "evaluate_subject", "evaluated"]
