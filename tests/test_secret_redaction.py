from __future__ import annotations

from pathlib import Path

from loguru import logger

from arenaagent.agent_base import AgentBase, AgentCfg
from arenaagent.utils import config as config_module
from arenaagent.utils.redaction import REDACTED, redact_sensitive


class LogOnlyAgent(AgentBase):
    def init(self, opt):
        return None

    def deinit(self):
        return None

    def run_step(self, subject, task_response):
        return {}

    def _connect(self) -> bool:
        return False


def _capture_messages() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    return messages, sink_id


def test_recursive_config_redaction_does_not_mutate_source() -> None:
    raw = {
        "client": {
            "api_key": "sk-super-secret-value",
            "access_token": "token-value",
            "name": "safe-model",
        }
    }

    safe = redact_sensitive(raw)

    assert safe["client"]["api_key"] == REDACTED
    assert safe["client"]["access_token"] == REDACTED
    assert safe["client"]["name"] == "safe-model"
    assert raw["client"]["api_key"] == "sk-super-secret-value"


def test_agent_load_never_logs_api_key() -> None:
    secret = "sk-agent-load-secret-123456"
    messages, sink_id = _capture_messages()
    try:
        agent = LogOnlyAgent(stub=None, channel=None, cfg=AgentCfg())
        agent.load({"name": "agent", "api_key": secret, "nested": {"token": "token-secret"}})
    finally:
        logger.remove(sink_id)

    rendered = "".join(messages)
    assert secret not in rendered
    assert "token-secret" not in rendered
    assert REDACTED in rendered


def test_load_config_logs_redacted_values(tmp_path: Path) -> None:
    secret = "sk-config-file-secret-123456"
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'[agent]\nname = "demo"\napi_key = "{secret}"\n', encoding="utf-8")
    messages, sink_id = _capture_messages()
    previous = config_module.CONFIG
    config_module.CONFIG = {}
    try:
        loaded = config_module.load_config(config_path)
    finally:
        config_module.CONFIG = previous
        logger.remove(sink_id)

    assert loaded["agent"]["api_key"] == secret
    rendered = "".join(messages)
    assert secret not in rendered
    assert REDACTED in rendered
