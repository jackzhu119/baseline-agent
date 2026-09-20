from __future__ import annotations

import re

from arenaagent.build_info import agent_build, source_revision


def test_agent_build_contains_traceable_revision() -> None:
    revision = source_revision()
    assert revision != "unknown"
    assert re.fullmatch(r"[0-9a-f]{7,12}", revision)
    assert agent_build() == f"competition-runtime-v4+{revision}"
