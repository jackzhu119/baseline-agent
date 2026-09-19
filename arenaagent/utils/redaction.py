from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

REDACTED = "<REDACTED>"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_SECRET_TEXT_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}\b", re.IGNORECASE),
)


def _sensitive_key(value: Any) -> bool:
    normalized = str(value).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def redact_text(value: str) -> str:
    """Remove common credential shapes from otherwise safe diagnostic text."""
    redacted = value
    for pattern in _SECRET_TEXT_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def redact_sensitive(value: Any) -> Any:  # noqa: PLR0911 - recursive type dispatcher
    """Return a non-mutating, log-safe representation of nested configuration data."""
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _sensitive_key(key) else redact_sensitive(child)
            for key, child in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: REDACTED if _sensitive_key(item.name) else redact_sensitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    if isinstance(value, set):
        return sorted((redact_sensitive(item) for item in value), key=str)
    if isinstance(value, str):
        return redact_text(value)
    return value
