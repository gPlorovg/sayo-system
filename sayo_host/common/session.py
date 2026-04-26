"""Lightweight session-related primitives shared between host services."""

from __future__ import annotations

import uuid
from typing import NewType

SessionId = NewType("SessionId", str)


def generate_session_id() -> SessionId:
    return SessionId(uuid.uuid4().hex)
