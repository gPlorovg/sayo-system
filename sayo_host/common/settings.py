"""Environment-driven settings shared by host services."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Required env variable missing: {name}")
    return value


def env_optional(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@dataclass(frozen=True, slots=True)
class HostSettings:
    ray_address: str
    ray_namespace: str
    registry_url: str
    node_id: str

    @classmethod
    def from_env(cls) -> "HostSettings":
        return cls(
            ray_address=_env("RAY_ADDRESS", "ray-head:6379"),
            ray_namespace=_env("RAY_NAMESPACE", "sayo"),
            registry_url=_env("REGISTRY_URL", "http://model-registry:8000"),
            node_id=_env("NODE_ID", "node-local"),
        )
