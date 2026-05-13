"""Master Router: Ray detached named actor `MasterRouter` (namespace=sayo).

Responsibilities:
  - placement: pick a node + actor for a new session (reuse if possible)
  - LRU eviction when no node has enough free VRAM
  - manifest cache (one fetch per model on cold path)

Hot path (`acquire_session`) never reads model.yaml; it consults only the
in-memory manifest cache populated from the Registry HTTP API.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field

import httpx
import ray
import structlog

from sayo_host.common.logging import configure_logging

logger = structlog.get_logger("router")


@dataclass
class ActorRecord:
    actor_name: str
    model_id: str
    node_id: str
    slot_resource: str
    vram_gb: float
    max_concurrent_sessions: int
    active_sessions: int = 0
    last_used: float = field(default_factory=time.time)


@dataclass
class NodeState:
    node_id: str
    last_heartbeat: float = field(default_factory=time.time)
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    actors: dict[str, ActorRecord] = field(default_factory=dict)


@dataclass
class Placement:
    session_id: str
    actor_name: str
    node_id: str
    model_id: str


@ray.remote
class MasterRouter:
    def __init__(self, registry_url: str) -> None:
        configure_logging("master-router")
        self._registry_url = registry_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=10.0)
        self._manifest_cache: dict[str, dict] = {}
        self._nodes: dict[str, NodeState] = {}
        self._placements: dict[str, Placement] = {}
        self._lock = asyncio.Lock()
        logger.info("MasterRouter ready", registry_url=self._registry_url)

    async def heartbeat(self, node_id: str, payload: dict) -> None:
        async with self._lock:
            node = self._nodes.setdefault(node_id, NodeState(node_id=node_id))
            node.last_heartbeat = time.time()
            node.vram_total_gb = float(payload.get("vram_total_gb", 0.0))
            node.vram_used_gb = float(payload.get("vram_used_gb", 0.0))
            # Worker Manager is source of truth for running docker-backed actors.
            alive = set(payload.get("actors") or [])
            for name in list(node.actors.keys()):
                if name not in alive:
                    record = node.actors.get(name)
                    if record is not None and (
                        record.active_sessions > 0
                        or (time.time() - record.last_used) < 30.0
                    ):
                        logger.info(
                            "keeping actor record",
                            node_id=node_id,
                            actor_name=name,
                            active_sessions=record.active_sessions,
                            last_used=record.last_used,
                            alive=sorted(alive),
                        )
                        continue
                    logger.warning(
                        "dropping router actor record (not in WM heartbeat)",
                        node_id=node_id,
                        actor_name=name,
                        alive=sorted(alive),
                    )
                    node.actors.pop(name, None)

    async def acquire_session(
        self, session_id: str, model_id: str, vad_cfg: dict | None = None
    ) -> str:
        del vad_cfg  # opaque to router; gateway forwards it directly to actor
        manifest = await self._fetch_manifest(model_id)
        async with self._lock:
            actor = self._find_reusable_actor(model_id, manifest)
            if actor is None:
                actor = await self._spawn_actor(manifest)
                logger.info(
                    "spawned new actor (no reusable slot)",
                    model_id=model_id,
                    actor_name=actor.actor_name,
                    max_sessions=int(manifest.get("max_concurrent_sessions", 1)),
                )
            else:
                logger.info(
                    "reusing actor",
                    model_id=model_id,
                    actor_name=actor.actor_name,
                    active_sessions=actor.active_sessions,
                    max_sessions=int(manifest.get("max_concurrent_sessions", 1)),
                )
            actor.active_sessions += 1
            actor.last_used = time.time()
            self._placements[session_id] = Placement(
                session_id=session_id,
                actor_name=actor.actor_name,
                node_id=actor.node_id,
                model_id=model_id,
            )
            logger.info(
                "session acquired",
                session_id=session_id,
                model_id=model_id,
                actor_name=actor.actor_name,
                node_id=actor.node_id,
            )
            return actor.actor_name

    async def release_session(self, session_id: str) -> None:
        async with self._lock:
            placement = self._placements.pop(session_id, None)
            if placement is None:
                logger.warning(
                    "release_session: unknown session_id (duplicate release?)",
                    session_id=session_id,
                )
                return
            node = self._nodes.get(placement.node_id)
            if node is None:
                return
            actor = node.actors.get(placement.actor_name)
            if actor is None:
                return
            actor.active_sessions = max(0, actor.active_sessions - 1)
            actor.last_used = time.time()
            logger.info(
                "session released",
                session_id=session_id,
                actor_name=actor.actor_name,
                active_sessions=actor.active_sessions,
            )

    async def admin_state(self) -> dict:
        async with self._lock:
            return {
                "nodes": {
                    nid: {
                        "vram_total_gb": n.vram_total_gb,
                        "vram_used_gb": n.vram_used_gb,
                        "last_heartbeat": n.last_heartbeat,
                        "actors": {
                            name: {
                                "model_id": a.model_id,
                                "active_sessions": a.active_sessions,
                                "max_concurrent_sessions": a.max_concurrent_sessions,
                                "last_used": a.last_used,
                                "vram_gb": a.vram_gb,
                            }
                            for name, a in n.actors.items()
                        },
                    }
                    for nid, n in self._nodes.items()
                },
                "sessions": list(self._placements.keys()),
                "models_cached": list(self._manifest_cache.keys()),
            }

    async def shutdown(self) -> None:
        await self._http.aclose()

    async def _fetch_manifest(self, model_id: str) -> dict:
        cached = self._manifest_cache.get(model_id)
        if cached is not None:
            return cached
        url = f"{self._registry_url}/v1/models/{model_id}"
        r = await self._http.get(url)
        if r.status_code != 200:
            raise RuntimeError(
                f"registry returned {r.status_code} for {model_id}: {r.text}"
            )
        manifest = r.json()
        self._manifest_cache[model_id] = manifest
        return manifest

    def _find_reusable_actor(self, model_id: str, manifest: dict) -> ActorRecord | None:
        max_sessions = int(manifest.get("max_concurrent_sessions", 1))
        for node in self._nodes.values():
            for actor in node.actors.values():
                if actor.model_id != model_id:
                    continue
                if actor.active_sessions < max_sessions:
                    return actor
        return None

    async def _spawn_actor(self, manifest: dict) -> ActorRecord:
        model_id = manifest["model_id"]
        vram_required = float(manifest.get("min_vram_gb", 0.0))
        node = self._pick_node(vram_required) or await self._evict_for_vram(
            vram_required
        )
        if node is None:
            raise RuntimeError(
                f"no node has {vram_required} GB VRAM available for {model_id}"
            )

        wm = ray.get_actor(f"WorkerManager:{node.node_id}", namespace="sayo")
        actor_name = await wm.spawn.remote(manifest)
        record = ActorRecord(
            actor_name=actor_name,
            model_id=model_id,
            node_id=node.node_id,
            slot_resource=actor_name.replace("actor_", "slot_"),
            vram_gb=vram_required,
            max_concurrent_sessions=int(manifest.get("max_concurrent_sessions", 1)),
        )
        node.actors[actor_name] = record
        node.vram_used_gb += vram_required
        return record

    def _pick_node(self, vram_required: float) -> NodeState | None:
        candidates = [
            n
            for n in self._nodes.values()
            if (n.vram_total_gb - n.vram_used_gb) >= vram_required
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda n: n.vram_total_gb - n.vram_used_gb, reverse=True)
        return candidates[0]

    async def _evict_for_vram(self, vram_required: float) -> NodeState | None:
        for node in self._nodes.values():
            idle = [a for a in node.actors.values() if a.active_sessions == 0]
            if not idle:
                continue
            idle.sort(key=lambda a: a.last_used)
            for actor in idle:
                wm = ray.get_actor(f"WorkerManager:{node.node_id}", namespace="sayo")
                try:
                    await wm.evict.remote(actor.actor_name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "evict failed",
                        actor_name=actor.actor_name,
                        error=str(exc),
                    )
                    continue
                node.actors.pop(actor.actor_name, None)
                node.vram_used_gb = max(0.0, node.vram_used_gb - actor.vram_gb)
                if (node.vram_total_gb - node.vram_used_gb) >= vram_required:
                    return node
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sayo Master Router bootstrap")
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "ray://ray-head:10001"),
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("RAY_NAMESPACE", "sayo"),
    )
    parser.add_argument(
        "--registry-url",
        default=os.environ.get("REGISTRY_URL", "http://model-registry:8000"),
    )
    args = parser.parse_args()

    configure_logging("router-bootstrap")
    ray.init(address=args.ray_address, namespace=args.namespace)
    actor_name = "MasterRouter"
    replace = os.environ.get("SAYO_REPLACE_MASTER_ROUTER", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if replace:
        try:
            old = ray.get_actor(actor_name, namespace=args.namespace)
            ray.kill(old, no_restart=True)
            logger.info(
                "MasterRouter killed before recreate (SAYO_REPLACE_MASTER_ROUTER)",
                actor_name=actor_name,
            )
        except ValueError:
            logger.info("no existing MasterRouter to replace", actor_name=actor_name)
        handle = MasterRouter.options(
            name=actor_name,
            namespace=args.namespace,
            lifetime="detached",
        ).remote(args.registry_url)
        logger.info(
            "MasterRouter created (replace)",
            actor_name=actor_name,
            namespace=args.namespace,
        )
    else:
        try:
            handle = ray.get_actor(actor_name, namespace=args.namespace)
            logger.info(
                "MasterRouter already in Ray cluster (container re-attach)",
                actor_name=actor_name,
                namespace=args.namespace,
            )
        except ValueError:
            handle = MasterRouter.options(
                name=actor_name,
                namespace=args.namespace,
                lifetime="detached",
            ).remote(args.registry_url)
            logger.info(
                "MasterRouter created",
                actor_name=actor_name,
                namespace=args.namespace,
            )

    from sayo_host.common.admin_http import serve_admin_state

    admin_port = int(os.environ.get("ADMIN_PORT", "8081"))

    def _snapshot() -> dict:
        return ray.get(handle.admin_state.remote(), timeout=5)

    serve_admin_state(admin_port, _snapshot)
    logger.info("admin HTTP started", port=admin_port)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
