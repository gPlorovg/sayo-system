"""Worker Manager: Ray detached named actor `WorkerManager:<node_id>`.

One per physical node (in single-node deployments: just `WorkerManager:node-local`).

Responsibilities:
  - spawn TranscriptActor containers via Docker SDK using actor_image_tag
  - evict actors (ray.kill + docker stop + docker rm)
  - heartbeat: VRAM stats via pynvml, list of running actors
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

import ray
import structlog

import docker
from sayo_host.common.logging import configure_logging

logger = structlog.get_logger("worker_manager")

DEFAULT_NETWORK = os.environ.get("SAYO_NETWORK", "sayo_net")
RAY_HEAD_HOST = os.environ.get("RAY_HEAD_HOST", "ray-head:6379")


@ray.remote
class WorkerManager:
    def __init__(
        self,
        node_id: str,
        ray_head_address: str,
        sayo_network: str,
        registry_url: str,
        distributed: bool,
    ) -> None:
        configure_logging(f"worker-manager:{node_id}")
        self._node_id = node_id
        self._ray_head_address = ray_head_address
        self._network = sayo_network
        self._registry_url = registry_url.rstrip("/")
        self._distributed = distributed
        self._docker = docker.from_env()
        self._actors: dict[str, dict] = {}
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            "WorkerManager ready",
            node_id=node_id,
            network=sayo_network,
            ray_head=ray_head_address,
            distributed=distributed,
        )

    async def spawn(self, manifest: dict) -> str:
        actor_image_tag = manifest["actor_image_tag"]
        model_id = manifest["model_id"]
        model_dir = manifest["model_dir"]
        max_concurrent_sessions = int(manifest.get("max_concurrent_sessions", 1))
        vram_gb = float(manifest.get("min_vram_gb", 0.0))

        if self._distributed:
            try:
                self._docker.images.get(actor_image_tag)
            except docker.errors.ImageNotFound:
                logger.info("docker pull actor", image=actor_image_tag)
                self._docker.images.pull(actor_image_tag)

        slot_resource = f"slot_{uuid.uuid4().hex}"
        actor_name = f"actor_{uuid.uuid4().hex}"
        device = "cuda:0" if vram_gb > 0 else "cpu"
        model_name = model_dir.rsplit("/", 1)[-1]

        cmd = [
            f"--ray-address={self._ray_head_address}",
            f"--actor-name={actor_name}",
            f"--namespace={os.environ.get('RAY_NAMESPACE', 'sayo')}",
            f"--model-name={model_name}",
            f"--model-dir={model_dir}",
            f"--device={device}",
            f"--slot-resource={slot_resource}",
            f"--max-concurrent-sessions={max_concurrent_sessions}",
        ]
        run_kwargs: dict = {
            "name": actor_name,
            "network": self._network,
            "detach": True,
            "command": cmd,
            "labels": {
                "org.sayo.actor.name": actor_name,
                "org.sayo.actor.model_id": model_id,
                "org.sayo.actor.node_id": self._node_id,
            },
        }
        if vram_gb > 0:
            run_kwargs["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        logger.info(
            "docker run actor",
            image=actor_image_tag,
            actor_name=actor_name,
            slot_resource=slot_resource,
        )
        container = self._docker.containers.run(actor_image_tag, **run_kwargs)
        await self._wait_for_actor(actor_name, timeout=60.0)

        self._actors[actor_name] = {
            "container_id": container.id,
            "model_id": model_id,
            "slot_resource": slot_resource,
            "started_at": time.time(),
        }
        return actor_name

    async def evict(self, actor_name: str) -> None:
        record = self._actors.pop(actor_name, None)
        if record is None:
            logger.warning("evict: unknown actor", actor_name=actor_name)
            return
        try:
            handle = ray.get_actor(actor_name, namespace="sayo")
            try:
                await handle.unload.remote()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "actor.unload failed", actor_name=actor_name, error=str(exc)
                )
            ray.kill(handle, no_restart=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ray.kill failed", actor_name=actor_name, error=str(exc))

        try:
            container = self._docker.containers.get(record["container_id"])
            container.stop(timeout=15)
            container.remove(force=True)
        except docker.errors.NotFound:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("docker rm failed", actor_name=actor_name, error=str(exc))
        logger.info("actor evicted", actor_name=actor_name)

    async def heartbeat_payload(self) -> dict:
        vram_total = vram_used = 0.0
        gpu_count = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            for i in range(gpu_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                vram_total += info.total / (1024**3)
                vram_used += info.used / (1024**3)
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
        return {
            "node_id": self._node_id,
            "vram_total_gb": vram_total,
            "vram_used_gb": vram_used,
            "gpu_count": gpu_count,
            "actors": list(self._actors.keys()),
        }

    async def admin_state(self) -> dict:
        payload = await self.heartbeat_payload()
        payload["actor_records"] = self._actors
        return payload

    async def _wait_for_actor(self, actor_name: str, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ray.get_actor(actor_name, namespace="sayo")
                return
            except ValueError:
                await asyncio.sleep(0.5)
        raise TimeoutError(f"actor {actor_name} did not register in {timeout:.0f}s")

    async def _heartbeat_loop(self) -> None:
        namespace = os.environ.get("RAY_NAMESPACE", "sayo")
        while True:
            try:
                router = ray.get_actor("MasterRouter", namespace=namespace)
                payload = await self.heartbeat_payload()
                await router.heartbeat.remote(self._node_id, payload)
            except ValueError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("heartbeat failed", error=str(exc))
            await asyncio.sleep(5.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sayo Worker Manager bootstrap")
    parser.add_argument("--node-id", default=os.environ.get("NODE_ID", "node-local"))
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "ray-head:10001"),
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("RAY_NAMESPACE", "sayo"),
    )
    parser.add_argument(
        "--ray-head",
        default=os.environ.get("RAY_HEAD_HOST", RAY_HEAD_HOST),
    )
    parser.add_argument(
        "--registry-url",
        default=os.environ.get("REGISTRY_URL", "http://model-registry:8000"),
    )
    parser.add_argument(
        "--network",
        default=os.environ.get("SAYO_NETWORK", DEFAULT_NETWORK),
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=os.environ.get("SAYO_DISTRIBUTED", "0") == "1",
    )
    args = parser.parse_args()

    configure_logging("worker-manager-bootstrap")
    ray.init(address=args.ray_address, namespace=args.namespace)
    handle = WorkerManager.options(
        name=f"WorkerManager:{args.node_id}",
        namespace=args.namespace,
        lifetime="detached",
    ).remote(
        node_id=args.node_id,
        ray_head_address=args.ray_head,
        sayo_network=args.network,
        registry_url=args.registry_url,
        distributed=args.distributed,
    )
    logger.info(
        "WorkerManager registered",
        actor=handle,
        node_id=args.node_id,
        namespace=args.namespace,
    )

    from sayo_host.common.admin_http import serve_admin_state

    admin_port = int(os.environ.get("ADMIN_PORT", "8082"))

    def _snapshot() -> dict:
        return ray.get(handle.admin_state.remote(), timeout=5)

    serve_admin_state(admin_port, _snapshot)
    logger.info("admin HTTP started", port=admin_port)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
