"""gRPC Gateway: client-facing entry point.

* HealthCheck queries Registry for the manifest list and surfaces it as
  `repeated ModelDescriptor`.
* StreamingRecognize:
    - first message must be StreamingConfig
    - acquire a TranscriptActor handle from MasterRouter
    - fire-and-forget actor.feed.remote(bytes)
    - async-iterate actor.results.options(num_returns="streaming").remote()
* Audio is NEVER decoded on the host side; bytes pass through unchanged.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import suppress
from typing import Any

import grpc
import httpx
import ray
import structlog

from proto import sayo_pb2, sayo_pb2_grpc
from sayo_host.common.logging import configure_logging

logger = structlog.get_logger("gateway")


_QUANT_MAP = {
    "pcm_s16le": sayo_pb2.AUDIO_QUANTIZATION_PCM_S16LE,
    "pcm_f32le": sayo_pb2.AUDIO_QUANTIZATION_PCM_F32LE,
}


def _quantization_to_proto(text: str) -> int:
    return _QUANT_MAP.get(text.lower(), sayo_pb2.AUDIO_QUANTIZATION_PCM_F32LE)


class SayoServiceServicer(sayo_pb2_grpc.SayoServiceServicer):
    def __init__(
        self,
        registry_url: str,
        ray_namespace: str,
    ) -> None:
        self._registry_url = registry_url.rstrip("/")
        self._namespace = ray_namespace
        self._http = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def HealthCheck(self, request, context):
        del request
        try:
            response = await self._http.get(f"{self._registry_url}/v1/models")
            response.raise_for_status()
            manifests = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("registry unavailable", error=str(exc))
            return sayo_pb2.HealthCheckResponse(
                ready=False, message=f"registry unavailable: {exc}"
            )

        descriptors = [
            sayo_pb2.ModelDescriptor(
                model_id=m["model_id"],
                description=m.get("description", ""),
                language_code=m.get("language_code", "en"),
                sample_rate_hertz=int(m.get("sample_rate_hertz", 16_000)),
                audio_quantization=_quantization_to_proto(
                    m.get("audio_quantization", "pcm_f32le")
                ),
                chunk_duration_ms=int(m.get("chunk_duration_ms", 0)),
                supports_interim_results=bool(m.get("supports_interim_results", True)),
                latency_ms=float(m.get("latency_ms", 0.0)),
            )
            for m in manifests
        ]
        return sayo_pb2.HealthCheckResponse(
            ready=True, message="ok", models=descriptors
        )

    async def StreamingRecognize(self, request_iterator, context):
        config: sayo_pb2.StreamingConfig | None = None
        actor: Any | None = None
        actor_name: str | None = None
        session_id = uuid.uuid4().hex
        feed_task: asyncio.Task | None = None
        results_iter = None
        router = None

        try:
            first = await request_iterator.__anext__()
            if not first.HasField("config"):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "first streaming message must be config",
                )
                return
            config = first.config

            manifest = await self._load_manifest(config.model_id)
            if manifest is None:
                await context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"unknown model_id: {config.model_id}",
                )
                return

            if int(config.sample_rate_hertz) != int(manifest["sample_rate_hertz"]):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "sample_rate_hertz mismatch with model manifest",
                )
                return
            expected_quant = _quantization_to_proto(
                manifest.get("audio_quantization", "pcm_f32le")
            )
            if int(config.audio_quantization) != int(expected_quant):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "audio_quantization mismatch with model manifest",
                )
                return

            router = ray.get_actor("MasterRouter", namespace=self._namespace)
            vad_cfg = {
                "threshold": float(config.vad_threshold or 0.0),
                "min_silence_ms": int(config.vad_min_silence_duration_ms or 0),
            }
            actor_name = await router.acquire_session.remote(
                session_id, config.model_id, vad_cfg
            )
            actor = ray.get_actor(actor_name, namespace=self._namespace)
            await actor.open_session.remote(
                session_id,
                vad_cfg,
                int(config.audio_quantization),
            )
            logger.info(
                "session started",
                session_id=session_id,
                model_id=config.model_id,
                actor_name=actor_name,
            )

            feed_task = asyncio.create_task(
                self._feed_loop(actor, session_id, request_iterator)
            )

            results_iter = actor.results.options(num_returns="streaming").remote(
                session_id
            )
            async for raw in results_iter:
                payload: dict = await raw if asyncio.iscoroutine(raw) else raw
                yield self._to_response(payload, config.model_id)

        except StopAsyncIteration:
            return
        finally:
            if feed_task is not None and not feed_task.done():
                feed_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await feed_task
            if actor is not None:
                with suppress(Exception):
                    await actor.close_session.remote(session_id)
            if router is not None and actor_name is not None:
                with suppress(Exception):
                    await router.release_session.remote(session_id)
            logger.info("session ended", session_id=session_id)

    async def _load_manifest(self, model_id: str) -> dict | None:
        try:
            response = await self._http.get(
                f"{self._registry_url}/v1/models/{model_id}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("registry fetch failed", model_id=model_id, error=str(exc))
            return None
        if response.status_code != 200:
            return None
        return response.json()

    @staticmethod
    async def _feed_loop(actor, session_id: str, request_iterator) -> None:
        async for request in request_iterator:
            if request.HasField("config"):
                continue
            if not request.HasField("audio_chunk"):
                continue
            actor.feed.remote(session_id, request.audio_chunk)

    @staticmethod
    def _to_response(
        payload: dict, model_id: str
    ) -> sayo_pb2.StreamingRecognizeResponse:
        metadata = {str(k): str(v) for k, v in (payload.get("metadata") or {}).items()}
        metadata.setdefault("model_id", model_id)
        latency_ms = float(payload.get("latency_ms", 0.0) or 0.0)
        if latency_ms:
            metadata.setdefault("latency_ms", f"{latency_ms:.1f}")
        return sayo_pb2.StreamingRecognizeResponse(
            transcript=str(payload.get("transcript", "")),
            is_final=bool(payload.get("is_final", False)),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            metadata=metadata,
        )


async def serve() -> None:
    configure_logging("gateway")
    registry_url = os.environ.get("REGISTRY_URL", "http://model-registry:8000")
    ray_address = os.environ.get("RAY_ADDRESS", "ray-head:10001")
    namespace = os.environ.get("RAY_NAMESPACE", "sayo")
    port = int(os.environ.get("GATEWAY_PORT", "50051"))

    ray.init(address=ray_address, namespace=namespace)

    server = grpc.aio.server()
    servicer = SayoServiceServicer(registry_url, ray_namespace=namespace)
    sayo_pb2_grpc.add_SayoServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    logger.info("gateway listening", port=port, registry_url=registry_url)
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await servicer.close()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
