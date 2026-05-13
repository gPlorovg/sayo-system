"""gRPC Gateway: client-facing entry point.

* HealthCheck queries Registry for the manifest list and surfaces it as
  `repeated ModelDescriptor`.
* StreamingRecognize:
    - first message must be StreamingConfig
    - yields metadata-only responses with connection_status until setup completes
    - acquire a TranscriptActor handle from MasterRouter
    - audio: concurrent ``_feed_loop`` → ``await feed.remote`` per coalesced chunk
    - transcripts: ``next_result`` RPCs from the actor
      (Ray Client cannot stream generators)
      prefetched so the next Ray round-trip overlaps gRPC ``yield`` to the client
* Raw audio bytes pass through the host unchanged (decoding only inside the actor).
* Debug: set ``SAYO_DISABLE_VAD=1`` on the gateway process to ignore client VAD settings
  and pass a disabled ``vad_cfg`` to the actor (no Silero gating).
"""

from __future__ import annotations

import asyncio
import os
import traceback
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


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


_QUANT_MAP = {
    "pcm_s16le": sayo_pb2.AUDIO_QUANTIZATION_PCM_S16LE,
    "pcm_f32le": sayo_pb2.AUDIO_QUANTIZATION_PCM_F32LE,
}


def _quantization_to_proto(text: str) -> int:
    return _QUANT_MAP.get(text.lower(), sayo_pb2.AUDIO_QUANTIZATION_PCM_F32LE)


def _connection_status_response(
    status: str,
    *,
    session_id: str,
    model_id: str | None = None,
    actor_name: str | None = None,
    detail: str | None = None,
    is_final: bool = False,
) -> sayo_pb2.StreamingRecognizeResponse:
    """Lifecycle hints for the client (see proto comment on metadata)."""
    md: dict[str, str] = {
        "connection_status": status,
        "session_id": session_id,
    }
    if model_id:
        md["model_id"] = model_id
    if actor_name:
        md["actor_name"] = actor_name
    if detail:
        md["detail"] = detail[:2048]
    return sayo_pb2.StreamingRecognizeResponse(
        transcript="",
        is_final=is_final,
        confidence=0.0,
        metadata=md,
    )


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
        router = None

        setup_status_stream_started = False
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

            chunk_ms = self._resolve_chunk_ms(manifest, config)
            bytes_per_sample = (
                2
                if int(config.audio_quantization)
                == sayo_pb2.AUDIO_QUANTIZATION_PCM_S16LE
                else 4
            )
            target_bytes = (
                int(int(config.sample_rate_hertz) * chunk_ms / 1000) * bytes_per_sample
            )
            min_bytes = bytes_per_sample * max(1, int(config.sample_rate_hertz) // 50)
            target_bytes = max(min_bytes, min(target_bytes, 1024 * 1024))

            yield _connection_status_response(
                "config_accepted",
                session_id=session_id,
                model_id=config.model_id,
            )
            setup_status_stream_started = True

            router = ray.get_actor("MasterRouter", namespace=self._namespace)
            vad_cfg = {
                "threshold": float(config.vad_threshold or 0.0),
                "min_silence_ms": int(config.vad_min_silence_duration_ms or 0),
            }
            if _env_truthy("SAYO_DISABLE_VAD"):
                vad_cfg = {"threshold": 0.0, "min_silence_ms": 0}
                logger.info(
                    "VAD disabled for debugging (SAYO_DISABLE_VAD)",
                    session_id=session_id,
                )
            yield _connection_status_response(
                "allocating_session",
                session_id=session_id,
                model_id=config.model_id,
            )
            actor_name = await router.acquire_session.remote(
                session_id=session_id,
                model_id=config.model_id,
                vad_cfg=vad_cfg,
            )
            yield _connection_status_response(
                "actor_reserved",
                session_id=session_id,
                model_id=config.model_id,
                actor_name=actor_name,
            )
            actor = ray.get_actor(actor_name, namespace=self._namespace)
            yield _connection_status_response(
                "session_opening",
                session_id=session_id,
                model_id=config.model_id,
                actor_name=actor_name,
            )

            await actor.open_session.remote(
                session_id=session_id,
                vad_cfg=vad_cfg,
                quantization=int(config.audio_quantization),
            )
            yield _connection_status_response(
                "connected",
                session_id=session_id,
                model_id=config.model_id,
                actor_name=actor_name,
            )
            logger.info(
                "session started",
                session_id=session_id,
                model_id=config.model_id,
                actor_name=actor_name,
                chunk_ms=chunk_ms,
                target_bytes=target_bytes,
                sample_rate_hertz=int(config.sample_rate_hertz),
            )

            feed_task = asyncio.create_task(
                self._feed_loop(
                    actor, session_id, request_iterator, target_bytes=target_bytes
                )
            )

            prefetch_task: asyncio.Task | None = None
            try:
                prefetch_task = asyncio.create_task(
                    self._actor_next_result(actor, session_id)
                )
                while True:
                    payload = await prefetch_task
                    if payload is None:
                        break
                    prefetch_task = asyncio.create_task(
                        self._actor_next_result(actor, session_id)
                    )
                    yield self._to_response(payload, config.model_id)
            finally:
                drain_prefetch = prefetch_task
                prefetch_task = None
                if drain_prefetch is not None:
                    if not drain_prefetch.done():
                        try:
                            await asyncio.shield(drain_prefetch)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "prefetch next_result drain",
                                session_id=session_id,
                                error=str(exc),
                            )
                    else:
                        with suppress(Exception):
                            drain_prefetch.result()

        except asyncio.CancelledError:
            raise
        except StopAsyncIteration:
            return
        except Exception as exc:  # noqa: BLE001
            if setup_status_stream_started and config is not None:
                yield _connection_status_response(
                    "error",
                    session_id=session_id,
                    model_id=config.model_id,
                    actor_name=actor_name,
                    detail=str(exc),
                    is_final=True,
                )
            logger.warning(
                "StreamingRecognize failed",
                session_id=session_id,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            await context.abort(grpc.StatusCode.INTERNAL, str(exc)[:1024])
        finally:
            pending_cancel: asyncio.CancelledError | None = None
            if feed_task is not None:
                if not feed_task.done():
                    try:
                        await asyncio.shield(
                            self._await_task_deadline(feed_task, deadline_s=30.0)
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "feed_task still running after deadline",
                            session_id=session_id,
                        )
                    except asyncio.CancelledError as exc:
                        pending_cancel = exc
                        try:
                            await asyncio.shield(
                                self._await_task_deadline(feed_task, deadline_s=30.0)
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "feed task failed before drain",
                            session_id=session_id,
                            error=str(exc),
                        )
                else:
                    with suppress(Exception):
                        feed_task.result()
            if actor is not None:
                with suppress(Exception):
                    await actor.close_session.remote(session_id=session_id)
            if router is not None and actor_name is not None:
                try:
                    await router.release_session.remote(session_id=session_id)
                    logger.info(
                        "session released to router",
                        session_id=session_id,
                        actor_name=actor_name,
                    )
                except Exception as rel_exc:  # noqa: BLE001
                    logger.warning(
                        "release_session failed (router may keep actor busy)",
                        session_id=session_id,
                        actor_name=actor_name,
                        error=str(rel_exc),
                    )
            logger.info("session ended", session_id=session_id)
            if pending_cancel is not None:
                raise pending_cancel

    @staticmethod
    async def _await_task_deadline(task: asyncio.Task, *, deadline_s: float) -> None:
        """Wait for ``task`` with a timeout without cancelling it if the deadline hits.

        ``asyncio.wait_for`` cancels the awaited task on timeout and when the parent
        coroutine is cancelled while waiting. That propagates to Ray Client in-flight
        ``.remote()`` futures and triggers ``InvalidStateError: CANCELLED`` in
        ``ray.util.client`` callbacks.
        """
        if task.done():
            await task
            return
        done, _ = await asyncio.wait(
            {task},
            timeout=deadline_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task not in done:
            raise TimeoutError
        await task

    @staticmethod
    def _resolve_chunk_ms(manifest: dict, config: sayo_pb2.StreamingConfig) -> int:
        """Align gateway coalescing with registry manifest."""
        for key in ("chunk_duration_ms", "chunk_ms"):
            raw = manifest.get(key)
            if raw is not None:
                try:
                    v = int(raw)
                    if v > 0:
                        return v
                except (TypeError, ValueError):
                    pass
        runtime = manifest.get("runtime")
        if isinstance(runtime, dict):
            for key in ("chunk_duration_ms", "chunk_ms"):
                raw = runtime.get(key)
                if raw is not None:
                    try:
                        v = int(raw)
                        if v > 0:
                            return v
                    except (TypeError, ValueError):
                        pass
        try:
            cfg_ms = int(getattr(config, "chunk_duration_ms", 0) or 0)
            if cfg_ms > 0:
                return cfg_ms
        except (TypeError, ValueError):
            pass
        return 560

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
    async def _actor_next_result(actor, session_id: str) -> dict | None:
        return await actor.next_result.remote(session_id=session_id)

    @staticmethod
    async def _feed_loop(
        actor, session_id: str, request_iterator, *, target_bytes: int
    ) -> None:
        chunk_count = 0
        buf = bytearray()
        async for request in request_iterator:
            if request.HasField("config"):
                continue
            if not request.HasField("audio_chunk"):
                continue
            chunk_count += 1
            n = len(request.audio_chunk)
            if chunk_count == 1:
                logger.info(
                    "gateway first audio chunk",
                    session_id=session_id,
                    bytes=n,
                )
            elif chunk_count % 200 == 0:
                logger.info(
                    "gateway audio progress",
                    session_id=session_id,
                    chunks=chunk_count,
                )
            buf.extend(request.audio_chunk)
            while len(buf) >= target_bytes:
                out = bytes(buf[:target_bytes])
                del buf[:target_bytes]
                # Ray Client: must await .remote() or the call may never be committed.
                await actor.feed.remote(session_id=session_id, chunk_bytes=out)

        # Flush tail on stream end (may help the model finalize).
        if buf:
            await actor.feed.remote(session_id=session_id, chunk_bytes=bytes(buf))

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
    ray_address = os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
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
