"""Ray actor that owns one loaded STT model and N concurrent sessions.

Heavy-actor pattern: this code runs INSIDE actor_image, the container itself
joins the Ray cluster as a worker node (see `bootstrap.py`). All per-session
state lives in `_SessionState`; the model adapter is shared.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import ray
import structlog

from sayo_image.transcript_actor.audio import chunk_to_float32

logger = structlog.get_logger("transcript_actor")

# Protobuf `AudioQuantization` numeric values (see proto/sayo.proto).
_PROTO_QUANT = {"pcm_s16le": 1, "pcm_f32le": 2}

_SENTINEL: Any = object()


@dataclass
class _SessionState:
    session_id: str
    audio_in: asyncio.Queue
    results_out: asyncio.Queue
    quantization: int
    vad: Any | None = None
    worker_thread: threading.Thread | None = None
    chunk_idx: int = 0
    bytes_in: int = 0
    last_feed_at: float = 0.0
    started_at: float = field(default_factory=time.time)
    closed: bool = False


@ray.remote
class TranscriptActor:
    def __init__(
        self,
        model_dir: str,
        device: str,
        max_concurrent_sessions: int = 4,
    ) -> None:
        from sayo_image.model_repository.model_repository import ModelRepository

        self._model_dir = model_dir
        self._device = device
        self._max_sessions = max(1, int(max_concurrent_sessions))

        self._repo = ModelRepository(model_dir)
        self._adapter = self._repo.create_adapter(device=device, auto_load=True)
        self._model_id = self._repo.entry.model_id
        self._sample_rate = self._repo.entry.sample_rate

        self._sessions: dict[str, _SessionState] = {}
        self._last_used = time.time()
        logger.info(
            "TranscriptActor ready",
            model_id=self._model_id,
            device=device,
            max_concurrent_sessions=self._max_sessions,
        )

    def _session_quantization(self, quantization: int | None) -> int:
        if quantization is not None:
            return int(quantization)
        rt = self._repo.raw.get("runtime") or {}
        key = str(rt.get("audio_quantization", "pcm_f32le")).lower()
        return int(_PROTO_QUANT.get(key, _PROTO_QUANT["pcm_f32le"]))

    async def open_session(
        self,
        session_id: str,
        vad_cfg: dict | None = None,
        quantization: int | None = None,
    ) -> dict:
        if session_id in self._sessions:
            raise RuntimeError(f"session already open: {session_id}")
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(
                f"actor at capacity ({self._max_sessions}), reject {session_id}"
            )

        loop = asyncio.get_running_loop()
        state = _SessionState(
            session_id=session_id,
            audio_in=asyncio.Queue(maxsize=256),
            results_out=asyncio.Queue(maxsize=256),
            quantization=self._session_quantization(quantization),
        )
        state.vad = self._maybe_build_vad(vad_cfg)

        worker = threading.Thread(
            target=self._run_adapter_stream,
            args=(state, loop),
            name=f"adapter-stream-{session_id[:8]}",
            daemon=True,
        )
        state.worker_thread = worker
        worker.start()

        self._sessions[session_id] = state
        self._last_used = time.time()
        logger.info(
            "session opened",
            session_id=session_id,
            model_id=self._model_id,
            vad="on" if state.vad else "off",
            quantization=state.quantization,
        )
        return {"actor_sample_rate": self._sample_rate, "model_id": self._model_id}

    async def feed(self, session_id: str, chunk_bytes: bytes) -> None:
        state = self._sessions.get(session_id)
        if state is None or state.closed:
            if state is None:
                logger.warning(
                    "feed ignored: unknown session",
                    session_id=session_id,
                    known=list(self._sessions.keys()),
                )
            return
        state.chunk_idx += 1
        state.bytes_in += len(chunk_bytes)
        state.last_feed_at = time.time()
        if state.chunk_idx == 1:
            logger.info(
                "first audio chunk",
                session_id=session_id,
                bytes=len(chunk_bytes),
            )
        try:
            state.audio_in.put_nowait(chunk_bytes)
        except asyncio.QueueFull:
            logger.warning(
                "audio_in full, dropping chunk",
                session_id=session_id,
                chunks=state.chunk_idx,
            )
        self._last_used = time.time()

    async def next_result(self, session_id: str) -> dict | None:
        """One transcript payload from the session stream, or None when finished.

        Use this from Ray Client; ``results(..., num_returns='streaming')`` is not
        supported on the client protocol.
        """
        state = self._sessions.get(session_id)
        if state is None:
            logger.warning(
                "next_result: no session (closed or unknown)",
                session_id=session_id,
            )
            return None
        item = await state.results_out.get()
        if item is _SENTINEL:
            return None
        if isinstance(item, BaseException):
            logger.error(
                "adapter error",
                session_id=session_id,
                error=str(item),
            )
            return {
                "transcript": "",
                "is_final": True,
                "confidence": 0.0,
                "latency_ms": 0.0,
                "metadata": {"error": str(item)},
            }
        return item

    def debug_state(self, session_id: str) -> dict:
        """Small introspection payload for debugging live audio delivery."""
        state = self._sessions.get(session_id)
        if state is None:
            return {
                "session_id": session_id,
                "exists": False,
                "sessions": list(self._sessions.keys()),
            }
        return {
            "session_id": session_id,
            "exists": True,
            "chunk_idx": int(state.chunk_idx),
            "bytes_in": int(state.bytes_in),
            "audio_in_qsize": int(getattr(state.audio_in, "qsize", lambda: -1)()),
            "results_out_qsize": int(getattr(state.results_out, "qsize", lambda: -1)()),
            "last_feed_age_s": float(time.time() - float(state.last_feed_at))
            if state.last_feed_at
            else None,
            "vad": bool(state.vad is not None),
            "quantization": int(state.quantization),
        }

    async def results(self, session_id: str) -> AsyncIterator[dict]:
        """Streaming generator (Ray cluster driver only; not Ray Client)."""
        state = self._sessions.get(session_id)
        if state is None:
            return
        while True:
            item = await state.results_out.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                logger.error(
                    "adapter error",
                    session_id=session_id,
                    error=str(item),
                )
                yield {
                    "transcript": "",
                    "is_final": True,
                    "confidence": 0.0,
                    "latency_ms": 0.0,
                    "metadata": {"error": str(item)},
                }
                break
            yield item

    async def close_session(self, session_id: str) -> None:
        state = self._sessions.pop(session_id, None)
        if state is None:
            return
        state.closed = True
        with suppress(asyncio.QueueFull):
            state.audio_in.put_nowait(_SENTINEL)
        loop = asyncio.get_running_loop()
        if state.worker_thread is not None:
            await loop.run_in_executor(None, state.worker_thread.join, 10.0)
        with suppress(asyncio.QueueFull):
            state.results_out.put_nowait(_SENTINEL)
        logger.info("session closed", session_id=session_id, model_id=self._model_id)
        self._last_used = time.time()

    def health(self) -> dict:
        return {
            "model_id": self._model_id,
            "device": self._device,
            "sessions": len(self._sessions),
            "max_concurrent_sessions": self._max_sessions,
            "last_used": self._last_used,
            "pid": os.getpid(),
        }

    def get_loaded_model(self) -> dict:
        return {
            "model_id": self._model_id,
            "model_dir": self._model_dir,
            "sample_rate": self._sample_rate,
            "info": dict(self._adapter.model_info or {}),
        }

    def mark_used(self) -> None:
        self._last_used = time.time()

    def unload(self) -> None:
        for sid in list(self._sessions):
            state = self._sessions.pop(sid, None)
            if state is None:
                continue
            state.closed = True
            with suppress(asyncio.QueueFull):
                state.audio_in.put_nowait(_SENTINEL)
        with suppress(Exception):
            self._adapter.unload()
        logger.info("TranscriptActor unloaded", model_id=self._model_id)

    def _maybe_build_vad(self, vad_cfg: dict | None) -> Any | None:
        if os.environ.get("SAYO_DISABLE_VAD", "").lower() in {"1", "true", "yes"}:
            logger.info("VAD disabled via SAYO_DISABLE_VAD")
            return None
        if not vad_cfg:
            return None
        threshold = float(vad_cfg.get("threshold", 0.0))
        min_silence_ms = int(vad_cfg.get("min_silence_ms", 0))
        if threshold <= 0.0 or min_silence_ms <= 0:
            return None
        try:
            from sayo_image.vad.silero import SileroVAD

            return SileroVAD(
                threshold=threshold,
                min_silence_ms=min_silence_ms,
                sample_rate=self._sample_rate,
            )
        except Exception as exc:
            logger.warning("VAD disabled", reason=str(exc))
            return None

    def _run_adapter_stream(
        self, state: _SessionState, loop: asyncio.AbstractEventLoop
    ) -> None:
        logger.info(
            "adapter stream thread started",
            session_id=state.session_id,
            model_id=self._model_id,
            quantization=state.quantization,
            vad=state.vad is not None,
        )

        def chunks_iter():
            # First N chunks bypass VAD so Silero state can warm up and we never starve
            # the adapter when the gate mis-classifies the start of an utterance.
            warmup_chunks = max(0, int(os.environ.get("SAYO_VAD_WARMUP_CHUNKS", "6")))
            fed = 0
            while True:
                fut = asyncio.run_coroutine_threadsafe(state.audio_in.get(), loop)
                item = fut.result()
                if item is _SENTINEL or state.closed:
                    return
                if not isinstance(item, (bytes, bytearray, memoryview)):
                    continue
                audio = chunk_to_float32(bytes(item), state.quantization)
                fed += 1
                if (
                    state.vad is not None
                    and fed > warmup_chunks
                    and not state.vad.is_speech(audio)
                ):
                    continue
                yield audio

        try:
            for result in self._adapter.transcribe_stream(chunks_iter()):
                payload = self._serialize_result(result, state)
                if payload is None:
                    continue
                asyncio.run_coroutine_threadsafe(
                    state.results_out.put(payload), loop
                ).result(timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "adapter stream failed",
                session_id=state.session_id,
                error=str(exc),
            )
            with suppress(Exception):
                asyncio.run_coroutine_threadsafe(
                    state.results_out.put(exc), loop
                ).result(timeout=5.0)
        finally:
            with suppress(Exception):
                asyncio.run_coroutine_threadsafe(
                    state.results_out.put(_SENTINEL), loop
                ).result(timeout=5.0)

    @staticmethod
    def _serialize_result(result: Any, state: _SessionState) -> dict | None:
        is_final = bool(getattr(result, "is_final", False))
        meta_raw = getattr(result, "metadata", None) or {}
        if not isinstance(meta_raw, dict):
            meta_raw = {}
        delta = str(getattr(result, "transcript", "") or "").strip()
        full_text = str(meta_raw.get("full_text", "") or "").strip()
        display = full_text or delta
        if not display and not is_final:
            return None
        metadata = {str(k): str(v) for k, v in meta_raw.items()}
        metadata.setdefault("chunk_idx", str(state.chunk_idx))
        return {
            "transcript": display,
            "is_final": is_final,
            "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
            "latency_ms": float(getattr(result, "latency_ms", 0.0) or 0.0),
            "metadata": metadata,
        }
