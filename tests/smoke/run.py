"""End-to-end smoke probe.

Steps:
  1. HealthCheck the Gateway, pick the first model, log its descriptor.
  2. Stream `--duration` seconds of silence at the model's sample rate
     and quantization through StreamingRecognize.
  3. Print the count of received responses and last transcript (if any).

Designed to run after `sayoctl register-model ...` populated the catalog.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import grpc
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from proto import sayo_pb2, sayo_pb2_grpc  # noqa: E402


def _silence(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


def _encode(audio: np.ndarray, quant: int) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    if quant == sayo_pb2.AUDIO_QUANTIZATION_PCM_S16LE:
        return (audio * 32767.0).astype(np.int16).tobytes()
    return audio.astype(np.float32, copy=False).tobytes()


async def run(host: str, port: int, duration_s: float) -> int:
    channel = grpc.aio.insecure_channel(f"{host}:{port}")
    stub = sayo_pb2_grpc.SayoServiceStub(channel)
    health = await stub.HealthCheck(sayo_pb2.HealthCheckRequest())
    if not health.ready or not health.models:
        print(
            f"FAIL: gateway not ready or empty catalog: ready={health.ready} "
            f"models={len(health.models)} message={health.message}"
        )
        await channel.close()
        return 2
    descriptor = health.models[0]
    print(
        "Using model:",
        descriptor.model_id,
        f"sr={descriptor.sample_rate_hertz}",
        f"chunk_ms={descriptor.chunk_duration_ms}",
        f"quant={descriptor.audio_quantization}",
    )

    cfg = sayo_pb2.StreamingConfig(
        model_id=descriptor.model_id,
        language_code=descriptor.language_code,
        interim_results=descriptor.supports_interim_results,
        sample_rate_hertz=descriptor.sample_rate_hertz,
        audio_quantization=descriptor.audio_quantization,
        chunk_duration_ms=descriptor.chunk_duration_ms,
        vad_threshold=0.0,
        vad_min_silence_duration_ms=0,
    )

    samples_per_chunk = max(
        1,
        int(descriptor.sample_rate_hertz * descriptor.chunk_duration_ms / 1000),
    )
    total_chunks = max(1, int(duration_s * 1000 / max(1, descriptor.chunk_duration_ms)))

    async def request_iter():
        yield sayo_pb2.StreamingRecognizeRequest(config=cfg)
        for _ in range(total_chunks):
            audio = _silence(samples_per_chunk)
            yield sayo_pb2.StreamingRecognizeRequest(
                audio_chunk=_encode(audio, descriptor.audio_quantization)
            )
            await asyncio.sleep(descriptor.chunk_duration_ms / 1000)

    received: list[sayo_pb2.StreamingRecognizeResponse] = []
    last_transcript = ""
    try:
        async for resp in stub.StreamingRecognize(request_iter()):
            received.append(resp)
            if resp.transcript:
                last_transcript = resp.transcript
    except grpc.aio.AioRpcError as exc:
        print(f"FAIL: gRPC error: {exc.code()}: {exc.details()}")
        await channel.close()
        return 3
    finally:
        await channel.close()

    print(
        f"OK: received {len(received)} responses; last_transcript={last_transcript!r}"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("SAYO_GATEWAY_HOST", "localhost"))
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SAYO_GATEWAY_PORT", "50051")),
    )
    p.add_argument("--duration", type=float, default=4.0)
    args = p.parse_args()
    sys.exit(asyncio.run(run(args.host, args.port, args.duration)))


if __name__ == "__main__":
    main()
