"""Audio decoding helpers used inside TranscriptActor only.

Decoding wire bytes -> float32 happens exactly once, just before the model
adapter sees them. Host services NEVER decode audio.
"""

from __future__ import annotations

import numpy as np

PCM_S16LE = 1
PCM_F32LE = 2


def chunk_to_float32(payload: bytes, quantization: int) -> np.ndarray:
    if quantization == PCM_S16LE:
        pcm16 = np.frombuffer(payload, dtype=np.int16)
        return (pcm16.astype(np.float32) / 32768.0).copy()
    return np.frombuffer(payload, dtype=np.float32).copy()
