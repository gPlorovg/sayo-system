"""Abstract base class for all STT model adapters.

Every STT engine packaged into a per-model docker image must subclass
`BaseSTTModel` and implement load / transcribe / transcribe_stream / unload.

This file is the binary-compatible interface shipped both inside actor_image
and inside the per-model image produced by the external build tool.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np


@dataclass
class STTConfig:
    """Configuration passed to a model adapter at load time."""

    model_id: str
    language_code: str = "en"
    sample_rate: int = 16_000
    device: str = "cuda"
    extra: dict = field(default_factory=dict)


@dataclass
class STTResult:
    """A single recognition result returned by the model."""

    transcript: str
    is_final: bool
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseSTTModel(ABC):
    """Abstract adapter interface for Speech-to-Text models."""

    @abstractmethod
    def load(self, config: STTConfig) -> None:
        """Heavy init: download/load weights into RAM/VRAM."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> STTResult:
        """Single-shot offline inference over a 1-D float32 mono buffer."""

    @abstractmethod
    def transcribe_stream(
        self,
        audio_chunks: Iterator[np.ndarray],
    ) -> Iterator[STTResult]:
        """Generator-based streaming inference."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free GPU/RAM."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True when the model is ready for inference."""

    @property
    @abstractmethod
    def model_info(self) -> dict:
        """Human-readable metadata dict (name, params, lang, ...)."""

    def transcribe_timed(self, audio: np.ndarray) -> STTResult:
        """`transcribe` plus auto-filled `latency_ms` and `metadata['rtf']`."""
        t0 = time.perf_counter()
        result = self.transcribe(audio)
        elapsed = time.perf_counter() - t0
        sample_rate = (
            self._config.sample_rate  # type: ignore[attr-defined]
            if hasattr(self, "_config")
            else 16_000
        )
        duration_s = len(audio) / sample_rate if sample_rate else 0.0
        result.latency_ms = elapsed * 1000
        result.metadata["rtf"] = (
            round(elapsed / duration_s, 4) if duration_s > 0 else 0.0
        )
        return result
