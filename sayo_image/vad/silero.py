"""Silero VAD wrapper that loads a pre-baked ONNX model from disk.

Weights are baked into actor_image at build time
(`/opt/silero/silero_vad.onnx`). No `torch.hub`, no network at runtime.
The class is stateful per-session and not thread-safe — instantiate one
per `_SessionState` inside `TranscriptActor`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_VAD_PATH = "/opt/silero/silero_vad.onnx"
SUPPORTED_SAMPLE_RATES = (8_000, 16_000)
WINDOW_SAMPLES = {8_000: 256, 16_000: 512}


def _resolve_model_path(path: str | os.PathLike[str] | None) -> Path:
    candidate = Path(path or os.environ.get("SILERO_VAD_PATH", DEFAULT_VAD_PATH))
    if not candidate.exists():
        raise FileNotFoundError(
            f"silero_vad.onnx not found at {candidate}. "
            "Bake it into actor_image at build time."
        )
    return candidate


class SileroVAD:
    """Speech / non-speech gate over int-valued window-aligned chunks."""

    def __init__(
        self,
        threshold: float,
        min_silence_ms: int,
        sample_rate: int,
        model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"SileroVAD supports {SUPPORTED_SAMPLE_RATES} Hz, got {sample_rate}"
            )
        import onnxruntime as ort

        self._threshold = float(threshold)
        self._min_silence_samples = int(sample_rate * max(0, min_silence_ms) / 1000)
        self._sample_rate = sample_rate
        self._window = WINDOW_SAMPLES[sample_rate]
        self._buffer = np.zeros(0, dtype=np.float32)
        self._silence_run = 0

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(_resolve_model_path(model_path)),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(sample_rate, dtype=np.int64)

    @property
    def threshold(self) -> float:
        return self._threshold

    def is_speech(self, chunk_f32: np.ndarray) -> bool:
        """Return True if the trailing window of `chunk_f32` looks like speech.

        We accumulate samples, run the ONNX VAD on full windows, and treat
        the chunk as speech if any window crossed the threshold OR the
        accumulated silence run is shorter than `min_silence_ms`.
        """
        if chunk_f32.size == 0:
            return self._silence_run < self._min_silence_samples

        self._buffer = np.concatenate(
            (self._buffer, np.asarray(chunk_f32, dtype=np.float32).reshape(-1))
        )
        any_speech = False
        while self._buffer.size >= self._window:
            window = self._buffer[: self._window].reshape(1, -1)
            self._buffer = self._buffer[self._window :]
            prob, self._state = self._sess.run(
                None,
                {"input": window, "state": self._state, "sr": self._sr},
            )
            speech_prob = float(prob[0, 0])
            if speech_prob >= self._threshold:
                any_speech = True
                self._silence_run = 0
            else:
                self._silence_run += self._window

        return any_speech or self._silence_run < self._min_silence_samples

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._silence_run = 0
