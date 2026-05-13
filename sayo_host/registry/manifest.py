"""ModelManifest dataclass + JSON (de)serialization.

Manifests are the public runtime contract of the Registry: Gateway and
Router consume them via HTTP and never touch model.yaml or docker.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _audio_quant_to_proto(value: str) -> str:
    return {
        "pcm_s16le": "AUDIO_QUANTIZATION_PCM_S16LE",
        "pcm_f32le": "AUDIO_QUANTIZATION_PCM_F32LE",
    }.get(value.lower(), "AUDIO_QUANTIZATION_PCM_F32LE")


@dataclass(slots=True)
class VADInfo:
    supported: bool = False
    default_threshold: float = 0.5
    default_min_silence_ms: int = 500


@dataclass(slots=True)
class ModelManifest:
    """Catalog entry for one registered model."""

    model_id: str
    source_image: str
    actor_image_tag: str
    model_dir: str
    description: str
    language_code: str
    sample_rate_hertz: int
    audio_quantization: str
    chunk_duration_ms: int
    supports_interim_results: bool
    latency_ms: float
    min_vram_gb: float
    max_concurrent_sessions: int
    vad: VADInfo = field(default_factory=VADInfo)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_audio_quant_proto(self) -> str:
        return _audio_quant_to_proto(self.audio_quantization)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelManifest":
        raw = dict(raw)
        raw.pop("weights", None)  # legacy catalog entries
        vad_raw = raw.get("vad") or {}
        vad = VADInfo(
            supported=bool(vad_raw.get("supported", False)),
            default_threshold=float(vad_raw.get("default_threshold", 0.5)),
            default_min_silence_ms=int(vad_raw.get("default_min_silence_ms", 500)),
        )
        return cls(
            model_id=str(raw["model_id"]),
            source_image=str(raw["source_image"]),
            actor_image_tag=str(raw["actor_image_tag"]),
            model_dir=str(raw["model_dir"]),
            description=str(raw.get("description", "")),
            language_code=str(raw.get("language_code", "en")),
            sample_rate_hertz=int(raw.get("sample_rate_hertz", 16_000)),
            audio_quantization=str(raw.get("audio_quantization", "pcm_f32le")),
            chunk_duration_ms=int(raw.get("chunk_duration_ms", 0)),
            supports_interim_results=bool(raw.get("supports_interim_results", True)),
            latency_ms=float(raw.get("latency_ms", 0.0)),
            min_vram_gb=float(raw.get("min_vram_gb", 0.0)),
            max_concurrent_sessions=int(raw.get("max_concurrent_sessions", 1)),
            vad=vad,
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ModelManifest":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
