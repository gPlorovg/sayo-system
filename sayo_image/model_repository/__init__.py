"""STT model interface baked inside actor_image.

This package contains ONLY the abstract interface (`BaseSTTModel`,
`STTConfig`, `STTResult`, `ModelRepository`, `ModelEntry`). Per-model
adapters (NeMo / GigaAM / etc.) live INSIDE the per-model docker image
and are imported at runtime via the same module path.

The version constant must stay binary-compatible with whatever the external
model-build tool ships in per-model images.
"""

from .base import BaseSTTModel, STTConfig, STTResult
from .model_repository import ModelEntry, ModelRepository

MODEL_REPOSITORY_VERSION = "1.0.0"

__all__ = [
    "BaseSTTModel",
    "STTConfig",
    "STTResult",
    "ModelRepository",
    "ModelEntry",
    "MODEL_REPOSITORY_VERSION",
]
