"""Single-model repository (model.yaml + adapter factory).

Lives inside actor_image; reads `/app/models/<MODEL_NAME>/model.yaml` and
builds a `BaseSTTModel` instance via the adapter declared inside the
per-model image (`model_repository.adapters.<adapter_name>`).
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from .base import BaseSTTModel, STTConfig

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ModelEntry:
    name: str
    model_id: str
    adapter: str
    sample_rate: int = 16_000
    language_code: str = "en"
    description: str = ""
    latency: float = 0.0
    runtime: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class ModelRepository:
    """Loader for a single model directory (`<models_dir>/<MODEL_NAME>/`)."""

    def __init__(self, model_dir: str | Path) -> None:
        self._model_dir = Path(model_dir).resolve()
        if not self._model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self._model_dir}")

        self._yaml_path = self._model_dir / "model.yaml"
        if not self._yaml_path.exists():
            raise FileNotFoundError(f"model.yaml not found: {self._yaml_path}")

        self._raw = self._load_yaml(self._yaml_path)
        self._entry = self._build_entry(self._raw)

    @classmethod
    def from_model_dir(
        cls, models_dir: str | Path, model_name: str
    ) -> "ModelRepository":
        return cls(Path(models_dir) / model_name)

    @property
    def entry(self) -> ModelEntry:
        return self._entry

    @property
    def name(self) -> str:
        return self._entry.name

    @property
    def root_dir(self) -> Path:
        return self._model_dir

    @property
    def raw(self) -> dict[str, Any]:
        return dict(self._raw)

    def get_weights_paths(self) -> list[Path]:
        weights_block = self._raw.get("weights", {})
        artifacts = (
            weights_block.get("artifacts", [])
            if isinstance(weights_block, dict)
            else []
        )
        if not isinstance(artifacts, list):
            return []

        paths: list[Path] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            rel_path = artifact.get("path")
            if isinstance(rel_path, str) and rel_path:
                paths.append((self.root_dir / rel_path).resolve())
        return paths

    def build_config(
        self,
        *,
        device: str,
        model_id: str | None = None,
        extra_overrides: dict[str, Any] | None = None,
    ) -> STTConfig:
        extra = dict(self._entry.extra)
        extra["weights"] = [str(p) for p in self.get_weights_paths()]
        if extra_overrides:
            extra.update(extra_overrides)

        return STTConfig(
            model_id=model_id or self._entry.model_id,
            language_code=self._entry.language_code,
            sample_rate=self._entry.sample_rate,
            device=device,
            extra=extra,
        )

    def create_adapter(
        self,
        *,
        device: str = "cuda",
        auto_load: bool = True,
        model_id: str | None = None,
        extra_overrides: dict[str, Any] | None = None,
    ) -> BaseSTTModel:
        adapter = self._create_adapter_instance(self._entry.adapter)
        config = self.build_config(
            device=device, model_id=model_id, extra_overrides=extra_overrides
        )
        if auto_load:
            adapter.load(config)
            logger.info(
                "Loaded model",
                model=self.name,
                adapter=self._entry.adapter,
                device=device,
            )
        return adapter

    def validate_files(self) -> list[str]:
        issues: list[str] = []
        for weight_path in self.get_weights_paths():
            if not weight_path.exists():
                issues.append(f"weight not found: {weight_path}")
        return issues

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid model.yaml at {path}: expected mapping.")
        return raw

    def _build_entry(self, raw: dict[str, Any]) -> ModelEntry:
        model_id = raw.get("id")
        adapter = raw.get("adapter")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                f"Invalid model.yaml at {self._yaml_path}: 'id' is required string."
            )
        if not isinstance(adapter, str) or not adapter:
            raise ValueError(
                f"Invalid model.yaml at {self._yaml_path}: \
                'adapter' is required string."
            )

        known = {
            "id",
            "adapter",
            "description",
            "language_code",
            "sample_rate",
            "latency",
            "runtime",
            "weights",
        }
        extra = {k: v for k, v in raw.items() if k not in known}
        runtime = raw.get("runtime", {})
        return ModelEntry(
            name=self._model_dir.name,
            model_id=model_id,
            adapter=adapter,
            description=str(raw.get("description", "")),
            language_code=str(raw.get("language_code", "en")),
            sample_rate=int(raw.get("sample_rate", 16_000)),
            latency=float(raw.get("latency", 0.0)),
            runtime=runtime if isinstance(runtime, dict) else {},
            extra=extra,
        )

    def _create_adapter_instance(self, adapter_name: str) -> BaseSTTModel:
        module_name = f"model_repository.adapters.{adapter_name}"
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import adapter module '{module_name}': {exc}"
            ) from exc

        adapter_bases: list[type] = [BaseSTTModel]
        root_pkg = module_name.split(".", 1)[0] if module_name else ""
        if root_pkg:
            try:
                ext_base_mod = importlib.import_module(f"{root_pkg}.base")
                ext_base = getattr(ext_base_mod, "BaseSTTModel", None)
                if (
                    isinstance(ext_base, type)
                    and ext_base is not BaseSTTModel
                    and ext_base not in adapter_bases
                ):
                    adapter_bases.append(ext_base)
            except ImportError:
                pass

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in adapter_bases:
                continue
            for candidate in adapter_bases:
                try:
                    if issubclass(cls, candidate):
                        return cls()
                except TypeError:
                    continue

        raise ValueError(
            f"No BaseSTTModel subclass found in adapter module '{module_name}'."
        )
