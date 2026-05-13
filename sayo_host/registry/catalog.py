"""Filesystem-backed manifests catalog.

Manifests live under `<root>/manifests/<model_id>.json`. The catalog keeps
an in-memory cache and reloads from disk on startup.
"""

from __future__ import annotations

import threading
from pathlib import Path

from sayo_host.registry.manifest import ModelManifest


class ManifestCatalog:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._dir = self._root / "manifests"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, ModelManifest] = {}
        self._load_from_disk()

    @property
    def manifests_dir(self) -> Path:
        return self._dir

    def _load_from_disk(self) -> None:
        for f in sorted(self._dir.glob("*.json")):
            try:
                m = ModelManifest.from_json_file(f)
                self._cache[m.model_id] = m
            except Exception:  # noqa: BLE001
                continue

    def list(self) -> list[ModelManifest]:
        with self._lock:
            return list(self._cache.values())

    def get(self, model_id: str) -> ModelManifest | None:
        with self._lock:
            return self._cache.get(model_id)

    def upsert(self, manifest: ModelManifest) -> None:
        with self._lock:
            self._cache[manifest.model_id] = manifest
            target = self._dir / f"{manifest.model_id}.json"
            target.write_text(manifest.to_json(), encoding="utf-8")

    def delete(self, model_id: str) -> bool:
        with self._lock:
            existed = self._cache.pop(model_id, None) is not None
            target = self._dir / f"{model_id}.json"
            if target.exists():
                target.unlink()
            return existed
