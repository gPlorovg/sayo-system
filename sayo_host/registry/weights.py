"""Weights downloader with sha256 verification and content-addressed cache."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger("registry.weights")


@dataclass(slots=True)
class WeightArtifact:
    """One weight blob declared in model.yaml."""

    relpath: str
    uri: str | None = None
    sha256: str | None = None
    required: bool = True


def parse_artifacts(raw: dict | None) -> list[WeightArtifact]:
    if not raw:
        return []
    items = raw.get("artifacts") if isinstance(raw, dict) else None
    out: list[WeightArtifact] = []
    if not isinstance(items, list):
        return out
    for entry in items:
        if not isinstance(entry, dict):
            continue
        relpath = entry.get("path") or entry.get("relpath")
        if not isinstance(relpath, str) or not relpath:
            continue
        out.append(
            WeightArtifact(
                relpath=relpath,
                uri=entry.get("uri") or None,
                sha256=entry.get("sha256") or None,
                required=bool(entry.get("required", True)),
            )
        )
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WeightsDownloader:
    """Download artifacts to `staging/<model_id>/<relpath>` using a sha cache."""

    def __init__(self, cache_root: str | Path, staging_root: str | Path) -> None:
        self._cache = Path(cache_root)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._staging = Path(staging_root)
        self._staging.mkdir(parents=True, exist_ok=True)

    def staging_dir(self, model_id: str) -> Path:
        path = self._staging / model_id
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fetch(
        self,
        model_id: str,
        artifact: WeightArtifact,
        target_root: Path,
    ) -> Path | None:
        """Place a verified artifact under `target_root/<relpath>`.

        Returns:
            local path of the staged file, or None if the artifact has no
            URI (it is expected to already be present inside the per-model
            image; we do not bake anything for it).
        """
        target = target_root / artifact.relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if not artifact.uri:
            logger.info(
                "weight has no uri; skip download",
                model_id=model_id,
                relpath=artifact.relpath,
            )
            return None

        cached = self._lookup_cache(artifact)
        if cached is not None:
            shutil.copy2(cached, target)
            logger.info(
                "weight from cache",
                model_id=model_id,
                relpath=artifact.relpath,
                sha256=artifact.sha256,
            )
            return target

        logger.info(
            "downloading weight",
            model_id=model_id,
            relpath=artifact.relpath,
            uri=artifact.uri,
        )
        with httpx.stream(
            "GET", artifact.uri, timeout=300.0, follow_redirects=True
        ) as r:
            r.raise_for_status()
            with open(target, "wb") as fp:
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    fp.write(chunk)

        actual = _sha256(target)
        if artifact.sha256 and actual.lower() != artifact.sha256.lower():
            target.unlink(missing_ok=True)
            raise ValueError(
                f"sha256 mismatch for {artifact.relpath}: "
                f"expected {artifact.sha256}, got {actual}"
            )
        if artifact.sha256:
            cache_path = self._cache / artifact.sha256
            shutil.copy2(target, cache_path)
        return target

    def _lookup_cache(self, artifact: WeightArtifact) -> Path | None:
        if not artifact.sha256:
            return None
        candidate = self._cache / artifact.sha256
        return candidate if candidate.exists() else None
