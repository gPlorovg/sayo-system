"""Wrapping pipeline: per-model image -> actor_image.

Steps (cold-path, executed by `POST /v1/admin/models`):
  1. docker pull <image_ref>
  2. docker inspect to read ENV MODEL_NAME -> /app/models/<MODEL_NAME>/model.yaml
  3. docker create + cp + rm to extract model.yaml as bytes
  4. parse model.yaml (yaml.safe_load) -> ModelEntry / weights / runtime
  5. download every weight artifact with a uri (sha256-verified)
  6. assemble build context with sayo_image/, vad_weights, proto/, weights/
  7. docker build -f Dockerfile.actor -t <internal_registry>/sayo-actor/<id>:<tag>
  8. optional docker push (distributed mode)
  9. persist ModelManifest in the catalog
"""

from __future__ import annotations

import datetime as _dt
import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import structlog
import yaml

import docker
from sayo_host.registry.catalog import ManifestCatalog
from sayo_host.registry.manifest import (
    ModelManifest,
    VADInfo,
    WeightArtifactInfo,
    WeightsInfo,
)
from sayo_host.registry.weights import WeightsDownloader, parse_artifacts

logger = structlog.get_logger("registry.wrapper")


def _default_repo_root() -> Path:
    """Locate the build context root (`/app` inside the registry image,
    `<repo>/sayo-backend-system/` for local dev)."""
    override = os.environ.get("SAYO_REPO_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[3]):
        if (candidate / "sayo_image").is_dir():
            return candidate
    return here.parents[2]


REPO_ROOT = _default_repo_root()
DEFAULT_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.actor"
DEFAULT_BUILD_SOURCES = (
    REPO_ROOT / "sayo_image",
    REPO_ROOT / "vad-weights",
    REPO_ROOT / "proto",
)


class WrapperError(RuntimeError):
    pass


class WrapperPipeline:
    """High-level orchestration of the registration pipeline."""

    def __init__(
        self,
        catalog: ManifestCatalog,
        weights: WeightsDownloader,
        docker_client: docker.DockerClient | None = None,
        internal_registry: str | None = None,
        dockerfile_path: Path | None = None,
        build_sources: tuple[Path, ...] = DEFAULT_BUILD_SOURCES,
    ) -> None:
        self._catalog = catalog
        self._weights = weights
        self._client = docker_client or docker.from_env()
        self._internal_registry = internal_registry or os.environ.get(
            "INTERNAL_DOCKER_REGISTRY"
        )
        self._dockerfile = dockerfile_path or DEFAULT_DOCKERFILE
        self._build_sources = build_sources

    def register(self, image_ref: str, force: bool = False) -> ModelManifest:
        logger.info("registration started", image_ref=image_ref)

        image = self._pull(image_ref)
        model_name = self._read_model_name(image)
        yaml_path = f"/app/models/{model_name}/model.yaml"
        raw_yaml = self._extract_file(image_ref, yaml_path)
        manifest_raw: dict[str, Any] = yaml.safe_load(raw_yaml) or {}

        model_id = str(manifest_raw.get("id") or model_name)
        if not force and self._catalog.get(model_id) is not None:
            logger.info("model already registered", model_id=model_id)
            existing = self._catalog.get(model_id)
            if existing is not None:
                return existing

        weights_artifacts = parse_artifacts(manifest_raw.get("weights"))
        staging_dir = self._weights.staging_dir(model_id)
        baked_artifacts: list[WeightArtifactInfo] = []
        for art in weights_artifacts:
            self._weights.fetch(model_id, art, staging_dir)
            baked_artifacts.append(
                WeightArtifactInfo(relpath=art.relpath, sha256=art.sha256)
            )

        actor_image_tag = self._build_actor_image(
            image_ref=image_ref,
            model_id=model_id,
            model_name=model_name,
            staging_dir=staging_dir,
            manifest_raw=manifest_raw,
        )

        if self._internal_registry:
            try:
                self._client.images.push(actor_image_tag)
                logger.info("actor image pushed", tag=actor_image_tag)
            except Exception as exc:  # noqa: BLE001
                logger.warning("push failed", tag=actor_image_tag, error=str(exc))

        manifest = self._build_manifest(
            model_id=model_id,
            model_name=model_name,
            image_ref=image_ref,
            actor_image_tag=actor_image_tag,
            manifest_raw=manifest_raw,
            baked_artifacts=baked_artifacts,
        )
        self._catalog.upsert(manifest)
        logger.info(
            "registration complete",
            model_id=model_id,
            actor_image_tag=actor_image_tag,
        )
        return manifest

    def unregister(self, model_id: str, remove_image: bool = False) -> bool:
        manifest = self._catalog.get(model_id)
        deleted = self._catalog.delete(model_id)
        if remove_image and manifest is not None:
            try:
                self._client.images.remove(manifest.actor_image_tag, force=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "rmi failed",
                    image=manifest.actor_image_tag,
                    error=str(exc),
                )
        return deleted

    def _pull(self, image_ref: str):
        logger.info("docker pull", image_ref=image_ref)
        return self._client.images.pull(image_ref)

    def _read_model_name(self, image) -> str:
        env = (image.attrs.get("Config", {}) or {}).get("Env") or []
        for entry in env:
            if entry.startswith("MODEL_NAME="):
                value = entry.split("=", 1)[1].strip()
                if value:
                    return value
        labels = (image.attrs.get("Config", {}) or {}).get("Labels") or {}
        if isinstance(labels, dict):
            value = labels.get("org.sayo.model.dir")
            if isinstance(value, str) and value:
                return value
        raise WrapperError(
            "Per-model image must declare ENV MODEL_NAME or label org.sayo.model.dir"
        )

    def _extract_file(self, image_ref: str, path: str) -> bytes:
        container = self._client.containers.create(image_ref, command="true")
        try:
            try:
                stream, _ = container.get_archive(path)
            except docker.errors.NotFound as exc:
                raise WrapperError(f"file not found in image: {path}") from exc
            buffer = io.BytesIO()
            for chunk in stream:
                buffer.write(chunk)
            buffer.seek(0)
            with tarfile.open(fileobj=buffer) as tf:
                names = tf.getnames()
                if not names:
                    raise WrapperError(f"empty archive for {path}")
                member = tf.getmember(names[0])
                fp = tf.extractfile(member)
                if fp is None:
                    raise WrapperError(f"cannot read {path} from image")
                return fp.read()
        finally:
            container.remove(force=True)

    def _build_actor_image(
        self,
        image_ref: str,
        model_id: str,
        model_name: str,
        staging_dir: Path,
        manifest_raw: dict[str, Any],
    ) -> str:
        with tempfile.TemporaryDirectory(prefix=f"sayo-build-{model_id}-") as tmp:
            ctx = Path(tmp)
            for src in self._build_sources:
                if not src.exists():
                    continue
                shutil.copytree(src, ctx / src.name, dirs_exist_ok=True)
            shutil.copy2(self._dockerfile, ctx / "Dockerfile.actor")

            weights_target = ctx / "weights-staging" / model_name
            weights_target.mkdir(parents=True, exist_ok=True)
            if any(staging_dir.iterdir()):
                shutil.copytree(staging_dir, weights_target, dirs_exist_ok=True)

            tag = self._compose_tag(model_id, image_ref)
            buildargs = {"MODEL_IMAGE": image_ref, "MODEL_NAME": model_name}
            labels = {
                "org.sayo.model.id": model_id,
                "org.sayo.model.name": model_name,
                "org.sayo.model.sample_rate": str(
                    int(manifest_raw.get("sample_rate", 16_000))
                ),
                "org.sayo.model.vram_gb": str(
                    float((manifest_raw.get("runtime") or {}).get("min_vram_gb", 0.0))
                ),
                "org.sayo.actor.from_image": image_ref,
                "org.sayo.actor.built_at": _dt.datetime.now(
                    _dt.timezone.utc
                ).isoformat(),
            }
            logger.info("docker build", tag=tag, context=str(ctx))
            self._client.images.build(
                path=str(ctx),
                dockerfile="Dockerfile.actor",
                tag=tag,
                buildargs=buildargs,
                labels=labels,
                rm=True,
                forcerm=True,
                pull=False,
            )
            return tag

    def _compose_tag(self, model_id: str, image_ref: str) -> str:
        digest_short = (
            image_ref.split("@sha256:")[-1][:12] if "@" in image_ref else "latest"
        )
        if self._internal_registry:
            return f"{self._internal_registry}/sayo-actor/{model_id}:{digest_short}"
        return f"sayo-actor/{model_id}:{digest_short}"

    def _build_manifest(
        self,
        model_id: str,
        model_name: str,
        image_ref: str,
        actor_image_tag: str,
        manifest_raw: dict[str, Any],
        baked_artifacts: list[WeightArtifactInfo],
    ) -> ModelManifest:
        runtime = manifest_raw.get("runtime") or {}
        vad_raw = manifest_raw.get("vad") or {}
        return ModelManifest(
            model_id=model_id,
            source_image=image_ref,
            actor_image_tag=actor_image_tag,
            model_dir=f"/app/models/{model_name}",
            description=str(manifest_raw.get("description", "")),
            language_code=str(manifest_raw.get("language_code", "en")),
            sample_rate_hertz=int(manifest_raw.get("sample_rate", 16_000)),
            audio_quantization=str(
                runtime.get("audio_quantization", "pcm_f32le")
            ).lower(),
            chunk_duration_ms=int(
                manifest_raw.get("chunk_ms") or runtime.get("chunk_duration_ms") or 560
            ),
            supports_interim_results=bool(
                runtime.get("supports_interim_results", True)
            ),
            latency_ms=float(manifest_raw.get("latency", 0.0)),
            min_vram_gb=float(runtime.get("min_vram_gb", 0.0)),
            max_concurrent_sessions=int(runtime.get("max_concurrent_sessions", 1)),
            vad=VADInfo(
                supported=bool(vad_raw.get("supported", False)),
                default_threshold=float(vad_raw.get("default_threshold", 0.5)),
                default_min_silence_ms=int(vad_raw.get("default_min_silence_ms", 500)),
            ),
            weights=WeightsInfo(baked=True, artifacts=baked_artifacts),
        )
