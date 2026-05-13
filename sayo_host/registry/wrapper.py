"""Wrapping pipeline: per-model image -> actor_image.

Steps (cold-path, executed by `POST /v1/admin/models`):
  1. resolve <image_ref>: use a local image if present (`docker image inspect`),
     otherwise `docker pull` (unless `local_only`, then error if missing locally)
  2. docker inspect to read ENV MODEL_NAME -> /app/models/<MODEL_NAME>/model.yaml
  3. docker create + cp + rm to extract model.yaml as bytes
  4. parse model.yaml (yaml.safe_load) -> runtime / vad / etc.
  5. assemble build context with sayo_image/, vad-weights/, proto/
  6. docker build -f Dockerfile.actor -t sayo-actor-<slug>:<tag>
  7. optional docker push (distributed mode)
  8. persist ModelManifest in the catalog

Model weights are expected inside the pulled per-model image; the registry only
adds Sayo actor layers (transcript_actor, Silero VAD asset, proto stubs, deps).
"""

from __future__ import annotations

import datetime as _dt
import io
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog
import yaml

import docker
from sayo_host.registry.catalog import ManifestCatalog
from sayo_host.registry.manifest import ModelManifest, VADInfo

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


EmitFn = Callable[[str, str], None] | None


def _emit(emit: EmitFn, phase: str, message: str) -> None:
    if emit and message is not None:
        emit(phase, message)


def _split_image_ref(image_ref: str) -> tuple[str, str | None]:
    """Split into (repository, tag). Digest references return (name, None)."""
    if "@" in image_ref:
        return image_ref.split("@", 1)[0], None
    if ":" in image_ref:
        left, right = image_ref.rsplit(":", 1)
        if "/" not in right:
            return left, right
    return image_ref, None


def _actor_slug_from_ref(image_ref: str, model_name: str, model_id: str) -> str:
    """sayo-model-nemo -> nemo; otherwise a safe slug from model_name / model_id."""
    repo, _tag = _split_image_ref(image_ref)
    short = repo.rsplit("/", 1)[-1]
    m = re.match(r"^sayo-model-(.+)$", short, re.IGNORECASE)
    if m:
        raw = m.group(1).lower()
        raw = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-._")
        if raw:
            return raw
    for candidate in (model_name, model_id):
        s = re.sub(r"[^a-z0-9._-]+", "-", str(candidate).lower()).strip("-._")
        if s:
            return s
    return "model"


def _actor_tag_suffix(image_ref: str, src_tag: str | None) -> str:
    if "@sha256:" in image_ref:
        digest = image_ref.split("@sha256:", 1)[-1].strip()[:12]
        return digest if digest else "latest"
    if src_tag:
        return src_tag
    return "latest"


class WrapperPipeline:
    """High-level orchestration of the registration pipeline."""

    def __init__(
        self,
        catalog: ManifestCatalog,
        docker_client: docker.DockerClient | None = None,
        internal_registry: str | None = None,
        dockerfile_path: Path | None = None,
        build_sources: tuple[Path, ...] = DEFAULT_BUILD_SOURCES,
    ) -> None:
        self._catalog = catalog
        self._client = docker_client or docker.from_env()
        self._internal_registry = internal_registry or os.environ.get(
            "INTERNAL_DOCKER_REGISTRY"
        )
        self._dockerfile = dockerfile_path or DEFAULT_DOCKERFILE
        self._build_sources = build_sources

    def register(
        self,
        image_ref: str,
        force: bool = False,
        *,
        local_only: bool = False,
        emit: EmitFn = None,
    ) -> ModelManifest:
        logger.info("registration started", image_ref=image_ref, local_only=local_only)
        _emit(emit, "start", f"image_ref={image_ref!r} local_only={local_only}")

        image = self._ensure_image(image_ref, local_only=local_only, emit=emit)
        model_name = self._read_model_name(image)
        yaml_path = f"/app/models/{model_name}/model.yaml"
        _emit(emit, "yaml", f"extract {yaml_path} (MODEL_NAME={model_name!r})")
        raw_yaml = self._extract_file(image_ref, yaml_path, emit=emit)
        manifest_raw: dict[str, Any] = yaml.safe_load(raw_yaml) or {}

        model_id = str(manifest_raw.get("id") or model_name)
        if not force and self._catalog.get(model_id) is not None:
            logger.info("model already registered", model_id=model_id)
            existing = self._catalog.get(model_id)
            if existing is not None:
                _emit(
                    emit,
                    "catalog",
                    f"model {model_id!r} already registered — returning existing",
                )
                return existing

        actor_image_tag = self._build_actor_image(
            image_ref=image_ref,
            model_id=model_id,
            model_name=model_name,
            manifest_raw=manifest_raw,
            emit=emit,
        )

        if self._internal_registry:
            _emit(emit, "push", f"pushing {actor_image_tag!r} …")
            try:
                self._client.images.push(actor_image_tag)
                logger.info("actor image pushed", tag=actor_image_tag)
                _emit(emit, "push", "push finished")
            except Exception as exc:  # noqa: BLE001
                logger.warning("push failed", tag=actor_image_tag, error=str(exc))
                _emit(emit, "push", f"push failed (non-fatal): {exc}")

        manifest = self._build_manifest(
            model_id=model_id,
            model_name=model_name,
            image_ref=image_ref,
            actor_image_tag=actor_image_tag,
            manifest_raw=manifest_raw,
        )
        _emit(emit, "catalog", f"save manifest for model_id={model_id!r}")
        self._catalog.upsert(manifest)
        logger.info(
            "registration complete",
            model_id=model_id,
            actor_image_tag=actor_image_tag,
        )
        _emit(emit, "catalog", f"registered actor_image_tag={actor_image_tag!r}")
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

    def _ensure_image(self, image_ref: str, *, local_only: bool, emit: EmitFn = None):
        if local_only:
            logger.info("docker image resolve (local only)", image_ref=image_ref)
            _emit(emit, "image", f"resolve local-only {image_ref!r}")
            try:
                return self._client.images.get(image_ref)
            except docker.errors.ImageNotFound as exc:
                raise WrapperError(
                    f"Image not found locally: {image_ref!r}. "
                    "Load or pull it first, or omit local_only / --local."
                ) from exc
        try:
            img = self._client.images.get(image_ref)
            logger.info("docker image (local)", image_ref=image_ref)
            _emit(emit, "image", f"using local image {image_ref!r}")
            return img
        except docker.errors.ImageNotFound:
            logger.info("docker pull", image_ref=image_ref)
            _emit(emit, "image", f"pulling {image_ref!r} …")
            try:
                return self._client.images.pull(image_ref)
            except docker.errors.NotFound as exc:
                raise WrapperError(
                    f"Could not resolve {image_ref!r}: not present on the Docker "
                    "daemon used by model-registry, and pulling it failed "
                    f"({getattr(exc, 'explanation', None) or str(exc)}). "
                    "Load or pull the image on that same host (the one that owns "
                    "/var/run/docker.sock mounted into the registry), using the "
                    "exact image_ref, or use a full registry/repo:tag if it lives "
                    "in a private registry."
                ) from exc

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

    def _extract_file(self, image_ref: str, path: str, *, emit: EmitFn = None) -> bytes:
        _emit(emit, "extract", f"docker create + get_archive {path!r}")
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
                data = fp.read()
                _emit(emit, "extract", f"read {len(data)} bytes from {path!r}")
                return data
        finally:
            container.remove(force=True)

    def _build_actor_image(
        self,
        image_ref: str,
        model_id: str,
        model_name: str,
        manifest_raw: dict[str, Any],
        *,
        emit: EmitFn = None,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix=f"sayo-build-{model_id}-") as tmp:
            ctx = Path(tmp)
            for src in self._build_sources:
                if not src.exists():
                    continue
                shutil.copytree(src, ctx / src.name, dirs_exist_ok=True)
            shutil.copy2(self._dockerfile, ctx / "Dockerfile.actor")

            slug = _actor_slug_from_ref(image_ref, model_name, model_id)
            tag = self._compose_actor_tag(slug, image_ref)
            actor_runtime = os.environ.get(
                "SAYO_ACTOR_RUNTIME_IMAGE", "sayo-actor-runtime:latest"
            )
            buildargs = {
                "MODEL_IMAGE": image_ref,
                "MODEL_NAME": model_name,
                "ACTOR_RUNTIME": actor_runtime,
            }
            labels = {
                "org.sayo.model.id": model_id,
                "org.sayo.model.name": model_name,
                "org.sayo.actor.slug": slug,
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
            _emit(emit, "docker_build", f"docker build -t {tag!r} (streaming logs) …")
            stream = self._client.api.build(
                path=str(ctx),
                dockerfile="Dockerfile.actor",
                tag=tag,
                buildargs=buildargs,
                labels=labels,
                rm=True,
                forcerm=True,
                pull=False,
                decode=True,
            )
            for chunk in stream:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise WrapperError(str(chunk["error"]))
                ed = chunk.get("errorDetail")
                if isinstance(ed, dict) and ed.get("message"):
                    raise WrapperError(str(ed["message"]))
                line = (chunk.get("stream") or "").rstrip("\r\n")
                if line:
                    msg = line if len(line) <= 8192 else line[:8192] + "…"
                    _emit(emit, "docker_build", msg)
                status = chunk.get("status")
                if status and not line:
                    img_id = chunk.get("id", "")
                    tail = f"{img_id} {status}".strip()
                    if tail:
                        _emit(emit, "docker_build", tail[:8192])
            _emit(emit, "docker_build", f"build finished tag={tag!r}")
            try:
                self._client.images.get(tag)
            except docker.errors.ImageNotFound as exc:
                raise WrapperError(
                    f"docker build reported success but image missing: {tag!r}"
                ) from exc
            return tag

    def _compose_actor_tag(self, slug: str, image_ref: str) -> str:
        _repo, src_tag = _split_image_ref(image_ref)
        suffix = _actor_tag_suffix(image_ref, src_tag)
        name = f"sayo-actor-{slug}"
        if self._internal_registry:
            reg = self._internal_registry.rstrip("/")
            return f"{reg}/{name}:{suffix}"
        return f"{name}:{suffix}"

    def _build_manifest(
        self,
        model_id: str,
        model_name: str,
        image_ref: str,
        actor_image_tag: str,
        manifest_raw: dict[str, Any],
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
        )
