"""FastAPI front-door for Sayo Model Registry.

Hot path (read-only):
  GET  /v1/models                  -> list manifests
  GET  /v1/models/{model_id}       -> single manifest
  GET  /v1/health                  -> liveness

Cold path (admin):
  POST   /v1/admin/models          -> wrap a per-model image into actor_image
  DELETE /v1/admin/models/{id}     -> remove from catalog (optionally rmi)
  GET    /v1/admin/state           -> internal snapshot for debugging
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sayo_host.common.logging import configure_logging
from sayo_host.registry.catalog import ManifestCatalog
from sayo_host.registry.weights import WeightsDownloader
from sayo_host.registry.wrapper import WrapperError, WrapperPipeline

configure_logging("model-registry")
logger = structlog.get_logger("registry.service")

CATALOG_ROOT = Path(os.environ.get("REGISTRY_ROOT", "/manifests"))
WEIGHTS_CACHE = Path(os.environ.get("WEIGHTS_CACHE_DIR", "/cache"))
WEIGHTS_STAGING = Path(os.environ.get("WEIGHTS_STAGING_DIR", "/staging"))

catalog = ManifestCatalog(CATALOG_ROOT)
weights = WeightsDownloader(cache_root=WEIGHTS_CACHE, staging_root=WEIGHTS_STAGING)
pipeline: WrapperPipeline | None = None


def _pipeline() -> WrapperPipeline:
    global pipeline
    if pipeline is None:
        pipeline = WrapperPipeline(catalog=catalog, weights=weights)
    return pipeline


class RegisterRequest(BaseModel):
    image_ref: str = Field(..., description="Per-model docker image reference")
    force: bool = False


class UnregisterResponse(BaseModel):
    deleted: bool


app = FastAPI(title="Sayo Model Registry", version="1.0.0")


@app.get("/v1/health")
def health() -> dict:
    return {"ready": True, "models": len(catalog.list())}


@app.get("/v1/models")
def list_models() -> list[dict]:
    return [m.to_dict() for m in catalog.list()]


@app.get("/v1/models/{model_id}")
def get_model(model_id: str) -> dict:
    manifest = catalog.get(model_id)
    if manifest is None:
        raise HTTPException(404, f"unknown model_id: {model_id}")
    return manifest.to_dict()


@app.post("/v1/admin/models")
def register_model(req: RegisterRequest) -> dict:
    try:
        manifest = _pipeline().register(req.image_ref, force=req.force)
    except WrapperError as exc:
        logger.warning("registration failed", error=str(exc), image_ref=req.image_ref)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("registration crashed", image_ref=req.image_ref)
        raise HTTPException(500, str(exc)) from exc
    return manifest.to_dict()


@app.delete("/v1/admin/models/{model_id}")
def unregister_model(model_id: str, remove_image: bool = False) -> UnregisterResponse:
    deleted = _pipeline().unregister(model_id, remove_image=remove_image)
    return UnregisterResponse(deleted=deleted)


@app.get("/v1/admin/state")
def admin_state() -> dict:
    return {
        "manifests_dir": str(catalog.manifests_dir),
        "weights_cache": str(WEIGHTS_CACHE),
        "weights_staging": str(WEIGHTS_STAGING),
        "models": [m.model_id for m in catalog.list()],
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "sayo_host.registry.service:app",
        host=os.environ.get("REGISTRY_HOST", "0.0.0.0"),
        port=int(os.environ.get("REGISTRY_PORT", "8000")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
