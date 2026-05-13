"""FastAPI front-door for Sayo Model Registry.

Hot path (read-only):
  GET  /v1/models                  -> list manifests
  GET  /v1/models/{model_id}       -> single manifest
  GET  /v1/health                  -> liveness

Cold path (admin):
  POST   /v1/admin/models          -> wrap a per-model image into actor_image
  POST   /v1/admin/models/register-stream -> same, NDJSON progress stream
  DELETE /v1/admin/models/{id}     -> remove from catalog (optionally rmi)
  GET    /v1/admin/state           -> internal snapshot for debugging
  GET    /admin                    -> small HTML admin UI
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from sayo_host.common.logging import configure_logging
from sayo_host.registry.catalog import ManifestCatalog
from sayo_host.registry.wrapper import WrapperError, WrapperPipeline

configure_logging("model-registry")
logger = structlog.get_logger("registry.service")

CATALOG_ROOT = Path(os.environ.get("REGISTRY_ROOT", "/manifests"))

catalog = ManifestCatalog(CATALOG_ROOT)
pipeline: WrapperPipeline | None = None

_ADMIN_HTML = Path(__file__).resolve().parent / "static" / "admin.html"


def _pipeline() -> WrapperPipeline:
    global pipeline
    if pipeline is None:
        pipeline = WrapperPipeline(catalog=catalog)
    return pipeline


class RegisterRequest(BaseModel):
    image_ref: str = Field(..., description="Per-model docker image reference")
    force: bool = False
    local_only: bool = Field(
        False,
        description="If true, never pull: the image must already exist locally",
    )


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


@app.get("/admin", include_in_schema=False)
def admin_ui() -> HTMLResponse:
    if not _ADMIN_HTML.is_file():
        raise HTTPException(500, "admin UI missing from package")
    return HTMLResponse(
        _ADMIN_HTML.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/v1/admin/models")
def register_model(req: RegisterRequest) -> dict:
    try:
        manifest = _pipeline().register(
            req.image_ref, force=req.force, local_only=req.local_only
        )
    except WrapperError as exc:
        logger.warning("registration failed", error=str(exc), image_ref=req.image_ref)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("registration crashed", image_ref=req.image_ref)
        raise HTTPException(500, str(exc)) from exc
    return manifest.to_dict()


@app.post("/v1/admin/models/register-stream")
def register_model_stream(req: RegisterRequest) -> StreamingResponse:
    """Register model while streaming NDJSON lines."""

    def ndjson_iter():
        events: queue.Queue[str | None] = queue.Queue()

        def emit(phase: str, message: str) -> None:
            line = json.dumps(
                {"phase": phase, "message": message},
                ensure_ascii=False,
            )
            events.put(line + "\n")

        def worker() -> None:
            try:
                manifest = _pipeline().register(
                    req.image_ref,
                    force=req.force,
                    local_only=req.local_only,
                    emit=emit,
                )
                events.put(
                    json.dumps(
                        {"phase": "done", "manifest": manifest.to_dict()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except WrapperError as exc:
                logger.warning(
                    "registration failed", error=str(exc), image_ref=req.image_ref
                )
                events.put(
                    json.dumps(
                        {"phase": "error", "error": str(exc), "kind": "wrapper"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("registration crashed", image_ref=req.image_ref)
                events.put(
                    json.dumps(
                        {"phase": "error", "error": str(exc), "kind": "internal"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        ndjson_iter(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/v1/admin/models/{model_id}")
def unregister_model(model_id: str, remove_image: bool = False) -> UnregisterResponse:
    deleted = _pipeline().unregister(model_id, remove_image=remove_image)
    return UnregisterResponse(deleted=deleted)


@app.get("/v1/admin/state")
def admin_state() -> dict:
    return {
        "manifests_dir": str(catalog.manifests_dir),
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
