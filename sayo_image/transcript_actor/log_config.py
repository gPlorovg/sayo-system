"""Stdout JSON structlog + stdlib logging for actor containers (no sayo_host import)."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def _service_processor(_logger: Any, _method: str, event_dict: dict) -> dict:
    event_dict.setdefault("service", "sayo-actor")
    return event_dict


def configure_actor_process_logging() -> None:
    """Call once from bootstrap before importing NeMo / TranscriptActor."""
    level = (os.environ.get("LOG_LEVEL") or "INFO").upper()
    numeric = getattr(logging, level, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            _service_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # NeMo / Lightning: INFO; torch internals stay quieter.
    for name in ("nemo", "lightning", "pytorch_lightning"):
        logging.getLogger(name).setLevel(
            logging.DEBUG if numeric <= logging.DEBUG else logging.INFO
        )
    for name in ("pytorch", "torch", "numba", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
