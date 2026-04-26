"""Tiny stdlib-based admin HTTP server.

Used by Router and Worker Manager bootstraps to expose `/admin/state`
without dragging in FastAPI/uvicorn (Registry already covers richer HTTP).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve_admin_state(
    port: int,
    snapshot: Callable[[], dict],
) -> ThreadingHTTPServer:
    """Start a daemon HTTP thread; return the server (call `.shutdown()` on exit)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") not in ("/admin/state", "/health"):
                self.send_error(404, "not found")
                return
            try:
                body = json.dumps(snapshot()).encode("utf-8")
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
            else:
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(
        target=server.serve_forever, name=f"admin-http-{port}", daemon=True
    )
    thread.start()
    return server
