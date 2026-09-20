#!/usr/bin/env python3
"""Tiny HTTP readiness stub for the brain container (default port 8090).

Keeps the container up without auto-training. Mission Control UI will later
use port 8080; this stub is only a health/ready placeholder.
"""

from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ReadyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"OK\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[ready_stub] {self.address_string()} - {fmt % args}")


def main() -> None:
    port = int(os.environ.get("READY_PORT", "8090"))
    host = os.environ.get("READY_HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), ReadyHandler)
    print(f"[ready_stub] listening on http://{host}:{port}/ (GET -> OK)")
    server.serve_forever()


if __name__ == "__main__":
    main()
