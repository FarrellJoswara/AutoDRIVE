"""Apply the reviewed per-container TCP port override to the official bridge."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_official_devkit_port.py <autodrive_bridge.py>")
    path = Path(sys.argv[1])
    source = path.read_text(encoding="utf-8")
    import_anchor = "import base64 # Base64 binary-to-text encoding/decoding scheme"
    port_import = "import os # Native parallel-run port override"
    if port_import not in source:
        if import_anchor not in source:
            raise SystemExit(f"Cannot find official bridge import anchor in {path}")
        source = source.replace(import_anchor, import_anchor + "\n" + port_import, 1)
    old_bind = "pywsgi.WSGIServer(('', 4567), app, handler_class=WebSocketHandler).serve_forever()"
    new_bind = "pywsgi.WSGIServer(('', int(os.environ.get('AICAR_DEVKIT_PORT', '4567'))), app, handler_class=WebSocketHandler).serve_forever()"
    if old_bind in source:
        source = source.replace(old_bind, new_bind, 1)
    elif new_bind not in source:
        raise SystemExit(f"Cannot find official bridge server binding in {path}")
    path.write_text(source, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
