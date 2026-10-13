"""Best-effort camera previews that never perform disk I/O on the control loop."""
from pathlib import Path
import threading
import numpy as np


class PreviewWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        self._thread = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # Collection has finished; drain its single outstanding file write
        # before callers archive or remove the run directory.
        if self._thread is not None:
            self._thread.join()

    def submit(self, frames):
        # At most one batch in flight. Dropping a preview is preferable to
        # delaying control or building an unbounded queue of old images.
        if self._thread is not None and self._thread.is_alive():
            return False
        batch = np.asarray(frames).copy()
        self._thread = threading.Thread(target=self._write, args=(batch,), daemon=True)
        self._thread.start()
        return True

    def _write(self, frames):
        try:
            from PIL import Image
            self.directory.mkdir(parents=True, exist_ok=True)
            for env_id, frame in enumerate(frames):
                if frame.shape[0] == 3:
                    frame = np.moveaxis(frame, 0, -1)
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    continue
                image = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8))
                image.thumbnail((320, 180))
                target = self.directory / f"env-{env_id}.jpg"
                temporary = target.with_suffix(".jpg.tmp")
                image.save(temporary, format="JPEG", quality=58, optimize=False)
                temporary.replace(target)
        except Exception:
            # Previews are not demonstration data; a failed display write must
            # not affect collection or vehicle control.
            pass
