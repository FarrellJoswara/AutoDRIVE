#!/usr/bin/env python3
"""Fast local checks for OCI extraction whiteouts and path confinement."""
import importlib.util
import io
import pathlib
import tarfile
import tempfile

spec = importlib.util.spec_from_file_location("native_linux_export", pathlib.Path(__file__).with_name("native-linux-export.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def write_layer(path, entries):
    with tarfile.open(path, "w:gz") as tf:
        for name, kind, payload in entries:
            ti = tarfile.TarInfo(name)
            if kind == "dir":
                ti.type = tarfile.DIRTYPE; ti.mode = 0o755; tf.addfile(ti)
            elif kind == "symlink":
                ti.type = tarfile.SYMTYPE; ti.linkname = payload; tf.addfile(ti)
            else:
                data = payload.encode(); ti.size = len(data); ti.mode = 0o644; tf.addfile(ti, io.BytesIO(data))

def main():
    with tempfile.TemporaryDirectory(prefix="native-linux-oci-check-") as td:
        base = pathlib.Path(td); root = base / "rootfs"; root.mkdir()
        first, second = base / "one.tar.gz", base / "two.tar.gz"
        write_layer(first, [("etc/", "dir", ""), ("etc/keep", "file", "keep"),
            ("etc/remove", "file", "remove"), ("etc/sub/old", "file", "old"),
            ("unsafe", "symlink", "/outside"), ("unsafe/marker", "file", "contained")])
        module.extract_layer(first, root)
        write_layer(second, [("etc/.wh.remove", "file", ""), ("etc/sub/.wh..wh..opq", "file", ""),
            ("etc/sub/new", "file", "new")])
        module.extract_layer(second, root)
        assert (root / "etc/keep").read_text() == "keep"
        assert not (root / "etc/remove").exists()
        assert not (root / "etc/sub/old").exists()
        assert (root / "etc/sub/new").read_text() == "new"
        assert (root / "outside/marker").read_text() == "contained"
        bad = base / "bad.tar.gz"
        write_layer(bad, [("../escape", "file", "no")])
        try:
            module.extract_layer(bad, root)
        except ValueError:
            pass
        else:
            raise AssertionError("archive traversal path was accepted")
        assert not (base / "escape").exists()
        print("OCI whiteout, opaque-directory, absolute-link confinement, and traversal checks passed")

if __name__ == "__main__": main()
