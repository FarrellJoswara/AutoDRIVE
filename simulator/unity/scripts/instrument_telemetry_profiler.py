"""Temporarily add CPU Profiler markers to the ignored AutoDRIVE project."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "simulator" / "unity" / "AutoDRIVE"
BACKUP = ROOT / "simulator" / "_build" / "telemetry-profiler-source-backup"
FILES = (
    Path("Assets/Scripts/DataCompressor.cs"),
    Path("Assets/Scripts/Socket.cs"),
)


def method_wrap(source: str, signature: str, marker: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    closing = None
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing is None:
        raise RuntimeError(f"could not find end of method {signature}")
    body = source[opening + 1:closing]
    return (source[:opening + 1] + "\n        using (" + marker + ".Auto())\n        {" +
            body + "\n        }\n    }" + source[closing + 1:])


def apply() -> None:
    if BACKUP.exists():
        raise RuntimeError(f"backup already exists; restore first: {BACKUP}")
    BACKUP.mkdir(parents=True)
    for relative in FILES:
        target = PROJECT / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        shutil.copy2(target, BACKUP / relative.name)

    compressor = PROJECT / FILES[0]
    text = compressor.read_text(encoding="utf-8")
    if "AiCar.DataCompressor.CompressArray" in text:
        raise RuntimeError("DataCompressor already has the temporary marker")
    text = text.replace("using UnityEngine;", "using UnityEngine;\nusing Unity.Profiling;", 1)
    text = text.replace(
        "public class DataCompressor\n{",
        'public class DataCompressor\n{\n'
        '    private static readonly ProfilerMarker CompressArrayMarker =\n'
        '        new ProfilerMarker("AiCar.DataCompressor.CompressArray");',
        1,
    )
    text = method_wrap(text, "public static string CompressArray(string[] data)", "CompressArrayMarker")
    compressor.write_text(text, encoding="utf-8", newline="")

    socket = PROJECT / FILES[1]
    text = socket.read_text(encoding="utf-8")
    if "AiCar.Socket.EmitTelemetryCallback" in text:
        raise RuntimeError("Socket already has the temporary marker")
    text = text.replace("using SocketIO;", "using SocketIO;\nusing Unity.Profiling;", 1)
    text = text.replace(
        "public class Socket : MonoBehaviour\n{",
        'public class Socket : MonoBehaviour\n{\n'
        '    private static readonly ProfilerMarker TelemetryCallbackMarker =\n'
        '        new ProfilerMarker("AiCar.Socket.EmitTelemetryCallback");',
        1,
    )
    callback = "        UnityMainThreadDispatcher.Instance().Enqueue(() =>\n        {"
    if callback not in text:
        raise RuntimeError("Socket telemetry callback anchor missing")
    begin = text.index(callback) + len(callback)
    end = text.index("\n        });", begin)
    callback_body = text[begin:end]
    text = (text[:begin] + "\n            using (TelemetryCallbackMarker.Auto())\n            {" +
            callback_body + "\n            }" + text[end:])
    socket.write_text(text, encoding="utf-8", newline="")
    print("Temporary telemetry profiler markers applied to", PROJECT)


def restore() -> None:
    if not BACKUP.is_dir():
        raise RuntimeError(f"no source backup found: {BACKUP}")
    for relative in FILES:
        target = PROJECT / relative
        saved = BACKUP / relative.name
        if not saved.is_file():
            raise FileNotFoundError(saved)
        shutil.copy2(saved, target)
    shutil.rmtree(BACKUP)
    print("Restored original AutoDRIVE telemetry sources")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "restore"))
    args = parser.parse_args()
    if args.action == "apply":
        apply()
    else:
        restore()


if __name__ == "__main__":
    main()
