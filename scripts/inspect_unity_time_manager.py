"""Inspect a Unity 2022 runtime TimeManager without loading or editing the player.

Supports serialized-file version 22 with stripped type trees, the format used by
the project's Linux players. Rejects unsupported layouts instead of guessing.
The four runtime class-5 floats correspond to Fixed Timestep, Maximum Allowed
Timestep, Time Scale, and Maximum Particle Timestep in TimeManager.asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


class Reader:
    def __init__(self, data: bytes, offset: int, limit: int, endian: str):
        self.data = data
        self.offset = offset
        self.limit = limit
        self.endian = endian

    def read(self, fmt: str):
        size = struct.calcsize(self.endian + fmt)
        if self.offset + size > self.limit:
            raise ValueError("serialized metadata is truncated")
        result = struct.unpack_from(self.endian + fmt, self.data, self.offset)
        self.offset += size
        return result[0] if len(result) == 1 else result

    def skip(self, size: int):
        if size < 0 or self.offset + size > self.limit:
            raise ValueError("invalid serialized metadata length")
        self.offset += size

    def c_string(self) -> str:
        end = self.data.find(b"\0", self.offset, self.limit)
        if end < 0:
            raise ValueError("unterminated Unity version string")
        value = self.data[self.offset:end].decode("ascii")
        self.offset = end + 1
        return value

    def align(self, alignment: int):
        self.skip((-self.offset) % alignment)


def inspect(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 48:
        raise ValueError(f"{path}: shorter than the version-22 header")
    _, _, version, _ = struct.unpack_from(">4I", data)
    if version != 22:
        raise ValueError(f"{path}: unsupported serialized version {version}")
    endian_flag = data[16]
    if endian_flag not in (0, 1):
        raise ValueError(f"{path}: invalid endian flag")
    endian = "<" if endian_flag == 0 else ">"
    metadata_size, file_size, data_offset, _ = struct.unpack_from(">IQQQ", data, 20)
    if file_size != len(data):
        raise ValueError(f"{path}: header file size does not match file")
    metadata_end = 48 + metadata_size
    if not 48 < metadata_end <= data_offset <= file_size:
        raise ValueError(f"{path}: invalid metadata/data bounds")
    reader = Reader(data, 48, metadata_end, endian)
    unity_version = reader.c_string()
    platform = reader.read("i")
    enable_type_tree = reader.read("?")
    if enable_type_tree:
        raise ValueError(f"{path}: type-tree layout is unsupported")
    type_count = reader.read("i")
    if not 0 < type_count < 100000:
        raise ValueError(f"{path}: invalid type count")
    classes = []
    for _ in range(type_count):
        class_id = reader.read("i")
        reader.read("?")  # stripped type
        reader.read("h")  # script type index
        if class_id == 114:
            reader.skip(16)  # MonoBehaviour script hash
        reader.skip(16)  # old type hash
        classes.append(class_id)
    object_count = reader.read("i")
    if not 0 < object_count < 10000000:
        raise ValueError(f"{path}: invalid object count")
    time_objects = []
    for _ in range(object_count):
        reader.align(4)
        path_id = reader.read("q")
        relative_offset = reader.read("q")
        size = reader.read("I")
        type_index = reader.read("i")
        if not 0 <= type_index < len(classes):
            raise ValueError(f"{path}: invalid object type index")
        offset = data_offset + relative_offset
        if relative_offset < 0 or offset + size > file_size:
            raise ValueError(f"{path}: invalid object data bounds")
        if classes[type_index] != 5:
            continue
        if size != 16:
            raise ValueError(f"{path}: unexpected TimeManager payload size {size}")
        values = struct.unpack_from(endian + "4f", data, offset)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}: nonfinite TimeManager values")
        time_objects.append({
            "class_id": 5,
            "path_id": path_id,
            "type_index": type_index,
            "file_offset": offset,
            "payload_size": size,
            "raw_hex": data[offset:offset + size].hex(),
            "settings": dict(zip((
                "fixed_timestep", "maximum_allowed_timestep", "time_scale",
                "maximum_particle_timestep",
            ), values)),
        })
    if len(time_objects) != 1:
        raise ValueError(f"{path}: expected one class-5 object, found {len(time_objects)}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "file_size": file_size,
        "serialized_version": version,
        "unity_version": unity_version,
        "platform": platform,
        "endianness": "little" if endian_flag == 0 else "big",
        "metadata_size": metadata_size,
        "data_offset": data_offset,
        "type_tree_present": enable_type_tree,
        "type_count": type_count,
        "object_count": object_count,
        "time_manager": time_objects[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inspections = [inspect(path) for path in args.paths]
    baseline = inspections[0]["time_manager"]["raw_hex"]
    result = {
        "method": "read-only serialized-file header, type/object table, class-5 payload parsing",
        "settings_match_first_payload": all(
            item["time_manager"]["raw_hex"] == baseline for item in inspections
        ),
        "limitations": [
            "Proves serialized initial settings; runtime code can subsequently override them.",
            "Class-5 payload layout is four float32 values; names follow the Unity TimeManager asset layout.",
            "Matching settings do not prove matching scene, physics/control code, or source build provenance.",
            "Different globalgamemanagers hashes can result from unrelated player settings or build GUIDs.",
        ],
        "inspections": inspections,
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
