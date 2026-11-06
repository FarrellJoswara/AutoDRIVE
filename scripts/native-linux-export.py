#!/usr/bin/env python3
"""Digest-pinned, user-space OCI image export with OCI whiteout semantics."""
import fcntl, hashlib, json, os, pathlib, posixpath, shutil, sys, tarfile, time, urllib.parse, urllib.request

REPO = "autodriveecosystem/autodrive_roboracer_sim"
RELEASES = {
    "practice": ("2026-iros-practice", "sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76"),
    "competition": ("2026-iros-compete", "sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d"),
}
ACCEPT = ", ".join(("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"))

def get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=90)

def digest_ok(data, digest):
    alg, expected = digest.split(":", 1)
    return hashlib.new(alg, data).hexdigest() == expected

def parts_of(name):
    if name.startswith("/"):
        raise ValueError(f"Absolute archive path rejected: {name!r}")
    parts = [x for x in name.split("/") if x not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"Archive path traversal rejected: {name!r}")
    return parts

def extract_layer(archive, root):
    """Apply whiteouts and extract with all links confined to virtual rootfs."""
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for m in members: parts_of(m.name)
        def resolve(parts):
            todo, out, hops = list(parts), [], 0
            while todo:
                item = todo.pop(0)
                if item in ("", "."): continue
                if item == "..":
                    if not out: raise ValueError("OCI symlink escapes image root")
                    out.pop(); continue
                candidate = root.joinpath(*out, item)
                if candidate.is_symlink():
                    hops += 1
                    if hops > 40: raise ValueError("OCI symlink loop")
                    target = os.readlink(candidate)
                    if target.startswith("/"): out = []
                    todo = target.split("/") + todo
                else: out.append(item)
            return out
        def path_for(parts): return root.joinpath(*resolve(parts))
        def unlink_entry(parts):
            path = path_for(parts[:-1]) / parts[-1]
            if path.is_symlink() or path.is_file(): path.unlink(missing_ok=True)
            elif path.is_dir(): shutil.rmtree(path)

        # Whiteouts apply to lower layers, before any current-layer additions.
        for m in members:
            p = parts_of(m.name)
            if not p or not p[-1].startswith(".wh."): continue
            if p[-1] == ".wh..wh..opq":
                directory = path_for(p[:-1])
                if directory.is_dir():
                    for child in directory.iterdir():
                        if child.is_symlink() or child.is_file(): child.unlink()
                        elif child.is_dir(): shutil.rmtree(child)
            else:
                target = p[-1][4:]
                if target in ("", ".", ".."):
                    raise ValueError(f"Invalid OCI whiteout target: {m.name!r}")
                unlink_entry(p[:-1] + [target])

        dir_modes = []
        for m in members:
            p = parts_of(m.name)
            if not p or p[-1].startswith(".wh."): continue
            parent = path_for(p[:-1]); parent.mkdir(parents=True, exist_ok=True)
            entry = parent / p[-1]
            if m.isdir():
                # Directory entries may traverse a lower-layer /bin symlink.
                target = path_for(p)
                if target.exists() and not target.is_dir(): unlink_entry(p); target = path_for(p)
                target.mkdir(parents=True, exist_ok=True); dir_modes.append((p, m.mode & 0o777))
            elif m.issym():
                if entry.is_symlink() or entry.exists(): unlink_entry(p)
                link = m.linkname
                target = resolve(link.split("/") if link.startswith("/") else p[:-1] + link.split("/"))
                relative = posixpath.relpath("/".join(target) or ".", "/".join(p[:-1]) or ".")
                os.symlink(relative, entry)
            elif m.islnk():
                source = path_for(parts_of(m.linkname))
                if not source.is_file(): raise ValueError(f"Invalid OCI hardlink target: {m.linkname!r}")
                if entry.is_symlink() or entry.exists(): unlink_entry(p)
                os.link(source, entry)
            elif m.isfile():
                if entry.is_symlink() or entry.exists(): unlink_entry(p)
                stream = tf.extractfile(m)
                if stream is None: raise ValueError(f"Cannot read OCI member {m.name!r}")
                with stream, entry.open("wb") as out: shutil.copyfileobj(stream, out, 1024 * 1024)
                entry.chmod(m.mode & 0o777)
            else: raise ValueError(f"Unsupported OCI member type: {m.name!r}")
        for p, mode in dir_modes: path_for(p).chmod(mode)

def root_digest(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob("*"), key=lambda x: x.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix().encode()
        if p.is_symlink(): h.update(b"L\0" + rel + b"\0" + os.readlink(p).encode() + b"\0")
        elif p.is_file():
            h.update(b"F\0" + rel + b"\0" + str(p.stat().st_mode & 0o777).encode() + b"\0")
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
            h.update(b"\0")
        elif p.is_dir(): h.update(b"D\0" + rel + b"\0" + str(p.stat().st_mode & 0o777).encode() + b"\0")
    return "sha256:" + h.hexdigest()

def download_layer(base, headers, layer, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    blob = cache_dir / (layer["digest"].replace(":", "-") + ".tar.gz")
    if blob.is_file() and blob.stat().st_size == layer["size"]:
        h = hashlib.sha256()
        with blob.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
        if "sha256:" + h.hexdigest() == layer["digest"]:
            return blob, False
    temp = cache_dir / (blob.name + f".part.{os.getpid()}")
    h = hashlib.sha256(); size = 0
    with get(f"{base}/blobs/{layer['digest']}", headers) as src, temp.open("wb") as out:
        for chunk in iter(lambda:src.read(1024*1024), b""):
            h.update(chunk); size += len(chunk); out.write(chunk)
    if size != layer["size"] or "sha256:" + h.hexdigest() != layer["digest"]:
        temp.unlink(missing_ok=True)
        raise ValueError(f"OCI layer size/digest mismatch for {layer['digest']}")
    os.replace(temp, blob)
    return blob, True

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", nargs="?", default=".native-linux/runtime")
    parser.add_argument("--release", choices=RELEASES, default="practice")
    parser.add_argument("--blob-cache", type=pathlib.Path)
    args = parser.parse_args()
    tag, digest = RELEASES[args.release]
    dest = pathlib.Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True); final_root = dest / "rootfs"; identity_path = dest / "IMAGE-IDENTITY.json"
    lock = (dest / ".export.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    token_url = "https://auth.docker.io/token?" + urllib.parse.urlencode({"service":"registry.docker.io", "scope":f"repository:{REPO}:pull"})
    token = json.load(get(token_url))["token"]; base = f"https://registry-1.docker.io/v2/{REPO}"
    headers = {"Authorization":f"Bearer {token}", "Accept":ACCEPT}
    with get(f"{base}/manifests/{tag}", headers) as r: raw, header_digest = r.read(), r.headers.get("Docker-Content-Digest")
    if header_digest != digest or not digest_ok(raw, digest): raise SystemExit(f"Pinned manifest digest mismatch: header={header_digest}")
    manifest = json.loads(raw)
    if "manifests" in manifest:
        matches = [m for m in manifest["manifests"] if m.get("platform",{}).get("os") == "linux" and m.get("platform",{}).get("architecture") == "amd64"]
        if len(matches) != 1: raise SystemExit(f"Expected one linux/amd64 manifest; found {len(matches)}")
        platform_digest = matches[0]["digest"]
        with get(f"{base}/manifests/{platform_digest}", headers) as r: platform_raw = r.read()
        if not digest_ok(platform_raw, platform_digest): raise SystemExit("Platform manifest byte digest mismatch")
        manifest = json.loads(platform_raw)
    config_desc = manifest["config"]
    with get(f"{base}/blobs/{config_desc['digest']}", headers) as r: config_raw = r.read()
    if len(config_raw) != config_desc["size"] or not digest_ok(config_raw, config_desc["digest"]):
        raise SystemExit("OCI config size/digest mismatch")
    config = json.loads(config_raw)
    if config.get("architecture") != "amd64" or config.get("os") != "linux":
        raise SystemExit(f"Expected official linux/amd64 config, got {config.get('os')}/{config.get('architecture')}")
    sim = final_root / "home/autodrive_simulator/AutoDRIVE Simulator.x86_64"
    if identity_path.is_file() and sim.is_file():
        cached = json.loads(identity_path.read_text())
        try: complete = json.loads((final_root / ".OCI-COMPLETE.json").read_text())
        except (OSError, ValueError): complete = {}
        cached_data = sim.parent / "Data"
        scene_ok = all((cached_data / filename).is_file() and hashlib.sha256((cached_data / filename).read_bytes()).hexdigest() == cached.get(key)
                       for filename, key in (("level0", "level0_sha256"), ("globalgamemanagers", "globalgamemanagers_sha256")))
        if (cached.get("tag") == tag and cached.get("digest") == digest and cached.get("manifest_verified") is True
                and cached.get("architecture") == "linux/amd64"
                and cached.get("rootfs_tree_digest") == root_digest(final_root)
                and complete.get("tag", tag) == tag and complete.get("digest") == digest and complete.get("verified_layers") == len(manifest["layers"])
                and cached.get("simulator_sha256") == hashlib.sha256(sim.read_bytes()).hexdigest() and scene_ok):
            print(f"Verified cached official runtime: {identity_path}"); return
        if cached.get("tag") != tag or cached.get("digest") != digest:
            raise SystemExit("Cached image identity differs from the selected pinned release; preserve it and select another runtime directory")
        print("Existing runtime is legacy/unverified; exporting into a fresh staging tree and preserving the old rootfs.")
    total = sum(x["size"] for x in manifest["layers"])
    cache_dir = (args.blob_cache or (dest / "blobs")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_lock = (cache_dir / ".cache.lock").open("a+")
    fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX)
    print(f"Official {REPO}:{tag}@{digest}; linux/amd64; {total:,} compressed bytes; reusing verified blobs when available", flush=True)
    root = dest / f"rootfs.staging.{os.getpid()}"
    root.mkdir(parents=False, exist_ok=False)
    (root / ".INCOMPLETE.json").write_text(json.dumps({"expected_tag": tag, "expected_digest": digest, "pid": os.getpid()}, indent=2) + "\n")
    for i, layer in enumerate(manifest["layers"], 1):
        blob, downloaded = download_layer(base, headers, layer, cache_dir)
        state = "downloaded+verified" if downloaded else "cached+verified"
        print(f"Applying layer {i}/{len(manifest['layers'])} ({state})", flush=True)
        extract_layer(blob, root)
    sim = root / "home/autodrive_simulator/AutoDRIVE Simulator.x86_64"
    if not sim.is_file(): raise SystemExit(f"Official simulator executable missing: {sim}")
    sim.chmod(sim.stat().st_mode | 0o111)
    identity = {"source":"Docker Hub AutoDRIVE Ecosystem publisher", "repository":REPO, "tag":tag,
        "digest":digest, "manifest_verified":True, "architecture":"linux/amd64", "config_digest":config_desc["digest"], "compressed_layer_bytes":total,
        "layer_digests":[x["digest"] for x in manifest["layers"]],
        "simulator_path":str(sim.relative_to(root)), "simulator_sha256":hashlib.sha256(sim.read_bytes()).hexdigest(),
        "level0_sha256":hashlib.sha256((sim.parent/"Data/level0").read_bytes()).hexdigest(),
        "globalgamemanagers_sha256":hashlib.sha256((sim.parent/"Data/globalgamemanagers").read_bytes()).hexdigest()}
    (root / ".INCOMPLETE.json").unlink()
    (root / ".OCI-COMPLETE.json").write_text(json.dumps({"tag":tag,"digest":digest,"verified_layers":len(manifest["layers"])},indent=2)+"\n")
    identity["rootfs_tree_digest"] = root_digest(root)
    identity["simulator_path"] = "home/autodrive_simulator/AutoDRIVE Simulator.x86_64"
    temp_identity = dest / f"IMAGE-IDENTITY.json.tmp.{os.getpid()}"
    temp_identity.write_text(json.dumps(identity,indent=2)+"\n")
    backup = None
    failed_root = dest / f"rootfs.failed.{int(time.time())}.{os.getpid()}"
    if final_root.exists():
        backup = dest / f"rootfs.previous.{int(time.time())}.{os.getpid()}"
        os.replace(final_root, backup)
        print(f"Preserved prior rootfs at {backup}")
    try:
        os.replace(root, final_root)
        os.replace(temp_identity, identity_path)
    except Exception:
        # Keep the failed promotion for diagnosis, restore the previous root,
        # and leave its identity marker paired with it.
        if final_root.exists():
            os.replace(final_root, failed_root)
            print(f"Preserved failed promoted rootfs at {failed_root}", file=sys.stderr)
        if backup is not None and backup.exists():
            os.replace(backup, final_root)
            print(f"Restored prior rootfs from {backup}", file=sys.stderr)
        raise
    if backup is not None:
        print(f"Preserved prior rootfs at {backup}")
    print(f"Simulator executable: {final_root / identity['simulator_path']}\nRuntime root: {final_root}")

if __name__ == "__main__": main()
