"""Archive immutable historical source and patch evidence for case reconstruction."""

from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path, PurePosixPath

from .storage import read_json, write_json


MAX_BYTES = 8 * 1024 * 1024
REVISION = re.compile(r"[0-9a-f]{40}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def _validate(repository: str, revision: str) -> None:
    if not REPOSITORY.fullmatch(repository) or not REVISION.fullmatch(revision):
        raise ValueError("repository or revision is not an immutable GitHub identifier")


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "TensorBound-source-archiver"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.url != url:
            raise ValueError("unexpected source redirect")
        content = response.read(MAX_BYTES + 1)
    if not content or len(content) > MAX_BYTES:
        raise ValueError("empty or oversized source response")
    return content


def _store(path: Path, content: bytes, metadata: dict) -> Path:
    metadata_path = path.with_name(path.name + ".meta.json")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing snapshot differs: {path}")
        if not metadata_path.exists():
            write_json(metadata_path, metadata)
        elif read_json(metadata_path) != metadata:
            raise ValueError(f"existing snapshot metadata differs: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    write_json(metadata_path, metadata)
    return path


def snapshot_historical_source(
    study_root: Path, repository: str, revision: str, source_path: str,
    expected_blob_id: str,
) -> Path:
    """Fetch a pinned GitHub source file and verify its Git blob identity."""
    _validate(repository, revision)
    pure_path = PurePosixPath(source_path)
    if (not source_path or source_path == "." or pure_path.is_absolute() or ".." in pure_path.parts
            or "." in pure_path.parts or pure_path.as_posix() != source_path
            or not re.fullmatch(r"[A-Za-z0-9_./+-]+", source_path)
            or not REVISION.fullmatch(expected_blob_id)):
        raise ValueError("invalid source path or expected blob ID")
    url = f"https://raw.githubusercontent.com/{repository}/{revision}/{source_path}"
    content = _fetch(url)
    blob_id = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if blob_id != expected_blob_id:
        raise ValueError("historical source Git blob differs from expected patch blob")
    path = study_root / "data/phase2/source" / repository / revision / source_path
    metadata = {
        "schema_version": "1.0.0", "repository": repository, "revision": revision,
        "source_path": source_path, "source_url": url, "git_blob_id": blob_id,
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
    }
    return _store(path, content, metadata)


def snapshot_historical_patch(study_root: Path, repository: str, revision: str) -> Path:
    """Fetch an immutable GitHub patch named by its full fixing commit."""
    _validate(repository, revision)
    url = f"https://github.com/{repository}/commit/{revision}.patch"
    content = _fetch(url)
    if not content.startswith(f"From {revision} ".encode()):
        raise ValueError("patch header differs from requested commit")
    path = study_root / "data/phase2/patches" / repository / f"{revision}.patch"
    metadata = {
        "schema_version": "1.0.0", "repository": repository, "revision": revision,
        "source_url": url, "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }
    return _store(path, content, metadata)
