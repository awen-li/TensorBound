"""Creation and discovery of a TensorBound study directory."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION
from .records import RECORD_FILES
from .storage import write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def record_path(study_dir: Path, kind: str) -> Path:
    phase = "phase1/input" if kind in {"system", "candidate"} else "phase2/output"
    return study_dir / "results" / phase / RECORD_FILES[kind]


def manifest_path(study_dir: Path) -> Path:
    return study_dir / "results" / "phase1" / "input" / "manifest.json"


def initialize_study(study_dir: Path, study_id: str, title: str) -> None:
    study_manifest = manifest_path(study_dir)
    if study_manifest.exists():
        raise FileExistsError(f"study already initialized: {study_manifest}")

    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "data").mkdir(exist_ok=True)
    for phase in range(1, 4):
        for boundary in ("input", "output"):
            directory = study_dir / "results" / f"phase{phase}" / boundary
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".gitkeep").touch()
    for kind, filename in RECORD_FILES.items():
        record_path(study_dir, kind).touch(exist_ok=False)

    write_json(
        study_manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "title": title,
            "created_at": utc_now(),
            "frozen_at": None,
        },
    )
