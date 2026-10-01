"""Repository-local temporary-directory policy for tests."""

from __future__ import annotations

import tempfile
from pathlib import Path


WORK_TMP = Path(__file__).resolve().parents[1] / ".work-tmp"


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    WORK_TMP.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="test-", dir=WORK_TMP)
