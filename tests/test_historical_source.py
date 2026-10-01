from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from _temp import temporary_directory
from tensorbound.historical_source import snapshot_historical_patch, snapshot_historical_source
from tensorbound.storage import read_json


class HistoricalSourceTests(unittest.TestCase):
    def test_pinned_source_checks_blob_and_is_idempotent(self) -> None:
        content = b"int example(void) { return 1; }\n"
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        with temporary_directory() as temporary, patch(
            "tensorbound.historical_source._fetch", return_value=content,
        ):
            root = Path(temporary)
            args = (root, "example/project", "a" * 40, "src/example.c", blob)
            path = snapshot_historical_source(*args)
            self.assertEqual(snapshot_historical_source(*args), path)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(read_json(path.with_name("example.c.meta.json"))["git_blob_id"], blob)
            with self.assertRaisesRegex(ValueError, "Git blob differs"):
                snapshot_historical_source(root, "example/project", "a" * 40, "src/other.c", "b" * 40)

    def test_rejects_unsafe_paths_and_wrong_patch_header(self) -> None:
        with temporary_directory() as temporary, patch(
            "tensorbound.historical_source._fetch", return_value=b"From " + b"b" * 40 + b" patch\n",
        ):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "invalid source path"):
                snapshot_historical_source(root, "example/project", "a" * 40, "../outside.c", "b" * 40)
            with self.assertRaisesRegex(ValueError, "patch header differs"):
                snapshot_historical_patch(root, "example/project", "a" * 40)


if __name__ == "__main__":
    unittest.main()
