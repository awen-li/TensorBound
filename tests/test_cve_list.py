from __future__ import annotations

import json
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from _temp import temporary_directory
from tensorbound.cve_list import collect_cve_list
from tensorbound.layout import initialize_study, record_path
from tensorbound.source_store import initialize_source_stores, validate_source_stores
from tensorbound.storage import append_jsonl, read_jsonl


REVISION = "a" * 40


class CVEListCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = temporary_directory()
        self.study = Path(self.temporary.name) / "study"
        initialize_study(self.study, "pilot", "Pilot")
        initialize_source_stores(self.study, "2026-09-24")
        append_jsonl(record_path(self.study, "system"), {
            "schema_version": "1.0.0", "record_type": "system", "system_id": "iree",
            "name": "IREE", "frontend": "MLIR", "compiler_layers": [], "runtime": "runtime",
            "device_backends": [], "async_model": "streams", "storage_model": "buffers",
            "completion_model": "semaphores", "version_range": {},
        })

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_archives_all_search_pages_and_pinned_records(self) -> None:
        calls = []

        def fetch(url: str, token: str | None):
            calls.append((url, token))
            if "/commits/" in url:
                payload = {"sha": REVISION, "commit": {"committer": {"date": "2026-09-24T12:00:00Z"}}}
            elif "/search/code?" in url:
                page = int(parse_qs(urlparse(url).query)["page"][0])
                cve_id = "CVE-2025-1000" if page == 1 else "CVE-2025-1001"
                payload = {"total_count": 2, "incomplete_results": False,
                           "items": [{"path": f"cves/2025/1xxx/{cve_id}.json"}]}
            else:
                cve_id = Path(url).stem
                payload = {"cveMetadata": {"cveId": cve_id, "state": "PUBLISHED",
                                           "datePublished": "2025-01-01T00:00:00Z"}}
            return json.dumps(payload).encode(), {}

        result = collect_cve_list(self.study, {"IREE": "iree"}, REVISION,
                                  per_page=1, token="test-token", fetch=fetch)
        self.assertEqual((result.requests, result.retrieved_search_hits, result.candidates), (5, 2, 2))
        self.assertEqual(result.incomplete_terms, ())
        self.assertTrue(all(token is None for url, token in calls if "raw.githubusercontent.com" in url))
        self.assertTrue(all(REVISION in url for url, _ in calls if "raw.githubusercontent.com" in url))
        rows = [row for _, row in read_jsonl(result.output)]
        self.assertEqual(rows[0]["screening_status"], "pending")
        self.assertEqual(rows[0]["system_ids"], ["iree"])
        report = validate_source_stores(self.study)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.raw_counts["cve-list"], 5)
        self.assertEqual(report.candidate_counts["cve-list"], 2)
        self.assertEqual(report.confirmed_counts["cve-list"], 0)

    def test_rejects_post_cutoff_revision(self) -> None:
        def fetch(url: str, token: str | None):
            return json.dumps({"sha": REVISION,
                               "commit": {"committer": {"date": "2026-09-25T00:00:00Z"}}}).encode(), {}

        with self.assertRaisesRegex(ValueError, "cutoff"):
            collect_cve_list(self.study, {"IREE": "iree"}, REVISION, fetch=fetch)


if __name__ == "__main__":
    unittest.main()
