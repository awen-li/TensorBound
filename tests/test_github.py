from __future__ import annotations

import json
import unittest
from pathlib import Path

from tensorbound.github import collect_github_issue_details, collect_github_issues
from tensorbound.layout import initialize_study, record_path
from tensorbound.storage import append_jsonl, read_jsonl
from _temp import temporary_directory


class GitHubCollectionTests(unittest.TestCase):
    def test_collection_deduplicates_queries_and_preserves_snapshots(self) -> None:
        with temporary_directory() as temporary:
            study = Path(temporary) / "study"
            initialize_study(study, "pilot", "Pilot")
            append_jsonl(
                record_path(study, "system"),
                {
                    "schema_version": "1.0.0",
                    "record_type": "system",
                    "system_id": "iree",
                    "name": "IREE",
                    "frontend": "MLIR",
                    "compiler_layers": ["Stream", "HAL"],
                    "runtime": "IREE runtime",
                    "device_backends": ["Vulkan"],
                    "async_model": "timeline semaphores",
                    "storage_model": "HAL buffers",
                    "completion_model": "timepoints",
                    "version_range": {"from": "a", "through": "b"},
                },
            )

            payload = json.dumps(
                {
                    "total_count": 1,
                    "incomplete_results": False,
                    "items": [
                        {
                            "number": 42,
                            "title": "async buffer lifetime",
                            "html_url": "https://github.com/iree-org/iree/issues/42",
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-02T00:00:00Z",
                            "state": "closed",
                            "labels": [{"name": "bug"}],
                        }
                    ],
                }
            ).encode()

            def fetch(_url: str, _token: str | None):
                return payload, {"X-RateLimit-Remaining": "9"}

            result = collect_github_issues(
                study,
                "iree",
                "iree-org/iree",
                ["async buffer", "stream race"],
                "2026-09-24",
                fetch=fetch,
            )

            self.assertEqual(result.requests, 2)
            self.assertEqual(result.unique_items, 1)
            self.assertEqual(result.added_candidates, 1)
            candidates = [record for _, record in read_jsonl(record_path(study, "candidate"))]
            self.assertEqual(candidates[0]["matched_queries"], ["async buffer", "stream race"])
            snapshots = list((study / "data" / "github" / "iree-org-iree").glob("**/page-001-*.json"))
            self.assertEqual(len(snapshots), 2)

    def test_page_limit_is_reported_as_incomplete(self) -> None:
        with temporary_directory() as temporary:
            study = Path(temporary) / "study"
            initialize_study(study, "pilot", "Pilot")
            append_jsonl(
                record_path(study, "system"),
                {
                    "schema_version": "1.0.0",
                    "record_type": "system",
                    "system_id": "iree",
                    "name": "IREE",
                    "frontend": "MLIR",
                    "compiler_layers": ["Stream"],
                    "runtime": "IREE runtime",
                    "device_backends": ["Vulkan"],
                    "async_model": "queues",
                    "storage_model": "buffers",
                    "completion_model": "timepoints",
                    "version_range": {},
                },
            )
            items = [
                {
                    "number": number,
                    "title": f"issue {number}",
                    "html_url": f"https://github.com/iree-org/iree/issues/{number}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                    "state": "closed",
                    "labels": [],
                }
                for number in range(1, 101)
            ]
            payload = json.dumps(
                {"total_count": 101, "incomplete_results": False, "items": items}
            ).encode()

            result = collect_github_issues(
                study,
                "iree",
                "iree-org/iree",
                ["async buffer"],
                "2026-09-24",
                fetch=lambda _url, _token: (payload, {}),
            )
            self.assertEqual(result.incomplete_queries, ("async buffer",))
            acquisition = next(read_jsonl(study / "data" / "github" / "acquisitions.jsonl"))[1]
            self.assertTrue(acquisition["truncated_by_page_limit"])

    def test_successful_pages_are_materialized_before_later_fetch_failure(self) -> None:
        with temporary_directory() as temporary:
            study = Path(temporary) / "study"
            initialize_study(study, "pilot", "Pilot")
            append_jsonl(
                record_path(study, "system"),
                {
                    "schema_version": "1.0.0",
                    "record_type": "system",
                    "system_id": "iree",
                    "name": "IREE",
                    "frontend": "MLIR",
                    "compiler_layers": ["Stream"],
                    "runtime": "IREE runtime",
                    "device_backends": ["Vulkan"],
                    "async_model": "queues",
                    "storage_model": "buffers",
                    "completion_model": "timepoints",
                    "version_range": {},
                },
            )
            payload = json.dumps(
                {
                    "total_count": 1,
                    "incomplete_results": False,
                    "items": [
                        {
                            "number": 42,
                            "title": "buffer lifetime",
                            "html_url": "https://github.com/iree-org/iree/issues/42",
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-02T00:00:00Z",
                            "state": "closed",
                            "labels": [],
                        }
                    ],
                }
            ).encode()
            requests = 0

            def fetch(_url: str, _token: str | None):
                nonlocal requests
                requests += 1
                if requests == 2:
                    raise OSError("rate limit")
                return payload, {}

            with self.assertRaises(OSError):
                collect_github_issues(
                    study,
                    "iree",
                    "iree-org/iree",
                    ["async buffer", "stream race"],
                    "2026-09-24",
                    fetch=fetch,
                )

            candidates = [record for _, record in read_jsonl(record_path(study, "candidate"))]
            self.assertEqual([candidate["candidate_id"] for candidate in candidates], ["github-iree-org-iree-42"])
            self.assertEqual(candidates[0]["matched_queries"], ["async buffer"])

    def test_resumed_collection_merges_query_provenance(self) -> None:
        with temporary_directory() as temporary:
            study = Path(temporary) / "study"
            initialize_study(study, "pilot", "Pilot")
            append_jsonl(
                record_path(study, "system"),
                {
                    "schema_version": "1.0.0",
                    "record_type": "system",
                    "system_id": "iree",
                    "name": "IREE",
                    "frontend": "MLIR",
                    "compiler_layers": ["Stream"],
                    "runtime": "IREE runtime",
                    "device_backends": ["Vulkan"],
                    "async_model": "queues",
                    "storage_model": "buffers",
                    "completion_model": "timepoints",
                    "version_range": {},
                },
            )

            def payload(number: int) -> bytes:
                return json.dumps(
                    {
                        "total_count": 1,
                        "incomplete_results": False,
                        "items": [
                            {
                                "number": number,
                                "title": "same issue",
                                "html_url": f"https://github.com/iree-org/iree/issues/{number}",
                                "created_at": "2024-01-01T00:00:00Z",
                                "updated_at": "2024-01-02T00:00:00Z",
                                "state": "closed",
                                "labels": [],
                            }
                        ],
                    }
                ).encode()

            collect_github_issues(
                study,
                "iree",
                "iree-org/iree",
                ["async buffer"],
                "2026-09-24",
                fetch=lambda _url, _token: (payload(42), {}),
            )
            collect_github_issues(
                study,
                "iree",
                "iree-org/iree",
                ["stream race"],
                "2026-09-24",
                start_page=2,
                fetch=lambda _url, _token: (payload(42), {}),
            )
            records = [record for _, record in read_jsonl(record_path(study, "candidate"))]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["matched_queries"], ["async buffer", "stream race"])

    def test_detail_collection_preserves_issue_and_comments(self) -> None:
        with temporary_directory() as temporary:
            study = Path(temporary) / "study"
            initialize_study(study, "pilot", "Pilot")
            append_jsonl(
                record_path(study, "candidate"),
                {
                    "schema_version": "1.0.0",
                    "record_type": "candidate",
                    "candidate_id": "github-iree-org-iree-42",
                    "system_id": "iree",
                    "title": "race",
                    "source_url": "https://github.com/iree-org/iree/issues/42",
                    "discovery_channels": ["github-issue-search"],
                    "discovered_at": "2024-01-01T00:00:00Z",
                    "status": "pending",
                    "repository": "iree-org/iree",
                    "source_number": 42,
                },
            )

            def fetch(url: str, _token: str | None):
                if "/comments?" in url:
                    return json.dumps([{"id": 1, "body": "fixed by PR"}]).encode(), {}
                return json.dumps({"number": 42, "body": "missing synchronization"}).encode(), {}

            result = collect_github_issue_details(
                study,
                ["github-iree-org-iree-42"],
                fetch=fetch,
            )
            self.assertEqual(result.requests, 2)
            snapshots = list(
                (study / "data" / "github" / "iree-org-iree" / "issues" / "42").glob("*.json")
            )
            self.assertEqual(len(snapshots), 2)

    def test_detail_collection_preserves_all_comment_pages(self) -> None:
        with temporary_directory() as temporary:
            study = Path(temporary) / "study"
            initialize_study(study, "pilot", "Pilot")
            append_jsonl(
                record_path(study, "candidate"),
                {
                    "schema_version": "1.0.0", "record_type": "candidate",
                    "candidate_id": "github-iree-org-iree-42", "system_id": "iree",
                    "title": "race", "source_url": "https://github.com/iree-org/iree/issues/42",
                    "discovery_channels": ["github-issue-search"],
                    "discovered_at": "2024-01-01T00:00:00Z", "status": "pending",
                    "repository": "iree-org/iree", "source_number": 42,
                },
            )

            def fetch(url: str, _token: str | None):
                if url.endswith("&page=1"):
                    return json.dumps([{"id": index} for index in range(100)]).encode(), {
                        "Link": '<https://api.github.com/page=2>; rel="next"',
                    }
                if url.endswith("&page=2"):
                    return json.dumps([{"id": 100}]).encode(), {}
                return json.dumps({"number": 42}).encode(), {}

            result = collect_github_issue_details(
                study, ["github-iree-org-iree-42"], fetch=fetch,
            )
            self.assertEqual(result.requests, 3)
            self.assertEqual(result.snapshots, 3)
            comments = list(
                (study / "data/github/iree-org-iree/issues/42").glob("comments*.json")
            )
            self.assertEqual(len(comments), 2)
            acquisitions = [row for _, row in read_jsonl(study / "data/github/acquisitions.jsonl")]
            self.assertEqual([row.get("page") for row in acquisitions if row["source"] == "github-issue-comments"], [1, 2])


if __name__ == "__main__":
    unittest.main()
