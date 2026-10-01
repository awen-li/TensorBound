"""Reproducible, high-recall GitHub issue collection."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .layout import record_path, utc_now
from .records import RECORD_ID_FIELDS
from .storage import append_jsonl, read_jsonl, write_bytes, write_jsonl

GITHUB_SEARCH_URL = "https://api.github.com/search/issues"


class CollectionError(RuntimeError):
    """Raised when an upstream collection response is invalid."""


@dataclass(frozen=True)
class CollectionResult:
    retrieved_items: int
    unique_items: int
    added_candidates: int
    skipped_existing: int
    requests: int
    incomplete_queries: tuple[str, ...]


@dataclass(frozen=True)
class DetailCollectionResult:
    candidates: int
    requests: int
    snapshots: int


Fetch = Callable[[str, str | None], tuple[bytes, Mapping[str, str]]]


def _default_fetch(url: str, token: str | None) -> tuple[bytes, Mapping[str, str]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "TensorBound-empirical-study",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), dict(response.headers.items())


def _candidate_id(repository: str, number: int) -> str:
    slug = repository.lower().replace("/", "-")
    return f"github-{slug}-{number}"


def _query_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _validate_repository(repository: str) -> None:
    pieces = repository.split("/")
    if len(pieces) != 2 or not all(piece and piece.replace("-", "").replace("_", "").isalnum() for piece in pieces):
        raise ValueError("repository must use the owner/name form")


def _validate_through(through: str) -> None:
    try:
        date.fromisoformat(through)
    except ValueError as error:
        raise ValueError("through must use YYYY-MM-DD") from error


def _known_ids(path: Path, identifier: str) -> set[str]:
    return {record[identifier] for _, record in read_jsonl(path) if identifier in record}


def _checkpoint_candidates(
    candidate_path: Path,
    existing_records: Mapping[str, dict[str, Any]],
    collected: Mapping[str, dict[str, Any]],
) -> None:
    """Persist fetched candidates without changing final collection counters."""
    checkpoint = {identifier: dict(record) for identifier, record in existing_records.items()}
    for identifier, candidate in collected.items():
        if identifier in checkpoint:
            previous_queries = checkpoint[identifier].get("matched_queries", [])
            checkpoint[identifier]["matched_queries"] = sorted(
                set(previous_queries) | set(candidate["matched_queries"])
            )
        else:
            record = dict(candidate)
            record["matched_queries"] = sorted(set(candidate["matched_queries"]))
            checkpoint[identifier] = record
    write_jsonl(candidate_path, (checkpoint[identifier] for identifier in sorted(checkpoint)))


def _decode_response(raw: bytes, url: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CollectionError(f"GitHub returned invalid JSON for {url}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise CollectionError(f"GitHub response for {url} has no items array")
    return payload


def collect_github_issues(
    study_dir: Path,
    system_id: str,
    repository: str,
    queries: Iterable[str],
    through: str,
    *,
    start_page: int = 1,
    max_pages: int = 1,
    token: str | None = None,
    fetch: Fetch = _default_fetch,
) -> CollectionResult:
    """Collect issue candidates and preserve every raw search response."""
    _validate_repository(repository)
    _validate_through(through)
    if start_page < 1:
        raise ValueError("start_page must be at least 1")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    normalized_queries = tuple(dict.fromkeys(query.strip() for query in queries if query.strip()))
    if not normalized_queries:
        raise ValueError("at least one non-empty query is required")

    system_ids = _known_ids(record_path(study_dir, "system"), RECORD_ID_FIELDS["system"])
    if system_id not in system_ids:
        raise ValueError(f"unknown system ID: {system_id}")

    candidate_path = record_path(study_dir, "candidate")
    existing_records = {
        record[RECORD_ID_FIELDS["candidate"]]: record
        for _, record in read_jsonl(candidate_path)
        if RECORD_ID_FIELDS["candidate"] in record
    }
    existing_ids = set(existing_records)
    acquisitions = study_dir / "data" / "github" / "acquisitions.jsonl"
    repository_slug = repository.lower().replace("/", "-")
    collected: dict[str, dict[str, Any]] = {}
    retrieved_items = 0
    requests = 0
    incomplete_queries: list[str] = []

    for query in normalized_queries:
        search_query = f"repo:{repository} is:issue created:<={through} {query}"
        query_key = _query_key(search_query)
        query_incomplete = False
        last_requested_page = start_page + max_pages - 1
        for page in range(start_page, last_requested_page + 1):
            parameters = urllib.parse.urlencode(
                {
                    "q": search_query,
                    "sort": "created",
                    "order": "asc",
                    "per_page": 100,
                    "page": page,
                }
            )
            url = f"{GITHUB_SEARCH_URL}?{parameters}"
            raw, headers = fetch(url, token)
            requests += 1
            payload = _decode_response(raw, url)
            digest = hashlib.sha256(raw).hexdigest()
            snapshot = (
                study_dir
                / "data"
                / "github"
                / repository_slug
                / query_key
                / f"page-{page:03d}-{digest}.json"
            )
            write_bytes(snapshot, raw)
            items = payload["items"]
            retrieved_items += len(items)
            query_incomplete = query_incomplete or bool(payload.get("incomplete_results"))
            total_count = payload.get("total_count")
            truncated_by_page_limit = (
                page == last_requested_page
                and len(items) == 100
                and isinstance(total_count, int)
                and total_count > page * 100
            )
            if truncated_by_page_limit:
                query_incomplete = True
            append_jsonl(
                acquisitions,
                {
                    "schema_version": "1.0.0",
                    "source": "github-search-issues",
                    "system_id": system_id,
                    "repository": repository,
                    "query": query,
                    "effective_query": search_query,
                    "through": through,
                    "page": page,
                    "url": url,
                    "retrieved_at": utc_now(),
                    "item_count": len(items),
                    "reported_total_count": total_count,
                    "incomplete_results": bool(payload.get("incomplete_results")),
                    "truncated_by_page_limit": truncated_by_page_limit,
                    "response_sha256": digest,
                    "snapshot": str(snapshot.relative_to(study_dir)),
                    "rate_limit_remaining": headers.get("X-RateLimit-Remaining"),
                    "rate_limit_reset": headers.get("X-RateLimit-Reset"),
                },
            )
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("number"), int):
                    raise CollectionError(f"GitHub response for {url} contains an invalid issue item")
                identifier = _candidate_id(repository, item["number"])
                candidate = collected.setdefault(
                    identifier,
                    {
                        "schema_version": "1.0.0",
                        "record_type": "candidate",
                        "candidate_id": identifier,
                        "system_id": system_id,
                        "title": item.get("title") or f"GitHub issue {item['number']}",
                        "source_url": item.get("html_url") or item.get("url"),
                        "discovery_channels": ["github-issue-search"],
                        "discovered_at": item.get("created_at") or utc_now(),
                        "status": "pending",
                        "repository": repository,
                        "source_number": item["number"],
                        "source_state": item.get("state"),
                        "updated_at": item.get("updated_at"),
                        "labels": [
                            label.get("name")
                            for label in item.get("labels", [])
                            if isinstance(label, dict) and isinstance(label.get("name"), str)
                        ],
                        "matched_queries": [],
                    },
                )
                candidate["matched_queries"].append(query)
            _checkpoint_candidates(candidate_path, existing_records, collected)
            if len(items) < 100:
                break
        if query_incomplete:
            incomplete_queries.append(query)

    added = 0
    skipped = 0
    for identifier in sorted(collected):
        if identifier in existing_ids:
            previous_queries = existing_records[identifier].get("matched_queries", [])
            existing_records[identifier]["matched_queries"] = sorted(
                set(previous_queries) | set(collected[identifier]["matched_queries"])
            )
            skipped += 1
            continue
        collected[identifier]["matched_queries"] = sorted(set(collected[identifier]["matched_queries"]))
        existing_records[identifier] = collected[identifier]
        added += 1

    write_jsonl(
        candidate_path,
        (existing_records[identifier] for identifier in sorted(existing_records)),
    )

    return CollectionResult(
        retrieved_items=retrieved_items,
        unique_items=len(collected),
        added_candidates=added,
        skipped_existing=skipped,
        requests=requests,
        incomplete_queries=tuple(incomplete_queries),
    )


def collect_github_issue_details(
    study_dir: Path,
    candidate_ids: Iterable[str],
    *,
    token: str | None = None,
    fetch: Fetch = _default_fetch,
) -> DetailCollectionResult:
    """Snapshot issue records and comments for selected GitHub candidates."""
    selected_ids = tuple(dict.fromkeys(candidate_ids))
    if not selected_ids:
        raise ValueError("at least one candidate ID is required")
    candidates = {
        record["candidate_id"]: record
        for _, record in read_jsonl(record_path(study_dir, "candidate"))
    }
    acquisitions = study_dir / "data" / "github" / "acquisitions.jsonl"
    requests = 0
    snapshots = 0

    for candidate_id in selected_ids:
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate ID: {candidate_id}")
        repository = candidate.get("repository")
        number = candidate.get("source_number")
        if not isinstance(repository, str) or not isinstance(number, int):
            raise ValueError(f"candidate {candidate_id} has no GitHub repository and issue number")
        _validate_repository(repository)
        repository_slug = repository.lower().replace("/", "-")
        base = f"https://api.github.com/repos/{repository}/issues/{number}"
        endpoints = (("issue", base),)
        for kind, url in endpoints:
            raw, headers = fetch(url, token)
            requests += 1
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CollectionError(f"GitHub returned invalid JSON for {url}: {error}") from error
            if kind == "issue" and not isinstance(payload, dict):
                raise CollectionError(f"GitHub issue response for {url} is not an object")
            if kind == "comments" and not isinstance(payload, list):
                raise CollectionError(f"GitHub comments response for {url} is not an array")
            digest = hashlib.sha256(raw).hexdigest()
            snapshot = (
                study_dir
                / "data"
                / "github"
                / repository_slug
                / "issues"
                / str(number)
                / f"{kind}-{digest}.json"
            )
            write_bytes(snapshot, raw)
            snapshots += 1
            append_jsonl(
                acquisitions,
                {
                    "schema_version": "1.0.0",
                    "source": f"github-issue-{kind}",
                    "candidate_id": candidate_id,
                    "system_id": candidate["system_id"],
                    "repository": repository,
                    "source_number": number,
                    "url": url,
                    "retrieved_at": utc_now(),
                    "response_sha256": digest,
                    "snapshot": str(snapshot.relative_to(study_dir)),
                    "item_count": len(payload) if isinstance(payload, list) else 1,
                    "rate_limit_remaining": headers.get("X-RateLimit-Remaining"),
                    "rate_limit_reset": headers.get("X-RateLimit-Reset"),
                },
            )

        page = 1
        while True:
            url = f"{base}/comments?per_page=100&page={page}"
            raw, headers = fetch(url, token)
            requests += 1
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CollectionError(f"GitHub returned invalid JSON for {url}: {error}") from error
            if not isinstance(payload, list):
                raise CollectionError(f"GitHub comments response for {url} is not an array")
            digest = hashlib.sha256(raw).hexdigest()
            name = f"comments-{digest}.json" if page == 1 else f"comments-page-{page:03d}-{digest}.json"
            snapshot = (
                study_dir / "data" / "github" / repository_slug / "issues"
                / str(number) / name
            )
            write_bytes(snapshot, raw)
            snapshots += 1
            append_jsonl(
                acquisitions,
                {
                    "schema_version": "1.0.0",
                    "source": "github-issue-comments",
                    "candidate_id": candidate_id,
                    "system_id": candidate["system_id"],
                    "repository": repository,
                    "source_number": number,
                    "page": page,
                    "url": url,
                    "retrieved_at": utc_now(),
                    "response_sha256": digest,
                    "snapshot": str(snapshot.relative_to(study_dir)),
                    "item_count": len(payload),
                    "rate_limit_remaining": headers.get("X-RateLimit-Remaining"),
                    "rate_limit_reset": headers.get("X-RateLimit-Reset"),
                },
            )
            link = next((value for key, value in headers.items() if key.lower() == "link"), "")
            if len(payload) < 100:
                break
            if link and 'rel="next"' not in link:
                break
            page += 1

    return DetailCollectionResult(
        candidates=len(selected_ids),
        requests=requests,
        snapshots=snapshots,
    )
