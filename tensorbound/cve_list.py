"""Cutoff-pinned discovery from all available CVE List search pages."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .github import _default_fetch
from .layout import record_path
from .source_store import _load_catalog, capture_source_query_page, capture_source_raw
from .storage import read_json, read_jsonl, write_json, write_jsonl


Fetch = Callable[[str, str | None], tuple[bytes, Mapping[str, str]]]
SEARCH_URL = "https://api.github.com/search/code"
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CVE_PATH_PATTERN = re.compile(r"^cves/\d{4}/[0-9x]+/CVE-\d{4}-\d+\.json$")


@dataclass(frozen=True)
class CVECollectionResult:
    requests: int
    retrieved_search_hits: int
    unique_paths: int
    candidates: int
    post_cutoff: int
    absent_at_revision: int
    incomplete_terms: tuple[str, ...]
    output: Path


def _search_url(term: str, page: int, per_page: int) -> str:
    query = f'repo:CVEProject/cvelistV5 path:cves "{term}"'
    return SEARCH_URL + "?" + urllib.parse.urlencode({
        "q": query, "per_page": per_page, "page": page,
    })


def _record_url(revision: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/CVEProject/cvelistV5/{revision}/{path}"


def collect_cve_list(
    study_root: Path,
    terms: Mapping[str, str],
    revision: str,
    *,
    per_page: int = 100,
    token: str | None = None,
    fetch: Fetch = _default_fetch,
) -> CVECollectionResult:
    """Archive CVE search pages and eligible record bytes without confirming cases.

    Search uses GitHub's current index to find paths, but record content is
    fetched from a fixed repository revision at or before the study cutoff.
    Incomplete search queries are reported, not silently treated as exhaustive.
    """
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError("revision must be a 40-character lowercase commit SHA")
    if not 1 <= per_page <= 100:
        raise ValueError("per_page must be 1..100")
    if not terms:
        raise ValueError("at least one search term is required")
    systems = {row["system_id"] for _, row in read_jsonl(record_path(study_root, "system"))}
    for term, system_id in terms.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", term):
            raise ValueError(f"invalid search term: {term}")
        if system_id not in systems:
            raise ValueError(f"unknown system ID: {system_id}")
    cutoff = date.fromisoformat(_load_catalog(study_root)["collection_through"])
    commit_url = f"https://api.github.com/repos/CVEProject/cvelistV5/commits/{revision}"
    commit_raw, _ = fetch(commit_url, token)
    try:
        commit = json.loads(commit_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid CVE List revision metadata: {error}") from error
    if not isinstance(commit, dict) or commit.get("sha") != revision:
        raise ValueError("CVE List revision metadata does not match requested SHA")
    commit_date = commit.get("commit", {}).get("committer", {}).get("date")
    if not isinstance(commit_date, str) or date.fromisoformat(commit_date[:10]) > cutoff:
        raise ValueError("CVE List revision is not at or before the study cutoff")
    capture_source_query_page(study_root, "cve-list", commit_url, commit_raw)
    plan = {
        "schema_version": "1.0.0", "source_id": "cve-list",
        "collection_through": cutoff.isoformat(), "revision": revision,
        "search_terms": dict(sorted(terms.items())), "per_page": per_page,
        "pagination": "all available pages",
        "search_limitation": "GitHub code search uses the current index and has an upstream 1000-result ceiling per query",
    }
    plan_path = study_root / "results" / "phase1" / "input" / "cve-list-plan.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("CVE List collection plan differs from frozen plan")
    else:
        write_json(plan_path, plan)
    output = study_root / "results" / "phase1" / "output" / "sources" / "cve-list" / "candidates.jsonl"
    prior = {row["cve_id"]: row for _, row in read_jsonl(output)}
    known: dict[str, dict[str, Any]] = {}
    requests = 1
    hits = post_cutoff = absent = 0
    incomplete: list[str] = []
    paths: dict[str, tuple[set[str], set[str]]] = {}
    query_totals: dict[str, int] = {}
    last_search_at: float | None = None

    for term, system_id in sorted(terms.items()):
        total = 0
        processed = 0
        page = 1
        incomplete_response = False
        while True:
            if fetch is _default_fetch and last_search_at is not None:
                time.sleep(max(0, 6.1 - (time.monotonic() - last_search_at)))
            url = _search_url(term, page, per_page)
            last_search_at = time.monotonic()
            raw, _ = fetch(url, token)
            requests += 1
            capture_source_query_page(study_root, "cve-list", url, raw)
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid CVE List search JSON: {url}: {error}") from error
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                raise ValueError(f"CVE List search response has no items array: {url}")
            if not isinstance(payload.get("total_count"), int) or payload["total_count"] < 0:
                raise ValueError(f"CVE List search response has no total_count: {url}")
            total = max(total, payload["total_count"])
            incomplete_response |= payload.get("incomplete_results") is True
            items = payload["items"]
            hits += len(items)
            processed += len(items)
            for item in items:
                if not isinstance(item, dict) or not CVE_PATH_PATTERN.fullmatch(str(item.get("path", ""))):
                    continue
                path = item["path"]
                match_terms, match_systems = paths.setdefault(path, (set(), set()))
                match_terms.add(term)
                match_systems.add(system_id)
            if not items or processed >= min(total, 1000):
                break
            page += 1
        query_totals[term] = total
        if processed < total or total > 1000 or incomplete_response:
            incomplete.append(term)

    def fetch_record(path: str) -> tuple[str, bytes | None, bool]:
        expected_id = Path(path).stem
        previous = prior.get(expected_id)
        record_url = _record_url(revision, path)
        if previous and previous.get("source_url") == record_url:
            snapshot = study_root / previous["snapshot"]
            if snapshot.is_file():
                cached = snapshot.read_bytes()
                if hashlib.sha256(cached).hexdigest() == previous.get("raw_sha256"):
                    return path, cached, False
        try:
            raw, _ = fetch(record_url, None)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return path, None, True
            raise
        return path, raw, True

    with ThreadPoolExecutor(max_workers=8) as executor:
        for path, record_raw, was_network in executor.map(fetch_record, sorted(paths)):
            requests += int(was_network)
            if record_raw is None:
                absent += 1
                continue
            record_url = _record_url(revision, path)
            try:
                record = json.loads(record_raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid CVE List record JSON: {record_url}: {error}") from error
            metadata = record.get("cveMetadata", {}) if isinstance(record, dict) else {}
            expected_id = Path(path).stem
            if not isinstance(metadata, dict) or metadata.get("cveId") != expected_id:
                raise ValueError(f"CVE record ID does not match its path: {record_url}")
            if metadata.get("state") != "PUBLISHED":
                continue
            published_at = metadata.get("datePublished")
            if not isinstance(published_at, str):
                raise ValueError(f"published CVE record lacks datePublished: {record_url}")
            if date.fromisoformat(published_at[:10]) > cutoff:
                post_cutoff += 1
                continue
            snapshot = capture_source_raw(study_root, "cve-list", expected_id,
                                          record_url, record_raw, published_at)
            match_terms, match_systems = paths[path]
            known[expected_id] = {
                "schema_version": "1.0.0", "source_id": "cve-list",
                "source_record_id": expected_id, "cve_id": expected_id,
                "source_url": record_url, "published_at": published_at,
                "revision": revision, "raw_sha256": hashlib.sha256(record_raw).hexdigest(),
                "snapshot": str(snapshot.relative_to(study_root)),
                "matched_terms": sorted(match_terms), "system_ids": sorted(match_systems),
                "screening_status": "pending",
            }

    write_jsonl(output, (known[cve_id] for cve_id in sorted(known)))
    write_json(output.with_name("collection-summary.json"), {
        "schema_version": "1.0.0", "source_id": "cve-list", "revision": revision,
        "collection_through": cutoff.isoformat(), "query_totals": query_totals,
        "retrieved_search_hits": hits, "unique_paths": len(paths),
        "candidate_count": len(known), "post_cutoff_count": post_cutoff,
        "absent_at_revision_count": absent, "incomplete_terms": incomplete,
        "complete_for_queries": not incomplete,
        "search_limitation": plan["search_limitation"],
    })
    return CVECollectionResult(requests, hits, len(paths), len(known), post_cutoff, absent,
                               tuple(incomplete), output)
