"""Independent raw and confirmed evidence stores for discovery sources."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .layout import utc_now
from .records import RecordError, validate_record
from .storage import append_jsonl, read_json, read_jsonl, write_bytes, write_json


SOURCE_SPECS = (
    ("github-issues", "discovery", "existing issue collection; raw data in data/github"),
    ("cve-list", "discovery", "official CVE records; CVE ID is not a case ID"),
)
SOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
SOURCE_HOSTS = {
    "cve-list": {"github.com", "api.github.com", "raw.githubusercontent.com"},
}
SOURCE_DOCUMENTATION = {
    "github-issues": "https://docs.github.com/en/rest/issues/issues",
    "cve-list": "https://github.com/CVEProject/cvelistV5",
}


@dataclass
class SourceValidation:
    errors: list[str] = field(default_factory=list)
    raw_counts: dict[str, int] = field(default_factory=dict)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    confirmed_counts: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors


def catalog_path(study_root: Path) -> Path:
    return study_root / "results" / "phase1" / "input" / "sources.json"


def _confirmed_root(study_root: Path, source_id: str) -> Path:
    return study_root / "results" / "phase2" / "output" / "sources" / source_id


def _acquisition_path(study_root: Path, source_id: str) -> Path:
    if source_id == "github-issues":
        return study_root / "data" / "github" / "acquisitions.jsonl"
    return study_root / "data" / "sources" / source_id / "acquisitions.jsonl"


def initialize_source_stores(study_root: Path, through: str) -> Path:
    """Freeze source identities; do not mix provider-specific evidence."""
    date.fromisoformat(through)
    if not (study_root / "results" / "phase1" / "input" / "manifest.json").exists():
        raise ValueError(f"not a TensorBound study directory: {study_root}")
    path = catalog_path(study_root)
    if path.exists():
        raise FileExistsError(f"source catalog already exists: {path}")
    catalog = {
        "schema_version": "1.0.0",
        "collection_through": through,
        "merge_policy": "No cross-source case merging without explicit root-cause adjudication",
        "sources": [
            {"source_id": source_id, "role": role, "scope_note": note,
             "documentation_url": SOURCE_DOCUMENTATION[source_id]}
            for source_id, role, note in SOURCE_SPECS
        ],
    }
    write_json(path, catalog)
    for source_id, _, _ in SOURCE_SPECS:
        root = _confirmed_root(study_root, source_id)
        root.mkdir(parents=True, exist_ok=True)
        (root / "cases.jsonl").touch(exist_ok=True)
        (root / "evidence.jsonl").touch(exist_ok=True)
    return path


def _load_catalog(study_root: Path) -> dict[str, Any]:
    catalog = read_json(catalog_path(study_root))
    if not isinstance(catalog, dict) or catalog.get("schema_version") != "1.0.0":
        raise ValueError("invalid source catalog")
    sources = catalog.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source catalog has no sources")
    ids = [row.get("source_id") for row in sources if isinstance(row, dict)]
    if len(ids) != len(sources) or any(
        not isinstance(source_id, str) or not SOURCE_ID_PATTERN.fullmatch(source_id)
        for source_id in ids
    ) or len(ids) != len(set(ids)):
        raise ValueError("source catalog has invalid or duplicate source IDs")
    if ids != [source_id for source_id, _, _ in SOURCE_SPECS]:
        raise ValueError("source catalog differs from the selected study sources")
    if any(source.get("role") != role for source, (_, role, _) in zip(sources, SOURCE_SPECS)):
        raise ValueError("source catalog has an invalid source role")
    date.fromisoformat(catalog["collection_through"])
    return catalog


def _require_source(study_root: Path, source_id: str) -> dict[str, Any]:
    catalog = _load_catalog(study_root)
    for source in catalog["sources"]:
        if source["source_id"] == source_id:
            return source
    raise ValueError(f"unknown source ID: {source_id}")


def _publication_day(value: str) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError("published_at must be an ISO date or datetime")
    if len(value) == 10:
        return date.fromisoformat(value)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _approved_source_url(source_id: str, url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in SOURCE_HOSTS[source_id]:
        return False
    path = parsed.path.lower()
    host = parsed.hostname
    if source_id == "cve-list":
        return ((host == "github.com" and path.startswith("/cveproject/cvelistv5/"))
                or (host == "api.github.com" and path.startswith("/repos/cveproject/cvelistv5/"))
                or (host == "api.github.com" and path == "/search/code"
                    and any(value.lower() == "repo:cveproject/cvelistv5"
                            for value in parse_qs(parsed.query).get("q", [""])[0].split()))
                or (host == "raw.githubusercontent.com" and path.startswith("/cveproject/cvelistv5/")))
    return True


def _snapshot_raw(study_root: Path, source_id: str, raw: bytes, media_type: str) -> tuple[Path, str]:
    if not raw:
        raise ValueError("raw response must not be empty")
    if media_type not in {"application/json", "text/html", "text/plain"}:
        raise ValueError("unsupported source media type")
    digest = hashlib.sha256(raw).hexdigest()
    extension = {"application/json": "json", "text/html": "html", "text/plain": "txt"}[media_type]
    snapshot = study_root / "data" / "sources" / source_id / "snapshots" / f"{digest}.{extension}"
    if snapshot.exists():
        if snapshot.read_bytes() != raw:
            raise ValueError(f"content-addressed snapshot collision: {snapshot}")
    else:
        write_bytes(snapshot, raw)
    return snapshot, digest


def capture_source_query_page(study_root: Path, source_id: str, source_url: str, raw: bytes) -> Path:
    """Archive an exact query response, including pages containing zero records."""
    _require_source(study_root, source_id)
    if source_id == "github-issues" or not _approved_source_url(source_id, source_url):
        raise ValueError(f"query URL must use an approved {source_id} organization endpoint")
    snapshot, digest = _snapshot_raw(study_root, source_id, raw, "application/json")
    acquisition_path = _acquisition_path(study_root, source_id)
    identity = ("query-page", source_url, digest)
    previous = [row for _, row in read_jsonl(acquisition_path)]
    if not any((row.get("record_kind"), row.get("source_url"), row.get("sha256")) == identity
               for row in previous):
        append_jsonl(acquisition_path, {
            "schema_version": "1.0.0", "source_id": source_id,
            "record_kind": "query-page", "source_url": source_url,
            "query_cutoff": _load_catalog(study_root)["collection_through"],
            "retrieved_at": utc_now(), "media_type": "application/json",
            "sha256": digest, "snapshot": str(snapshot.relative_to(study_root)),
        })
    return snapshot


def capture_source_raw(
    study_root: Path,
    source_id: str,
    source_record_id: str,
    source_url: str,
    raw: bytes,
    published_at: str,
    media_type: str = "application/json",
) -> Path:
    """Store one exact source response and append its acquisition provenance."""
    _require_source(study_root, source_id)
    if source_id == "github-issues":
        raise ValueError("github-issues uses its existing dedicated collector")
    if not source_record_id.strip():
        raise ValueError("source record ID and raw response are required")
    if not _approved_source_url(source_id, source_url):
        raise ValueError(f"source URL must use an approved {source_id} organization host")
    published = _publication_day(published_at)
    cutoff = date.fromisoformat(_load_catalog(study_root)["collection_through"])
    if published > cutoff:
        raise ValueError("source record was published after the frozen cutoff")
    snapshot, digest = _snapshot_raw(study_root, source_id, raw, media_type)
    acquisition_path = _acquisition_path(study_root, source_id)
    previous = [row for _, row in read_jsonl(acquisition_path)]
    identity = (source_record_id, source_url, digest)
    if not any((row.get("source_record_id"), row.get("source_url"), row.get("sha256")) == identity for row in previous):
        append_jsonl(acquisition_path, {
            "schema_version": "1.0.0",
            "source_id": source_id,
            "record_kind": "source-record",
            "source_record_id": source_record_id,
            "source_url": source_url,
            "published_at": published_at,
            "retrieved_at": utc_now(),
            "media_type": media_type,
            "sha256": digest,
            "snapshot": str(snapshot.relative_to(study_root)),
        })
    return snapshot


def publish_source_case(study_root: Path, source_id: str, bundle: dict[str, Any]) -> str:
    """Publish a case together with its source-backed evidence."""
    source = _require_source(study_root, source_id)
    if source["role"] != "discovery":
        raise ValueError(f"{source_id} is not a case-discovery source")
    case = validate_record("case", bundle.get("case"))
    evidence_items = bundle.get("evidence")
    if not isinstance(evidence_items, list) or not evidence_items:
        raise ValueError("source case requires evidence records")
    evidence = [validate_record("evidence", item) for item in evidence_items]
    if case.get("source_id") != source_id or not case["case_id"].startswith(source_id + "-"):
        raise ValueError("case ID and source_id must identify this source")
    record_ids = case.get("source_record_ids")
    if not isinstance(record_ids, list) or not record_ids or not all(isinstance(value, str) and value for value in record_ids):
        raise ValueError("case requires source_record_ids")
    if not isinstance(case.get("confirmed_by"), str) or not case["confirmed_by"].strip():
        raise ValueError("source case requires a named evidence confirmer")
    if "human_confirmed" in case or "human_reviewed" in case:
        raise ValueError("human review belongs to the later independent-review stage")
    if case["evidence_level"] not in {"E2", "E3"}:
        raise ValueError("source case requires E2 or E3 evidence")
    acquisitions = [row for _, row in read_jsonl(_acquisition_path(study_root, source_id))]
    available_ids = {row.get("source_record_id", row.get("candidate_id")) for row in acquisitions
                     if row.get("record_kind") != "query-page"}
    if not set(record_ids) <= available_ids:
        raise ValueError("case cites source records without raw acquisitions")
    raw_hashes = {row.get("sha256", row.get("response_sha256")) for row in acquisitions}
    if not any(item.get("sha256") in raw_hashes for item in evidence):
        raise ValueError("case requires at least one evidence digest from this source's raw store")
    ids = [item["evidence_id"] for item in evidence]
    if len(ids) != len(set(ids)) or set(case["evidence_ids"]) != set(ids):
        raise ValueError("case evidence IDs do not match bundle")
    if any(item["subject_type"] != "case" or item["subject_id"] != case["case_id"] for item in evidence):
        raise ValueError("evidence subject does not match case")

    root = _confirmed_root(study_root, source_id)
    case_path = root / "cases.jsonl"
    evidence_path = root / "evidence.jsonl"
    if any(row.get("case_id") == case["case_id"] for _, row in read_jsonl(case_path)):
        raise ValueError(f"duplicate source case ID: {case['case_id']}")
    existing_evidence = {row.get("evidence_id") for _, row in read_jsonl(evidence_path)}
    if existing_evidence & set(ids):
        raise ValueError("duplicate source evidence ID")
    for item in evidence:
        append_jsonl(evidence_path, item)
    append_jsonl(case_path, case)
    return case["case_id"]


def validate_source_stores(study_root: Path) -> SourceValidation:
    """Check source separation, snapshot integrity, and confirmed-case links."""
    catalog = _load_catalog(study_root)
    report = SourceValidation()
    for source in catalog["sources"]:
        source_id = source["source_id"]
        acquisition_path = _acquisition_path(study_root, source_id)
        acquisitions = [row for _, row in read_jsonl(acquisition_path)]
        report.raw_counts[source_id] = len(acquisitions)
        report.candidate_counts[source_id] = 0
        if source_id == "github-issues":
            report.candidate_counts[source_id] = sum(
                1 for _ in read_jsonl(study_root / "results" / "phase1" / "input" / "candidates.jsonl")
            )
        available_ids = {row.get("source_record_id", row.get("candidate_id")) for row in acquisitions
                         if row.get("record_kind") != "query-page"}
        raw_hashes = {row.get("sha256", row.get("response_sha256")) for row in acquisitions}
        if source_id != "github-issues":
            for row in acquisitions:
                if row.get("source_id") != source_id or not isinstance(row.get("snapshot"), str):
                    report.errors.append(f"{source_id}: invalid acquisition provenance")
                    continue
                if not isinstance(row.get("source_url"), str) or not _approved_source_url(source_id, row["source_url"]):
                    report.errors.append(f"{source_id}: unapproved acquisition URL")
                if row.get("record_kind") == "query-page":
                    if row.get("query_cutoff") != catalog["collection_through"]:
                        report.errors.append(f"{source_id}: query page uses a different cutoff")
                elif row.get("record_kind") in {None, "source-record"}:
                    try:
                        if _publication_day(row.get("published_at")) > date.fromisoformat(catalog["collection_through"]):
                            report.errors.append(f"{source_id}: post-cutoff source record")
                    except ValueError:
                        report.errors.append(f"{source_id}: invalid publication date")
                    if not isinstance(row.get("source_record_id"), str) or not row["source_record_id"]:
                        report.errors.append(f"{source_id}: missing source record ID")
                else:
                    report.errors.append(f"{source_id}: unknown acquisition kind")
                snapshot = (study_root / row["snapshot"]).resolve()
                expected_dir = (study_root / "data" / "sources" / source_id).resolve()
                if not snapshot.is_relative_to(expected_dir):
                    report.errors.append(f"{source_id}: snapshot escapes source raw store")
                elif not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != row.get("sha256"):
                    report.errors.append(f"{source_id}: missing or corrupted raw snapshot: {row.get('snapshot')}")
        if source_id == "cve-list":
            candidate_path = study_root / "results" / "phase1" / "output" / "sources" / "cve-list" / "candidates.jsonl"
            candidates = [row for _, row in read_jsonl(candidate_path)]
            report.candidate_counts[source_id] = len(candidates)
            raw_records = {(row.get("source_record_id"), row.get("sha256")) for row in acquisitions
                           if row.get("record_kind") != "query-page"}
            seen_ids: set[str] = set()
            for candidate in candidates:
                cve_id = candidate.get("cve_id")
                if cve_id in seen_ids:
                    report.errors.append(f"cve-list: duplicate candidate: {cve_id}")
                seen_ids.add(cve_id)
                if (candidate.get("source_id") != "cve-list" or candidate.get("source_record_id") != cve_id
                    or (cve_id, candidate.get("raw_sha256")) not in raw_records):
                    report.errors.append(f"cve-list: candidate lacks matching raw acquisition: {cve_id}")
                if candidate.get("screening_status") not in {"pending", "uncertain", "accepted", "excluded"}:
                    report.errors.append(f"cve-list: invalid candidate screening status: {cve_id}")
        root = _confirmed_root(study_root, source_id)
        cases = [row for _, row in read_jsonl(root / "cases.jsonl")]
        evidence = [row for _, row in read_jsonl(root / "evidence.jsonl")]
        report.confirmed_counts[source_id] = len(cases)
        evidence_by_id = {row.get("evidence_id"): row for row in evidence}
        if len(evidence_by_id) != len(evidence):
            report.errors.append(f"{source_id}: duplicate evidence IDs")
        for raw_item in evidence:
            try:
                item = validate_record("evidence", raw_item)
            except RecordError as error:
                report.errors.append(f"{source_id}: {error}")
                continue
            if item.get("subject_type") != "case":
                report.errors.append(f"{source_id}: evidence subject must be case: {item['evidence_id']}")
        seen_cases: set[str] = set()
        for raw_case in cases:
            try:
                case = validate_record("case", raw_case)
            except RecordError as error:
                report.errors.append(f"{source_id}: {error}")
                continue
            case_id = case["case_id"]
            if case_id in seen_cases:
                report.errors.append(f"{source_id}: duplicate case ID: {case_id}")
            seen_cases.add(case_id)
            if source["role"] != "discovery":
                report.errors.append(f"{source_id}: non-discovery source contains a case: {case_id}")
            if (case.get("source_id") != source_id or not case_id.startswith(source_id + "-")
                or not isinstance(case.get("confirmed_by"), str) or not case["confirmed_by"].strip()
                or "human_confirmed" in case or "human_reviewed" in case
                or case.get("evidence_level") not in {"E2", "E3"}):
                report.errors.append(f"{source_id}: case lacks source identity or qualified evidence confirmation: {case_id}")
            record_ids = case.get("source_record_ids")
            if not isinstance(record_ids, list) or not record_ids or not all(isinstance(value, str) and value for value in record_ids) or not set(record_ids) <= available_ids:
                report.errors.append(f"{source_id}: case lacks raw source records: {case_id}")
            if not any(evidence_by_id.get(evidence_id, {}).get("sha256") in raw_hashes for evidence_id in case["evidence_ids"]):
                report.errors.append(f"{source_id}: case lacks raw-backed evidence: {case_id}")
            for evidence_id in case["evidence_ids"]:
                item = evidence_by_id.get(evidence_id)
                if item is None or item.get("subject_type") != "case" or item.get("subject_id") != case_id:
                    report.errors.append(f"{source_id}: missing case evidence: {case_id}/{evidence_id}")
    return report
