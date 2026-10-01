"""Record-level validation for TensorBound study data."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from . import SCHEMA_VERSION

RECORD_ID_FIELDS = {
    "system": "system_id",
    "binding": "binding_id",
    "candidate": "candidate_id",
    "case": "case_id",
    "evidence": "evidence_id",
}

RECORD_FILES = {
    "system": "systems.jsonl",
    "binding": "bindings.jsonl",
    "candidate": "candidates.jsonl",
    "case": "cases.jsonl",
    "evidence": "evidence.jsonl",
}

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]*$")


class RecordError(ValueError):
    """Raised when a study record violates the stable schema."""


def _is_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_string(item) for item in value)


def _is_optional_string(value: Any) -> bool:
    return value is None or _is_string(value)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _require(record: Mapping[str, Any], field: str, predicate: Callable[[Any], bool], expected: str) -> None:
    if field not in record:
        raise RecordError(f"missing required field: {field}")
    if not predicate(record[field]):
        raise RecordError(f"{field} must be {expected}")


def _require_choice(record: Mapping[str, Any], field: str, choices: set[str]) -> None:
    _require(record, field, _is_string, "a non-empty string")
    if record[field] not in choices:
        options = ", ".join(sorted(choices))
        raise RecordError(f"{field} must be one of: {options}")


def _validate_common(kind: str, record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise RecordError(f"schema_version must be {SCHEMA_VERSION}")
    if record.get("record_type") != kind:
        raise RecordError(f"record_type must be {kind}")
    identifier = RECORD_ID_FIELDS[kind]
    _require(record, identifier, _is_string, "a non-empty string")
    if not _ID_PATTERN.fullmatch(record[identifier]):
        raise RecordError(f"{identifier} contains unsupported characters")


def _validate_system(record: Mapping[str, Any]) -> None:
    for field in ("name", "frontend", "runtime", "async_model", "storage_model", "completion_model"):
        _require(record, field, _is_string, "a non-empty string")
    for field in ("compiler_layers", "device_backends"):
        _require(record, field, _is_string_list, "a list of non-empty strings")
    _require(record, "version_range", _is_mapping, "an object")


def _validate_binding(record: Mapping[str, Any]) -> None:
    for field in (
        "system_id", "source_entity", "destination_entity", "relation",
        "implementation_representation", "layer", "semantic_meaning",
    ):
        _require(record, field, _is_string, "a non-empty string")
    _require_choice(record, "explicitness", {"explicit", "implicit", "mixed"})
    _require_choice(record, "availability", {"static", "dynamic", "hybrid"})
    _require_choice(record, "cardinality", {"one-to-one", "one-to-many", "many-to-one", "many-to-many"})
    _require(record, "evidence_ids", _is_string_list, "a list of evidence IDs")
    if not record["evidence_ids"]:
        raise RecordError("evidence_ids must contain at least one evidence record")


def _validate_candidate(record: Mapping[str, Any]) -> None:
    for field in ("system_id", "title", "source_url", "discovered_at"):
        _require(record, field, _is_string, "a non-empty string")
    _require(record, "discovery_channels", _is_string_list, "a list of non-empty strings")
    _require_choice(record, "status", {"pending", "uncertain", "accepted", "excluded"})
    if "duplicate_of" in record and record["duplicate_of"] is not None:
        _require(record, "duplicate_of", _is_string, "a candidate ID or null")


def _validate_case(record: Mapping[str, Any]) -> None:
    for field in ("system_id", "trigger", "symptom", "root_cause", "reproduction_status", "interpretation", "candidate_abstraction"):
        _require(record, field, _is_string, "a non-empty string")
    for field in ("value", "storage", "execution_context", "expected_completion"):
        _require(record, field, _is_optional_string, "a non-empty string or null")
    for field in (
        "candidate_ids", "operations", "affected_versions", "fixed_versions",
        "runtime_consequences", "security_consequences", "evidence_ids",
        "new_concepts",
    ):
        _require(record, field, _is_string_list, "a list of non-empty strings")
    if not record["candidate_ids"]:
        raise RecordError("candidate_ids must contain at least one candidate")
    if not record["evidence_ids"]:
        raise RecordError("evidence_ids must contain at least one evidence record")
    _require_choice(record, "evidence_level", {"E1", "E2", "E3"})
    _require(record, "transition", _is_mapping, "an object")
    for field in ("operation", "entity_before", "entity_after", "precondition", "completion_requirement"):
        _require(record["transition"], field, _is_optional_string, "a non-empty string or null")


def _validate_evidence(record: Mapping[str, Any]) -> None:
    for field in ("subject_id", "kind", "uri", "description", "observed_fact", "collected_at"):
        _require(record, field, _is_string, "a non-empty string")
    _require_choice(record, "subject_type", set(RECORD_ID_FIELDS) - {"evidence"})
    if "sha256" in record and record["sha256"] is not None:
        digest = record["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RecordError("sha256 must be a lowercase 64-character hexadecimal digest or null")


_VALIDATORS: dict[str, Callable[[Mapping[str, Any]], None]] = {
    "system": _validate_system,
    "binding": _validate_binding,
    "candidate": _validate_candidate,
    "case": _validate_case,
    "evidence": _validate_evidence,
}


def validate_record(kind: str, record: Any) -> dict[str, Any]:
    """Validate one record and return a plain mutable mapping."""
    if kind not in _VALIDATORS:
        raise RecordError(f"unknown record kind: {kind}")
    if not isinstance(record, Mapping):
        raise RecordError("record must be a JSON object")
    _validate_common(kind, record)
    _VALIDATORS[kind](record)
    return dict(record)
