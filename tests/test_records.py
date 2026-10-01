from __future__ import annotations

import unittest

from tensorbound.records import RecordError, validate_record


class RecordValidationTests(unittest.TestCase):
    def test_system_record_is_valid(self) -> None:
        record = {
            "schema_version": "1.0.0",
            "record_type": "system",
            "system_id": "iree",
            "name": "IREE",
            "frontend": "MLIR",
            "compiler_layers": ["Flow", "Stream", "HAL"],
            "runtime": "IREE runtime",
            "device_backends": ["Vulkan"],
            "async_model": "timeline semaphores",
            "storage_model": "HAL buffers",
            "completion_model": "timepoints",
            "version_range": {"from": "pilot", "through": "pilot"},
        }
        self.assertEqual(validate_record("system", record)["system_id"], "iree")

    def test_case_requires_candidate(self) -> None:
        record = {
            "schema_version": "1.0.0",
            "record_type": "case",
            "case_id": "case-1",
            "candidate_ids": [],
            "system_id": "iree",
            "affected_versions": ["v1"],
            "fixed_versions": ["v2"],
            "value": "v",
            "operations": ["dispatch"],
            "storage": "buffer",
            "execution_context": "queue",
            "expected_completion": "timepoint",
            "transition": {
                "operation": "reuse",
                "entity_before": "v1->b",
                "entity_after": "v2->b",
                "precondition": "none",
                "completion_requirement": "dispatch complete",
            },
            "trigger": "trigger",
            "symptom": "symptom",
            "root_cause": "root cause",
            "runtime_consequences": ["wrong result"],
            "security_consequences": [],
            "reproduction_status": "not attempted",
            "evidence_level": "E2",
            "evidence_ids": ["ev-1"],
            "interpretation": "interpretation",
            "candidate_abstraction": "candidate abstraction",
            "new_concepts": [],
        }
        with self.assertRaisesRegex(RecordError, "at least one candidate"):
            validate_record("case", record)

    def test_unknown_teb_components_remain_explicit(self) -> None:
        record = {
            "schema_version": "1.0.0",
            "record_type": "case",
            "case_id": "case-unknown",
            "candidate_ids": ["candidate-1"],
            "system_id": "iree",
            "affected_versions": [],
            "fixed_versions": [],
            "value": None,
            "operations": [],
            "storage": "buffer",
            "execution_context": None,
            "expected_completion": None,
            "transition": {
                "operation": "reuse",
                "entity_before": None,
                "entity_after": None,
                "precondition": None,
                "completion_requirement": None,
            },
            "trigger": "trigger",
            "symptom": "symptom",
            "root_cause": "root cause",
            "runtime_consequences": [],
            "security_consequences": [],
            "reproduction_status": "not attempted",
            "evidence_level": "E1",
            "evidence_ids": ["ev-1"],
            "interpretation": "incomplete reconstruction",
            "candidate_abstraction": "unresolved",
            "new_concepts": ["other"],
        }
        validated = validate_record("case", record)
        self.assertIsNone(validated["value"])

    def test_unknown_evidence_level_is_rejected(self) -> None:
        evidence = {
            "schema_version": "1.0.0",
            "record_type": "evidence",
            "evidence_id": "ev-1",
            "subject_type": "case",
            "subject_id": "case-1",
            "kind": "patch",
            "uri": "https://example.invalid/patch",
            "description": "fix",
            "observed_fact": "adds a dependency",
            "collected_at": "2026-09-24T00:00:00Z",
            "sha256": "not-a-digest",
        }
        with self.assertRaisesRegex(RecordError, "sha256"):
            validate_record("evidence", evidence)


if __name__ == "__main__":
    unittest.main()
