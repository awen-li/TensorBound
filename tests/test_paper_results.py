import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_verifier", ROOT / "scripts/verify_artifact.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class PaperResultsTests(unittest.TestCase):
    def test_exact_hypergeometric_counts(self):
        # All six samples of two objects from a four-object population.
        self.assertEqual(V.hypergeometric_cdf_numerator(4, 2, 2, 0), 1)
        self.assertEqual(V.hypergeometric_cdf_numerator(4, 2, 2, 1), 5)
        self.assertEqual(V.hypergeometric_cdf_numerator(4, 2, 2, 2), math.comb(4, 2))
        self.assertEqual(V.hypergeometric_cdf_numerator(4, 2, 2, -1), 0)

    def test_finite_population_interval(self):
        self.assertEqual(V.missed_interval(6729, 600, 4), (14, 111))
        self.assertEqual(V.missed_interval(6729, 600, 4, 20)[1], 99)

    def test_complete_census_and_extreme_observations(self):
        self.assertEqual(V.missed_interval(10, 10, 3), (3, 3))
        self.assertEqual(V.missed_interval(10, 10, 0), (0, 0))
        self.assertEqual(V.missed_interval(10, 10, 10), (10, 10))
        with self.assertRaises(V.ArtifactError):
            V.missed_interval(10, 11, 3)

    def test_human_totals_keep_unavailable_records_explicit(self):
        result = V.verify_phase1_human_totals()
        self.assertEqual(result["sample_records"], 600)
        self.assertEqual(result["sample_agreements"], 592)
        self.assertIsNone(result["missing_records"]["unique_restored_records"])
        self.assertEqual(result["missing_records"]["sample_membership"], "unavailable_in_release")

    def test_original_assignments_and_secondary_mask(self):
        result = V.verify_archived_assignments()
        self.assertEqual(result["actor_type"], "unknown")
        self.assertEqual(result["secondary_pairs"], 2235)
        self.assertEqual(result["automatic_primary_family_omissions"], 149)
        self.assertEqual(result["secondary_positive_agreement_percent"], 59.0)

    def test_pre_fix_sites_are_separate_from_repaired_paths(self):
        self.assertEqual(V.verify_source_corrections()["corrected_records"], 3)

    def test_result_index_contains_actual_members(self):
        index = V.result_index()
        mapping = index["report_cause_mapping"]
        self.assertEqual(len(mapping), 149)
        self.assertEqual(sum(row["count"] for row in mapping.values()), 162)
        self.assertIsNone(index["phase1_human_tables"]["record_ids"])
        for name in ("relation_transition_matrix", "failure_repair_matrix", "obligation_repair_matrix"):
            rows = index[name].values()
            ids = [i for row in rows for i in row["ids"]]
            self.assertEqual(len(ids), 149)
            self.assertEqual(len(set(ids)), 149)


if __name__ == "__main__":
    unittest.main()
