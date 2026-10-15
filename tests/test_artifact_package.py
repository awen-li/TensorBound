import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ArtifactPackageTests(unittest.TestCase):
    def test_raw_exports_have_no_notification_capability_links(self):
        prefix = b"https://github.com/notifications/unsubscribe-auth/"
        marker = prefix + b"[REDACTED]"
        for path in (ROOT / "data" / "raw").rglob("*"):
            if path.is_file():
                self.assertNotIn(prefix, path.read_bytes().replace(marker, b""), str(path))

    def test_package_has_no_intermediate_results_or_review_scripts(self):
        allowed_scripts = {"verify_artifact.py"}
        self.assertEqual({p.name for p in (ROOT / "scripts").glob("*.py")}, allowed_scripts)
        allowed_modules = {"__init__.py", "__main__.py", "cli.py", "github.py", "cve_list.py",
                           "source_store.py", "layout.py", "records.py", "storage.py", "historical_source.py"}
        self.assertEqual({p.name for p in (ROOT / "tensorbound").glob("*.py")}, allowed_modules)
        forbidden = ("template", "materialization", "reconciliation", "review-protocol",
                     "review-validation", "tosem-", "selection-stage-history", "taxonomy-recoding")
        for path in (ROOT / "results").rglob("*"):
            name = str(path.relative_to(ROOT))
            self.assertFalse(any(term in name for term in forbidden), name)
        for name in (".agent", ".agents", ".codex", "TensorBound-paper"):
            self.assertFalse((ROOT / name).exists(), name)

    def test_published_artifact_is_internally_consistent(self):
        import importlib.util

        path = ROOT / "scripts/verify_artifact.py"
        spec = importlib.util.spec_from_file_location("verify_artifact", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        summary = module.verify()
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["cohort"]["reports"], 264)
        self.assertEqual(summary["cohort"]["distinct_repaired_causes"], 149)


if __name__ == "__main__":
    unittest.main()
