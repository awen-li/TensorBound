from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from _temp import temporary_directory
from tensorbound.cli import build_parser, frame_queries, main
from tensorbound.github import CollectionResult


ROOT = Path(__file__).resolve().parents[1]


class CollectionCLITests(unittest.TestCase):
    def test_plan_preserves_all_five_predicates_and_final_cutoff(self):
        spec = json.loads((ROOT / 'results/collection-frame.json').read_text())
        plan = frame_queries(spec)
        self.assertEqual(len(plan), 5)
        for repo in plan:
            if repo['repository'] in ('iree-org/iree', 'jax-ml/jax'):
                self.assertEqual(len(repo['queries']), 19)
                self.assertIn('created:2026-01-01..2026-09-24', repo['queries'][-1])
            else:
                self.assertEqual(len(repo['queries']), 1)
        self.assertIn('label:"module: correctness (silent)"', plan[1]['queries'][0])

    def test_plan_needs_no_workspace_or_network(self):
        with temporary_directory() as tmp:
            target = Path(tmp) / 'not-created'
            with redirect_stdout(StringIO()):
                code = main(['collect-frame', str(target), '--spec', str(ROOT / 'results/collection-frame.json'), '--plan'])
            self.assertEqual(code, 0)
            self.assertFalse(target.exists())

    def test_initialization_registers_collector_systems(self):
        with temporary_directory() as tmp:
            target = Path(tmp) / 'collection'
            with redirect_stdout(StringIO()):
                self.assertEqual(main(['init', str(target), '--through', '2026-06-24']), 0)
            systems = [json.loads(line) for line in
                       (target / 'results/phase1/input/systems.jsonl').read_text().splitlines()]
            self.assertEqual({r['system_id'] for r in systems}, {'iree', 'pytorch-inductor-triton', 'xla-jax-pjrt'})

    def test_refuses_to_write_into_published_artifact(self):
        with redirect_stderr(StringIO()):
            self.assertEqual(main(['init', str(ROOT), '--through', '2026-06-24']), 1)

    def test_no_review_commands_are_published(self):
        help_text = build_parser().format_help()
        for command in ('prepare-review', 'compare-reviews', 'build-consensus', 'prepare-reconstruction'):
            self.assertNotIn(command, help_text)

    def test_incomplete_query_returns_nonzero_status(self):
        with temporary_directory() as tmp:
            result = CollectionResult(retrieved_items=1000, unique_items=1000,
                                      added_candidates=1000, skipped_existing=0,
                                      requests=10, incomplete_queries=('label:bug',))
            with patch('tensorbound.cli.collect_github_issues', return_value=result), redirect_stdout(StringIO()):
                status = main(['collect-github', tmp, '--system-id', 'iree',
                               '--repository', 'iree-org/iree', '--query', 'label:bug',
                               '--through', '2026-06-24'])
            self.assertEqual(status, 2)


if __name__ == '__main__':
    unittest.main()
