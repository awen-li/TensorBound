"""Data-collection commands for the published TensorBound artifact."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import json
import os
from pathlib import Path
import sys
import urllib.error

from . import __version__
from .cve_list import collect_cve_list
from .github import CollectionError, collect_github_issue_details, collect_github_issues
from .layout import initialize_study, record_path
from .source_store import initialize_source_stores
from .storage import append_jsonl


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='initialize a separate collection workspace')
    init.add_argument('study_dir', type=Path)
    init.add_argument('--through', required=True)
    init.add_argument('--system-id', action='append')
    github = sub.add_parser('collect-github', help='archive GitHub issue search responses')
    github.add_argument('study_dir', type=Path)
    github.add_argument('--system-id', required=True)
    github.add_argument('--repository', required=True)
    github.add_argument('--query', action='append', required=True)
    github.add_argument('--through', required=True)
    github.add_argument('--start-page', type=int, default=1)
    github.add_argument('--max-pages', type=int, default=10)
    details = sub.add_parser('collect-github-details', help='archive selected issues and comments')
    details.add_argument('study_dir', type=Path)
    details.add_argument('--candidate-id', action='append', required=True)
    cve = sub.add_parser('collect-cve-list', help='archive CVE records at a pinned revision')
    cve.add_argument('study_dir', type=Path)
    cve.add_argument('--revision', required=True)
    cve.add_argument('--term', action='append', required=True, help='TERM:SYSTEM_ID')
    frame = sub.add_parser('collect-frame', help='collect queries from a frame specification')
    frame.add_argument('study_dir', type=Path)
    frame.add_argument('--spec', type=Path, required=True)
    frame.add_argument('--source', choices=['github', 'cve-list'], default='github')
    frame.add_argument('--plan', action='store_true', help='print the plan without network or writes')
    return parser


def frame_queries(spec):
    cutoff = date.fromisoformat(spec['collection_boundary'])
    queries = []
    for repo in spec['github']['repositories']:
        labels = ' '.join('label:' + json.dumps(label, ensure_ascii=False) for label in repo['labels_all'])
        parts = [labels]
        if repo['partitioned']:
            first = date.fromisoformat(repo['partition_start'])
            parts = []
            for year in range(first.year, cutoff.year + 1):
                start = max(first, date(year, 1, 1))
                end = min(cutoff, date(year, 12, 31))
                parts.append(f'{labels} created:{start.isoformat()}..{end.isoformat()}')
        queries.append({'repository': repo['repository'], 'system_id': repo['system'], 'queries': parts})
    return queries


def protect_artifact(study_dir):
    if (study_dir / 'results/artifact-manifest.json').exists() or (study_dir / 'data/final/cases.csv').exists():
        raise ValueError('Use a separate collection directory; published data is read-only.')


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        protect_artifact(args.study_dir)
        token = os.environ.get('GITHUB_TOKEN')
        if args.command == 'init':
            date.fromisoformat(args.through)
            initialize_study(args.study_dir, 'tensorbound-collection', 'TensorBound data collection')
            # Collectors require registered system identifiers, not causal classifications.
            for system_id in dict.fromkeys(args.system_id or ['iree', 'pytorch-inductor-triton', 'xla-jax-pjrt']):
                append_jsonl(record_path(args.study_dir, 'system'), {'system_id': system_id})
            initialize_source_stores(args.study_dir, args.through)
            result = {'workspace': str(args.study_dir), 'through': args.through}
        elif args.command == 'collect-github':
            result = asdict(collect_github_issues(args.study_dir, args.system_id, args.repository,
                args.query, args.through, start_page=args.start_page, max_pages=args.max_pages, token=token))
        elif args.command == 'collect-github-details':
            result = asdict(collect_github_issue_details(args.study_dir, args.candidate_id, token=token))
        elif args.command == 'collect-cve-list':
            terms = {}
            for item in args.term:
                term, system = item.split(':', 1)
                if term in terms and terms[term] != system:
                    raise ValueError(f'conflicting system mapping for {term}')
                terms[term] = system
            result = asdict(collect_cve_list(args.study_dir, terms, args.revision, token=token))
        else:
            spec = json.loads(args.spec.read_text())
            if args.source == 'github':
                plan = frame_queries(spec)
                if args.plan:
                    result = {'through': spec['collection_boundary'], 'queries': plan}
                else:
                    result = {'collections': [asdict(collect_github_issues(args.study_dir,
                        item['system_id'], item['repository'], item['queries'],
                        spec['collection_boundary'], max_pages=10, token=token)) for item in plan]}
            else:
                frame = spec['cve_list']
                terms = {item['term']: item['system'] for item in frame['terms']}
                if args.plan:
                    result = {'through': spec['collection_boundary'], 'revision': frame['revision'], 'terms': terms}
                else:
                    result = asdict(collect_cve_list(args.study_dir, terms, frame['revision'], token=token))
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        runs = result.get('collections', [result])
        return 2 if any(run.get('incomplete_queries') or run.get('incomplete_terms') for run in runs) else 0
    except (OSError, ValueError, KeyError, CollectionError, urllib.error.URLError) as error:
        print(f'collection failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
