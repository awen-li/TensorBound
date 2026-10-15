# Optional data collection

Collection is separate from reproducing the frozen results. Live queries can
return different records as project labels, reports, and search indexes change.
Keep new output under `.work-tmp/`; the CLI refuses to collect into a published
artifact directory.

## GitHub issues

```sh
python3 -m tensorbound init .work-tmp/collection --through 2026-09-24
python3 -m tensorbound collect-frame .work-tmp/collection \
  --spec results/collection-frame.json --source github --plan
python3 -m tensorbound collect-frame .work-tmp/collection \
  --spec results/collection-frame.json --source github
```

`--plan` prints the queries without network requests or writes. The collector
uses the five repository predicates in the frame specification. IREE and JAX
use disjoint creation-year intervals; each query retrieves up to ten 100-item
pages. Queries reaching a page limit or reporting incomplete results are marked
incomplete, and the CLI returns exit status 2. Their query windows must be
subdivided before claiming complete retrieval.

For an individual query:

```sh
python3 -m tensorbound collect-github .work-tmp/collection \
  --system-id iree --repository iree-org/iree \
  --through 2026-09-24 --query 'label:"bug 🐞" created:2025-01-01..2025-12-31'
```

To collect issue bodies and comment pages after discovery:

```sh
python3 -m tensorbound collect-github-details .work-tmp/collection \
  --candidate-id github-iree-org-iree-10483
```

The candidate must already exist in that workspace. Raw responses, hashes,
timestamps, effective queries, and rate-limit metadata are recorded. Set
`GITHUB_TOKEN` through your normal credential mechanism when authentication is
needed. Never put a token in a command argument, tracked file, or data bundle.

## CVE List

The CVE collector searches the official repository and retrieves record content
at an explicit commit. The commit must be at or before the workspace cutoff.

```sh
python3 -m tensorbound collect-frame .work-tmp/collection \
  --spec results/collection-frame.json --source cve-list --plan
```

Inspect the plan before a live run. The released frame preserves a September
24, 2026 cutoff and pins CVE content to a revision dated September 24, 2026.
The collector rejects a revision newer than its workspace cutoff. The frozen
evidence and results are preserved as supplied; changing the cutoff or revision
for a new collection produces a new dataset, not an exact recreation of the
frozen one.

For a new collection, initialize a separate workspace with the intended cutoff
and supply a compatible 40-character commit SHA:

```sh
python3 -m tensorbound collect-cve-list .work-tmp/collection \
  --revision <compatible-40-character-commit> \
  --term IREE:iree --term PJRT:xla-jax-pjrt \
  --term XLA:xla-jax-pjrt --term jaxlib:xla-jax-pjrt \
  --term PyTorch:pytorch-inductor-triton --term Triton:pytorch-inductor-triton
```

Replace the placeholder before running the command. Search uses the current
GitHub index even when record content is pinned. The published CVE frame
contains 693 retrieved records from 713 reported hits; the 20-record retrieval
gap remains recorded in the frame specification.
