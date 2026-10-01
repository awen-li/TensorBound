#!/usr/bin/env python3
"""Recompute and validate the published TensorBound artifact summary."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ArtifactError(RuntimeError):
    """Raised when published artifact files disagree."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_collection_frame() -> dict[str, int | str]:
    frame = read_json("results/collection-frame.json")
    github = frame["github"]
    repositories = github["repositories"]
    boundary = frame["collection_boundary"]
    github_records = sum(item["unique_records"] for item in repositories)
    cve_records = frame["cve_list"]["retrieved_unique_records"]

    require(len(repositories) == 5, "collection frame must contain five repositories")
    require(github_records == github["unique_records"], "GitHub repository counts disagree")
    require(
        all(f"created:<={boundary}" in item["whole_range_query"] for item in repositories),
        "a GitHub query does not use the declared collection boundary",
    )
    require(
        github_records + cve_records == frame["total_records"],
        "GitHub and CVE counts do not sum to the total collection frame",
    )
    return {
        "boundary": boundary,
        "github_records": github_records,
        "cve_records": cve_records,
        "total_records": frame["total_records"],
    }


def verify_case_bundles(rows: list[dict[str, str]]) -> None:
    aliases = {
        "iree-org/iree": "iree",
        "pytorch/pytorch": "pytorch",
        "triton-lang/triton": "triton",
        "jax-ml/jax": "jax",
        "openxla/xla": "xla",
    }
    for row in rows:
        require(row["repository"] in aliases, f"unknown repository: {row['repository']}")
        case_root = ROOT / "data/final" / aliases[row["repository"]] / row["case_id"]
        require((case_root / "report.md").is_file(), f"missing case report: {case_root}")
        patches = [item.strip() for item in row["fix_patches"].split(";") if item.strip()]
        for patch in patches:
            require((case_root / "fixes" / patch).is_file(), f"missing patch: {patch}")


def verify_cohort() -> dict[str, object]:
    cases = read_csv("data/final/cases.csv")
    keys = [(row["repository"], row["case_id"]) for row in cases]
    statuses = Counter(row["analysis_status"] for row in cases)
    causes = {row["cause_id"] for row in cases if row["cause_id"]}
    repair_backed = sum(bool(row["cause_id"]) for row in cases)
    repositories = Counter(row["repository"] for row in cases)
    stacks = {
        "iree": repositories["iree-org/iree"],
        "pytorch-inductor-triton": (
            repositories["pytorch/pytorch"] + repositories["triton-lang/triton"]
        ),
        "xla-jax-pjrt": repositories["jax-ml/jax"] + repositories["openxla/xla"],
    }

    require(len(cases) == 264, "final cohort must contain 264 reports")
    require(len(set(keys)) == len(keys), "final cohort contains duplicate report keys")
    require(repair_backed == 162, "repair-backed report count must be 162")
    require(len(causes) == 149, "distinct repaired-cause count must be 149")
    require(statuses["reviewed_no_traceable_landed_repair"] == 102, "report-only count must be 102")
    require(stacks == {"iree": 74, "pytorch-inductor-triton": 99, "xla-jax-pjrt": 91}, "stack counts disagree")
    verify_case_bundles(cases)
    return {
        "reports": len(cases),
        "repair_backed_reports": repair_backed,
        "distinct_repaired_causes": len(causes),
        "reports_without_traceable_repair": statuses["reviewed_no_traceable_landed_repair"],
        "reports_by_stack": stacks,
    }


def verify_analysis_outputs() -> dict[str, int]:
    expected_rows = {
        "results/rq-cause-analysis.csv": 149,
        "results/rq2-provenance-analysis.csv": 149,
        "results/rq5-report-impacts.csv": 264,
        "results/rq6-repair-relevant-facts.csv": 149,
        "results/rq6-diagnostic-attempts.csv": 4,
        "results/report-accountability.csv": 264,
    }
    observed: dict[str, int] = {}
    for relative_path, expected in expected_rows.items():
        count = len(read_csv(relative_path))
        require(count == expected, f"{relative_path} has {count} rows; expected {expected}")
        observed[Path(relative_path).name] = count

    cause_rows = read_csv("results/rq-cause-analysis.csv")
    require(len({row["cause_id"] for row in cause_rows}) == 149, "cause table has duplicate IDs")
    impact_rows = read_csv("results/rq5-report-impacts.csv")
    require(
        len({(row["repository"], row["case_id"]) for row in impact_rows}) == 264,
        "report-impact table does not cover 264 unique reports",
    )
    return observed


def verify_manifest() -> int:
    manifest = read_json("results/artifact-manifest.json")
    expected = set()
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        require(path.resolve().is_relative_to(ROOT), "manifest path leaves the artifact")
        require(entry["path"] not in expected, "duplicate manifest path")
        expected.add(entry["path"])
        require(path.is_file(), f"manifest file is missing: {entry['path']}")
        require(path.stat().st_size == entry["bytes"], f"size mismatch: {entry['path']}")
        require(sha256(path) == entry["sha256"], f"hash mismatch: {entry['path']}")
    observed = {str(p.relative_to(ROOT)) for folder in ("data", "results")
                for p in (ROOT / folder).rglob("*") if p.is_file() and p.name != "artifact-manifest.json"}
    require(expected == observed, "data inventory differs from the manifest")
    return len(manifest["files"])


def jsonl(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def verify_results():
    causes = read_csv("results/rq-cause-analysis.csv")
    ids = {r["cause_id"] for r in causes}
    summary = read_json("results/rq-analysis-summary.json")
    dimensions = ("relation_family", "transition_family", "obligation_family", "failure_mode", "repair_strategy")
    codebook = read_json("results/taxonomy-codebook.json")
    for field in dimensions:
        counts = dict(Counter(r[field] for r in causes))
        require(counts == summary[field], f"RQ distribution differs: {field}")
        require(set(counts) == set(codebook["dimensions"][field]), f"codebook differs: {field}")
        for family, definition in codebook["dimensions"][field].items():
            example = definition["positive_example"]
            require(any(r["cause_id"] == example["cause_id"] and r[field] == family for r in causes),
                    f"codebook example differs: {family}")
            require((ROOT / example["case_path"]).is_file(), f"missing example: {family}")
    for name, x, y in (("failure_by_repair", "failure_mode", "repair_strategy"),
                       ("obligation_by_repair", "obligation_family", "repair_strategy"),
                       ("relation_by_obligation", "relation_family", "obligation_family"),
                       ("relation_by_transition", "relation_family", "transition_family")):
        counts = {value: dict(Counter(r[y] for r in causes if r[x] == value)) for value in {r[x] for r in causes}}
        require(counts == summary["cross_tabs"][name], f"cross-tab differs: {name}")
    provenance = read_csv("results/rq2-provenance-analysis.csv")
    ps = read_json("results/rq2-provenance-summary.json")
    require({r["cause_id"] for r in provenance} == ids, "RQ2 cause coverage differs")
    for field in ("fact_origin_layer", "fact_consumer_layer", "layer_scope", "semantic_boundary_count"):
        require(dict(Counter(r[field] for r in provenance)) == ps[field], f"RQ2 differs: {field}")
    impacts = read_csv("results/rq5-report-impacts.csv")
    consequence = Counter(r["primary_observed_consequence"] for r in impacts)
    require(dict(consequence) == read_json("results/rq5-impact-summary.json")["primary_observed_consequence"],
            "report consequence counts differ")
    pairs = Counter()
    secondary_causes = secondary_labels = 0
    for row in causes:
        labels = [value.strip() for value in row["secondary_obligation_families"].split(";") if value.strip()]
        secondary_causes += bool(labels)
        secondary_labels += len(labels)
        pairs.update((row["obligation_family"], label) for label in labels)
    secondary = summary["secondary_obligation_analysis"]
    require(secondary_causes == secondary["causes_with_consensus_secondary"], "secondary cause count differs")
    require(secondary_labels == secondary["consensus_secondary_labels"], "secondary label count differs")
    require(dict(pairs) == {(r["primary"], r["secondary"]): r["count"] for r in secondary["primary_secondary_pairs"]},
            "primary-secondary pairs differ")
    attempts = read_csv("results/rq6-diagnostic-attempts.csv")
    ds = read_json("results/rq6-summary.json")
    for field, counts in ds["diagnostic_outcomes"].items():
        require(dict(Counter(r[field] for r in attempts)) == counts, f"diagnostic counts differ: {field}")
    return {"families": {f: len(summary[f]) for f in dimensions},
            "cross_layer_paths": sum(r["layer_scope"] == "cross-layer" for r in provenance),
            "secondary_causes": secondary_causes, "secondary_labels": secondary_labels,
            "diagnostic_attempts": len(attempts)}


def verify_sensitivity():
    selection = read_csv("results/selection.csv")
    require(len(selection) == len({r["record_id"] for r in selection}) == 7509, "selection coverage differs")
    require(dict(Counter(r["final_membership"] for r in selection)) == {"retained": 264, "excluded": 7245},
            "selection membership differs")
    retained = {r["record_id"] for r in selection if r["final_membership"] == "retained"}
    cases = read_csv("data/final/cases.csv")
    require(retained == {f"github-{r['repository'].replace('/', '-')}-{r['case_id']}" for r in cases},
            "selection differs from the cohort")
    audit = jsonl("results/statistics/exclusion-audit.jsonl")
    positives = [r for r in audit if r["final_codebook_decision"] == "in_scope"]
    require(len(audit) == len({r["record_id"] for r in audit}) == 2516 and len(positives) == 19,
            "audit counts differ")
    omissions = read_csv("results/statistics/omitted-positive-sensitivity.csv")
    require({r["record_id"] for r in omissions} == {r["record_id"] for r in positives}, "omission IDs differ")
    published = read_json("results/statistics/selection-sensitivity-summary.json")
    primary = Counter(r["primary_observed_consequence"] for r in read_csv("results/rq5-report-impacts.csv"))
    observed = primary + Counter(r["primary_consequence"] for r in omissions)
    weighted = Counter(primary)
    for row in omissions:
        weighted[row["primary_consequence"]] += float(row["sampling_weight"])
    require(dict(observed) == published["combined_exact_report_consequence_scenario"], "exact sensitivity differs")
    for family, count in weighted.items():
        require(abs(count - published["combined_weighted_report_consequence_scenario"][family]) < 1e-9,
                f"weighted sensitivity differs: {family}")
    require(abs(sum(weighted.values()) - published["estimated_total_positive_reports"]) < 1e-9,
            "weighted denominator differs")
    evidence = read_json("results/evidence-sensitivity.json")
    causes = read_csv("results/rq-cause-analysis.csv")
    support = {r["cause_id"]: r for r in read_csv("results/historical-alignment.csv")}
    require(set(support) == {r["cause_id"] for r in causes}, "support coverage differs")
    stronger = [r for r in causes if support[r["cause_id"]]["landed_patch"] == "yes"
                and support[r["cause_id"]]["source_contrast"] == "yes"
                and any(support[r["cause_id"]][f] == "yes" for f in ("regression_test", "developer_diagnosis"))]
    require(len(stronger) == 136, "stronger-evidence denominator differs")
    for field, distribution in evidence["stronger_evidence_distributions"].items():
        full = Counter(r[field] for r in causes)
        subset = Counter(r[field] for r in stronger)
        for row in distribution["families"]:
            family = row["family"]
            require((row["full_count"], row["stronger_count"]) == (full[family], subset[family]),
                    f"sensitivity count differs: {family}")
            require(abs(row["change_percentage_points"] - 100 * (subset[family]/136 - full[family]/149)) < 1e-9,
                    f"sensitivity percentage differs: {family}")
    patches = jsonl("results/repair-patch-index.jsonl")
    require(len(patches) == 167 and {p["cause_id"] for p in patches} == set(support), "patch coverage differs")
    for patch in patches:
        path = ROOT / patch["path"]
        require(sha256(path) == patch["sha256"], f"patch hash differs: {path}")
        files = sorted({line.split(" b/", 1)[1] for line in path.read_text().splitlines()
                        if line.startswith("diff --git ") and " b/" in line})
        require(files == patch["changed_files"], f"changed-file list differs: {path}")
    return {"audited_exclusions": len(audit), "observed_omissions": len(omissions),
            "observed_in_scope": sum(observed.values()), "weighted_in_scope": sum(weighted.values()),
            "stronger_evidence_causes": len(stronger), "archived_patches": len(patches)}


def verify() -> dict[str, object]:
    return {
        "status": "ok",
        "collection": verify_collection_frame(),
        "cohort": verify_cohort(),
        "analysis_rows": verify_analysis_outputs(),
        "data_manifest_files": verify_manifest(),
        "results": verify_results(),
        "sensitivity": verify_sensitivity(),
    }


def main() -> int:
    try:
        summary = verify()
    except (ArtifactError, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"artifact verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
