#!/usr/bin/env python3
"""Recompute and validate the published TensorBound artifact summary."""

from __future__ import annotations

import csv
import hashlib
import json
import re
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


def verify_causal_selection() -> dict[str, object]:
    cases = read_csv("data/final/cases.csv")
    causes = {row["cause_id"]: row for row in read_csv("results/rq-cause-analysis.csv")}
    support = {row["cause_id"]: row for row in read_csv("results/historical-alignment.csv")}
    audit = read_csv("results/causal-selection-audit.csv")
    summary = read_json("results/causal-selection-audit-summary.json")
    repair_linked = [row for row in cases if row["analysis_status"].startswith("repair_backed_")]
    report_only = [row for row in cases
                   if row["analysis_status"] == "reviewed_no_traceable_landed_repair"]
    statuses = Counter(row["analysis_status"] for row in cases)
    report_counts = Counter(row["cause_id"] for row in repair_linked)

    require(len(repair_linked) == 162 and len(report_only) == 102,
            "causal-selection cohort partition differs")
    require(statuses["repair_backed_primary"] == 149 and statuses["repair_backed_duplicate"] == 13,
            "causal-selection primary or duplicate count differs")
    require(len(audit) == 149 and {row["cause_id"] for row in audit} == set(causes) == set(support),
            "causal-selection audit cause coverage differs")

    for row in audit:
        cause_id = row["cause_id"]
        cause = causes[cause_id]
        evidence = support[cause_id]
        expected = {
            "primary_report_id": cause["primary_report_id"],
            "repair_linked_report_count": str(report_counts[cause_id]),
            "first_invalid_decision_site": evidence["first_invalid_decision_site"],
            "primary_obligation": cause["obligation_family"],
            "secondary_obligations": cause["secondary_obligation_families"],
            "landed_patch": evidence["landed_patch"],
            "source_contrast": evidence["source_contrast"],
            "regression_test": evidence["regression_test"],
            "developer_diagnosis": evidence["developer_diagnosis"],
            "public_reproduction_evidence": evidence["public_reproduction_evidence"],
            "support_basis": evidence["support_basis"],
            "case_path": cause["case_path"],
        }
        require(all(row[field] == value for field, value in expected.items()),
                f"causal-selection evidence differs: {cause_id}")
        require(row["final_decision_status"] == "resolved_unique"
                and row["final_primary_obligation_status"] == "resolved_unique",
                f"unresolved final causal selection: {cause_id}")
        require(bool(row["primary_obligation_basis"].strip()),
                f"missing primary-obligation basis: {cause_id}")
        case_path = ROOT / row["case_path"]
        require(case_path.is_file(), f"missing causal-selection case: {cause_id}")
        match = re.search(r"^## Causal basis\s*$\n+(.*?)(?=\n## |\Z)",
                          case_path.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL)
        require(match is not None, f"missing causal basis in case report: {cause_id}")
        causal_basis = re.sub(r"\s+", " ", match.group(1)).strip()
        require(row["causal_basis"] == causal_basis,
                f"causal-selection basis differs from case report: {cause_id}")

    final = {
        "unique_first_invalid_decisions": len(audit),
        "unresolved_first_invalid_decisions": 0,
        "unique_primary_obligations": len(audit),
        "unresolved_primary_obligations": 0,
    }
    require(summary["final_causal_selection"] == final, "final causal-selection counts differ")
    require(summary["cohort_accounting"] == {
        "confirmed_reports": 264,
        "repair_linked_reports": 162,
        "primary_cause_reports": 149,
        "duplicate_reports": 13,
        "distinct_repaired_causes": 149,
        "reports_without_traceable_landed_repair": 102,
    }, "causal-selection cohort summary differs")

    historical = summary["historical_candidate_accounting"]
    require(historical["status"] == "not_recorded"
            and historical["initial_multiple_operation_candidates"] is None
            and historical["initial_multiple_obligation_candidates"] is None
            and historical["resolved_from_initial_multiple_candidates"] is None,
            "historical candidate availability differs")

    weaker = [row for row in audit
              if row["regression_test"] == "no" and row["developer_diagnosis"] == "no"]
    weaker_compound = [row for row in weaker if row["secondary_obligations"]]
    require(len(weaker) == 13 and len(weaker_compound) == 6,
            "weaker causal-selection evidence counts differ")
    require(sum(not row["secondary_obligations"] for row in weaker) == 7
            and sum(row["public_reproduction_evidence"] == "yes" for row in weaker) == 6,
            "weaker causal-selection support differs")
    require(summary["stronger_evidence_accounting"] == {
        "landed_patch_and_source_contrast": 149,
        "also_regression_or_developer_diagnosis": 136,
        "without_regression_or_developer_diagnosis": 13,
        "weaker_single_obligation": 7,
        "weaker_compound_obligation": 6,
        "weaker_with_public_reproduction": 6,
    }, "stronger-evidence causal-selection summary differs")

    no_repair = summary["reports_without_traceable_repair"]
    require(no_repair["reports"] == 102
            and no_repair["open"] == sum(not row["closed_at"].strip() for row in report_only) == 70
            and no_repair["closed"] == sum(bool(row["closed_at"].strip()) for row in report_only) == 32,
            "report-only open or closed counts differ")
    require(no_repair["with_observed_decision"] == sum(bool(row["observed_decision"].strip()) for row in report_only) == 102
            and no_repair["with_reported_proposal_or_workaround"]
            == sum(bool(row["reported_repair_or_workaround"].strip()) for row in report_only) == 100,
            "report-only evidence counts differ")
    missing = {
        "fix_patch": sum(not row["fix_patches"].strip() for row in report_only),
        "fix_url": sum(not row["fix_urls"].strip() for row in report_only),
        "semantic_repair": sum(not row["repair"].strip() for row in report_only),
    }
    require(no_repair["missing_fields"] == missing == {
        "fix_patch": 102, "fix_url": 102, "semantic_repair": 102,
    }, "report-only missing-field counts differ")
    require(summary["schema_version"] == "1.0.0" and summary["status"] == "complete",
            "causal-selection summary status differs")
    return final


def verify_human_validation_coverage() -> dict[str, object]:
    coverage_record = read_json("results/human-validation-coverage.json")
    coverage = coverage_record["coverage"]
    outcome = coverage_record["outcome"]
    cases = read_csv("data/final/cases.csv")
    causes = read_csv("results/rq-cause-analysis.csv")
    repair_linked = [row for row in cases if row["analysis_status"].startswith("repair_backed_")]
    duplicates = [row for row in cases if row["analysis_status"] == "repair_backed_duplicate"]
    required_fields = {
        "first invalid decision",
        "relation family",
        "transition family",
        "primary validity obligation",
        "secondary obligations",
        "failure family",
        "repair strategy and repair site",
        "physical effect and observed consequence",
        "provenance origin and consumer endpoints",
        "duplicate links",
    }
    require(coverage_record["schema_version"] == "1.0.0", "human-validation schema differs")
    require(coverage_record["reviewers"] == 2, "human reviewer count differs")
    require(coverage["cause_records"] == coverage["cause_records_reviewed_by_each_human"] == len(causes) == 149,
            "human cause-record coverage differs")
    require(coverage["repair_linked_report_assignments"] == len(repair_linked) == 162,
            "human repair-linked coverage differs")
    require(coverage["duplicate_report_links"] == len(duplicates) == 13,
            "human duplicate-link coverage differs")
    require(set(coverage["fields_checked"]) == required_fields,
            "human validation field coverage differs")
    require(len(coverage["evidence_inspected"]) == 5,
            "human validation evidence coverage differs")
    require(outcome["both_reviewers_confirmed_all_final_records"] is True
            and outcome["issues_resolved_by_evidence_based_correction"] is True
            and outcome["unresolved_final_cause_records"] == 0,
            "human validation outcome differs")
    return {
        "reviewers": coverage_record["reviewers"],
        "cause_records_reviewed_by_each": coverage["cause_records_reviewed_by_each_human"],
        "repair_linked_report_assignments": coverage["repair_linked_report_assignments"],
        "duplicate_report_links": coverage["duplicate_report_links"],
        "unresolved_final_cause_records": outcome["unresolved_final_cause_records"],
    }


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
        "causal_selection": verify_causal_selection(),
        "human_validation": verify_human_validation_coverage(),
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
