#!/usr/bin/env python3
"""Recompute and validate the published TensorBound artifact summary."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
    baselines = read_json("results/source-baselines.json")
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
    expected_baselines = {
        "iree-org/iree": "c83444ba6ca9c0f54e182d8f17d81f016fa7810c",
        "jax-ml/jax": "c54c660a6095898a054304115afde04c4a0e9621",
        "openxla/xla": "8a4dadd840e3964149b177199fa4750c6778776b",
        "pytorch/pytorch": "6d03f2608c11ac75af45b126b1b75590db3b4d22",
        "triton-lang/triton": "654b1d1499569f6f66ad4db13f6478305874f8f4",
    }
    require(
        baselines == {
            "schema_version": "1.0.0",
            "collection_boundary": boundary,
            "repositories": expected_baselines,
        },
        "common source baselines differ",
    )
    require(
        all(re.fullmatch(r"[0-9a-f]{40}", revision)
            for revision in baselines["repositories"].values()),
        "a common source baseline is not a full commit identifier",
    )
    selection = read_csv("results/selection.csv")
    selection_ids = [row["record_id"] for row in selection]
    require(len(selection_ids) == len(set(selection_ids)) == frame["total_records"],
            "collection identifiers are missing or repeated")
    for repository in repositories:
        prefix = "github-" + repository["repository"].replace("/", "-") + "-"
        require(sum(identifier.startswith(prefix) for identifier in selection_ids) == repository["unique_records"],
                f"collection repository identifiers differ: {repository['repository']}")
    cve_ids = {row["record_id"] for row in selection if row["record_id"].startswith("CVE-")}
    sources = [json.loads(path.read_text(encoding="utf-8"))
               for path in (ROOT / "data/raw/cve-list").glob("*/source.json")]
    require(len(sources) == len(cve_ids) == cve_records,
            "CVE source inventory and selection count differ")
    require({row["source_record_id"] for row in sources} == cve_ids,
            "CVE source identifiers differ from the selection frame")
    require(len({row["source_url"] for row in sources}) == cve_records,
            "CVE source paths are not unique")
    require(all(frame["cve_list"]["revision"] in row["source_url"] for row in sources),
            "CVE source revision differs")
    return {
        "boundary": boundary,
        "github_records": github_records,
        "cve_records": cve_records,
        "source_baselines": len(expected_baselines),
        "total_records": frame["total_records"],
        "cve_unique_source_paths": len(sources),
        "cve_capture_start": min(row["retrieved_at"] for row in sources),
        "cve_capture_end": max(row["retrieved_at"] for row in sources),
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
        require(row["final_decision_status"] == "selected_repair_supported"
                and row["final_primary_obligation_status"] == "resolved_for_coding",
                f"unresolved final repair-centered coding: {cause_id}")
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
        "selected_first_invalid_decisions": len(audit),
        "unresolved_first_invalid_decisions": 0,
        "selected_primary_obligations": len(audit),
        "unresolved_primary_obligations": 0,
    }
    require(summary["final_repair_centered_coding"] == final,
            "final repair-centered coding counts differ")
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


def verify_scope_boundary_examples() -> dict[str, object]:
    examples = read_csv("results/scope-boundary-examples.csv")
    selection = {row["record_id"]: row["frozen_primary_cohort_membership"]
                 for row in read_csv("results/selection.csv")}
    cases = {
        f"github-{row['repository'].replace('/', '-')}-{row['case_id']}"
        for row in read_csv("data/final/cases.csv")
    }
    paired_groups = {"offset-semantics", "numeric-semantics", "compiler-state-ownership"}

    require(len(examples) == 7, "scope-boundary example count differs")
    for row in examples:
        require(selection.get(row["record_id"]) == row["membership"],
                f"scope-boundary membership differs: {row['record_id']}")
        require(bool(row["decisive_evidence"].strip()) and bool(row["selection_reason"].strip()),
                f"scope-boundary explanation is empty: {row['record_id']}")
        evidence_paths = [item.strip() for item in row["evidence_paths"].split(";") if item.strip()]
        require(evidence_paths, f"scope-boundary evidence is empty: {row['record_id']}")
        require(all((ROOT / item).is_file() for item in evidence_paths),
                f"scope-boundary evidence path is missing: {row['record_id']}")

    for group in paired_groups:
        pair = [row for row in examples if row["example_group"] == group]
        require(len(pair) == 2 and {row["role"] for row in pair} == {"included", "excluded"},
                f"scope-boundary pair differs: {group}")
        require({row["membership"] for row in pair} == {"retained", "excluded"}
                and {row["evidence_status"] for row in pair} == {"mechanism_established"},
                f"scope-boundary pair status differs: {group}")
        retained = next(row for row in pair if row["membership"] == "retained")
        require(retained["record_id"] in cases, f"scope-boundary retained case is absent: {group}")

    limited = [row for row in examples if row["example_group"] == "documentation-limitation"]
    require(len(limited) == 1
            and limited[0]["role"] == "evidence_limited"
            and limited[0]["membership"] == "excluded"
            and limited[0]["evidence_status"] == "mechanism_not_established",
            "mechanism-documentation example differs")
    return {
        "paired_construct_boundaries": len(paired_groups),
        "paired_records": 2 * len(paired_groups),
        "evidence_limited_records": len(limited),
    }


def verify_manifest() -> int:
    manifest = read_json("results/artifact-manifest.json")
    release = read_json("results/release-info.json")
    require(sha256(ROOT / "scripts/verify_artifact.py") == release["analysis_script_sha256"],
            "analysis script differs from the release fingerprint")
    require(sha256(ROOT / "results/taxonomy-codebook.json") == release["codebook_sha256"],
            "codebook differs from the release fingerprint")
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


def verify_redactions() -> dict[str, int]:
    prefix = b"https://github.com/notifications/unsubscribe-auth/"
    marker = prefix + b"[REDACTED]"
    marked = 0
    scanned = 0
    for path in (ROOT / "data/raw").rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        content = path.read_bytes()
        marked += content.count(marker)
        require(prefix not in content.replace(marker, b""),
                f"unredacted notification capability link: {path.relative_to(ROOT)}")
    return {"raw_files_scanned": scanned, "redaction_markers": marked}


def jsonl(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def hypergeometric_cdf_numerator(population, positives, sample, observed):
    """Exact count of samples containing at most `observed` positives."""
    require(0 <= positives <= population and 0 <= sample <= population,
            "invalid hypergeometric population or sample")
    lo = max(0, sample - (population - positives))
    hi = min(observed, sample, positives)
    return sum(math.comb(positives, k) * math.comb(population - positives, sample - k)
               for k in range(lo, hi + 1))


def first_true(lo, hi, predicate):
    """First integer satisfying a monotone predicate; hi + 1 if none."""
    while lo <= hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid - 1
        else:
            lo = mid + 1
    return lo


def missed_interval(population, sample, observed, tail_denominator=40):
    """Invert exact tails (1/40 for equal-tailed 95%, 1/20 for one-sided)."""
    require(0 <= observed <= sample <= population, "invalid observed audit count")
    denominator = math.comb(population, sample)
    lower = first_true(0, population, lambda m:
        (denominator - hypergeometric_cdf_numerator(population, m, sample, observed - 1))
        * tail_denominator >= denominator)
    upper = first_true(0, population, lambda m:
        hypergeometric_cdf_numerator(population, m, sample, observed)
        * tail_denominator < denominator) - 1
    return lower, upper


def verify_phase1_human_totals():
    """Arithmetic only: the release has no individual human decisions."""
    data = read_json("results/statistics/phase1-human-review.json")
    n = sum(row["reviewed"] for row in data["samples"])
    m = sum(row["binding_failures"] for row in data["samples"])
    agreements = sum(row["pre_discussion_agreements"] for row in data["samples"])
    population = data["initial_exclusions"]
    require(len(data["samples"]) == 3 and all(row["reviewed"] == 200 for row in data["samples"]),
            "reported sample sizes differ")
    for row in data["samples"]:
        require(0 <= row["binding_failures"] <= row["reviewed"]
                and 0 <= row["pre_discussion_agreements"] <= row["reviewed"],
                "human audit totals are out of range")
    previous = 780
    for row in data["review_steps"]:
        require(row["input"] == previous, "selection step inputs do not join")
        require(0 <= row["restored"] <= row["proposed_removals"] <= row["input"],
                "selection removal counts are out of range")
        require(row["final_removals"] == row["proposed_removals"] - row["restored"]
                and row["output"] == row["input"] - row["final_removals"],
                "selection step does not balance")
        previous = row["output"]
    final = data["final_set"]["reports"]
    require(previous == final == len(read_csv("data/final/cases.csv")) == 264,
            "final human-review total differs from retained membership")
    require(data["final_set"]["confirmed_by_both_before_discussion"] == final,
            "final-set confirmation count differs")
    parts = data["detailed_review_removals"]
    require(sum(row["proposed"] for row in parts) == data["review_steps"][0]["proposed_removals"]
            and sum(row["restored"] for row in parts) == data["review_steps"][0]["restored"],
            "appendix detailed-review groups differ")
    denominator = math.comb(population, n)
    missed = population * m / n
    lower, upper = missed_interval(population, n, m)
    one_sided_upper = missed_interval(population, n, m, 20)[1]
    null_missed = math.ceil(population * 0.05)
    threshold = first_true(0, n, lambda k:
        hypergeometric_cdf_numerator(population, null_missed, n, k) * 20 > denominator) - 1
    power = hypergeometric_cdf_numerator(population, round(population * 0.03), n, threshold) / denominator
    observed = {
        "sample_records": n,
        "sample_binding_failures": m,
        "sample_agreements": agreements,
        "estimated_missed": round(missed, 2),
        "missed_equal_tailed_95_interval": [lower, upper],
        "correctly_excluded_percent": round(100 * (1 - m / n), 2),
        "correctly_excluded_95_interval_percent": [round(100 * (1 - v / population), 2)
                                                   for v in (upper, lower)],
        "correctly_excluded_one_sided_95_lower_percent": round(100 * (1 - one_sided_upper / population), 2),
        "acceptance_threshold": threshold,
        "power_at_97_percent_correct": round(power, 2),
        "initial_recall_percent": round(100 * final / (final + missed), 2),
        "initial_recall_95_interval_percent": [round(100 * final / (final + v), 2) for v in (upper, lower)],
        "removal_proposals": sum(row["proposed_removals"] for row in data["review_steps"]),
        "restoration_decisions": sum(row["restored"] for row in data["review_steps"]),
        "final_removals": sum(row["final_removals"] for row in data["review_steps"]),
    }
    require(observed == data["expected"], "Phase 1 arithmetic differs from published values")
    return {"verification_scope": data["verification_scope"],
            "missing_records": data["missing_records"], **observed}


def historical_primary_obligations():
    """Keep archived secondary-mask calculations tied to their original primary labels."""
    return {row["cause_id"]: row["old_value"]
            for row in read_json("results/coding-corrections.json")["corrections"]
            if row["field"] == "obligation_family"}


def verify_archived_assignments():
    rows = jsonl("results/statistics/archived-taxonomy-assignments.jsonl")
    causes = {row["cause_id"]: row for row in read_csv("results/rq-cause-analysis.csv")}
    require(len(rows) == len(causes) == 149 and {row["cause_id"] for row in rows} == set(causes),
            "archived assignment coverage differs")
    published = read_json("results/statistics/taxonomy-agreement-summary.json")
    primary_results = {}
    for field, expected in published["fields"].items():
        a = Counter(row["pass_a"][field] for row in rows)
        b = Counter(row["pass_b"][field] for row in rows)
        count = sum(row["pass_a"][field] == row["pass_b"][field] for row in rows)
        chance = sum(a[label] * b[label] for label in set(a) | set(b)) / len(rows) ** 2
        kappa = (count / len(rows) - chance) / (1 - chance)
        require(count == expected["agreements"] and abs(kappa - expected["cohens_kappa"]) < 1e-12,
                f"archived assignment agreement differs: {field}")
        primary_results[field] = {"agreements": count, "kappa": kappa}
    secondary = read_json("results/statistics/secondary-obligation-summary.json")
    historical_primary = historical_primary_obligations()
    vocabulary = set(secondary["label_vocabulary"])
    counts = Counter()
    retained_labels = compound = exact_sets = 0
    for row in rows:
        cause = causes[row["cause_id"]]
        primary = historical_primary.get(row["cause_id"], cause["obligation_family"])
        # The source assignment arrays are unchanged; the mask is a derived view.
        a = set(row["pass_a"]["secondary_obligation_families"]) - {primary}
        b = set(row["pass_b"]["secondary_obligation_families"]) - {primary}
        require(a <= vocabulary and b <= vocabulary, "unknown secondary obligation")
        counts.update(("present" if label in a else "absent") + "|" +
                      ("present" if label in b else "absent") for label in vocabulary)
        jointly_assigned = a & b
        retained = {label.strip() for label in cause["secondary_obligation_families"].split(";") if label.strip()}
        require(jointly_assigned == retained, f"retained secondary labels differ: {row['cause_id']}")
        retained_labels += len(jointly_assigned)
        compound += bool(jointly_assigned)
        exact_sets += a == b
    require(dict(counts) == secondary["pair_counts"], "secondary binary comparison differs")
    require(exact_sets == secondary["exact_set_agreements"], "secondary set agreement differs")
    positive = 2 * counts["present|present"]
    disagreements = counts["present|absent"] + counts["absent|present"]
    return {"actor_type": "unknown", "primary_agreement": primary_results,
            "secondary_pairs": sum(counts.values()), "secondary_pair_counts": dict(counts),
            "automatic_primary_family_omissions": len(rows),
            "secondary_overall_agreement_percent": round(100 * (1 - disagreements / sum(counts.values())), 1),
            "secondary_positive_agreement_percent": round(100 * positive / (positive + disagreements), 1),
            "retained_secondary_labels": retained_labels, "causes_with_secondary_labels": compound}


def verify_source_corrections():
    corrections = read_json("results/source-location-corrections.json")["corrections"]
    alignment = {row["cause_id"]: row for row in read_csv("results/historical-alignment.csv")}
    paths = {row["cause_id"]: row for row in jsonl("results/rq-cause-paths.jsonl")}
    for row in corrections:
        cause = alignment[row["cause_id"]]
        require(cause["first_invalid_decision_site"] == row["new_site"]
                and cause["pre_fix_revision"] == row["pre_fix_revision"],
                f"source-location correction not propagated: {row['cause_id']}")
        require(paths[row["cause_id"]]["pre_fix_decision_site"] == row["new_site"]
                and paths[row["cause_id"]]["path_kind"] == "required_or_repaired_propagation",
                "pre-fix decision and repaired propagation path are not distinguished")
        for reference in row["evidence"]:
            require((ROOT / reference.split("#", 1)[0]).is_file(), f"missing source evidence: {reference}")
        if "pre_fix_patch_text" in row:
            patches = [ROOT / ref for ref in row["evidence"] if ref.endswith(".patch")]
            require(any(row["pre_fix_patch_text"] in path.read_text() for path in patches),
                    "corrected operation is not in the pre-fix patch context")
    return {"corrected_records": len(corrections), "independent_human_validation": False}


def verify_coding_corrections():
    log = read_json("results/coding-corrections.json")
    require(log["historical_human_validation"] is False,
            "revision-time coding check is mislabeled as historical human review")
    causes = {row["cause_id"]: row for row in read_csv("results/rq-cause-analysis.csv")}
    alignment = {row["cause_id"]: row for row in read_csv("results/historical-alignment.csv")}
    audit = {row["cause_id"]: row for row in read_csv("results/causal-selection-audit.csv")}
    paths = {row["cause_id"]: row for row in jsonl("results/rq-cause-paths.jsonl")}
    seen = set()
    for correction in log["corrections"]:
        cause_id = correction["cause_id"]
        require(cause_id not in seen, "duplicate coding correction")
        seen.add(cause_id)
        row = causes[cause_id]
        require(row[correction["field"]] == correction["new_value"],
                f"coding correction not propagated: {cause_id}")
        for field, change in correction.get("additional_label_changes", {}).items():
            require(row[field] == change["new_value"],
                    f"additional coding correction not propagated: {cause_id}/{field}")
        require(all(row[field] == label for field, label in correction["preserved_labels"].items()),
                f"unrelated labels changed: {cause_id}")
        site = correction["first_invalid_decision_site"]
        require(alignment[cause_id]["first_invalid_decision_site"] == site
                and audit[cause_id]["first_invalid_decision_site"] == site
                and paths[cause_id]["pre_fix_decision_site"] == site,
                f"coding correction source anchors differ: {cause_id}")
        require(alignment[cause_id]["pre_fix_revision"] == correction["pre_fix_revision"],
                f"coding correction revision differs: {cause_id}")
        for path in correction["evidence"]:
            require((ROOT / path).is_file(), f"coding correction evidence missing: {path}")
    require(seen == {"cause-iree-9033-buffer-retains-allocator",
                     "cause-iree-9392-callback-cleanup-identity"},
            "unexpected coding-correction scope")
    patch = (ROOT / "data/final/iree/9033/fixes/iree-9033-fix.patch").read_text()
    require("+    iree_hal_allocator_retain(device_allocator);" in patch
            and "+      iree_hal_allocator_release(device_allocator_to_release);" in patch
            and "+  def testGcShutdownFiasco(self):" in patch,
            "allocator retain/release or supporting regression is absent")
    return {"corrected_records": len(seen), "new_human_review": False}


def verify_results():
    causes = read_csv("results/rq-cause-analysis.csv")
    ids = {r["cause_id"] for r in causes}
    summary = read_json("results/rq-analysis-summary.json")
    agreement = read_json("results/statistics/taxonomy-agreement-summary.json")
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
    marginals = agreement["failure_mode_marginals"]
    require(all(sum(counts.values()) == 149 for counts in marginals.values()),
            "failure-mode marginal denominator differs")
    # Preserve historical reconciliation; apply documented revision-time corrections.
    current_failures = Counter(marginals["reconciled"])
    for correction in read_json("results/coding-corrections.json")["corrections"]:
        if correction["field"] == "failure_mode":
            current_failures[correction["old_value"]] -= 1
            current_failures[correction["new_value"]] += 1
    require(dict(current_failures) == dict(Counter(r["failure_mode"] for r in causes)),
            "corrected failure-mode marginals differ")
    require(marginals["pass_a"] == {
        "incorrect mapping or translation": 34,
        "lost or omitted relation": 29,
        "missing eligibility check": 31,
        "missing ordering or enforcement": 7,
        "omitted operation": 3,
        "premature invalidation or stale state": 1,
        "premature or unordered transition": 13,
        "unsound assumption or inference": 31,
    } and marginals["pass_b"] == {
        "incorrect mapping or translation": 41,
        "lost or omitted relation": 29,
        "missing eligibility check": 37,
        "missing ordering or enforcement": 9,
        "omitted operation": 4,
        "premature invalidation or stale state": 3,
        "premature or unordered transition": 10,
        "unsound assumption or inference": 16,
    }, "archived failure-mode marginals differ")
    for name, x, y in (("failure_by_repair", "failure_mode", "repair_strategy"),
                       ("obligation_by_repair", "obligation_family", "repair_strategy"),
                       ("relation_by_obligation", "relation_family", "obligation_family"),
                       ("relation_by_transition", "relation_family", "transition_family")):
        counts = {value: dict(Counter(r[y] for r in causes if r[x] == value)) for value in {r[x] for r in causes}}
        require(counts == summary["cross_tabs"][name], f"cross-tab differs: {name}")
    provenance = read_csv("results/rq2-provenance-analysis.csv")
    ps = read_json("results/rq2-provenance-summary.json")
    require({r["cause_id"] for r in provenance} == ids, "RQ2 cause coverage differs")
    for field in ("fact_origin_layer", "fact_consumer_layer", "endpoint_category_relation", "semantic_boundary_count"):
        require(dict(Counter(r[field] for r in provenance)) == ps[field], f"RQ2 differs: {field}")
    impacts = read_csv("results/rq5-report-impacts.csv")
    consequence = Counter(r["primary_observed_consequence"] for r in impacts)
    impact_summary = read_json("results/rq5-impact-summary.json")
    require(dict(consequence) == impact_summary["primary_observed_consequence"],
            "report consequence counts differ")
    published_by_repository = impact_summary["primary_observed_consequence_by_repository"]
    repositories = {r["repository"] for r in impacts}
    consequence_families = set(impact_summary["primary_observed_consequence"])
    require(set(published_by_repository) == repositories,
            "repository consequence frames differ")
    for repository, published in published_by_repository.items():
        repository_rows = [r for r in impacts if r["repository"] == repository]
        observed = Counter(r["primary_observed_consequence"] for r in repository_rows)
        require(set(published) == consequence_families | {"reports"},
                f"repository consequence families differ: {repository}")
        expected = {family: published[family] for family in consequence_families}
        require(published["reports"] == len(repository_rows) and
                expected == {family: observed.get(family, 0) for family in consequence_families},
                f"repository consequence counts differ: {repository}")
    pairs = Counter()
    historical_primary = historical_primary_obligations()
    secondary_causes = secondary_labels = 0
    for row in causes:
        labels = [value.strip() for value in row["secondary_obligation_families"].split(";") if value.strip()]
        secondary_causes += bool(labels)
        secondary_labels += len(labels)
        primary = historical_primary.get(row["cause_id"], row["obligation_family"])
        pairs.update((primary, label) for label in labels)
    secondary = summary["secondary_obligation_analysis"]
    require(secondary_causes == secondary["causes_with_consensus_secondary"], "secondary cause count differs")
    require(secondary_labels == secondary["consensus_secondary_labels"], "secondary label count differs")
    require(dict(pairs) == {(r["primary"], r["secondary"]): r["count"] for r in secondary["primary_secondary_pairs"]},
            "primary-secondary pairs differ")
    attempts = read_csv("results/rq6-diagnostic-attempts.csv")
    ds = read_json("results/rq6-summary.json")
    require(ds["primary_obligation"] == summary["obligation_family"]
            and ds["repair_strategy"] == summary["repair_strategy"]
            and ds["obligation_by_repair_strategy"] == summary["cross_tabs"]["obligation_by_repair"],
            "RQ6 primary-obligation or repair summary differs")
    for field, counts in ds["diagnostic_outcomes"].items():
        require(dict(Counter(r[field] for r in attempts)) == counts, f"diagnostic counts differ: {field}")
    return {"families": {f: len(summary[f]) for f in dimensions},
            "different_endpoint_categories": sum(
                r["endpoint_category_relation"] == "different" for r in provenance),
            "secondary_causes": secondary_causes, "secondary_labels": secondary_labels,
            "diagnostic_attempts": len(attempts)}


def verify_sensitivity():
    selection = read_csv("results/selection.csv")
    require(len(selection) == len({r["record_id"] for r in selection}) == 7509, "selection coverage differs")
    require(dict(Counter(r["frozen_primary_cohort_membership"] for r in selection)) == {"retained": 264, "excluded": 7245},
            "selection membership differs")
    retained = {r["record_id"] for r in selection if r["frozen_primary_cohort_membership"] == "retained"}
    cases = read_csv("data/final/cases.csv")
    require(retained == {f"github-{r['repository'].replace('/', '-')}-{r['case_id']}" for r in cases},
            "selection differs from the cohort")
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
    boundary = read_json("results/statistics/construct-boundary-sensitivity.json")
    require(boundary["boundary_family"] == "value--operation/identity",
            "construct-boundary family differs")
    restricted = [r for r in causes if r["relation_family"] != boundary["boundary_family"]]
    full_failures = dict(Counter(r["failure_mode"] for r in causes))
    restricted_failures = dict(Counter(r["failure_mode"] for r in restricted))
    require(boundary["full_causes"] == len(causes)
            and boundary["boundary_family_causes"] == len(causes) - len(restricted)
            and boundary["restricted_causes"] == len(restricted)
            and boundary["failure_mode"]["full"] == full_failures
            and boundary["failure_mode"]["restricted"] == restricted_failures,
            "construct-boundary sensitivity differs")
    leading = boundary["leading_three_restricted"]
    actual_leading = sorted(restricted_failures,
                            key=lambda family: (-restricted_failures[family], family))[:3]
    require(leading["families"] == actual_leading == [
        "missing eligibility check",
        "lost or omitted relation",
        "incorrect mapping or translation",
    ], "construct-boundary leading families differ")
    leading_count = sum(restricted_failures[family] for family in leading["families"])
    require(leading["causes"] == leading_count
            and abs(leading["percent"] - 100 * leading_count / len(restricted)) < 1e-9,
            "construct-boundary leading-family sensitivity differs")
    patches = jsonl("results/repair-patch-index.jsonl")
    require(len(patches) == 167 and {p["cause_id"] for p in patches} == set(support), "patch coverage differs")
    for patch in patches:
        path = ROOT / patch["path"]
        require(sha256(path) == patch["sha256"], f"patch hash differs: {path}")
        files = sorted({line.split(" b/", 1)[1] for line in path.read_text().splitlines()
                        if line.startswith("diff --git ") and " b/" in line})
        require(files == patch["changed_files"], f"changed-file list differs: {path}")
    return {"stronger_evidence_causes": len(stronger),
            "construct_boundary_restricted_causes": len(restricted),
            "archived_patches": len(patches)}


def verify_headline_results():
    causes = read_csv("results/rq-cause-analysis.csv")
    reports = read_csv("results/rq5-report-impacts.csv")
    corresponding = {"missing eligibility check": "strengthen eligibility check",
                     "incorrect mapping or translation": "correct mapping or representation",
                     "lost or omitted relation": "preserve or propagate relation"}
    leading = [row for row in causes if row["failure_mode"] in corresponding]
    matching = [row for row in leading if row["repair_strategy"] == corresponding[row["failure_mode"]]]
    flag_fields = ("memory_access_signal", "concurrency_signal", "availability_signal")
    flagged = [row for row in reports if any(row[field] == "yes" for field in flag_fields)]
    flags = {field: sum(row[field] == "yes" for row in reports) for field in flag_fields}
    require(len(leading) == 103 and len(matching) == 94, "headline mechanism/repair counts differ")
    require(list(flags.values()) == [23, 14, 24] and len(flagged) == 60,
            "report flag counts or union differ")
    other_reports = [row for row in reports if row["repository"] != "pytorch/pytorch"]
    other_causes = [row for row in causes if not row["primary_report_id"].startswith("github-pytorch-pytorch-")]
    require(len(other_reports) == 193 and len(other_causes) == 112,
            "PyTorch-repository exclusion membership differs")
    incorrect = sum(row["primary_observed_consequence"] == "incorrect result or state" for row in other_reports)
    remaining_leading = sum(row["failure_mode"] in corresponding for row in other_causes)
    require(incorrect == 103 and remaining_leading == 76, "repository sensitivity counts differ")
    attempts = read_csv("results/rq6-diagnostic-attempts.csv")
    require(len(attempts) == 4 and len({row["cause_id"] for row in attempts}) == 3,
            "diagnostic attempt/cause grouping differs")
    for row in attempts:
        if row["failing_path_coverage"] == "no":
            require(all(row[field] == "not_evaluated" for field in
                        ("effect_detection", "effect_localization", "cause_localization", "cause_explanation")),
                    "uncovered diagnostic path has an evaluated outcome")
    return {"leading_failure_causes": len(leading), "corresponding_repairs": len(matching),
            "flag_counts": flags, "flagged_report_union": len(flagged),
            "without_pytorch_repository": {"reports": len(other_reports), "causes": len(other_causes),
                "incorrect_results": incorrect, "leading_failure_causes": remaining_leading}}


def result_index():
    """Return memberships behind numerical displays; never invent human IDs."""
    cases = read_csv("data/final/cases.csv")
    causes = read_csv("results/rq-cause-analysis.csv")
    reports = read_csv("results/rq5-report-impacts.csv")
    def group(rows, fields, identifier):
        grouped = {}
        for row in rows:
            key = " | ".join(row[field] for field in fields)
            grouped.setdefault(key, []).append(identifier(row))
        return {key: {"count": len(ids), "ids": sorted(ids)} for key, ids in sorted(grouped.items())}
    cause_id = lambda row: row["cause_id"]
    report_id = lambda row: "github-" + row["repository"].replace("/", "-") + "-" + row["case_id"]
    output = {
        "collection": {"input": "results/selection.csv", "ids": [row["record_id"] for row in read_csv("results/selection.csv")]},
        "phase1_human_tables": {"input": "results/statistics/phase1-human-review.json",
                                "unit": "reported review decisions", "record_ids": None,
                                "status": "aggregate_only; individual human records unavailable"},
        "report_cause_mapping": group([row for row in cases if row["cause_id"]], ["cause_id"], report_id),
        "relation_by_ecosystem": group(causes, ["relation_family", "stack"], cause_id),
        "provenance_endpoints": group(read_csv("results/rq2-provenance-analysis.csv"),
                                      ["fact_origin_layer", "fact_consumer_layer"], cause_id),
        "obligation_by_relation": group(causes, ["obligation_family", "relation_family"], cause_id),
        "relation_transition_matrix": group(causes, ["relation_family", "transition_family"], cause_id),
        "failure_repair_matrix": group(causes, ["failure_mode", "repair_strategy"], cause_id),
        "obligation_repair_matrix": group(causes, ["obligation_family", "repair_strategy"], cause_id),
        "consequences": group(reports, ["primary_observed_consequence"], report_id),
        "diagnostics": read_csv("results/rq6-diagnostic-attempts.csv"),
        "label_comparison": {"input": "results/statistics/archived-taxonomy-assignments.jsonl",
                             "mask": "remove final primary family from both secondary-label sets"},
        "figure_overview": {"reports": len(cases), "repair_linked": sum(bool(row["cause_id"]) for row in cases),
                            "causes": len(causes), "check": "source-drawn workflow, not an inferred graph"},
        "figure_initialization": {"case": "data/final/iree/12161/report.md",
                                  "check": "manually compare the pre-fix publication and repaired write/check/publication"}
    }
    support = {row["cause_id"]: row for row in read_csv("results/historical-alignment.csv")}
    output["sensitivity_memberships"] = {
        "test_or_diagnosis": sorted(row["cause_id"] for row in causes
                                   if any(support[row["cause_id"]][field] == "yes" for field in ("regression_test", "developer_diagnosis"))),
        "without_pytorch_reports": sorted(report_id(row) for row in reports if row["repository"] != "pytorch/pytorch"),
        "without_pytorch_causes": sorted(row["cause_id"] for row in causes if not row["primary_report_id"].startswith("github-pytorch-pytorch-")),
        "without_operation_identity": sorted(row["cause_id"] for row in causes if row["relation_family"] != "value--operation/identity")
    }
    return output


def verify() -> dict[str, object]:
    return {
        "status": "ok",
        "collection": verify_collection_frame(),
        "cohort": verify_cohort(),
        "analysis_rows": verify_analysis_outputs(),
        "causal_selection": verify_causal_selection(),
        "human_validation": verify_human_validation_coverage(),
        "scope_boundaries": verify_scope_boundary_examples(),
        "data_manifest_files": verify_manifest(),
        "redactions": verify_redactions(),
        "results": verify_results(),
        "sensitivity": verify_sensitivity(),
        "phase1_reported_human_totals": verify_phase1_human_totals(),
        "archived_taxonomy_assignments": verify_archived_assignments(),
        "source_location_corrections": verify_source_corrections(),
        "coding_corrections": verify_coding_corrections(),
        "headlines": verify_headline_results(),
    }


def main() -> int:
    try:
        summary = verify()
        if sys.argv[1:] == ["--result-index"]:
            summary = result_index()
        elif sys.argv[1:]:
            raise ArtifactError("usage: verify_artifact.py [--result-index]")
    except (ArtifactError, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"artifact verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
