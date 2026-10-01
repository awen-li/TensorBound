# Confirmed TensorBound cohort

This directory contains the 264 reports retained after screening and review.
Cases are grouped by repository and report number. Each case directory contains
a human-readable `report.md` with the archived report evidence and the study's
case-level reconstruction.

The cohort has two repair-coverage states:

- 162 reports link to a traceable landed repair. They represent 149 distinct
  repaired causes after causal deduplication, and their case directories also
  contain the archived patch files under `fixes/`.
- 102 reports establish an in-scope binding failure and its observed effect,
  but the archived public artifacts do not identify a traceable landed repair.
  Their case files preserve the observed decision, effect, consequence, and
  any reported proposal or workaround. They contain no `fixes/` directory.

[`cases.csv`](cases.csv) is the cohort index. Blank repair and root-cause fields
for the 102 report-only cases mean that the corresponding relationship was not
established from a landed repair; they do not mean that the report is outside
the confirmed cohort.
