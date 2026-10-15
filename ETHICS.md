# Ethics and Responsible Use

TensorBound studies public technical records from open-source compiler and
runtime projects. The study did not collect private communications, interact
with users, or create new failures in deployed systems. Archived records
support audits of the reported classifications and repair-centered accounts.

The artifact preserves public repository and report identifiers because they
are needed to connect claims to source evidence and landed repairs. It focuses
on software mechanisms, not on evaluating individual contributors. Users of
the artifact should follow the source projects' terms and avoid using the data
to rank or profile people.

Public issue exports can contain capability-bearing notification links. The
release replaces those links with a fixed redaction marker while preserving
the surrounding technical discussion. The verifier scans the package for the
known capability-link pattern.

Some reports describe crashes, memory errors, or concurrency faults. The paper
does not treat those symptoms alone as demonstrated security vulnerabilities;
security impact requires an explicit protected asset, attacker capability, and
deployment context.
