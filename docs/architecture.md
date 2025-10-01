# Architecture overview

This repository packages the shared storage model and deterministic
workflow required by the VAAnnotate specification. The implementation
adopts the Python + SQLite variant of the recommended architecture and
lays the groundwork for portable Admin and Reviewer clients.

## Project layout

Every project lives entirely under a single root directory. The
command-line tooling in this repository creates the following structure:

```
[project root]/
├── project.db
├── corpus/
│   └── corpus.db
├── phenotypes/
│   └── <pheno_id>/
│       └── rounds/<round_number>/
│           ├── round_config.json
│           ├── manifest.csv
│           ├── assignments/<reviewer_id>/assignment.db
│           ├── imports/
│           └── reports/
└── admin_tools/
```

This mirrors the prescribed layout and ensures that all mutable state is
captured in SQLite databases with WAL journaling.

## Database boundaries

* `project.db` — authoritative metadata for projects, phenotypes, label
  sets, rounds, reviewers, and assignment provenance.
* `corpus/corpus.db` — immutable clinical text corpus. Canonicalization
  and hashing are handled on ingestion; sampling verifies hashes before
  assigning units.
* `phenotypes/<id>/rounds/<n>/assignments/<reviewer>/assignment.db` —
  per-reviewer writable stores used by the client application. The
  schema matches the specification and includes audit tables and
  rationale capture.

Each database is initialized with WAL mode and `FOREIGN KEY` enforcement
so the files remain safe on shared SMB volumes.

## Deterministic sampling

The `generate-round` command materializes the deterministic sampling
pipeline:

1. Apply patient and note filters with optional regex matching and
   windowing around `date_index`.
2. Build candidate units (single-document or multi-document) while
   tracking strata keys defined by the round config.
3. Shuffle within each stratum using a cryptographic seed derived from
   the global round seed plus the stratum identifier. Overlap units are
   replicated across reviewer pools.
4. Split remaining units evenly via round-robin assignment and then
   shuffle reviewer orderings using a reviewer-specific derived seed.
5. Persist deterministic manifests and assignment databases, logging the
   full configuration in both `project.db` and `round_config.json`.

## Next steps

The current codebase ships the storage and deterministic sampling layer.
The Windows GUI clients (Admin and Reviewer) can be implemented against
these primitives. Client binaries should honour the assignment schema
and sync pattern (local cache + heartbeat) described in the original
specification; placeholders for imports, reports, and adjudication
outputs already exist in the generated folder structure.
