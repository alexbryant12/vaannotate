# UI requirements reference

The storage layer implemented in this repository is designed to back two
Windows desktop applications:

1. **Admin app (WPF/WinUI or Qt)**
   * Dashboard summarizing projects, phenotypes, rounds, completion, and
     import status.
   * Corpus explorer with pre-computed analytics (counts, year
     histograms, notetype distributions).
   * Round configuration wizard covering label set selection, patient and
     note filters (including regex with timeouts), stratification,
     reviewer setup, overlap quotas, and summary validation.
   * Import manager that validates reviewer submissions, enforces hashes,
     and stages imported `assignment.db` files under
     `phenotypes/<id>/rounds/<n>/imports/`.
   * Adjudication suite presenting inter-annotator agreement metrics,
     disagreement browsers, rationale visualization, and final gold data
     export.

2. **Reviewer client (portable exe)**
   * Tri-pane layout with phenotype/round context, progress tracking, and
     sticky rules popover.
   * Middle note viewer supports adjustable font size and span
     highlighting tied to labels. Span offsets are preserved using the
     immutable corpus hashes stored in `assignment.db`.
   * Right pane renders label controls based on the schema stored in the
     assignment database, including gating expressions and NA policies.
   * Keyboard shortcuts (`Ctrl+Enter`, `Ctrl+L`, `Alt+1-9`, `Ctrl+H`)
     accelerate annotation.
   * Autosave on every change plus explicit submission that toggles the
     assignment status to `submitted` and emits a receipt JSON file.
   * Local caching strategy: the client copies `assignment.db` to a local
     cache, syncs edits back to the network share, and maintains
     heartbeat locks.

These requirements are documented here to ensure the eventual desktop
applications implement the mandated behaviours while using the storage
primitives and deterministic manifests defined in this codebase.
