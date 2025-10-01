# VAAnnotate

VAAnnotate is a file-based toolkit for managing multi-annotator NLP phenotyping projects in fully disconnected environments. The project now ships with a PySide6 administrative console and an annotator client that mirror the architecture described in the specification while keeping all state inside a single project folder.

## Features

* SQLite-backed project, corpus, assignment, and aggregate databases stored inside a project directory tree.
* Corpus ingestion utilities that canonicalize clinical text and capture content hashes for immutability checks.
* Round generation that applies deterministic filtering, stratification, overlap, and reviewer assignment logic while writing manifests and reviewer-specific SQLite bundles.
* Import and aggregation helpers for combining reviewer submissions into round-level databases and exports.
* Agreement metrics including percent agreement, Cohen's κ (two reviewers), and Fleiss' κ (three or more reviewers) surfaced through the Admin GUI.
* PySide6-based annotator client that provides tri-pane navigation, autosave, rationale capture, and completion tracking.

## Installation

Create a Python virtual environment (Python 3.11+) and install the package dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Admin workflow

1. **Launch the Admin application**
   ```bash
   python -m vaannotate.AdminApp.main
   ```
   Choose or create a project directory. The Admin dashboard exposes project management, phenotype creation, round generation, and agreement metric dashboards.

2. **Stage a corpus** by placing a `corpus.db` file in the project's `/corpus` folder or by using future ingestion tooling. The Admin UI automatically creates the SQLite schema on first open.

3. **Create phenotypes and label sets** from the Phenotypes tab. Label sets are versioned; newly generated rounds can auto-provision a starter boolean label if a label set is absent.

4. **Generate rounds** from the Rounds tab. Configure reviewers, seeds, overlap, and sample size, then generate manifests and reviewer assignments. The Admin app writes `manifest.csv`, reviewer folders with `assignment.db`, and accompanying `label_schema.json` descriptors.

5. **Review agreement metrics** in the IAA tab. The dashboard computes percent agreement, Cohen's κ, and Fleiss' κ using embedded deterministic formulas and prepares the reports directory.

The Admin app can be packaged with PyInstaller for disconnected VA workstations; all state lives inside the selected project directory.

## Annotator workflow

Annotators open the client inside their assignment directory or pass the path on the command line:

```bash
python -m vaannotate.ClientApp.main ./project_root/phenotypes/<pheno_id>/rounds/<round_n>/assignments/<reviewer_id>
```

The client loads the assignment SQLite bundle locally, presents unit navigation, renders note text, and displays the label form. Changes are autosaved with audit log events, highlights can be captured directly from the note viewer, and completion is tracked per unit. Submitting creates a `submitted.json` receipt file that the Admin app imports.

## Project layout

Generated projects follow the directory structure described in the specification, including per-round manifests, reviewer folders, aggregate databases, and export directories. All SQLite databases are created in WAL mode to support concurrent read access on shared drives.

## Tests

At present the repository does not ship automated tests. Future enhancements should include deterministic sampling unit tests, agreement metric validation against published examples, and integration tests that exercise the end-to-end round workflow.
