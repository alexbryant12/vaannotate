# VAAnnotate

VAAnnotate is a file-based toolkit for managing multi-annotator NLP phenotyping projects in fully disconnected environments. The project provides a command-line admin workflow and a lightweight console annotator client that store all state inside a single project folder, mirroring the architecture described in the specification.

## Features

* SQLite-backed project, corpus, assignment, and aggregate databases stored inside a project directory tree.
* Corpus ingestion utilities that canonicalize clinical text and capture content hashes for immutability checks.
* Round generation that applies deterministic filtering, stratification, overlap, and reviewer assignment logic while writing manifests and reviewer-specific SQLite bundles.
* Import and aggregation helpers for combining reviewer submissions into round-level databases and exports.
* Agreement metrics including percent agreement, Cohen's κ (two reviewers), and Fleiss' κ (three or more reviewers) surfaced through the admin CLI.
* Console-based annotator client that updates assignment databases, records completion timestamps, and writes audit events.

## Installation

Create a Python virtual environment (Python 3.11+) and install the package dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Admin workflow

1. **Initialize a project**
   ```bash
   python -m vaannotate.admin_cli init ./demo_project --project-id demo --name "Demo Project"
   ```

2. **Import a staged corpus** (expects CSV files with the schema documented in the specification):
   ```bash
   python -m vaannotate.admin_cli import-corpus ./demo_project --patients-csv patients.csv --documents-csv documents.csv
   ```

3. **Register reviewers and create phenotypes/label sets** using JSON descriptors:
   ```bash
   python -m vaannotate.admin_cli addreviewer ./demo_project --reviewer-id r1 --name "Reviewer 1"
   python -m vaannotate.admin_cli addphenotype ./demo_project --pheno-id ph1 --project-id demo --name "Phenotype" --level single_doc
   python -m vaannotate.admin_cli createlabelset ./demo_project --config-json labelset.json
   ```

4. **Generate a round** from a frozen configuration JSON:
   ```bash
   python -m vaannotate.admin_cli generateround ./demo_project --pheno-id ph1 --config-json round_config.json
   ```

5. **Import assignment submissions and build aggregates** once annotators return their SQLite files:
   ```bash
   python -m vaannotate.admin_cli importassignment ./demo_project --pheno-id ph1 --round-number 1 --reviewer-id r1
   python -m vaannotate.admin_cli aggregate ./demo_project --pheno-id ph1 --round-number 1
   ```

6. **Compute inter-annotator agreement** and export merged annotations:
   ```bash
   python -m vaannotate.admin_cli iaa ./demo_project --pheno-id ph1 --round-number 1 --label-id has_phenotype
   python -m vaannotate.admin_cli export-annotations ./demo_project --pheno-id ph1 --round-number 1
   ```

## Annotator workflow

Annotators open the console client inside their assignment directory:

```bash
python -m vaannotate.annotator_cli open-assignment ./demo_project/phenotypes/ph1/rounds/round_1/assignments/r1
```

The client walks through each unit in randomized order, prompting for label values, marking completion timestamps, and writing event logs for auditing. Submissions are ready once all units are complete.

## Project layout

Generated projects follow the directory structure described in the specification, including per-round manifests, reviewer folders, aggregate databases, and export directories. All SQLite databases are created in WAL mode to support concurrent read access on shared drives.

## Tests

At present the repository does not ship automated tests. Future enhancements should include deterministic sampling unit tests, agreement metric validation against published examples, and integration tests that exercise the end-to-end round workflow.
