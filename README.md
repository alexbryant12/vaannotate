# VAAnnotate

VAAnnotate is a self-contained toolkit for building offline clinical NLP
annotation projects that follow the requirements laid out in the VA
multi-annotator specification. The repository provides:

* A deterministic project layout rooted in a single shared folder with
  isolated SQLite databases for each component.
* Command-line tools for project initialization, corpus loading,
  phenotype and label set registration, and deterministic round
  generation.
* Schema definitions for project metadata, round manifests, reviewer
  assignments, and clinical corpora that match the detailed specification.

The initial implementation focuses on the administrative data model and
sampling pipeline so that downstream client applications can operate on
consistent storage primitives. A graphical Admin or Reviewer client can
be layered on top of the data model without changing persisted formats.

## Getting started

1. **Initialize a project folder**

   ```bash
   vaannotate /path/to/project init <project_id> "Project Name" <created_by>
   ```

   This creates the canonical folder structure, `project.db`, and
   `corpus/corpus.db`.

2. **Load a corpus**

   Prepare two CSV files: `patients.csv` and `documents.csv`. The schema
   must match the specification (see `docs/corpus_schema.md`). Then run:

   ```bash
   vaannotate /path/to/project load-corpus patients.csv documents.csv
   ```

3. **Create a phenotype and label set**

   ```bash
   vaannotate /path/to/project create-phenotype <project_id> <pheno_id> "Name" single_doc "Description"
   vaannotate /path/to/project register-labelset labelset.json
   ```

   The `labelset.json` file should match the examples in
   `docs/labelsets.md`.

4. **Generate a round**

   Create a round configuration JSON file that mirrors the full
   specification (filters, stratification, reviewers, overlap). Generate
   the round via:

   ```bash
   vaannotate /path/to/project generate-round <pheno_id> 1 config.json <created_by>
   ```

   The command writes `round_config.json`, `manifest.csv`, and a
   reviewer-specific `assignment.db` under the phenotype folder.

## Documentation

* [`docs/architecture.md`](docs/architecture.md) — Summarizes how the
  repository maps to the high-level design requirements.
* [`docs/corpus_schema.md`](docs/corpus_schema.md) — Reference for the
  immutable corpus tables and indexes.
* [`docs/project_workflow.md`](docs/project_workflow.md) — Step-by-step
  walkthrough of the Admin workflow and generated artifacts.
* [`docs/labelsets.md`](docs/labelsets.md) — Guidance and examples for
  authoring label set JSON files.

These documents also note the follow-on work required to ship the WPF
clients specified in the requirements.
