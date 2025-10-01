# Project workflow

The CLI commands in this repository reproduce the administrative
workflow required to stage rounds, distribute assignments, and maintain
complete provenance.

## 1. Project bootstrap

1. `vaannotate <root> init <project_id> <name> <created_by>`
2. Optionally place Admin and Reviewer binaries under
   `<root>/admin_tools/` and `<root>/phenotypes/.../assignments/<id>/`.

## 2. Corpus ingestion

1. Prepare `patients.csv` and `documents.csv` using the fields described
   in `docs/corpus_schema.md`.
2. Run `vaannotate <root> load-corpus patients.csv documents.csv`.
3. The loader canonicalizes text, computes SHA256 hashes, and enforces
   WAL journaling for robustness on shared drives.

## 3. Phenotype and label sets

1. `vaannotate <root> create-phenotype <project_id> <pheno_id> <name> <level> <description>`
2. Author a label set JSON payload containing label definitions and
   option lists. Example structure:

   ```json
   {
     "labelset_id": "ls_diabetes_v1",
     "pheno_id": "ph_diabetes",
     "version": 1,
     "created_by": "admin",
     "labels": [
       {
         "label_id": "has_phenotype",
         "name": "Has Phenotype",
         "type": "boolean",
         "required": true,
         "options": [
           {"option_id": "opt_yes", "value": "Yes", "display": "Yes"},
           {"option_id": "opt_no", "value": "No", "display": "No"}
         ]
       }
     ]
   }
   ```

3. Register the label set via `vaannotate <root> register-labelset labelset.json`.

## 4. Round creation

1. Author a `round_config.json` mirroring the specification (filters,
   stratification, reviewers, overlap, RNG seed, labelset reference).
2. Run `vaannotate <root> generate-round <pheno_id> <round_number> round_config.json <created_by>`.
3. Outputs created under
   `phenotypes/<pheno_id>/rounds/<round_number>/` include:
   * `round_config.json` — Frozen configuration.
   * `manifest.csv` — Deterministic unit manifest with strata and overlap
     metadata.
   * `assignments/<reviewer_id>/assignment.db` — SQLite file ready for
     the reviewer client to consume.
   * `imports/` and `reports/` — empty placeholders awaiting submission
     imports and adjudication outputs.

## 5. Future steps

* **Reviewer client** — Should mount `assignment.db`, implement the tri
  pane UI described in the specification, and sync results back to the
  shared assignment folder.
* **Import & adjudication** — Admin tooling will ingest submitted
  assignments into `imports/`, compute agreement metrics, and populate
  `round_aggregate.db` and export artifacts under `reports/`.

These later stages are outlined in the specification and can be
implemented using the schemas and deterministic manifests produced by
this repository.
