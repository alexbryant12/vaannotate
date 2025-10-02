# Admin Runbook — Project Toy

## Hydrate the Toy Workspace
> The demo workspace and executables are generated on demand (no binaries are tracked in Git).

1. From an activated Conda prompt in the repo root, run `python tools\seed_toy_project.py` to regenerate the workspace.
2. The script rebuilds `demo/Project_Toy/`, creates reviewer assignments, and places placeholder clients into each assignment folder.
3. When you build real executables, copy them into `dist/` before sharing the project.

## Overview of Folder Layout
- `phenotypes/<pheno_id>/corpus/` — phenotype-specific corpus database copied from the source you selected in the Admin UI (**do not edit directly**).
- `project.db` — metadata database (phenotypes, reviewers, rounds).
- `phenotypes/ph_diabetes/rounds/round_1/` — Round 1 workspace with manifests, assignments, reports.
- `reports/` — export targets (`reports/exports/` holds gold data after finalization).
- `scripts/` — optional helper launchers (not required when running from Conda).
- `dist/` — locally built Admin and Client executables.

> **Do not** move or rename `project.db` or anything inside `phenotypes/` once assignments are distributed.

## Launching the Admin App
1. Open an Anaconda Prompt and activate the environment used for VAAnnotate:
   ```
   conda activate vaannotate
   ```
2. Launch the Admin client:
   ```
   python -m vaannotate.AdminApp.main
   ```
3. Use **File → Open project folder…** to point the application at `\\server\share\Project_Toy`.

## Target Corpus Summary
- Select a phenotype’s **Corpus** node in the project tree to review seeded patients (ICNs 2001–2010).
- Notes span 2015–2024 with mixed STA3Ns (506, 515) and now include 100 long-form entries.

## Round Wizard Walkthrough
- In the project tree expand **ph_diabetes** and select **Round 1**.
- Review stored configuration (`round_config.json`) showing:
  - Patient filters: years 2018–2024, STA3Ns 506/515.
  - Note filters: PRIMARY CARE and ENDOCRINOLOGY notes matching `(metformin|insulin|hba1c\s*\d+(\.\d+)?)` (case-insensitive).
  - Stratification by `note_year`, two samples per stratum.
  - Reviewers `r_alex`, `r_blake` with overlap of 1 unit (independent reviews).
- Confirm deterministic RNG seed `133742`.

## Reviewer Folders
- Each reviewer assignment lives under `phenotypes/ph_diabetes/rounds/round_1/assignments/<reviewer_id>/`.
- Contents per reviewer:
- `assignment.db` — SQLite assignment package.
- `label_schema.json` — label metadata for the client.
- `client.exe` — placeholder annotator app (regenerated from `dist/ClientApp.exe`; replace with your real build if available).
- `scripts/run_client.ps1` — optional launcher; not required when using Conda commands.
  - `logs/` — client runtime logs.

### Sharing with Annotators
- Provide the UNC path to each reviewer folder.
- Annotators will see only `assignment.db`, `client.exe`, `label_schema.json`, `scripts/`, and `logs/`.

## Importing Submissions
1. Collect `assignment.db` from reviewers (or let the Admin app pull directly from shared folders).
2. Select **Round 1** in the project tree and use the import controls to ingest each reviewer.
3. Watch for integrity messages:
   - *"Text hash mismatch"* indicates the corpus changed; regenerate the round before proceeding.
   - *"Assignment locked"* means a client is still open; verify and remove stale `.lock` files only after closure.

## Reviewing Disagreements
1. After imports, select the phenotype’s **IAA** node.
2. The overlapped unit is intentionally split (Has_phenotype Yes vs No) to demonstrate adjudication.
3. Review evidence, select final values, and record adjudicator notes if needed.

## Finalizing the Round
1. Confirm all assignments show **Imported** status.
2. Resolve every disagreement, then choose **Finalize round**.
3. Final adjudications are stored in `round_aggregate.db`.

## Exporting Gold Data
- Use the **Exports** button to generate CSV/JSON outputs.
- Exports are written to `phenotypes/ph_diabetes/rounds/round_1/reports/exports/`.
- Share or archive the exports as the project gold standard.
