# Admin Runbook — Project Toy

## Hydrate the Toy Workspace
> The demo workspace and executables are generated on demand (no binaries are tracked in Git).

1. From the repo root, run `python tools\seed_toy_project.py`, **or** double-click `scripts\new_toy_project.ps1` from Windows.
2. The script rebuilds `demo/Project_Toy/`, creates reviewer assignments, and places placeholder clients into each assignment folder.
3. When you build real executables, copy them into `dist/` before sharing the project.

## Overview of Folder Layout
- `corpus/` — canonical patient/document SQLite database (**do not edit directly**).
- `project.db` — metadata database (phenotypes, reviewers, rounds).
- `phenotypes/ph_diabetes/rounds/round_1/` — Round 1 workspace with manifests, assignments, reports.
- `reports/` — export targets (`reports/exports/` holds gold data after finalization).
- `scripts/` — helper PowerShell launchers (keep with the project).
- `dist/` — locally built Admin and Client executables.

> **Do not** move or rename `project.db`, `corpus/`, or anything inside `phenotypes/` once assignments are distributed.

## Launching the Admin App
1. From Windows, open PowerShell.
2. Run `scripts\run_admin.ps1 -Project \\server\share\Project_Toy`.
   - Alternatively, launch directly via `dist\AdminApp.exe --project \\server\share\Project_Toy` (after building the executable locally).
3. The Admin application opens against the provided UNC project path.

## Target Corpus Summary
- Navigate to the **Corpus** or **Phenotypes** tabs to review seeded patients (ICNs 1001–1003).
- Notes span 2018–2024 with mixed STA3Ns (506, 515).

## Round Wizard Walkthrough
- Open **Rounds → ph_diabetes → Round 1**.
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
- `scripts/run_client.ps1` — helper launcher for that folder.
  - `logs/` — client runtime logs.

### Sharing with Annotators
- Provide the UNC path to each reviewer folder.
- Annotators will see only `assignment.db`, `client.exe`, `label_schema.json`, `scripts/`, and `logs/`.

## Importing Submissions
1. Collect `assignment.db` from reviewers (or let the Admin app pull directly from shared folders).
2. In Admin → Rounds → Round 1, choose **Import assignment** for each reviewer.
3. Watch for integrity messages:
   - *"Text hash mismatch"* indicates the corpus changed; regenerate the round before proceeding.
   - *"Assignment locked"* means a client is still open; verify and remove stale `.lock` files only after closure.

## Reviewing Disagreements
1. After imports, open the **Disagreements** tab.
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
