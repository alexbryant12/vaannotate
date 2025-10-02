# Creating a New VAAnnotate Project and Phenotype

The steps below walk through the entire setup process – from preparing the
folder structure to launching the Admin app and defining a phenotype. The goal
is to give a non-technical coordinator a checklist they can follow without
needing to write code.

## 1. Prepare the workspace folder

1. Pick a permanent location on your shared drive where everyone on the project
   team has read/write access (for example, `\\\\research-fs01\\Projects`).
2. Create a new folder for the study (e.g. `\\\\research-fs01\\Projects\\PH_HeartFailure`).
3. Inside that folder create the following empty structure – the Admin app will
   fill in the files after the first launch:
   ```
   PH_HeartFailure\
     corpus\
     dist\
     phenotypes\
     reports\
     scripts\
   ```
4. Copy the PowerShell helper scripts from the repository (`scripts\*.ps1`) into
   the new `scripts\` folder so that team members can launch the tools without
   opening a terminal.

## 2. Build the Admin and Client executables

If you already have fresh copies of `AdminApp.exe` and `ClientApp.exe`, you can
skip this section. Otherwise:

1. On a Windows workstation with Python 3.11 installed, open PowerShell in the
   repository root.
2. Create and activate a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```
3. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Build the Admin executable:
   ```powershell
   scripts\build_admin.ps1
   ```
5. Build the Client executable:
   ```powershell
   scripts\build_client.ps1
   ```
6. After both commands succeed you will have `dist\AdminApp.exe` and
   `dist\ClientApp.exe`. Copy these two files into the `dist\` folder you created
   inside the project workspace.

## 3. Launch the Admin app against the new project folder

1. From Windows Explorer, right-click the new `scripts\run_admin.ps1` inside the
   project folder and choose **Run with PowerShell**. When prompted, supply the
   UNC path to the project root (for example,
   `\\\\research-fs01\\Projects\\PH_HeartFailure`).
2. On first launch the Admin application will create the SQLite databases:
   - `project.db` at the project root for metadata
   - `corpus\corpus.db` for patient notes
3. The window will open on the **Projects** tab. Fill in a friendly project name
   and the “Created by” field, then click **Create Project**. The record appears
   immediately in the list on the left.

## 4. Load or create the corpus

1. Prepare two CSV files: `patients.csv` (patient ICNs and STA3Ns) and
   `documents.csv` (one row per note, including the full text).
2. In PowerShell run the import helper, replacing the paths with your files:
   ```powershell
   python -m vaannotate.admin_cli import-corpus \
     "\\research-fs01\Projects\PH_HeartFailure" \
     --patients-csv "C:\Data\patients.csv" \
     --documents-csv "C:\Data\documents.csv"
   ```
3. Return to the Admin app and switch to the **Corpus** tab. You should now see
   the document counts and a top-50 preview confirming the data loaded
   correctly.

## 5. Create a phenotype definition

1. Go to the **Phenotypes** tab.
2. Choose the project you just created from the drop-down menu.
3. Enter a descriptive phenotype name, pick the level (`single_doc` for
   individual notes or `multi_doc` for patient-level reviews), and write a brief
   description for annotators.
4. Click **Create phenotype**. The phenotype appears in the list on the left and
   is now ready for round configuration.

## 6. Next steps (optional)

- Use the **Rounds** tab to configure reviewers, sampling, and manifests.
- Copy `dist\ClientApp.exe` into each reviewer assignment folder once rounds are
  generated (rename the copy to `client.exe` for convenience).
- Share the project folder path with the team. Annotators only need their
  assignment subfolder; admins launch the project via `scripts\run_admin.ps1`.

Following the checklist above gets you from an empty network folder to a fully
initialized VAAnnotate project with at least one phenotype ready for round
setup. Save this guide with the project so anyone can repeat the process.
