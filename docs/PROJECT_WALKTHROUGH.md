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
4. Keep a copy of the repository's `scripts/` folder in the workspace if you
   still distribute Windows shortcuts, but the steps below assume you will run
   the tooling from an activated Conda prompt.

## 2. Build the Admin and Client executables

If you already have fresh copies of `AdminApp.exe` and `ClientApp.exe`, you can
skip this section. Otherwise:

1. On a workstation with Conda (Anaconda or Miniconda) installed, open an
   **Anaconda Prompt** in the repository root.
2. Create the environment (run once):
   ```
   conda create -n vaannotate python=3.11
   ```
3. Activate the environment and install dependencies:
   ```
   conda activate vaannotate
   pip install -r requirements.txt
   pip install pyinstaller
   ```
4. Build the Admin executable from the activated environment:
   ```
   pyinstaller --noconfirm --clean --name AdminApp --onefile --windowed \
     vaannotate/AdminApp/main.py
   ```
5. Build the Client executable:
   ```
   pyinstaller --noconfirm --clean --name ClientApp --onefile --windowed \
     vaannotate/ClientApp/main.py
   ```
6. After both commands succeed you will have `dist/AdminApp.exe` and
   `dist/ClientApp.exe`. Copy these two files into the `dist/` folder you created
   inside the project workspace.

## 3. Launch the Admin app against the new project folder

1. From the activated Conda prompt, launch the Admin application:
   ```
   conda activate vaannotate
   python -m vaannotate.AdminApp.main
   ```
   Use **File → Open project folder…** to browse to the UNC path for the new
   project (for example, `\\\\research-fs01\\Projects\\PH_HeartFailure`).
2. On first launch the Admin application will create the SQLite databases:
   - `project.db` at the project root for metadata
   - `corpus\corpus.db` for patient notes
3. The window will open on the **Projects** tab. Fill in a friendly project name
   and the “Created by” field, then click **Create Project**. The record appears
   immediately in the list on the left.

## 4. Load or create the corpus

1. Prepare two CSV files: `patients.csv` (patient ICNs and STA3Ns) and
   `documents.csv` (one row per note, including the full text).
2. In the activated Conda prompt run the import helper, replacing the paths with
   your files:
   ```
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
- Copy `dist/ClientApp.exe` into each reviewer assignment folder once rounds are
  generated (rename the copy to `client.exe` for convenience).
- Share the project folder path with the team. Annotators only need their
  assignment subfolder; admins launch the project from an activated Conda prompt
  with `python -m vaannotate.AdminApp.main`.

Following the checklist above gets you from an empty network folder to a fully
initialized VAAnnotate project with at least one phenotype ready for round
setup. Save this guide with the project so anyone can repeat the process.
