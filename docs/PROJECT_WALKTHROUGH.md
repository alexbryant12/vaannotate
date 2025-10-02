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
2. On first launch the Admin application will ensure the project metadata
   database (`project.db`) exists. Corpora are now tracked per phenotype and are
   stored within `phenotypes/<pheno_id>/corpus/corpus.db` once you assign one to
   a phenotype.
3. The window opens with a project tree on the left and detail views on the
   right. The top-level node represents the project; right-click it at any time
   to add new phenotypes.

## 4. Load or create the corpus

1. Prepare two CSV files: `patients.csv` (patient ICNs and STA3Ns) and
   `documents.csv` (one row per note, including the full text).
2. In the activated Conda prompt run the import helper, replacing the paths with
   your files:
   ```
   python -m vaannotate.admin_cli import-corpus \
     "\\research-fs01\Projects\PH_HeartFailure" \
     --patients-csv "C:\Data\patients.csv" \
     --documents-csv "C:\Data\documents.csv" \
     --corpus-db "phenotypes/ph_diabetes/corpus/source.db"
  ```
   The destination can be any path; relative values are resolved against the project
   folder so you can stage per-phenotype corpora ahead of time.
3. After the import, add a phenotype and select this corpus when prompted. Once
   saved, highlight the phenotype’s **Corpus** node in the project tree to see
   document counts and a top-50 preview confirming the data loaded correctly.

## 5. Create a phenotype definition

1. In the project tree, right-click the project node and choose **Add phenotype…**.
2. Enter a descriptive phenotype name, pick the level (`single_doc` for
   individual notes or `multi_doc` for patient-level reviews), add a short
   description, and browse to the corpus database that should back this
   phenotype. The Admin app copies the selected corpus into the phenotype’s
   folder and locks it in for that phenotype.
3. Click **OK**. The phenotype appears under the project node along with child
   items for the corpus, rounds, and the IAA dashboard.

## 6. Next steps (optional)

- Right-click a phenotype in the tree and choose **Add round…** to configure
  reviewers, sampling, and manifests.
- Copy `dist/ClientApp.exe` into each reviewer assignment folder once rounds are
  generated (rename the copy to `client.exe` for convenience).
- Share the project folder path with the team. Annotators only need their
  assignment subfolder; admins launch the project from an activated Conda prompt
  with `python -m vaannotate.AdminApp.main`.

Following the checklist above gets you from an empty network folder to a fully
initialized VAAnnotate project with at least one phenotype ready for round
setup. Save this guide with the project so anyone can repeat the process.
