# Building Portable Windows Executables

> After cloning, run `python tools\seed_toy_project.py` from an activated Conda prompt to regenerate `demo/Project_Toy/` and placeholders — binaries are not committed to Git.

1. **Prerequisites**
   - Windows 10 or later
   - Anaconda or Miniconda with Python 3.11 available
2. **Create a Conda environment**
   ```
   conda create -n vaannotate python=3.11
   ```
3. **Activate the environment**
   ```
   conda activate vaannotate
   ```
4. **Install dependencies**
   ```
   pip install -r requirements.txt
   pip install pyinstaller
   ```
5. **Build AdminApp**
   ```
   pyinstaller --noconfirm --clean --name AdminApp --onefile --windowed \
     vaannotate/AdminApp/main.py
   ```
6. **Build ClientApp**
   ```
   pyinstaller --noconfirm --clean --name ClientApp --onefile --windowed \
     vaannotate/ClientApp/main.py
   ```
7. **Locate the outputs**
   - Executables are written to `dist\AdminApp.exe` and `dist\ClientApp.exe`.
8. **Deploy to reviewers**
   - Copy `ClientApp.exe` into each assignment folder and rename to `client.exe`.
   - Share `AdminApp.exe` with the admin team or keep it in `dist/` for the launch scripts.
   - Re-run `python tools\seed_toy_project.py` after building to refresh placeholder assignments if needed.
