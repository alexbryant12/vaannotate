# Building Portable Windows Executables

> After cloning, run `python tools\seed_toy_project.py` (or `scripts\new_toy_project.ps1`) to regenerate `demo/Project_Toy/` and placeholders — binaries are not committed to Git.

1. **Prerequisites**
   - Windows 10 or later
   - Python 3.11 installed (`py -3.11`)
2. **Create a virtual environment**
   ```powershell
   py -3.11 -m venv .venv
   ```
3. **Activate the environment**
   ```powershell
   .venv\Scripts\activate
   ```
4. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
5. **Build AdminApp**
   ```powershell
   pyinstaller --noconfirm --clean --name AdminApp --onefile --windowed ^
     --add-data "Shared;Shared" --add-data "DataAccess;DataAccess" ^
     AdminApp\main.py
   ```
6. **Build ClientApp**
   ```powershell
   pyinstaller --noconfirm --clean --name ClientApp --onefile --windowed ^
     --add-data "Shared;Shared" --add-data "DataAccess;DataAccess" ^
     ClientApp\main.py
   ```
7. **Locate the outputs**
   - Executables are written to `dist\AdminApp.exe` and `dist\ClientApp.exe`.
8. **Deploy to reviewers**
   - Copy `ClientApp.exe` into each assignment folder and rename to `client.exe`.
   - Share `AdminApp.exe` with the admin team or keep it in `dist/` for the launch scripts.
   - Re-run `python tools\seed_toy_project.py` after building to refresh placeholder assignments if needed.
