$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path

Push-Location $repoRoot
try {
    pyinstaller --noconfirm --clean --name AdminApp --onefile --windowed ^
        --add-data "Shared;Shared" --add-data "DataAccess;DataAccess" ^
        AdminApp\main.py
} finally {
    Pop-Location
}
