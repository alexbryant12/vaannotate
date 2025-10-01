Param(
    [Parameter(Mandatory = $true)]
    [string]$Project
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
$adminExe = Join-Path $repoRoot "dist\AdminApp.exe"

if (-not (Test-Path $adminExe)) {
    throw "AdminApp.exe not found at $adminExe. Build the binaries before launching."
}

Write-Host "Launching AdminApp for project path $Project"
Start-Process -FilePath $adminExe -ArgumentList "--project `"$Project`"" -WorkingDirectory (Split-Path $adminExe)
