Param(
    [string]$ProjectRelative = "demo/Project_Toy"
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
$projectPath = Join-Path $repoRoot $ProjectRelative

if (Test-Path $projectPath) {
    Write-Host "Removing existing project at $projectPath"
    Remove-Item $projectPath -Recurse -Force
}

$seedScript = Join-Path $repoRoot "tools\seed_toy_project.py"
Write-Host "Seeding project via $seedScript"
python $seedScript --project $ProjectRelative

$adminExe = Join-Path $repoRoot "dist\AdminApp.exe"
if (Test-Path $adminExe) {
    $absProject = (Resolve-Path $projectPath).Path
    Write-Host "Launching AdminApp for $absProject"
    Start-Process -FilePath $adminExe -ArgumentList "--project `"$absProject`"" -WorkingDirectory (Split-Path $adminExe)
} else {
    Write-Warning "AdminApp.exe not found in dist/. Build the executables first."
}
