$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$assignmentDir = Split-Path -Parent $scriptRoot

$assignmentDb = Join-Path $assignmentDir "assignment.db"
if (-not (Test-Path $assignmentDb)) {
    throw "assignment.db not found at $assignmentDb"
}

$clientExe = Join-Path $assignmentDir "client.exe"
if (-not (Test-Path $clientExe)) {
    throw "client.exe not found at $clientExe"
}

Write-Host "Launching annotator client for $assignmentDb"
Start-Process -FilePath $clientExe -ArgumentList "`"$assignmentDb`"" -WorkingDirectory $assignmentDir
