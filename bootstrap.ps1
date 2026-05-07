# Sentinel — one-script bootstrap.
#
# This is the ONLY file you need to copy to a fresh PC. It fetches
# the rest of the repo, then chains into setup.ps1.
#
# Three source modes (pick whichever fits your situation):
#
#   Git clone:
#     .\bootstrap.ps1 -RepoUrl https://github.com/you/sentinel.git
#
#   From a local zip:
#     .\bootstrap.ps1 -ZipPath C:\Users\You\Downloads\sentinel.zip
#
#   From an existing folder (robocopy mirror):
#     .\bootstrap.ps1 -SourceDir D:\old-pc\sentinel
#
#   Interactive (no args, just run it):
#     .\bootstrap.ps1
#
# Default install location is $HOME\sentinel; override with -TargetDir.

[CmdletBinding(DefaultParameterSetName = "Interactive")]
param(
    [Parameter(ParameterSetName = "Git")]
    [string]$RepoUrl,

    [Parameter(ParameterSetName = "Zip")]
    [string]$ZipPath,

    [Parameter(ParameterSetName = "Folder")]
    [string]$SourceDir,

    [string]$TargetDir = (Join-Path $env:USERPROFILE "sentinel"),

    [switch]$SkipSetup    # fetch only, don't run setup.ps1
)

$ErrorActionPreference = "Stop"

function Write-Banner {
    Clear-Host
    Write-Host @"
+============================================================+
|                                                            |
|   SENTINEL bootstrap                                       |
|   Fetches the repo, then chains into setup.ps1.            |
|                                                            |
+============================================================+

"@ -ForegroundColor Cyan
}

function Test-Cmd {
    param([string]$Name)
    $null = Get-Command $Name -ErrorAction SilentlyContinue
    return $?
}

function Ensure-Git {
    if (Test-Cmd "git") { return }
    Write-Host "  ..  Git not found. Installing via winget..." -ForegroundColor Yellow
    winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install Git. Install manually from https://git-scm.com/download/win and re-run."
    }
    # winget doesn't refresh PATH for the current session; locate git.exe directly.
    $gitExe = "C:\Program Files\Git\cmd\git.exe"
    if (Test-Path $gitExe) {
        $env:Path = "C:\Program Files\Git\cmd;" + $env:Path
    } else {
        throw "Git installed but not on PATH. Open a new terminal and re-run bootstrap."
    }
}

# ---------------------------------------------------------------------
Write-Banner

# Resolve source mode -- prompt if nothing given.
$mode = $PSCmdlet.ParameterSetName
if ($mode -eq "Interactive") {
    Write-Host "Pick a source for the Sentinel repo:" -ForegroundColor Yellow
    Write-Host "  1) Git clone (you have a repo URL)"
    Write-Host "  2) Local zip file"
    Write-Host "  3) Existing folder on this machine (e.g. external drive)"
    Write-Host ""
    $choice = Read-Host "Enter 1, 2, or 3"
    switch ($choice) {
        "1" { $mode = "Git";    $RepoUrl   = Read-Host "Git URL (https or ssh)" }
        "2" { $mode = "Zip";    $ZipPath   = Read-Host "Path to sentinel.zip" }
        "3" { $mode = "Folder"; $SourceDir = Read-Host "Path to existing sentinel folder" }
        default { throw "Unknown choice '$choice'." }
    }
    $defaultTarget = $TargetDir
    $entered = Read-Host "Install location [$defaultTarget]"
    if ($entered) { $TargetDir = $entered }
}

# Refuse to nuke an existing target.
if (Test-Path $TargetDir) {
    $items = @(Get-ChildItem -Path $TargetDir -Force -ErrorAction SilentlyContinue)
    if ($items.Count -gt 0) {
        Write-Host ""
        Write-Host "  !!  Target dir already exists and is non-empty: $TargetDir" -ForegroundColor Yellow
        $proceed = Read-Host "Type 'yes' to continue into it (no clobber attempted), anything else aborts"
        if ($proceed -ne "yes") { exit 1 }
    }
} else {
    New-Item -ItemType Directory -Path $TargetDir | Out-Null
}

Write-Host ""
Write-Host "Fetching into $TargetDir ..." -ForegroundColor Cyan

switch ($mode) {
    "Git" {
        Ensure-Git
        $items = @(Get-ChildItem -Path $TargetDir -Force -ErrorAction SilentlyContinue)
        if ($items.Count -gt 0 -and (Test-Path (Join-Path $TargetDir ".git"))) {
            Write-Host "  ..  Existing repo found, pulling latest..." -ForegroundColor Yellow
            Push-Location $TargetDir
            try { & git pull --ff-only } finally { Pop-Location }
        } else {
            # Clone INTO the target (must be empty).
            & git clone $RepoUrl $TargetDir
            if ($LASTEXITCODE -ne 0) { throw "git clone failed." }
        }
    }
    "Zip" {
        if (-not (Test-Path $ZipPath)) { throw "Zip not found: $ZipPath" }
        Write-Host "  ..  Extracting $ZipPath ..." -ForegroundColor Yellow
        Expand-Archive -Path $ZipPath -DestinationPath $TargetDir -Force
        # If the zip contained a single top-level dir, flatten it.
        $top = @(Get-ChildItem -Path $TargetDir -Directory)
        if ($top.Count -eq 1 -and (Get-ChildItem -Path $TargetDir -File).Count -eq 0) {
            Get-ChildItem -Path $top[0].FullName -Force | Move-Item -Destination $TargetDir
            Remove-Item -Path $top[0].FullName -Force
        }
    }
    "Folder" {
        if (-not (Test-Path $SourceDir)) { throw "Source dir not found: $SourceDir" }
        Write-Host "  ..  robocopy from $SourceDir (excluding caches + DBs)..." -ForegroundColor Yellow
        # robocopy: /MIR mirror, /XD exclude dirs, /XF exclude files
        & robocopy $SourceDir $TargetDir /MIR `
            /XD __pycache__ .git logs backups .venv `
            /XF *.pyc *.db *.db-shm *.db-wal `
            /NFL /NDL /NJH /NJS | Out-Null
        # robocopy uses non-zero exit codes for "OK with copies"; treat 0-7 as success.
        if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE)." }
    }
}

# Sanity: ensure setup.ps1 is present.
$setup = Join-Path $TargetDir "setup.ps1"
if (-not (Test-Path $setup)) {
    Write-Host ""
    Write-Host "  XX  setup.ps1 not found inside $TargetDir." -ForegroundColor Red
    Write-Host "      Either the source did not contain it, or extraction landed in a subfolder." -ForegroundColor Red
    Write-Host "      Inspect $TargetDir and run setup.ps1 manually." -ForegroundColor Red
    exit 1
}

Write-Host "  OK  Repo fetched into $TargetDir" -ForegroundColor Green

if ($SkipSetup) {
    Write-Host ""
    Write-Host "Stopping after fetch (--SkipSetup). Run setup later with:" -ForegroundColor Yellow
    Write-Host "  cd `"$TargetDir`"; .\setup.ps1" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Chaining into setup.ps1 ..." -ForegroundColor Cyan
Set-Location $TargetDir
& $setup
exit $LASTEXITCODE
