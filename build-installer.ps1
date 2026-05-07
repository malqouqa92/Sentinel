# Build Sentinel-Setup.exe -- a self-extracting installer that
# bundles the entire repo source + chains into setup.ps1 on extract.
#
# Output is a SINGLE .exe (~1-2 MB). Source code is embedded as a
# base64-encoded zip inside the script. Heavy deps (Python, Ollama,
# Qwen weights, Claude CLI) are NOT bundled -- setup.ps1 downloads
# them at runtime via winget / ollama pull / npm.
#
# Usage:
#   .\build-installer.ps1                    # default output
#   .\build-installer.ps1 -OutputFile foo.exe

[CmdletBinding()]
param(
    [string]$OutputFile = ".\Sentinel-Setup.exe",
    [string]$Template = ".\scripts\installer\_template.ps1"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Exclusions: artifacts that don't belong in a fresh-install bundle.
$EXCLUDE_DIRS = @(
    '.git', '__pycache__', 'logs', 'backups', '.venv',
    'node_modules', '.pytest_cache'
)
$EXCLUDE_GLOBS = @(
    '*.pyc', '*.pyo',
    '*.db', '*.db-shm', '*.db-wal', '*.db-journal',
    'Sentinel-Setup.exe',
    'knowledge.db.*',
    '*.regressed-snapshot*',
    '*.stresstest-pre*',
    'h_journal.db',
    '_test_*.ps1',
    '_test_*.py'
)

# ---------------------------------------------------------------------
# 1. Ensure NuGet + ps2exe
# ---------------------------------------------------------------------
if (-not (Get-Module -ListAvailable -Name ps2exe)) {
    Write-Host "Installing NuGet provider + ps2exe (one time)..." -ForegroundColor Yellow
    Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 `
        -Force -Scope CurrentUser | Out-Null
    Install-Module -Name ps2exe -Scope CurrentUser -Force `
        -AllowClobber -SkipPublisherCheck
}
Import-Module ps2exe -Force

# ---------------------------------------------------------------------
# 2. Build source zip (exclude artifacts)
# ---------------------------------------------------------------------
Write-Host "Building source zip..." -ForegroundColor Cyan
$tempZip = Join-Path $env:TEMP "sentinel-bundle-$([System.Guid]::NewGuid().ToString('N')).zip"

Add-Type -AssemblyName System.IO.Compression -ErrorAction SilentlyContinue
Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue

$archive = [System.IO.Compression.ZipFile]::Open(
    $tempZip, [System.IO.Compression.ZipArchiveMode]::Create
)
$repoRoot = (Get-Location).Path
$fileCount = 0
try {
    Get-ChildItem -Path $repoRoot -Recurse -File -Force | ForEach-Object {
        $abs = $_.FullName
        $rel = $abs.Substring($repoRoot.Length + 1) -replace '\\', '/'

        # Filter by excluded dirs (any path component matches).
        $parts = $rel.Split('/')
        $skip = $false
        foreach ($d in $EXCLUDE_DIRS) {
            if ($parts -contains $d) { $skip = $true; break }
        }
        if (-not $skip) {
            foreach ($g in $EXCLUDE_GLOBS) {
                if ($_.Name -like $g) { $skip = $true; break }
            }
        }
        if ($skip) { return }

        $entry = $archive.CreateEntry(
            $rel, [System.IO.Compression.CompressionLevel]::Optimal
        )
        $stream = $entry.Open()
        try {
            $bytes = [System.IO.File]::ReadAllBytes($abs)
            $stream.Write($bytes, 0, $bytes.Length)
        } finally {
            $stream.Dispose()
        }
        $script:fileCount++
    }
} finally {
    $archive.Dispose()
}

$zipSize = (Get-Item $tempZip).Length
Write-Host ("  OK  $fileCount files, $([math]::Round($zipSize/1MB, 2)) MB zipped") -ForegroundColor Green

# ---------------------------------------------------------------------
# 3. Base64-encode + chunk for PowerShell 5.1 parser limits
# ---------------------------------------------------------------------
Write-Host "Base64-encoding..." -ForegroundColor Cyan
$bytes = [System.IO.File]::ReadAllBytes($tempZip)
$blob = [Convert]::ToBase64String($bytes)  # NO line breaks
Write-Host ("  OK  $([math]::Round($blob.Length/1MB, 2)) MB base64 string") -ForegroundColor Green

# Split into chunks of $chunkSize chars. Each chunk becomes a single-
# quoted PowerShell string in the generated array literal. WinPS 5.1
# parser handles arrays of many small strings far better than one
# multi-MB here-string.
$chunkSize = 4000
$sb = [System.Text.StringBuilder]::new($blob.Length + ($blob.Length / $chunkSize * 4))
for ($i = 0; $i -lt $blob.Length; $i += $chunkSize) {
    $end = [Math]::Min($i + $chunkSize, $blob.Length)
    $piece = $blob.Substring($i, $end - $i)
    [void]$sb.Append("'").Append($piece).AppendLine("'")
}
$chunkBlock = $sb.ToString().TrimEnd("`r`n")
$chunkCount = [math]::Ceiling($blob.Length / $chunkSize)
Write-Host ("  OK  Split into $chunkCount chunks of $chunkSize chars") -ForegroundColor Green

# ---------------------------------------------------------------------
# 4. Generate runtime script from template
# ---------------------------------------------------------------------
if (-not (Test-Path $Template)) {
    throw "Template not found: $Template"
}
Write-Host "Generating installer script..." -ForegroundColor Cyan
$tpl = Get-Content $Template -Raw -Encoding UTF8
$generated = $tpl.Replace('@@BUNDLE_CHUNKS@@', $chunkBlock)
$tempPs = Join-Path $env:TEMP "sentinel-installer-$([System.Guid]::NewGuid().ToString('N')).ps1"
[System.IO.File]::WriteAllText($tempPs, $generated, [System.Text.UTF8Encoding]::new($false))

# ---------------------------------------------------------------------
# 5. Compile via ps2exe
# ---------------------------------------------------------------------
Write-Host "Compiling .exe via ps2exe (this can take ~30s with a big bundle)..." -ForegroundColor Cyan
Invoke-PS2EXE `
    -inputFile $tempPs `
    -outputFile $OutputFile `
    -title "Sentinel Setup" `
    -description "Sentinel local agent framework -- self-extracting installer" `
    -company "Sentinel" `
    -version "1.0.0.0" `
    -noConsole:$false

# Cleanup temp files
Remove-Item $tempZip, $tempPs -Force -ErrorAction SilentlyContinue

if (Test-Path $OutputFile) {
    $f = Get-Item $OutputFile
    Write-Host ""
    Write-Host ("Built: " + $f.FullName) -ForegroundColor Green
    Write-Host ("Size:  " + [math]::Round($f.Length / 1MB, 2) + " MB") -ForegroundColor Green
    Write-Host ""
    Write-Host "Distribute this single file. On the target PC:" -ForegroundColor Yellow
    Write-Host "  1. Double-click Sentinel-Setup.exe" -ForegroundColor Yellow
    Write-Host "  2. SmartScreen warns 'unrecognized publisher' -- click 'More info' -> 'Run anyway'" -ForegroundColor Yellow
    Write-Host "  3. Pick install location, follow setup.ps1 prompts" -ForegroundColor Yellow
} else {
    throw "Build did not produce $OutputFile"
}
