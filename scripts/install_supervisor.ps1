# Phase 11 supervisor: register Sentinel as a Windows Task Scheduler entry
# so it auto-starts at user logon and restarts on unexpected exit.
#
# This script does NOT install itself automatically. Run it once manually:
#
#     powershell -ExecutionPolicy Bypass -File scripts\install_supervisor.ps1
#
# It registers a user-level task (no admin rights required) named
# "Sentinel OS" that:
#   - Triggers at user logon
#   - Runs python.exe main.py from the project directory, hidden window
#   - Restarts up to 3 times on failure, 1-minute interval between
#   - Starts when next available if the trigger was missed (laptop sleep, etc.)
#
# To uninstall later:
#     Unregister-ScheduledTask -TaskName "Sentinel OS" -Confirm:$false

$ErrorActionPreference = "Stop"

# Resolve project root from the location of this script.
$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Resolve-Path (Join-Path $scriptDir "..")

# Locate python.exe -- prefer the project venv if one exists, else fall back
# to the interpreter currently on PATH.
$pythonExe = $null
$venvPython = Join-Path $projectDir ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $pythonExe = $cmd.Source }
}
if (-not $pythonExe) {
    Write-Error "Could not find python.exe. Activate your venv or add python to PATH."
    exit 1
}

Write-Host "Project dir: $projectDir"
Write-Host "Python:      $pythonExe"

$taskName = "Sentinel OS"

# Drop any existing entry first so this script is idempotent.
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Existing '$taskName' task found -- removing it first."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction `
    -Execute $pythonExe `
    -Argument "main.py" `
    -WorkingDirectory $projectDir.Path

$trigger = New-ScheduledTaskTrigger -AtLogon

$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)  # 0 = no time limit

# Run as the current user (no admin needed for user-level tasks).
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Sentinel OS -- agentic engine. Auto-starts at logon, restarts on failure." | Out-Null

Write-Host ""
Write-Host "[OK] Registered '$taskName' to start at logon."
Write-Host "     To start it now:    Start-ScheduledTask -TaskName '$taskName'"
Write-Host "     To verify:          Get-ScheduledTask -TaskName '$taskName' | Get-ScheduledTaskInfo"
Write-Host "     To uninstall:       Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
