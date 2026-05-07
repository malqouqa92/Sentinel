# Sentinel — first-run setup for Windows.
# Idempotent: safe to re-run after a partial failure. Each step
# checks whether it's already done before acting.
#
# Usage:
#   .\setup.ps1                  # full setup
#   .\setup.ps1 -SkipClaudeCli   # don't install Claude CLI
#   .\setup.ps1 -NoTelegram      # skip the .env prompt
#
# Tested on Windows 11 + PowerShell 5.1. Run from the repo root.

[CmdletBinding()]
param(
    [switch]$SkipClaudeCli,
    [switch]$NoTelegram
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
function Write-Section {
    param([string]$Title, [int]$Step, [int]$Total = 10)
    Write-Host ""
    Write-Host ("[$Step/$Total] " + $Title) -ForegroundColor Cyan
    Write-Host ("-" * 64) -ForegroundColor DarkGray
}
function Write-OK    { param([string]$Msg) Write-Host "  OK  $Msg" -ForegroundColor Green }
function Write-Skip  { param([string]$Msg) Write-Host "  --  $Msg (already done)" -ForegroundColor DarkGray }
function Write-Doing { param([string]$Msg) Write-Host "  ..  $Msg" -ForegroundColor Yellow }
function Write-Warn  { param([string]$Msg) Write-Host "  !!  $Msg" -ForegroundColor Yellow }
function Write-Bad   { param([string]$Msg) Write-Host "  XX  $Msg" -ForegroundColor Red }

function Test-Cmd {
    param([string]$Name)
    $null = Get-Command $Name -ErrorAction SilentlyContinue
    return $?
}

# ---------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------
Clear-Host
Write-Host @"
+============================================================+
|                                                            |
|   SENTINEL  -  local agent framework setup                 |
|                                                            |
|   Constraints:  RTX 3050 / 4 GB VRAM  +  16 GB RAM         |
|   Backends:     Ollama (local)  +  Claude CLI (ceiling)    |
|   Interface:    Telegram bot                               |
|                                                            |
+============================================================+

This script walks you through 10 steps. Each step is idempotent --
safe to re-run if anything fails partway. Total time on a fresh PC:
roughly 5-15 min, mostly the model pulls (~3 GB download).

"@ -ForegroundColor Cyan

Read-Host "Press Enter to begin (Ctrl-C to abort)"

# ---------------------------------------------------------------------
# 1. Python 3.12
# ---------------------------------------------------------------------
Write-Section "Python 3.12" 1
$pythonOK = $false
if (Test-Cmd "py") {
    $v = (& py -3.12 --version 2>&1)
    if ($LASTEXITCODE -eq 0 -and $v -match "Python 3\.12") {
        Write-Skip "Python 3.12 ($v)"
        $pythonOK = $true
    }
}
if (-not $pythonOK) {
    Write-Doing "Installing Python 3.12 via winget..."
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Bad "winget failed. Install Python 3.12 manually from python.org and re-run this script."
        exit 1
    }
    Write-OK "Python 3.12 installed (you may need to open a new terminal so PATH refreshes)"
}

# ---------------------------------------------------------------------
# 2. Ollama
# ---------------------------------------------------------------------
Write-Section "Ollama" 2
if (Test-Cmd "ollama") {
    Write-Skip ("Ollama: " + (& ollama --version))
} else {
    Write-Doing "Installing Ollama via winget..."
    winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Bad "winget failed. Install Ollama manually from https://ollama.com/download/windows and re-run."
        exit 1
    }
    Write-OK "Ollama installed"
    Write-Doing "Waiting 8s for the Ollama service to come up..."
    Start-Sleep -Seconds 8
}

# ---------------------------------------------------------------------
# 3. Ollama env vars (VRAM-friendly defaults from CLAUDE.md)
# ---------------------------------------------------------------------
Write-Section "Ollama env vars (4 GB VRAM defaults)" 3
$envVars = @{
    "OLLAMA_FLASH_ATTENTION" = "1"
    "OLLAMA_KV_CACHE_TYPE"   = "q8_0"
    "OLLAMA_NUM_PARALLEL"    = "1"
    "OLLAMA_MAX_LOADED_MODELS" = "1"
    "OLLAMA_KEEP_ALIVE"      = "2m"
}
foreach ($k in $envVars.Keys) {
    $current = [System.Environment]::GetEnvironmentVariable($k, "User")
    if ($current -eq $envVars[$k]) {
        Write-Skip "$k = $($envVars[$k])"
    } else {
        & setx $k $envVars[$k] | Out-Null
        Write-OK "$k = $($envVars[$k])"
    }
}
Write-Warn "These take effect in NEW terminals. The current terminal still uses old values."

# ---------------------------------------------------------------------
# 4. Pull Qwen models (~3 GB)
# ---------------------------------------------------------------------
Write-Section "Pull Qwen models (~3 GB total)" 4
$models = @("qwen3:1.7b", "qwen2.5-coder:3b")
$installed = (& ollama list) -split "`n" | ForEach-Object { ($_ -split "\s+")[0] }
foreach ($m in $models) {
    if ($installed -contains $m) {
        Write-Skip $m
    } else {
        Write-Doing "Pulling $m (this can take a while)..."
        & ollama pull $m
        if ($LASTEXITCODE -ne 0) {
            Write-Bad "Pull failed for $m. Check internet + try `ollama pull $m` manually, then re-run."
            exit 1
        }
        Write-OK $m
    }
}

# ---------------------------------------------------------------------
# 5. Build sentinel-brain Modelfile
# ---------------------------------------------------------------------
Write-Section "Build sentinel-brain (custom Modelfile)" 5
$brainPresent = ((& ollama list) -match "^sentinel-brain")
if ($brainPresent) {
    Write-Skip "sentinel-brain already built"
} else {
    Write-Doing "Building sentinel-brain from Modelfile.brain..."
    & ollama create sentinel-brain -f .\Modelfile.brain
    if ($LASTEXITCODE -ne 0) {
        Write-Bad "ollama create failed. Inspect Modelfile.brain and try again."
        exit 1
    }
    Write-OK "sentinel-brain built"
}

# ---------------------------------------------------------------------
# 6. Python deps
# ---------------------------------------------------------------------
Write-Section "Python dependencies (pip install -r requirements.txt)" 6
Write-Doing "Installing into system Python user site (no venv per project convention)..."
& py -3.12 -m pip install --user -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Bad "pip install failed. Inspect requirements.txt and try again."
    exit 1
}
Write-OK "Dependencies installed"

# ---------------------------------------------------------------------
# 7. PyInstaller (for /gwenask app builds)
# ---------------------------------------------------------------------
Write-Section "PyInstaller (for /gwenask -> .exe pipeline)" 7
$pyiCheck = (& py -3.12 -m PyInstaller --version 2>$null)
if ($LASTEXITCODE -eq 0) {
    Write-Skip "PyInstaller $pyiCheck"
} else {
    Write-Doing "Installing PyInstaller..."
    & py -3.12 -m pip install --user pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "PyInstaller install failed. /gwenask -> .exe will not work, but the bot still runs."
    } else {
        Write-OK "PyInstaller installed"
    }
}

# ---------------------------------------------------------------------
# 8. Claude CLI (ceiling for /code teaching)
# ---------------------------------------------------------------------
Write-Section "Claude CLI (ceiling backend for /code teaching)" 8
if ($SkipClaudeCli) {
    Write-Skip "Skipped (--SkipClaudeCli flag)"
} elseif (Test-Cmd "claude") {
    Write-Skip ("Claude CLI: " + (& claude --version))
} else {
    if (-not (Test-Cmd "npm")) {
        Write-Doing "Installing Node.js (required for Claude CLI) via winget..."
        winget install --id OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Node install failed. Skipping Claude CLI. Bot will run on Qwen only."
        }
    }
    if (Test-Cmd "npm") {
        Write-Doing "Installing @anthropic-ai/claude-code globally..."
        & npm install -g "@anthropic-ai/claude-code"
        if ($LASTEXITCODE -eq 0) {
            Write-OK "Claude CLI installed"
            Write-Host ""
            Write-Host "  >> Now run 'claude login' in a NEW terminal to authenticate." -ForegroundColor Magenta
            Write-Host "  >> Sentinel only invokes the local CLI; it never sends API calls itself." -ForegroundColor Magenta
        } else {
            Write-Warn "npm install failed. Bot will fall back to Qwen only for /code."
        }
    }
}

# ---------------------------------------------------------------------
# 9. Telegram credentials (.env)
# ---------------------------------------------------------------------
Write-Section "Telegram credentials" 9
$envFile = Join-Path $RepoRoot ".env"
if ($NoTelegram) {
    Write-Skip "Skipped (--NoTelegram flag). Set SENTINEL_TELEGRAM_TOKEN + SENTINEL_TELEGRAM_USER_ID before running main.py."
} elseif ((Test-Path $envFile) -and (Get-Content $envFile -Raw) -match "SENTINEL_TELEGRAM_TOKEN=\S") {
    Write-Skip ".env already has a Telegram token"
} else {
    Write-Host @"

  +-- HOW TO GET A BOT TOKEN ---------------------------------+
  |  1. Open Telegram, search for @BotFather and start a chat. |
  |  2. Send /newbot and follow the prompts (name + username). |
  |  3. BotFather replies with an HTTP API token like:         |
  |       1234567890:AABBCCdd_eeFFggHHii-jjKKllMMnnOOppQqRrSs  |
  |  4. Paste that token below.                                |
  +------------------------------------------------------------+

  +-- HOW TO GET YOUR USER ID --------------------------------+
  |  1. In Telegram, search for @userinfobot.                  |
  |  2. Send /start.                                           |
  |  3. It replies with 'Id: <number>'. Paste the number below.|
  |  +-- (only this user ID will be authorized to use the bot)+
  +------------------------------------------------------------+

"@ -ForegroundColor Yellow

    $token = Read-Host "Telegram bot token"
    $userId = Read-Host "Your Telegram user ID (numeric)"
    if ($token -and $userId) {
        $content = "SENTINEL_TELEGRAM_TOKEN=$token`r`nSENTINEL_TELEGRAM_USER_ID=$userId`r`n"
        [System.IO.File]::WriteAllText($envFile, $content, [System.Text.Encoding]::ASCII)
        Write-OK ".env written ($envFile)"
        Write-Host "  >> Add to your shell profile or use a launcher that sources .env, OR" -ForegroundColor Magenta
        Write-Host "  >> set both vars manually:  setx SENTINEL_TELEGRAM_TOKEN `"$token`"" -ForegroundColor Magenta
    } else {
        Write-Warn "Empty input -- skipping .env. You'll need to set the env vars manually before running main.py."
    }
}

# ---------------------------------------------------------------------
# 10. KB seed + smoke test
# ---------------------------------------------------------------------
Write-Section "KB preload + smoke test" 10
$preload = Join-Path $RepoRoot "tools\preload_kb.py"
if (Test-Path $preload) {
    Write-Doing "Seeding curated KB patterns (idempotent, ~5s)..."
    & py -3.12 $preload
    if ($LASTEXITCODE -eq 0) {
        Write-OK "KB seeded"
    } else {
        Write-Warn "KB seed had a non-zero exit -- inspect tools/preload_kb.py output above."
    }
} else {
    Write-Skip "tools/preload_kb.py not present (skipping)"
}

Write-Doing "Running Ollama smoke test..."
$smoke = & py -3.12 -c @"
from core.llm import OllamaClient
try:
    out = OllamaClient().generate(model='qwen2.5-coder:3b',
                                   prompt='Say OK in one word.',
                                   timeout=60)
    print('SMOKE_OK:' + out.strip()[:40])
except Exception as e:
    print('SMOKE_FAIL:' + type(e).__name__ + ':' + str(e)[:120])
"@
if ($smoke -match "SMOKE_OK") {
    Write-OK "Ollama generate works ($smoke)"
} else {
    Write-Warn "Smoke test did not pass: $smoke"
}

# ---------------------------------------------------------------------
# Final card
# ---------------------------------------------------------------------
Write-Host ""
Write-Host @"
+============================================================+
|                                                            |
|              S E T U P    C O M P L E T E                  |
|                                                            |
+============================================================+

Next steps:

  1. Open a NEW terminal so updated PATH + env vars take effect.

  2. (If you installed Claude CLI) authenticate:
        claude login

  3. Start the bot:
        py -3.12 main.py

  4. In Telegram, message your bot:
        /help          -- list commands
        /dashboard     -- system health snapshot
        /gwenask <idea>  -- have local Qwen author a recipe
        /gwen <recipe>   -- execute a literal recipe (paste-mangle-immune)

Documentation:
  - CLAUDE.md   -- architecture + design decisions
  - PHASES.md   -- per-phase change log + footguns
  - README of /gwen + /gwenask is in the Phase 18 PHASES.md entry.

Save points:
  - phase18-logoff-20260507  (latest stable)
  Revert: git reset --hard phase18-logoff-20260507

"@ -ForegroundColor Green
