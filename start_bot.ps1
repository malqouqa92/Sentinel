# Launch the Sentinel Telegram bot. Reads credentials from .env.bot
# (UTF-8, KEY=VALUE per line). Copy .env.example -> .env.bot and fill
# in your token + user ID before running this script.

Set-Location -LiteralPath $PSScriptRoot

$envFile = Join-Path $PSScriptRoot ".env.bot"
if (-not (Test-Path $envFile)) {
    Write-Host "ERROR: .env.bot not found." -ForegroundColor Red
    Write-Host "Copy .env.example -> .env.bot and fill in your bot token + user ID:" -ForegroundColor Yellow
    Write-Host "  Copy-Item .env.example .env.bot" -ForegroundColor Yellow
    Write-Host "  notepad .env.bot" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

# Parse KEY=VALUE lines (skip blanks + comments)
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        $key = $matches[1].Trim()
        $val = $matches[2].Trim().Trim('"').Trim("'")
        Set-Item -Path "env:$key" -Value $val
    }
}

if (-not $env:SENTINEL_TELEGRAM_TOKEN -or `
    $env:SENTINEL_TELEGRAM_TOKEN -eq "PUT_YOUR_BOT_TOKEN_HERE") {
    Write-Host "ERROR: SENTINEL_TELEGRAM_TOKEN not set in .env.bot" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

py -3.12 main.py
