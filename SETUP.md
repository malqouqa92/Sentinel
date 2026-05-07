# Sentinel — Setup Guide

Complete reference for installing Sentinel on a new Windows PC. Three
tools are provided, each one layer of friendliness above the next.
Pick whichever matches the situation.

---

## Tools at a Glance

| File | Role | When to use |
|---|---|---|
| `setup.ps1` | Idempotent installer for an already-cloned repo. Walks 10 steps: Python → Ollama → models → deps → Claude CLI → credentials. | You already have the source on the new PC and just need to install dependencies. |
| `bootstrap.ps1` | Fetches the repo (Git clone / zip extract / folder copy), then chains into `setup.ps1`. | You want a single PowerShell file to copy to the new PC; it pulls everything else. |
| `Sentinel-Setup.exe` | **Self-extracting** installer (~1.2 MB). Source code is embedded; double-click extracts and chains into setup.ps1. No Git URL or zip needed. | You want a single clickable file that contains everything except the heavy downloads (Python, Ollama, Qwen weights). |
| `build-installer.ps1` | Rebuilds `Sentinel-Setup.exe` from `bootstrap.ps1`. | After editing `bootstrap.ps1`. |

All four files live at the repo root.

---

## Decision Tree

```
Need to install Sentinel on a new PC?
│
├─ Have you already copied the source folder over?
│   ├─ Yes  →  cd into it, run setup.ps1                       (5-15 min)
│   └─ No
│
└─ Pick a single-file bootstrap:
    ├─ Want a clickable .exe?  →  Copy Sentinel-Setup.exe, double-click
    ├─ OK with PowerShell?     →  Copy bootstrap.ps1, run it
    └─ Source is on GitHub and box has internet?
        →  iex (irm https://your-repo/bootstrap.ps1)
        (works only if you push bootstrap.ps1 to a public URL)
```

---

## What setup.ps1 Does (10 Steps)

Every step is **idempotent** — re-run safely after a partial failure.
Each step prints `OK / -- / .. / !! / XX` for `did / already done /
working on it / warning / failed`.

| # | Step | Tool | What it installs |
|---|---|---|---|
| 1 | Python 3.12 | winget `Python.Python.3.12` | Required for the bot. |
| 2 | Ollama | winget `Ollama.Ollama` | Local LLM runtime. Sleeps 8 s for the service. |
| 3 | Ollama env vars | `setx` | `OLLAMA_FLASH_ATTENTION=1`, `KV_CACHE_TYPE=q8_0`, `NUM_PARALLEL=1`, `MAX_LOADED_MODELS=1`, `KEEP_ALIVE=2m`. Tuned for 4 GB VRAM. |
| 4 | Pull models | `ollama pull` | `qwen3:1.7b` (brain) + `qwen2.5-coder:3b` (worker). ~3 GB total. |
| 5 | sentinel-brain | `ollama create -f Modelfile.brain` | Custom variant with `num_ctx=8192, num_predict=1024, temperature=0.1`. |
| 6 | Python deps | `pip install --user -r requirements.txt` | pydantic, python-telegram-bot, jobspy, croniter, etc. |
| 7 | PyInstaller | `pip install --user pyinstaller` | Optional — needed for `/gwenask` → `.exe` pipeline. |
| 8 | Claude CLI | `winget OpenJS.NodeJS.LTS` then `npm install -g @anthropic-ai/claude-code` | Optional ceiling. Skip with `-SkipClaudeCli`. After install, run `claude login` in a fresh terminal. |
| 9 | Telegram credentials | Inline prompt | Writes `.env` with `SENTINEL_TELEGRAM_TOKEN` + `SENTINEL_TELEGRAM_USER_ID`. Skip with `-NoTelegram`. |
| 10 | KB seed + smoke test | `tools/preload_kb.py` + `OllamaClient` ping | Seeds 39 curated patterns, then runs a 1-token Qwen generate. |

Total time on a fresh PC: ~5-15 min, dominated by the model pulls.

---

## What bootstrap.ps1 Does

Three source modes. Pick on the command line, or run with no args
for an interactive menu.

```powershell
# Git clone (auto-installs Git via winget if missing)
.\bootstrap.ps1 -RepoUrl https://github.com/you/sentinel.git

# Local zip — auto-flattens single-top-dir layouts
.\bootstrap.ps1 -ZipPath C:\Users\You\Downloads\sentinel.zip

# Existing folder (e.g. external drive) — robocopy /MIR with excludes
.\bootstrap.ps1 -SourceDir D:\old-pc\sentinel

# Override default install path ($HOME\sentinel)
.\bootstrap.ps1 -RepoUrl ... -TargetDir D:\sentinel

# Fetch only, don't run setup.ps1
.\bootstrap.ps1 -RepoUrl ... -SkipSetup
```

After fetch, `bootstrap.ps1` chains into the target's `setup.ps1`
automatically. Use `-SkipSetup` to stop after fetch.

**Folder mode excludes** (when robocopy mirroring from an existing
install): `__pycache__/`, `.git/`, `logs/`, `backups/`, `.venv/`,
`*.pyc`, `*.db`, `*.db-shm`, `*.db-wal`. So the new PC starts clean
without your old run-time state. Bring `knowledge.db` separately if
you want to carry over /code learnings.

---

## What Sentinel-Setup.exe Does

A **self-extracting** installer — single `.exe` that contains the
entire repo source (zip + base64-encoded inside the script, compiled
by [ps2exe](https://github.com/MScholtes/PS2EXE)). On a fresh PC
with no internet for Git access, you only need to copy this one file.

Sequence on double-click:
1. Banner + brief intro shown.
2. Asks where to install (default `$HOME\sentinel`).
3. Refuses to clobber a non-empty target without `yes` confirmation.
4. Decodes embedded bundle → temp zip → `Expand-Archive` → target dir.
5. Chains into `setup.ps1` to install Python / Ollama / models / etc.

**Size:** ~1.2 MB. Composition:
- ~0.85 MB compressed source zip (254 files)
- ~1.13 MB after base64 inflation (33%), split into ~300 array chunks
- ~0.4 MB ps2exe .NET wrapper overhead

**Why chunked array (not a single here-string):** Windows PowerShell 5.1's
parser fails on a single-quoted here-string with >~1 MB body — silently
treats subsequent lines as code, producing `Missing expression after
unary operator '+'` errors when base64 lines start with `+`. The build
splits the base64 into 4000-char chunks stored in a `string[]` array,
joined at runtime with `-join ''`. Round-trip verified.

**What's NOT bundled** (downloads at runtime):
- Python 3.12 (~30 MB) — winget
- Ollama runtime (~150 MB) — winget
- Qwen models (~3 GB) — `ollama pull`
- Node.js + Claude CLI (~50 MB) — winget + npm
- Python packages (~20 MB) — pip

**Caveats:**
- **SmartScreen warning** — without a code-signing cert (~$300/yr from
  DigiCert / Sectigo), Windows shows "Unrecognized publisher" on first
  run. User clicks "More info" → "Run anyway". Unavoidable.
- **Antivirus** — ps2exe wrappers occasionally trip Defender heuristics.
  Set up a developer exception or distribute `bootstrap.ps1` instead
  if you hit this.
- **Source freezes at build time** — the .exe contains source as it was
  when you last ran `build-installer.ps1`. After every meaningful repo
  change, rebuild:
  ```powershell
  .\build-installer.ps1
  ```
  Build time: ~30 sec for the 2.4 MB output.

### When to use the self-extracting .exe vs bootstrap.ps1

| Scenario | Use |
|---|---|
| Target PC has no internet (truly offline install) | `Sentinel-Setup.exe` (everything you need is in the file, except the runtime downloads which you can do offline by pre-pulling Python/Ollama/models on another machine) |
| You want exactly one file to copy | `Sentinel-Setup.exe` |
| Repo lives on GitHub, target has internet | `bootstrap.ps1 -RepoUrl ...` (always pulls latest, no rebuild after every commit) |
| Source on USB / external drive | `bootstrap.ps1 -SourceDir D:\sentinel` |

---

## Telegram Credentials

`setup.ps1` step 9 prompts for these, with inline instructions. If you
need them ahead of time:

### Bot token (from @BotFather)
1. Open Telegram, search **@BotFather**, start a chat.
2. Send `/newbot`, follow prompts for the bot's name and username.
3. BotFather replies with a token like `1234567890:AABBCC...`.
4. Paste into setup.ps1 prompt, **or** save manually:
   ```
   setx SENTINEL_TELEGRAM_TOKEN "1234567890:AABB..."
   ```

### Your user ID (from @userinfobot)
1. In Telegram, search **@userinfobot**, send `/start`.
2. It replies with `Id: <number>`.
3. Paste into setup.ps1 prompt, **or** save manually:
   ```
   setx SENTINEL_TELEGRAM_USER_ID "123456789"
   ```

Only the user IDs you list are authorized to use the bot. The token
is stored in `.env` (UTF-8) at the repo root. **Don't commit `.env`** —
it's already in `.gitignore`.

---

## What's Carried Over vs Auto-Created

When moving to a new PC, **bring**:

| Item | Why |
|---|---|
| The repo (`setup.ps1`, `bootstrap.ps1`, `core/`, `skills/`, `agents/`, `interfaces/`, `tools/`, `tests/`, `requirements.txt`, `Modelfile.brain`, `workspace/persona/*.md`) | Source code, configs, persona files. |
| `knowledge.db` (optional) | /code learnings + Phase-15g curated KB. New PC will reseed via `preload_kb.py` if missing. |
| Edited `workspace/persona/*.md` (optional) | Carry over your customisations of `PROMPT_BRIEF.md`, `QWENCODER.md`, `CODE_TIERS.md`. |

**Don't bring** (auto-creates or auto-downloads):
- `sentinel.db`, `memory.db` (auto-init on `database.init_db()`)
- Ollama models (`setup.ps1` pulls them)
- `__pycache__/`, `logs/`, `backups/`
- `Sentinel-Setup.exe` (rebuild from `build-installer.ps1`)
- `.env` (regenerate via `setup.ps1` step 9)

---

## Startup Banner

`main.py` prints a short intro on every boot, before any error. If
the bot fails to start because of a missing token, the banner shows
first so you have context, and the error includes the exact `setx`
command to fix it.

```
+============================================================+
|   SENTINEL  -  local agent framework                       |
|   Backends: Ollama (Qwen 3 1.7B + Coder 3B) + Claude CLI   |
|   Interface: Telegram bot                                  |
+============================================================+

Quick reference (Telegram):
  /help              list all commands
  /dashboard         live system health snapshot
  /gwenask <idea>    have local Qwen author a recipe
  /gwen <recipe>     execute a literal recipe
  /code <problem>    Qwen with Claude-CLI ceiling
  /commit            commit working-tree changes

Docs:  CLAUDE.md (architecture)  |  PHASES.md (change log)
Logs:  logs/sentinel.jsonl       |  Health: 127.0.0.1:18700/health
```

The banner is plain ASCII so it renders identically in
PowerShell, cmd, Windows Terminal, or piped to a file.

---

## After Setup Finishes

```powershell
# 1. Open a NEW terminal so updated PATH + env vars take effect.

# 2. Authenticate Claude CLI (if installed):
claude login

# 3. Launch the bot:
py -3.12 main.py
```

In Telegram, message your bot:
- `/help` — list all commands
- `/dashboard` — system health snapshot
- `/gwenask build me a stopwatch` — local Qwen authors a recipe
- Paste the recipe back as `/gwen STEP 1: ...` to build the .exe

Apps land at `C:\Users\<You>\Desktop\Sentinel-Demos\apps\<Name>_Setup.exe`.

---

## Troubleshooting

### "winget not found"
Windows 10 versions before 1809 don't have winget. Either upgrade
Windows or install Python and Ollama manually, then re-run
`setup.ps1` — the steps that detect existing installs will skip
themselves.

### "Python 3.12 installed but `py -3.12` says 'not found'"
PATH wasn't refreshed in the current terminal. **Open a new
terminal** and re-run.

### "ollama list" hangs
Ollama service didn't start. `Get-Service Ollama` to check status.
`Restart-Service Ollama`, wait 5 s, retry.

### Smoke test fails with `LLMError: Model qwen2.5-coder:3b not available`
Step 4 didn't complete. Re-run `setup.ps1` — step 4 retries the
pull. Check disk space (need ~3 GB free).

### Bot starts but Telegram says "Unauthorized"
Wrong token in `.env`. Verify by visiting
`https://api.telegram.org/bot<TOKEN>/getMe` — should return
`{"ok": true, "result": {...}}`. If `ok: false`, the token is bad —
get a new one from @BotFather.

### Bot starts but ignores your messages
`SENTINEL_TELEGRAM_USER_ID` doesn't match your actual ID. Re-check
with @userinfobot, update `.env` (or run `setx`), restart bot.

### SmartScreen blocks `Sentinel-Setup.exe`
Click "More info" → "Run anyway". Without code-signing this is
unavoidable. If you want to remove the warning permanently, buy a
code-signing cert (DigiCert, Sectigo, ~$300/yr) and re-run
`build-installer.ps1` with the cert.

### Defender flags `Sentinel-Setup.exe` as suspicious
ps2exe-built `.exes` sometimes trip heuristics. Add a Defender
exclusion for the file, or distribute `bootstrap.ps1` directly
instead.

---

## Cleanup / Uninstall

There's no formal uninstaller. To remove Sentinel:

```powershell
# 1. Stop the bot (Ctrl+C in its terminal, or)
Get-Process py | Where-Object { $_.CommandLine -like '*main.py*' } | Stop-Process

# 2. Remove the repo
Remove-Item -Recurse -Force C:\path\to\sentinel

# 3. Optionally remove Ollama and pulled models
ollama rm qwen3:1.7b qwen2.5-coder:3b sentinel-brain
winget uninstall Ollama.Ollama

# 4. Remove env vars (if you want a complete cleanup)
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", $null, "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", $null, "User")
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", $null, "User")
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", $null, "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", $null, "User")
[Environment]::SetEnvironmentVariable("SENTINEL_TELEGRAM_TOKEN", $null, "User")
[Environment]::SetEnvironmentVariable("SENTINEL_TELEGRAM_USER_ID", $null, "User")
```

Python 3.12, Node.js, and Claude CLI stay installed (they're useful
for plenty of other things) — uninstall manually via `winget
uninstall ...` or Settings → Apps if you want them gone.
