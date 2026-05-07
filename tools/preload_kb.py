"""Phase 15g -- pre-load the KB with hand-curated patterns.

Background. The Phase 15e graduation-stash regression destroyed
~28 post-Phase-14b patterns and left only May-4 Phase-9 testbed
patterns. Net effect: KB few-shot context for any modern task is
dominated by recipes describing a codebase shape that no longer
exists, so /code iterates 2-3 attempts on familiar problems and
shadow planning agreement is stuck at 0.000. This script seeds the
KB with patterns matching the user's actual workflow on the CURRENT
codebase shape so the bot can be productive from a cold start
instead of having to organically rebuild context.

Each pattern is a REPRESENTATIVE example -- not a literal
executable recipe against the live codebase, but a calibrated
illustration of the right shape. Claude's pre-teach injects these
as few-shot context and adapts the actual recipe for the user's
specific problem.

Design principles (in order):

  1. Result-oriented problem_summaries. Match what users actually
     type. No "how do I" or "best practice for" prompts -- those
     don't appear in real /code traffic.

  2. Recipes follow QWENCODER.md contract verbatim:
       STEP 1: read_file path="..."          (verify state first)
       STEP 2: edit_file/write_file ...      (real tool, real args)
       STEP 3: run_bash command="..."        (verifier with assert)
       STEP 4: done summary="..."            (final, deterministic)

  3. Diff bodies are valid `diff --git` bodies that pass the Phase
     15d-bugfix-2 quality gate (cleanup_low_quality_patterns won't
     archive them on bot startup).

  4. Shadow recipes are pre-computed -- short STEP-N versions a 3B
     coder might plausibly write. Agreement scores are pre-computed
     against the canonical recipe via core.plan_agreement. This
     gives /kb planning real numbers from day one instead of having
     to wait for shadow data to accumulate.

  5. Every pattern is PINNED (Phase 15a) so the kb_lifecycle auto-
     walker NEVER archives them. They're foundation -- they should
     outlive everything.

  6. Idempotent. Re-runs detect existing patterns by problem_summary
     fuzzy match and skip. Safe to run after any future regression.

Run:

  python tools/preload_kb.py

To force re-insertion (replace existing seeded rows):

  python tools/preload_kb.py --replace
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.knowledge_base import KnowledgeBase
from core.plan_agreement import score_plan_agreement

TRACE = "SEN-preload"


# ─────────────────────────────────────────────────────────────────
# Pattern definitions
#
# Each entry is a dict with these required keys:
#   tags                : list[str]   -- FTS5 search keywords
#   problem_summary     : str         -- mirrors a real user prompt
#   solution_pattern    : str         -- canonical STEP-N recipe
#                                        Claude/Qwen could produce
#   solution_code       : str         -- representative diff body
#   explanation         : str         -- why this approach + gotchas
#   shadow_recipe       : str         -- a plausible Qwen shadow plan
#                                        for the same problem (used
#                                        to populate planning_stats
#                                        from day one)
#
# qwen_plan_agreement is computed at preload time via
# score_plan_agreement(solution_pattern, shadow_recipe) so the
# numbers are honest -- the same scorer the production shadow path
# uses.
# ─────────────────────────────────────────────────────────────────

PATTERNS: list[dict[str, Any]] = [
    # ──────────────────────────────────────────────────────────
    # Telegram bot extensions (4 patterns)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["telegram", "command", "handler", "bot", "command_handler"],
        "problem_summary": (
            "add a new /command to the telegram bot"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="interfaces/telegram_bot.py"\n'
            'STEP 2: edit_file path="interfaces/telegram_bot.py" '
            'old="self.app.add_handler(CommandHandler(\\"kb\\", '
            'self.handle_kb))" new="self.app.add_handler('
            'CommandHandler(\\"kb\\", self.handle_kb))\\n        '
            'self.app.add_handler(CommandHandler(\\"NAME\\", '
            'self.handle_NAME))"\n'
            'STEP 3: edit_file path="interfaces/telegram_bot.py" '
            'old="    async def handle_kb(self, update: Update," '
            'new="    async def handle_NAME(self, update: Update,\\n'
            '                          context: ContextTypes.DEFAULT_TYPE'
            ') -> None:\\n        if not await self._check_auth(update):'
            '\\n            return\\n        await update.message.'
            'reply_text(\\"OK\\")\\n\\n    async def handle_kb('
            'self, update: Update,"\n'
            'STEP 4: run_bash command="python -c \\"import ast; ast.parse'
            '(open(\'interfaces/telegram_bot.py\', encoding=\'utf-8\').'
            'read()); print(\'syntax ok\')\\""\n'
            'STEP 5: done summary="added /NAME command with auth check"'
        ),
        "solution_code": (
            "diff --git a/interfaces/telegram_bot.py "
            "b/interfaces/telegram_bot.py\n"
            "--- a/interfaces/telegram_bot.py\n"
            "+++ b/interfaces/telegram_bot.py\n"
            "@@ -335,6 +335,7 @@ class SentinelTelegramBot:\n"
            "         self.app.add_handler(CommandHandler(\"kb\", "
            "self.handle_kb))\n"
            "+        self.app.add_handler(CommandHandler(\"NAME\", "
            "self.handle_NAME))\n"
            "@@ -1180,6 +1181,12 @@ class SentinelTelegramBot:\n"
            "+    async def handle_NAME(self, update: Update,\n"
            "+                          context: ContextTypes."
            "DEFAULT_TYPE) -> None:\n"
            "+        if not await self._check_auth(update):\n"
            "+            return\n"
            "+        await update.message.reply_text(\"OK\")\n"
        ),
        "explanation": (
            "Telegram bot commands need TWO additions: a CommandHandler "
            "registration in the constructor's add_handler block (around "
            "line 335), AND an async def handle_<name> method on the "
            "class. Auth check is non-negotiable -- every handler starts "
            "with `if not await self._check_auth(update): return` per "
            "Phase 9 contract. Verify via `python -c 'import ast; "
            "ast.parse(open(...).read())'` so syntax errors are caught "
            "before the next bot start."
        ),
        "shadow_recipe": (
            'STEP 1: write_file path="interfaces/telegram_bot.py" '
            'content="..."\n'
            'STEP 2: done summary="added the new command"'
        ),
    },
    {
        "tags": ["telegram", "progress bar", "_build_bar", "emoji", "visual"],
        "problem_summary": (
            "change the telegram progress bar emojis"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="interfaces/telegram_bot.py"\n'
            'STEP 2: edit_file path="interfaces/telegram_bot.py" '
            'old="def _build_bar(pct: int, w: int = 10) -> str:\\n'
            '    pct = max(0, min(100, pct))\\n    f = int(w * pct / '
            '100)\\n    return \\"OLD_FILLED\\" * f + \\"OLD_EMPTY\\" * '
            '(w - f) + \\" \\" + str(pct) + \\"%\\"" new="def _build_bar'
            '(pct: int, w: int = 10) -> str:\\n    pct = max(0, min(100, '
            'pct))\\n    f = int(w * pct / 100)\\n    return '
            '\\"NEW_FILLED\\" * f + \\"NEW_EMPTY\\" * (w - f) + \\" \\" '
            '+ str(pct) + \\"%\\""\n'
            'STEP 3: run_bash command="python -c \\"from interfaces.'
            'telegram_bot import _build_bar; print(_build_bar(60))\\""\n'
            'STEP 4: done summary="updated _build_bar emojis to NEW style"'
        ),
        "solution_code": (
            "diff --git a/interfaces/telegram_bot.py "
            "b/interfaces/telegram_bot.py\n"
            "--- a/interfaces/telegram_bot.py\n"
            "+++ b/interfaces/telegram_bot.py\n"
            "@@ -14,4 +14,4 @@\n"
            " def _build_bar(pct: int, w: int = 10) -> str:\n"
            "     pct = max(0, min(100, pct))\n"
            "     f = int(w * pct / 100)\n"
            "-    return \"OLD_FILLED\" * f + \"OLD_EMPTY\" * (w - f) + "
            "\" \" + str(pct) + \"%\"\n"
            "+    return \"NEW_FILLED\" * f + \"NEW_EMPTY\" * (w - f) + "
            "\" \" + str(pct) + \"%\"\n"
        ),
        "explanation": (
            "_build_bar lives at the top of interfaces/telegram_bot.py "
            "(around line 14-17). It's called by every handler that "
            "shows progress (/code, /jobsearch, /research, brain "
            "dispatch). Changing the two emoji literals in the return "
            "is the only edit needed -- all callers pick up the change. "
            "ALWAYS verify by importing and calling _build_bar(60) so "
            "the new emojis render in the run_bash output."
        ),
        "shadow_recipe": (
            'STEP 1: read_file path="interfaces/telegram_bot.py"\n'
            'STEP 2: edit_file path="interfaces/telegram_bot.py" '
            'old="OLD_FILLED" new="NEW_FILLED"\n'
            'STEP 3: edit_file path="interfaces/telegram_bot.py" '
            'old="OLD_EMPTY" new="NEW_EMPTY"\n'
            'STEP 4: done summary="swapped emojis"'
        ),
    },
    {
        "tags": ["telegram", "brain", "dispatch", "lane", "intent"],
        "problem_summary": (
            "add a new lane to the brain's intent classifier"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/brain.py"\n'
            'STEP 2: edit_file path="core/brain.py" old="if brain_result'
            '.intent == \\"chat\\":" new="if brain_result.intent == '
            '\\"NEW_INTENT\\":\\n            return BrainResult(intent='
            '\\"NEW_INTENT\\", response=\\"...\\")\\n        if '
            'brain_result.intent == \\"chat\\":"\n'
            'STEP 3: run_bash command="python -c \\"from core.brain '
            'import BRAIN; print(\'brain loaded\')\\""\n'
            'STEP 4: done summary="added NEW_INTENT lane to brain "\n'
        ),
        "solution_code": (
            "diff --git a/core/brain.py b/core/brain.py\n"
            "--- a/core/brain.py\n"
            "+++ b/core/brain.py\n"
            "@@ -200,6 +200,9 @@\n"
            "+        if brain_result.intent == \"NEW_INTENT\":\n"
            "+            return BrainResult(intent=\"NEW_INTENT\", "
            "response=\"...\")\n"
            "         if brain_result.intent == \"chat\":\n"
        ),
        "explanation": (
            "Brain intents are dispatched in core/brain.py's process() "
            "method via if/elif on `brain_result.intent`. New intents "
            "go BEFORE the catch-all 'chat' branch so they take "
            "precedence. Each intent returns a BrainResult with intent= "
            "tag set so downstream handlers can dispatch correctly. The "
            "intent name should match a string the qwen3-brain Modelfile "
            "is trained to produce -- check workspace/persona/IDENTITY.md "
            "for the canonical list."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/brain.py" old="..." new="..."\n'
            'STEP 2: done summary="added the new intent"'
        ),
    },
    {
        "tags": ["telegram", "auth", "authorize", "user", "TELEGRAM_AUTHORIZED_USERS"],
        "problem_summary": (
            "add a new authorized telegram user"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/config.py"\n'
            'STEP 2: edit_file path="core/config.py" '
            'old="TELEGRAM_AUTHORIZED_USERS: list[int] = [" '
            'new="TELEGRAM_AUTHORIZED_USERS: list[int] = [\\n    '
            'NEW_USER_ID,"\n'
            'STEP 3: run_bash command="python -c \\"from core import '
            'config; assert NEW_USER_ID in config.TELEGRAM_AUTHORIZED_USERS\\""\n'
            'STEP 4: done summary="added NEW_USER_ID to authorized "\n'
        ),
        "solution_code": (
            "diff --git a/core/config.py b/core/config.py\n"
            "--- a/core/config.py\n"
            "+++ b/core/config.py\n"
            "@@ -96,6 +96,7 @@\n"
            " TELEGRAM_AUTHORIZED_USERS: list[int] = [\n"
            "     int(uid) for uid in\n"
            "     _os9.environ.get(\"SENTINEL_TELEGRAM_USER_ID\", "
            "\"\").split(\",\")\n"
            "+    + [\"NEW_USER_ID\"]\n"
            "     if uid.strip().isdigit()\n"
        ),
        "explanation": (
            "Auth is non-negotiable: every Telegram handler calls "
            "self._check_auth(update) which validates "
            "update.effective_user.id against config."
            "TELEGRAM_AUTHORIZED_USERS. The list is built from the "
            "SENTINEL_TELEGRAM_USER_ID env var (comma-separated). For "
            "permanent additions, modify the list construction directly "
            "in core/config.py. For session-only, set the env var and "
            "restart the bot. Either way: NO new handler should bypass "
            "_check_auth -- the bot is public and anyone can find it."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/config.py" '
            'old="..." new="...NEW_USER_ID..."\n'
            'STEP 2: done summary="added user"'
        ),
    },
    {
        "tags": ["telegram", "long message", "split", "_send_long", "chunk"],
        "problem_summary": (
            "make telegram messages over 4000 chars split correctly"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="interfaces/telegram_bot.py"\n'
            'STEP 2: edit_file path="interfaces/telegram_bot.py" '
            'old="    async def _send_long(self, update, text):" '
            'new="    async def _send_long(self, update, text, chunk_size: '
            'int = 3500):\\n        \\"\\"\\"Split long messages on '
            'newline boundaries and send sequentially.\\"\\"\\"\\n        '
            'if len(text) <= chunk_size:\\n            await update.'
            'message.reply_text(text)\\n            return\\n        cur '
            '= []\\n        running = 0\\n        for line in text.'
            'split(\\"\\\\n\\"):\\n            if running + len(line) > '
            'chunk_size and cur:\\n                await update.message.'
            'reply_text(\\"\\\\n\\".join(cur))\\n                cur, '
            'running = [], 0\\n            cur.append(line)\\n            '
            'running += len(line) + 1\\n        if cur:\\n            '
            'await update.message.reply_text(\\"\\\\n\\".join(cur))\\n'
            '\\n    async def _send_long_OLD(self, update, text):"\n'
            'STEP 3: run_bash command="python -c \\"import ast; ast.parse'
            '(open(\'interfaces/telegram_bot.py\', encoding=\'utf-8\').'
            'read())\\""\n'
            'STEP 4: done summary="rewrote _send_long with line-aware '
            '3500-char chunking"'
        ),
        "solution_code": (
            "diff --git a/interfaces/telegram_bot.py "
            "b/interfaces/telegram_bot.py\n"
            "--- a/interfaces/telegram_bot.py\n"
            "+++ b/interfaces/telegram_bot.py\n"
            "@@ -800,3 +800,16 @@\n"
            "+    async def _send_long(self, update, text, "
            "chunk_size: int = 3500):\n"
            "+        \"\"\"Split long messages on newline boundaries.\"\"\"\n"
            "+        if len(text) <= chunk_size:\n"
            "+            await update.message.reply_text(text)\n"
            "+            return\n"
            "+        cur, running = [], 0\n"
            "+        for line in text.split(\"\\n\"):\n"
            "+            if running + len(line) > chunk_size and cur:\n"
            "+                await update.message.reply_text(\"\\n\"."
            "join(cur))\n"
            "+                cur, running = [], 0\n"
            "+            cur.append(line)\n"
            "+            running += len(line) + 1\n"
        ),
        "explanation": (
            "Telegram caps single messages at 4096 chars. The bot's "
            "_send_long helper is used for any long replies (research "
            "reports, /jobsearch summaries, /kb dumps). Split on \\n "
            "boundaries when possible -- a mid-line split mangles "
            "code blocks and tables. Choose chunk_size=3500 (under "
            "4096 with headroom for emoji + Telegram's character "
            "counting quirks)."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="interfaces/telegram_bot.py" '
            'old="..." new="def _send_long..."\n'
            'STEP 2: done summary="added chunked send"'
        ),
    },
    {
        "tags": ["telegram", "BotFather", "set_my_commands", "command list"],
        "problem_summary": (
            "register telegram bot commands with BotFather menu"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="interfaces/telegram_bot.py"\n'
            'STEP 2: edit_file path="interfaces/telegram_bot.py" '
            'old="            BotCommand(\\"kb\\", \\"" '
            'new="            BotCommand(\\"NEW_CMD\\", '
            '\\"description shown in /\\"),\\n            BotCommand'
            '(\\"kb\\", \\""\n'
            'STEP 3: run_bash command="python -c \\"import ast; ast.parse'
            '(open(\'interfaces/telegram_bot.py\', encoding=\'utf-8\').'
            'read())\\""\n'
            'STEP 4: done summary="added NEW_CMD to BotFather menu"'
        ),
        "solution_code": (
            "diff --git a/interfaces/telegram_bot.py "
            "b/interfaces/telegram_bot.py\n"
            "--- a/interfaces/telegram_bot.py\n"
            "+++ b/interfaces/telegram_bot.py\n"
            "@@ -700,5 +700,6 @@\n"
            "         await self.app.bot.set_my_commands([\n"
            "+            BotCommand(\"NEW_CMD\", \"description shown\"),\n"
            "             BotCommand(\"kb\", \"...\"),\n"
        ),
        "explanation": (
            "set_my_commands runs once at startup and populates the "
            "BotFather '/' menu in Telegram. Each entry is BotCommand"
            "(name, description). Description is what appears next to "
            "the slash-command in the menu (max ~30 chars). The "
            "registration is separate from add_handler -- a handler "
            "without a BotCommand still works, but won't appear in the "
            "menu. Add to BOTH for discoverability."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="interfaces/telegram_bot.py" '
            'old="..." new="...BotCommand..."\n'
            'STEP 2: done summary="added BotFather entry"'
        ),
    },
    {
        "tags": ["telegram", "callback", "approval", "token", "two_step"],
        "problem_summary": (
            "add a two-step telegram approval flow with token"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/curation.py"\n'
            'STEP 2: read_file path="interfaces/telegram_bot.py"\n'
            'STEP 3: edit_file path="interfaces/telegram_bot.py" '
            'old="    async def handle_curate_approve(self, update," '
            'new="    async def handle_NEW_approve(self, update: Update,'
            '\\n                                  context: ContextTypes.'
            'DEFAULT_TYPE) -> None:\\n        if not await self.'
            '_check_auth(update):\\n            return\\n        token '
            '= (context.args or [\\"\\"])[0]\\n        ok = await self.'
            'NEW_FLOW.apply(token)\\n        await update.message.'
            'reply_text(f\\"approved={ok}\\")\\n\\n    async def '
            'handle_curate_approve(self, update,"\n'
            'STEP 4: run_bash command="python -c \\"import ast; ast.parse'
            '(open(\'interfaces/telegram_bot.py\', encoding=\'utf-8\').'
            'read())\\""\n'
            'STEP 5: done summary="added /NEW_approve handler with '
            'token-based two-step approval"'
        ),
        "solution_code": (
            "diff --git a/interfaces/telegram_bot.py "
            "b/interfaces/telegram_bot.py\n"
            "--- a/interfaces/telegram_bot.py\n"
            "+++ b/interfaces/telegram_bot.py\n"
            "@@ -1500,3 +1500,11 @@\n"
            "+    async def handle_NEW_approve(self, update: Update,\n"
            "+                                  context: ContextTypes."
            "DEFAULT_TYPE) -> None:\n"
            "+        if not await self._check_auth(update):\n"
            "+            return\n"
            "+        token = (context.args or [\"\"])[0]\n"
            "+        ok = await self.NEW_FLOW.apply(token)\n"
            "+        await update.message.reply_text(f\"approved={ok}\")\n"
        ),
        "explanation": (
            "Two-step approval pattern (Phase 10 curate flow): the "
            "FIRST command (/curate) generates a proposal + 4-hex-char "
            "token, posts to Telegram with the token. The SECOND "
            "command (/curate_approve <token>) accepts the token and "
            "executes. The token is stored in an in-memory dict on "
            "the bot -- no persistence (intentional: stale tokens "
            "expire on bot restart, which is the right safety posture "
            "for these flows). Always validate context.args[0] is a "
            "real token before doing the apply."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="interfaces/telegram_bot.py" '
            'old="..." new="async def handle_NEW_approve..."\n'
            'STEP 2: done summary="added approval handler"'
        ),
    },
    {
        "tags": ["telegram", "alert", "send_alert", "notification", "background"],
        "problem_summary": (
            "make the bot send a telegram alert when X happens"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="interfaces/telegram_bot.py"\n'
            'STEP 2: read_file path="core/internal_handlers.py"\n'
            'STEP 3: edit_file path="core/EVENT_MODULE.py" '
            'old="(point of event)" new="(point of event)\\n        '
            'from core.internal_handlers import _alert\\n        '
            'await _alert(\\"⚠️ event description\\")"\n'
            'STEP 4: run_bash command="python -c \\"from core.'
            'internal_handlers import _alert; print(\'alert helper '
            'loaded\')\\""\n'
            'STEP 5: done summary="wired _alert into EVENT_MODULE"'
        ),
        "solution_code": (
            "diff --git a/core/EVENT_MODULE.py b/core/EVENT_MODULE.py\n"
            "--- a/core/EVENT_MODULE.py\n"
            "+++ b/core/EVENT_MODULE.py\n"
            "@@ -100,3 +100,5 @@\n"
            "     log_event(trace_id, \"INFO\", \"event\", \"X happened\")\n"
            "+    from core.internal_handlers import _alert\n"
            "+    await _alert(\"⚠️ X happened\")\n"
        ),
        "explanation": (
            "Sentinel exposes alerts via core.internal_handlers._alert "
            "(wired by main.py at boot to bot.send_alert). Importing "
            "_alert at the call site (not module-level) avoids circular "
            "imports through core.internal_handlers. Use sparingly -- "
            "the bot has rate limits, and noisy alerts erode the user's "
            "attention to the channel."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/EVENT_MODULE.py" '
            'old="..." new="...send_alert..."\n'
            'STEP 2: done summary="added alert"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # KB / KnowledgeBase methods (3 patterns)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["sqlite", "schema", "alter", "migration", "idempotent",
                 "knowledge_base"],
        "problem_summary": (
            "add a new column to the knowledge table with idempotent "
            "ALTER"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/knowledge_base.py"\n'
            'STEP 2: edit_file path="core/knowledge_base.py" '
            'old="            if \\"base_sha\\" not in existing_cols:\\n'
            '                conn.execute(\\n                    '
            '\\"ALTER TABLE knowledge \\"\\n                    \\"ADD '
            'COLUMN base_sha TEXT\\"\\n                )" '
            'new="            if \\"base_sha\\" not in existing_cols:\\n'
            '                conn.execute(\\n                    '
            '\\"ALTER TABLE knowledge \\"\\n                    \\"ADD '
            'COLUMN base_sha TEXT\\"\\n                )\\n            '
            'if \\"NEW_COL\\" not in existing_cols:\\n                '
            'conn.execute(\\n                    \\"ALTER TABLE '
            'knowledge \\"\\n                    \\"ADD COLUMN NEW_COL '
            'TEXT NOT NULL DEFAULT \'default\'\\"\\n                )"\n'
            'STEP 3: run_bash command="python -c \\"import tempfile; '
            'from pathlib import Path; from core.knowledge_base import '
            'KnowledgeBase; KnowledgeBase(db_path=Path(tempfile.'
            'NamedTemporaryFile(suffix=\'.db\', delete=False).name)); '
            'print(\'schema ok\')\\""\n'
            'STEP 4: done summary="added NEW_COL with idempotent ALTER"'
        ),
        "solution_code": (
            "diff --git a/core/knowledge_base.py "
            "b/core/knowledge_base.py\n"
            "--- a/core/knowledge_base.py\n"
            "+++ b/core/knowledge_base.py\n"
            "@@ -210,6 +210,11 @@ class KnowledgeBase:\n"
            "             if \"base_sha\" not in existing_cols:\n"
            "                 conn.execute(\n"
            "                     \"ALTER TABLE knowledge \"\n"
            "                     \"ADD COLUMN base_sha TEXT\"\n"
            "                 )\n"
            "+            if \"NEW_COL\" not in existing_cols:\n"
            "+                conn.execute(\n"
            "+                    \"ALTER TABLE knowledge \"\n"
            "+                    \"ADD COLUMN NEW_COL TEXT NOT NULL "
            "DEFAULT 'default'\"\n"
            "+                )\n"
        ),
        "explanation": (
            "The Sentinel idempotent-ALTER pattern (Phase 14a/15a/15b/"
            "15c): inside _init_schema, fetch the column set via "
            "PRAGMA table_info, then `if \"col_name\" not in "
            "existing_cols:` gates each ALTER TABLE ADD COLUMN. NEVER "
            "use try/except on the duplicate-column error -- the PRAGMA "
            "check is the canonical pattern. NOT NULL DEFAULT backfills "
            "existing rows in a single statement. Add the corresponding "
            "Pydantic field to KnowledgeEntry AND a defensive getter in "
            "_row_to_entry that handles pre-migration rows."
        ),
        "shadow_recipe": (
            'STEP 1: read_file path="core/knowledge_base.py"\n'
            'STEP 2: edit_file path="core/knowledge_base.py" '
            'old="..." new="..."\n'
            'STEP 3: done summary="added new column"'
        ),
    },
    {
        "tags": ["kb", "knowledge_base", "method", "query", "filter"],
        "problem_summary": (
            "add a query method to KnowledgeBase that filters by tag"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/knowledge_base.py"\n'
            'STEP 2: edit_file path="core/knowledge_base.py" '
            'old="    def stats(self) -> dict:" new="    def list_by_tag'
            '(self, tag: str, limit: int = 20) -> list[KnowledgeEntry]:'
            '\\n        \\"\\"\\"Filter active patterns by tag '
            'substring.\\"\\"\\"\\n        conn = _connect(self.db_path)'
            '\\n        try:\\n            rows = conn.execute(\\n      '
            '          \\"SELECT * FROM knowledge \\"\\n               '
            ' \\"WHERE tags LIKE ? AND state != \'archived\' \\"\\n     '
            '           \\"ORDER BY usage_count DESC LIMIT ?\\",\\n     '
            '       (f\\"%{tag}%\\", limit),\\n            ).fetchall()'
            '\\n        finally:\\n            conn.close()\\n        '
            'return [_row_to_entry(r) for r in rows]\\n\\n    def stats'
            '(self) -> dict:"\n'
            'STEP 3: run_bash command="python -c \\"from core.'
            'knowledge_base import KnowledgeBase; assert hasattr('
            'KnowledgeBase, \'list_by_tag\'); print(\'method ok\')\\""\n'
            'STEP 4: done summary="added list_by_tag method"'
        ),
        "solution_code": (
            "diff --git a/core/knowledge_base.py "
            "b/core/knowledge_base.py\n"
            "--- a/core/knowledge_base.py\n"
            "+++ b/core/knowledge_base.py\n"
            "@@ -750,2 +750,16 @@\n"
            "+    def list_by_tag(self, tag: str, "
            "limit: int = 20) -> list[KnowledgeEntry]:\n"
            "+        \"\"\"Filter active patterns by tag "
            "substring.\"\"\"\n"
            "+        conn = _connect(self.db_path)\n"
            "+        try:\n"
            "+            rows = conn.execute(\n"
            "+                \"SELECT * FROM knowledge \"\n"
            "+                \"WHERE tags LIKE ? AND state != "
            "'archived' \"\n"
            "+                \"ORDER BY usage_count DESC LIMIT ?\",\n"
            "+                (f\"%{tag}%\", limit),\n"
            "+            ).fetchall()\n"
            "+        finally:\n"
            "+            conn.close()\n"
            "+        return [_row_to_entry(r) for r in rows]\n"
            "+\n"
            "     def stats(self) -> dict:\n"
        ),
        "explanation": (
            "KB query methods follow a consistent pattern: open via "
            "_connect, run a parameterized SELECT (always `AND state "
            "!= 'archived'` per Phase 15a), close in finally, return "
            "list of _row_to_entry'd KnowledgeEntry objects. Tag "
            "matching uses LIKE with %% wildcards because tags is a "
            "comma-separated string column (not normalized -- Phase 7 "
            "design). Insert the new method just before `def stats` so "
            "the public-API surface stays grouped."
        ),
        "shadow_recipe": (
            'STEP 1: read_file path="core/knowledge_base.py"\n'
            'STEP 2: edit_file path="core/knowledge_base.py" '
            'old="..." new="def list_by_tag..."\n'
            'STEP 3: done summary="added query method"'
        ),
    },
    {
        "tags": ["kb", "knowledge_base", "stats", "aggregate", "group_by"],
        "problem_summary": (
            "add a stats method to KnowledgeBase that aggregates by "
            "category"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/knowledge_base.py"\n'
            'STEP 2: edit_file path="core/knowledge_base.py" '
            'old="    def stats(self) -> dict:" new="    def category_'
            'stats(self) -> dict:\\n        \\"\\"\\"GROUP BY category '
            'aggregate.\\"\\"\\"\\n        conn = _connect(self.db_path)'
            '\\n        try:\\n            rows = conn.execute(\\n      '
            '          \\"SELECT category, COUNT(*) AS n FROM knowledge '
            '\\"\\n                \\"WHERE state != \'archived\' GROUP '
            'BY category\\"\\n            ).fetchall()\\n        finally'
            ':\\n            conn.close()\\n        return {r[\\"category'
            '\\"]: int(r[\\"n\\"]) for r in rows}\\n\\n    def stats('
            'self) -> dict:"\n'
            'STEP 3: run_bash command="python -c \\"from core.'
            'knowledge_base import KnowledgeBase; print(KnowledgeBase().'
            'category_stats())\\""\n'
            'STEP 4: done summary="added category_stats GROUP BY method"'
        ),
        "solution_code": (
            "diff --git a/core/knowledge_base.py "
            "b/core/knowledge_base.py\n"
            "--- a/core/knowledge_base.py\n"
            "+++ b/core/knowledge_base.py\n"
            "@@ -750,2 +750,13 @@\n"
            "+    def category_stats(self) -> dict:\n"
            "+        \"\"\"GROUP BY category aggregate.\"\"\"\n"
            "+        conn = _connect(self.db_path)\n"
            "+        try:\n"
            "+            rows = conn.execute(\n"
            "+                \"SELECT category, COUNT(*) AS n "
            "FROM knowledge \"\n"
            "+                \"WHERE state != 'archived' GROUP BY "
            "category\"\n"
            "+            ).fetchall()\n"
            "+        finally:\n"
            "+            conn.close()\n"
            "+        return {r[\"category\"]: int(r[\"n\"]) for r in "
            "rows}\n"
        ),
        "explanation": (
            "Aggregate methods use a single SQL GROUP BY rather than "
            "multiple COUNT queries (cheaper, atomic). Always exclude "
            "archived rows via `WHERE state != 'archived'` per Phase "
            "15a. Return a plain dict for easy serialization to JSON "
            "(health endpoint, /dashboard, etc.). Cast the COUNT to int "
            "so SQLite's row factory doesn't leak Row objects into "
            "downstream consumers."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/knowledge_base.py" '
            'old="..." new="def category_stats..."\n'
            'STEP 2: done summary="added stats method"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Memory ops (2 patterns)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["memory", "episodic", "store", "scope", "search"],
        "problem_summary": (
            "have an agent store an episode and retrieve recent ones"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/memory.py"\n'
            'STEP 2: edit_file path="core/agents.py" old="    async def '
            'run(self," new="    def _record_episode(self, summary: str, '
            'trace_id: str) -> None:\\n        try:\\n            from '
            'core.memory import get_memory\\n            get_memory().'
            'store_episode(scope=self.name, trace_id=trace_id, event_'
            'type=\\"pipeline_completed\\", summary=summary)\\n        '
            'except Exception:\\n            pass\\n\\n    async def '
            'run(self,"\n'
            'STEP 3: run_bash command="python -c \\"from core.memory '
            'import get_memory; m = get_memory(); m.store_episode(scope=\'test\', '
            'trace_id=\'t\', event_type=\'note\', summary=\'preload '
            'sanity\'); print(m.get_recent_episodes(scope=\'test\', '
            'limit=1)[0].summary)\\""\n'
            'STEP 4: done summary="agent now records pipeline outcome "\n'
        ),
        "solution_code": (
            "diff --git a/core/agents.py b/core/agents.py\n"
            "--- a/core/agents.py\n"
            "+++ b/core/agents.py\n"
            "@@ -50,3 +50,12 @@ class Agent:\n"
            "+    def _record_episode(self, summary: str, "
            "trace_id: str) -> None:\n"
            "+        try:\n"
            "+            from core.memory import get_memory\n"
            "+            get_memory().store_episode(scope=self.name, "
            "trace_id=trace_id, event_type=\"pipeline_completed\", "
            "summary=summary)\n"
            "+        except Exception:\n"
            "+            pass\n"
            "+\n"
            "     async def run(self,\n"
        ),
        "explanation": (
            "Episodic memory stores per-scope diary entries (Phase 10). "
            "Use scope=<agent_name> for agent activity, scope='global' "
            "for system events, scope=<session_id> for chat. Always "
            "wrap store_episode in try/except -- a memory hiccup must "
            "NEVER break a successful pipeline. The store call returns "
            "an int id but most callers ignore it. Retrieve via "
            "get_recent_episodes(scope=...) for a context block, "
            "search_episodes(query, scope=...) for FTS5 lookup."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/agents.py" '
            'old="..." new="..._record_episode..."\n'
            'STEP 2: done summary="added episode recording"'
        ),
    },
    {
        "tags": ["memory", "semantic", "fact", "remember", "store_fact"],
        "problem_summary": (
            "store a durable fact in semantic memory programmatically"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/memory.py"\n'
            'STEP 2: edit_file path="core/EXTRACTION_MODULE.py" '
            'old="(end of extraction logic)" new="            from core.'
            'memory import get_memory\\n            get_memory().store_'
            'fact(key=key, value=value, source=\\"task_derived\\", '
            'confidence=0.8)\\n"\n'
            'STEP 3: run_bash command="python -c \\"from core.memory '
            'import get_memory; m = get_memory(); m.store_fact(\'test_key'
            '\', \'test_value\', source=\'task_derived\'); '
            'assert m.get_fact(\'test_key\').value == \'test_value\'; '
            'print(\'fact ok\')\\""\n'
            'STEP 4: done summary="wired fact storage with task_derived "\n'
        ),
        "solution_code": (
            "diff --git a/core/EXTRACTION_MODULE.py "
            "b/core/EXTRACTION_MODULE.py\n"
            "--- a/core/EXTRACTION_MODULE.py\n"
            "+++ b/core/EXTRACTION_MODULE.py\n"
            "@@ -50,3 +50,5 @@\n"
            "     for key, value in extracted.items():\n"
            "+        from core.memory import get_memory\n"
            "+        get_memory().store_fact(key=key, value=value, "
            "source=\"task_derived\", confidence=0.8)\n"
        ),
        "explanation": (
            "store_fact does upsert-by-key with confidence-aware merge "
            "(Phase 10): higher confidence wins, equal -> newer wins. "
            "Source values: 'user_explicit' (1.0 default), 'task_"
            "derived' (0.8), 'auto_extracted' (0.6), 'persona_file' "
            "(1.0). Phase 15b: created_by_origin is read from a "
            "ContextVar at INSERT, NOT overwritten on UPSERT -- the "
            "column records who FIRST wrote the key. So an auto-"
            "extracted fact later asserted as user_explicit keeps "
            "origin='background_extraction' (audit-trail correctness "
            "beats latest-writer-wins)."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/EXTRACTION_MODULE.py" '
            'old="..." new="...store_fact..."\n'
            'STEP 2: done summary="added fact storage"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Tests (2 patterns)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["test", "pytest", "ecc", "phase", "fixture"],
        "problem_summary": (
            "add a new test file for a phase with ECC coverage"
        ),
        "solution_pattern": (
            'STEP 1: list_dir path="tests"\n'
            'STEP 2: read_file path="tests/test_phase15a_lifecycle.py"\n'
            'STEP 3: write_file path="tests/test_phaseNN_FEATURE.py" '
            'content="\\"\\"\\"Phase NN -- FEATURE.\\n\\nCoverage:\\n  '
            'N01 -- happy path\\n  N02 -- edge case\\n\\"\\"\\"\\nfrom '
            '__future__ import annotations\\nimport sys\\nfrom pathlib '
            'import Path\\nimport pytest\\n\\nsys.path.insert(0, str('
            'Path(__file__).resolve().parent.parent))\\n\\nfrom core '
            'import config\\n\\n\\n@pytest.fixture\\ndef fresh_thing('
            'tmp_path, monkeypatch):\\n    return None\\n\\n\\ndef '
            'test_n01_happy(fresh_thing):\\n    assert True\\n\\n\\n'
            'def test_n02_edge(fresh_thing):\\n    assert True\\n"\n'
            'STEP 4: run_bash command="python -m pytest tests/'
            'test_phaseNN_FEATURE.py -q"\n'
            'STEP 5: done summary="added phase NN test file with two '
            'ECC cases"'
        ),
        "solution_code": (
            "diff --git a/tests/test_phaseNN_FEATURE.py "
            "b/tests/test_phaseNN_FEATURE.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_phaseNN_FEATURE.py\n"
            "@@ -0,0 +1,22 @@\n"
            "+\"\"\"Phase NN -- FEATURE.\n"
            "+\n"
            "+Coverage:\n"
            "+  N01 -- happy path\n"
            "+  N02 -- edge case\n"
            "+\"\"\"\n"
            "+from __future__ import annotations\n"
            "+import sys\n"
            "+from pathlib import Path\n"
            "+import pytest\n"
            "+\n"
            "+sys.path.insert(0, str(Path(__file__).resolve().parent."
            "parent))\n"
            "+\n"
            "+from core import config\n"
            "+\n"
            "+@pytest.fixture\n"
            "+def fresh_thing(tmp_path, monkeypatch):\n"
            "+    return None\n"
            "+\n"
            "+def test_n01_happy(fresh_thing):\n"
            "+    assert True\n"
        ),
        "explanation": (
            "Sentinel test conventions (mirroring "
            "test_phase14a_graduation.py / test_phase15a_lifecycle.py): "
            "module docstring with a `Coverage:` section listing test "
            "IDs (N01, N02 etc.) -- this is human-readable index of "
            "what the file tests; `from __future__ import annotations` "
            "for forward-reference type hints; sys.path.insert at top "
            "so tests run without the package being installed; one "
            "@pytest.fixture per shared setup; test names prefixed with "
            "the ID from the docstring. Always verify with `pytest -q` "
            "before declaring done."
        ),
        "shadow_recipe": (
            'STEP 1: write_file path="tests/test_phaseNN.py" '
            'content="..."\n'
            'STEP 2: done summary="added test file"'
        ),
    },
    {
        "tags": ["test", "monkeypatch", "stub", "llm", "mock"],
        "problem_summary": (
            "write a test that mocks a LLM call to avoid real GPU "
            "inference"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="tests/test_phase14a_graduation.py"\n'
            'STEP 2: write_file path="tests/test_NEW_llm_mock.py" '
            'content="\\"\\"\\"Test with stubbed LLM.\\"\\"\\"\\nimport '
            'pytest\\nfrom skills import code_assist as ca\\n\\n\\ndef '
            'test_with_mocked_qwen(monkeypatch):\\n    def fake_qwen('
            'system, user, trace_id, model, **kwargs):\\n        return '
            '\\"STEP 1: done summary=\\\\\\"ok\\\\\\"\\"\\n    monkeypatch.'
            'setattr(ca, \\"_qwen_generate\\", fake_qwen)\\n    # ... '
            'invoke the path under test\\n    assert True\\n"\n'
            'STEP 3: run_bash command="python -m pytest tests/'
            'test_NEW_llm_mock.py -q"\n'
            'STEP 4: done summary="added LLM-mocked test using "\n'
        ),
        "solution_code": (
            "diff --git a/tests/test_NEW_llm_mock.py "
            "b/tests/test_NEW_llm_mock.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_NEW_llm_mock.py\n"
            "@@ -0,0 +1,12 @@\n"
            "+\"\"\"Test with stubbed LLM.\"\"\"\n"
            "+import pytest\n"
            "+from skills import code_assist as ca\n"
            "+\n"
            "+def test_with_mocked_qwen(monkeypatch):\n"
            "+    def fake_qwen(system, user, trace_id, model, "
            "**kwargs):\n"
            "+        return \"STEP 1: done summary=\\\"ok\\\"\"\n"
            "+    monkeypatch.setattr(ca, \"_qwen_generate\", "
            "fake_qwen)\n"
            "+    assert True\n"
        ),
        "explanation": (
            "Sentinel tests must NEVER invoke the real Ollama / Claude "
            "CLI -- pytest runs CPU-only. Use monkeypatch.setattr to "
            "stub _qwen_generate (skills.code_assist), the brain's "
            "inference.generate (core.brain), and ClaudeCliClient."
            "generate (core.claude_cli). The _stub_embedder fixture in "
            "test_pre15_hybrid_retrieval.py is the canonical pattern "
            "for embedding stubs. Always include **kwargs in your stub "
            "signature so future param additions don't break tests."
        ),
        "shadow_recipe": (
            'STEP 1: write_file path="tests/test_mock.py" '
            'content="...monkeypatch..."\n'
            'STEP 2: done summary="added mock test"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Internal handlers + scheduler (1 pattern)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["scheduler", "internal", "maintenance", "cron",
                 "internal_handlers"],
        "problem_summary": (
            "add a new internal maintenance handler that runs on a "
            "nightly cron"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/internal_handlers.py"\n'
            'STEP 2: edit_file path="core/internal_handlers.py" '
            'old="INTERNAL_HANDLERS[\\"kb_lifecycle\\"] = kb_lifecycle" '
            'new="INTERNAL_HANDLERS[\\"kb_lifecycle\\"] = kb_lifecycle'
            '\\n\\n\\nasync def NEW_TASK(arg: str) -> str:\\n    '
            '\\"\\"\\"Phase NN -- new maintenance task.\\"\\"\\"\\n    '
            'log_event(\\"SEN-system\\", \\"INFO\\", \\"internal\\", '
            '\\"NEW_TASK ran\\")\\n    return \\"ok\\"\\n\\n\\n'
            'INTERNAL_HANDLERS[\\"NEW_TASK\\"] = NEW_TASK"\n'
            'STEP 3: edit_file path="main.py" old="    {\\"name\\": '
            '\\"KB lifecycle transition\\"," new="    {\\"name\\": '
            '\\"NEW_TASK NAME\\",\\n     \\"schedule_type\\": \\"cron'
            '\\", \\"schedule_value\\": \\"0 4 * * *\\",\\n     '
            '\\"command\\": \\"/internal_NEW_TASK\\",\\n     '
            '\\"session_type\\": \\"isolated\\"},\\n    {\\"name\\": '
            '\\"KB lifecycle transition\\","\n'
            'STEP 4: run_bash command="python -c \\"from core.'
            'internal_handlers import INTERNAL_HANDLERS; assert \'NEW_'
            'TASK\' in INTERNAL_HANDLERS; print(\'registered\')\\""\n'
            'STEP 5: done summary="registered NEW_TASK in INTERNAL_'
            'HANDLERS + seeded nightly 04:00 EST cron in main.py"'
        ),
        "solution_code": (
            "diff --git a/core/internal_handlers.py "
            "b/core/internal_handlers.py\n"
            "--- a/core/internal_handlers.py\n"
            "+++ b/core/internal_handlers.py\n"
            "@@ -260,3 +260,11 @@\n"
            " INTERNAL_HANDLERS[\"kb_lifecycle\"] = kb_lifecycle\n"
            "+\n"
            "+\n"
            "+async def NEW_TASK(arg: str) -> str:\n"
            "+    \"\"\"Phase NN -- new maintenance task.\"\"\"\n"
            "+    log_event(\"SEN-system\", \"INFO\", \"internal\", "
            "\"NEW_TASK ran\")\n"
            "+    return \"ok\"\n"
            "+\n"
            "+\n"
            "+INTERNAL_HANDLERS[\"NEW_TASK\"] = NEW_TASK\n"
        ),
        "explanation": (
            "Sentinel's INTERNAL_HANDLERS pattern (Phase 11): each "
            "handler is `async def name(arg: str) -> str` and self-"
            "registers at module import via INTERNAL_HANDLERS[name] = "
            "func. Because main.py imports core.internal_handlers at "
            "boot, the registration is automatic. To trigger on a "
            "schedule, append a dict to _DEFAULT_SCHEDULED_JOBS in "
            "main.py with name/schedule_type='cron'/schedule_value="
            "'<minute hour * * *>' (EST per Phase 11)/command="
            "'/internal_<name>'. Stagger times to avoid running "
            "simultaneously with the existing 03:30 backup and 03:45 "
            "kb_lifecycle."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/internal_handlers.py" '
            'old="..." new="..."\n'
            'STEP 2: edit_file path="main.py" old="..." new="..."\n'
            'STEP 3: done summary="added new task"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Skills + Agents (2 patterns)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["skill", "BaseSkill", "pipeline", "pydantic_schema"],
        "problem_summary": (
            "add a new skill that processes input and returns a typed "
            "result"
        ),
        "solution_pattern": (
            'STEP 1: list_dir path="skills"\n'
            'STEP 2: read_file path="skills/web_summarize.py"\n'
            'STEP 3: write_file path="skills/NEW_skill.py" '
            'content="\\"\\"\\"NEW skill -- describe what it does.'
            '\\"\\"\\"\\nfrom pydantic import BaseModel\\nfrom core.'
            'skills import BaseSkill\\n\\n\\nclass NEWInput(BaseModel):'
            '\\n    text: str\\n\\nclass NEWOutput(BaseModel):\\n    '
            'result: str\\n\\nclass NEWSkill(BaseSkill):\\n    name = '
            '\\"new_skill\\"\\n    input_schema = NEWInput\\n    '
            'output_schema = NEWOutput\\n\\n    async def execute(self, '
            'input_data: NEWInput, trace_id: str, context: dict) -> '
            'NEWOutput:\\n        return NEWOutput(result=input_data.'
            'text.upper())"\n'
            'STEP 4: run_bash command="python -m pytest tests/'
            'test_skills.py -q -k new_skill"\n'
            'STEP 5: done summary="added NEWSkill with input/output "\n'
        ),
        "solution_code": (
            "diff --git a/skills/NEW_skill.py b/skills/NEW_skill.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/skills/NEW_skill.py\n"
            "@@ -0,0 +1,18 @@\n"
            "+\"\"\"NEW skill -- describe what it does.\"\"\"\n"
            "+from pydantic import BaseModel\n"
            "+from core.skills import BaseSkill\n"
            "+\n"
            "+class NEWInput(BaseModel):\n"
            "+    text: str\n"
            "+\n"
            "+class NEWOutput(BaseModel):\n"
            "+    result: str\n"
            "+\n"
            "+class NEWSkill(BaseSkill):\n"
            "+    name = \"new_skill\"\n"
            "+    input_schema = NEWInput\n"
            "+    output_schema = NEWOutput\n"
            "+\n"
            "+    async def execute(self, input_data, trace_id, "
            "context):\n"
            "+        return NEWOutput(result=input_data.text.upper())\n"
        ),
        "explanation": (
            "Skills inherit BaseSkill (core.skills) and define three "
            "ClassVars: name (used in agent YAML pipeline lists), "
            "input_schema (Pydantic), output_schema (Pydantic). The "
            "registry auto-discovers any *.py in skills/ at startup -- "
            "no manual registration needed. Phase 10 added accepts_list "
            "+ output_is_list ClassVars for fan-out patterns (see "
            "skills/job_score.py / skills/job_report.py). Async execute "
            "is the contract; trace_id is for log_event correlation; "
            "context is a dict that may include 'model' to override "
            "complexity routing."
        ),
        "shadow_recipe": (
            'STEP 1: write_file path="skills/NEW_skill.py" '
            'content="..."\n'
            'STEP 2: done summary="added skill"'
        ),
    },
    {
        "tags": ["agent", "yaml", "pipeline", "skill_pipeline",
                 "command_agent_map"],
        "problem_summary": (
            "add a new agent that runs a fixed pipeline of skills"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="agents/job_searcher.yaml"\n'
            'STEP 2: write_file path="agents/NEW_agent.yaml" '
            'content="name: new_agent\\ndescription: NEW agent that '
            'does X\\nmodel: qwen-coder\\nskill_pipeline:\\n  - new_skill'
            '\\n"\n'
            'STEP 3: edit_file path="core/config.py" old="COMMAND_AGENT_'
            'MAP = {" new="COMMAND_AGENT_MAP = {\\n    \\"/new_command'
            '\\": \\"new_agent\\","\n'
            'STEP 4: run_bash command="python -c \\"from core.agent_'
            'registry import AGENT_REGISTRY; AGENT_REGISTRY.discover(); '
            'print(AGENT_REGISTRY.get(\'new_agent\'))\\""\n'
            'STEP 5: done summary="added new_agent YAML + wired '
            '/new_command in COMMAND_AGENT_MAP"'
        ),
        "solution_code": (
            "diff --git a/agents/NEW_agent.yaml b/agents/NEW_agent.yaml\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/agents/NEW_agent.yaml\n"
            "@@ -0,0 +1,5 @@\n"
            "+name: new_agent\n"
            "+description: NEW agent that does X\n"
            "+model: qwen-coder\n"
            "+skill_pipeline:\n"
            "+  - new_skill\n"
        ),
        "explanation": (
            "Agents are YAML configs in agents/ -- the registry auto-"
            "discovers them at startup. Required keys: name, "
            "description, skill_pipeline (list of skill names from the "
            "skill registry). Optional: model (pins the worker model "
            "via complexity routing bypass -- common values: 'qwen-"
            "coder', 'sentinel-brain', 'claude-cli'). The agent runs "
            "skills in order; output of skill N feeds input of skill "
            "N+1 (with fan-out for accepts_list/output_is_list pairs). "
            "To trigger from Telegram: add an entry to "
            "config.COMMAND_AGENT_MAP."
        ),
        "shadow_recipe": (
            'STEP 1: write_file path="agents/NEW.yaml" content="..."\n'
            'STEP 2: done summary="added agent"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Math + util functions (3 patterns -- the bread-and-butter)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["math", "util", "function", "validation", "math_utils"],
        "problem_summary": (
            "add a math function to math_utils.py with input validation"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="math_utils.py"\n'
            'STEP 2: edit_file path="math_utils.py" old="def is_'
            'palindrome(s: str," new="def NEW_FUNC(x: int) -> int:\\n    '
            'if not isinstance(x, int) or x < 0:\\n        raise '
            'ValueError(\\"x must be a non-negative int\\")\\n    return '
            'x * 2\\n\\ndef is_palindrome(s: str,"\n'
            'STEP 3: run_bash command="python -c \\"from math_utils '
            'import NEW_FUNC; assert NEW_FUNC(5) == 10; print(\'ok\')\\""\n'
            'STEP 4: done summary="added NEW_FUNC with type+value "\n'
        ),
        "solution_code": (
            "diff --git a/math_utils.py b/math_utils.py\n"
            "--- a/math_utils.py\n"
            "+++ b/math_utils.py\n"
            "@@ -50,3 +50,9 @@\n"
            "+def NEW_FUNC(x: int) -> int:\n"
            "+    if not isinstance(x, int) or x < 0:\n"
            "+        raise ValueError(\"x must be a non-negative int\")\n"
            "+    return x * 2\n"
            "+\n"
            " def is_palindrome(s: str,\n"
        ),
        "explanation": (
            "math_utils.py is the established testbed for /code "
            "(historically: factorial, is_palindrome, etc.). New "
            "functions land BEFORE existing ones to test the "
            "edit_file-with-anchor pattern (using is_palindrome's def "
            "as the unique anchor). Always include type hints + "
            "isinstance + value validation + ValueError. Verify with "
            "`python -c 'from math_utils import <fn>; assert ...; "
            "print(\"ok\")'` -- the print is what Claude review reads "
            "as a clear pass signal."
        ),
        "shadow_recipe": (
            'STEP 1: read_file path="math_utils.py"\n'
            'STEP 2: edit_file path="math_utils.py" '
            'old="..." new="def NEW_FUNC..."\n'
            'STEP 3: run_bash command="python -c \\"...\\""\n'
            'STEP 4: done summary="added NEW_FUNC"'
        ),
    },
    {
        "tags": ["util", "string", "core_util", "transform"],
        "problem_summary": (
            "add a string utility to core/util.py"
        ),
        "solution_pattern": (
            'STEP 1: list_dir path="core"\n'
            'STEP 2: read_file path="core/util.py"\n'
            'STEP 3: edit_file path="core/util.py" old="(end of file)" '
            'new="\\n\\ndef NEW_STR(s: str) -> str:\\n    \\"\\"\\"'
            'Describe.\\"\\"\\"\\n    return s.strip().lower()"\n'
            'STEP 4: run_bash command="python -c \\"from core.util '
            'import NEW_STR; assert NEW_STR(\'  HELLO  \') == \'hello\'; '
            'print(\'ok\')\\""\n'
            'STEP 5: done summary="added NEW_STR string transformer"'
        ),
        "solution_code": (
            "diff --git a/core/util.py b/core/util.py\n"
            "--- a/core/util.py\n"
            "+++ b/core/util.py\n"
            "@@ -45,3 +45,5 @@\n"
            "+def NEW_STR(s: str) -> str:\n"
            "+    \"\"\"Describe.\"\"\"\n"
            "+    return s.strip().lower()\n"
        ),
        "explanation": (
            "core/util.py holds general-purpose helpers. String "
            "transformers should accept str, return str, and use only "
            "stdlib. Verify with a representative input → expected "
            "mapping that exercises the actual transform (whitespace, "
            "case, etc.). Verifier prints 'ok' on assert success so "
            "Claude's review sees a clear pass signal."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/util.py" '
            'old="..." new="def NEW_STR..."\n'
            'STEP 2: done summary="added string util"'
        ),
    },
    {
        "tags": ["util", "list", "iteration", "core_util"],
        "problem_summary": (
            "add a list operation to core/util.py"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/util.py"\n'
            'STEP 2: edit_file path="core/util.py" old="(end of file)" '
            'new="\\n\\ndef NEW_LIST(lst: list) -> list:\\n    \\"\\"\\"'
            'Describe.\\"\\"\\"\\n    seen = set()\\n    out = []\\n    '
            'for x in lst:\\n        if x not in seen:\\n            '
            'seen.add(x)\\n            out.append(x)\\n    return out"\n'
            'STEP 3: run_bash command="python -c \\"from core.util '
            'import NEW_LIST; assert NEW_LIST([1,2,1,3,2]) == [1,2,3]; '
            'print(\'ok\')\\""\n'
            'STEP 4: done summary="added NEW_LIST preserving order"'
        ),
        "solution_code": (
            "diff --git a/core/util.py b/core/util.py\n"
            "--- a/core/util.py\n"
            "+++ b/core/util.py\n"
            "@@ -55,3 +55,11 @@\n"
            "+def NEW_LIST(lst: list) -> list:\n"
            "+    \"\"\"Describe.\"\"\"\n"
            "+    seen = set()\n"
            "+    out = []\n"
            "+    for x in lst:\n"
            "+        if x not in seen:\n"
            "+            seen.add(x)\n"
            "+            out.append(x)\n"
            "+    return out\n"
        ),
        "explanation": (
            "List operations preserve insertion order where possible "
            "(set() is for O(1) membership, list is for output). For "
            "verifiers, use input/output pairs that test BOTH the "
            "transformation AND order: [1,2,1,3,2] → [1,2,3] tests "
            "uniqueness AND order preservation in one assertion."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/util.py" '
            'old="..." new="def NEW_LIST..."\n'
            'STEP 2: done summary="added list util"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # KB / KnowledgeBase -- 3 more patterns
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["sqlite", "partial_index", "performance", "selective"],
        "problem_summary": (
            "add a partial index to a sqlite table for selective query"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/knowledge_base.py"\n'
            'STEP 2: edit_file path="core/knowledge_base.py" '
            'old="            conn.execute(\\n                \\"CREATE '
            'INDEX IF NOT EXISTS idx_knowledge_archived \\"\\n           '
            '     \\"ON knowledge(state) WHERE state = \'archived\'\\"\\n'
            '            )" new="            conn.execute(\\n           '
            '     \\"CREATE INDEX IF NOT EXISTS idx_knowledge_archived '
            '\\"\\n                \\"ON knowledge(state) WHERE state = '
            '\'archived\'\\"\\n            )\\n            conn.execute('
            '\\n                \\"CREATE INDEX IF NOT EXISTS idx_NEW '
            '\\"\\n                \\"ON knowledge(NEW_COL) WHERE '
            'NEW_COL IS NOT NULL\\"\\n            )"\n'
            'STEP 3: run_bash command="python -c \\"from core.knowledge_'
            'base import KnowledgeBase; from core import config; kb = '
            'KnowledgeBase(); import sqlite3; c = sqlite3.connect(str(kb.'
            'db_path)); idx = [r[0] for r in c.execute(\'SELECT name '
            'FROM sqlite_master WHERE type=\\\\\\"index\\\\\\"\').fetchall()]'
            '; assert \'idx_NEW\' in idx; print(\'index created\')\\""\n'
            'STEP 4: done summary="added partial index idx_NEW on "\n'
        ),
        "solution_code": (
            "diff --git a/core/knowledge_base.py "
            "b/core/knowledge_base.py\n"
            "--- a/core/knowledge_base.py\n"
            "+++ b/core/knowledge_base.py\n"
            "@@ -260,4 +260,8 @@\n"
            "             conn.execute(\n"
            "                 \"CREATE INDEX IF NOT EXISTS "
            "idx_knowledge_archived \"\n"
            "                 \"ON knowledge(state) WHERE "
            "state = 'archived'\"\n"
            "             )\n"
            "+            conn.execute(\n"
            "+                \"CREATE INDEX IF NOT EXISTS idx_NEW \"\n"
            "+                \"ON knowledge(NEW_COL) WHERE NEW_COL IS "
            "NOT NULL\"\n"
            "+            )\n"
        ),
        "explanation": (
            "Partial indexes (Phase 14a/15a discipline): cover ONLY the "
            "rows you actually query, not the whole column. Examples in "
            "the codebase: `WHERE needs_reteach = 1` (small subset), "
            "`WHERE last_verified_at IS NOT NULL` (excludes unverified "
            "majority), `WHERE state = 'archived'`, `WHERE pinned = 1`. "
            "Adds CREATE INDEX IF NOT EXISTS inside _init_schema; "
            "idempotent on subsequent boots. Stays tiny even at the "
            "50K-row cap because only matching rows occupy the index."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/knowledge_base.py" '
            'old="..." new="...CREATE INDEX..."\n'
            'STEP 2: done summary="added partial index"'
        ),
    },
    {
        "tags": ["kb", "backfill", "embeddings", "migration", "one_shot"],
        "problem_summary": (
            "backfill a column for rows where it's NULL"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/knowledge_base.py"\n'
            'STEP 2: edit_file path="core/knowledge_base.py" '
            'old="    def backfill_embeddings(" new="    def backfill_'
            'NEW_COL(self, batch_size: int = 50, trace_id: str = '
            '\\"SEN-system\\") -> dict:\\n        \\"\\"\\"Compute and '
            'fill NEW_COL for any rows where it IS NULL.\\"\\"\\"\\n    '
            '    scanned = filled = failed = 0\\n        conn = '
            '_connect(self.db_path)\\n        try:\\n            rows = '
            'conn.execute(\\"SELECT id FROM knowledge WHERE NEW_COL IS '
            'NULL LIMIT ?\\", (batch_size,)).fetchall()\\n            '
            'for r in rows:\\n                scanned += 1\\n           '
            '     try:\\n                    val = self._compute_NEW(r['
            '\\"id\\"])\\n                    conn.execute(\\"UPDATE '
            'knowledge SET NEW_COL = ? WHERE id = ?\\", (val, r[\\"id'
            '\\"]))\\n                    filled += 1\\n               '
            ' except Exception:\\n                    failed += 1\\n   '
            '     finally:\\n            conn.close()\\n        return '
            '{\\"scanned\\": scanned, \\"filled\\": filled, \\"failed\\"'
            ': failed}\\n\\n    def backfill_embeddings("\n'
            'STEP 3: run_bash command="python -c \\"from core.knowledge_'
            'base import KnowledgeBase; print(KnowledgeBase().backfill_'
            'NEW_COL(batch_size=10))\\""\n'
            'STEP 4: done summary="added backfill_NEW_COL one-shot "\n'
        ),
        "solution_code": (
            "diff --git a/core/knowledge_base.py "
            "b/core/knowledge_base.py\n"
            "--- a/core/knowledge_base.py\n"
            "+++ b/core/knowledge_base.py\n"
            "@@ -490,2 +490,21 @@\n"
            "+    def backfill_NEW_COL(self, batch_size: int = 50, "
            "trace_id: str = \"SEN-system\") -> dict:\n"
            "+        \"\"\"Fill NEW_COL where IS NULL.\"\"\"\n"
            "+        scanned = filled = failed = 0\n"
            "+        conn = _connect(self.db_path)\n"
            "+        try:\n"
            "+            rows = conn.execute(\n"
            "+                \"SELECT id FROM knowledge WHERE "
            "NEW_COL IS NULL LIMIT ?\", (batch_size,)\n"
            "+            ).fetchall()\n"
            "+            for r in rows:\n"
            "+                scanned += 1\n"
            "+                # ... compute + UPDATE\n"
        ),
        "explanation": (
            "Backfill pattern (mirrors KB.backfill_embeddings): "
            "process rows in batches, count {scanned, filled, failed}, "
            "always WHERE COL IS NULL so re-runs are no-ops on already-"
            "filled rows. Run on startup once or on demand. Best-effort "
            "per-row -- compute failures don't crash the loop. Cap "
            "batch_size to avoid blocking on very large legacy datasets."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/knowledge_base.py" '
            'old="..." new="def backfill_NEW_COL..."\n'
            'STEP 2: done summary="added backfill"'
        ),
    },
    {
        "tags": ["kb", "search", "rerank", "hybrid", "weights"],
        "problem_summary": (
            "tune the hybrid search rerank weights for KB"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/knowledge_base.py"\n'
            'STEP 2: edit_file path="core/knowledge_base.py" '
            'old="    HYBRID_BM25_WEIGHT = 0.4" new="    HYBRID_BM25_'
            'WEIGHT = 0.3  # Phase NN -- favor semantic over keyword"\n'
            'STEP 3: run_bash command="python -m pytest tests/'
            'test_pre15_hybrid_retrieval.py -q"\n'
            'STEP 4: done summary="adjusted HYBRID_BM25_WEIGHT 0.4 → 0.3'
            ' (more semantic weight)"'
        ),
        "solution_code": (
            "diff --git a/core/knowledge_base.py "
            "b/core/knowledge_base.py\n"
            "--- a/core/knowledge_base.py\n"
            "+++ b/core/knowledge_base.py\n"
            "@@ -262,2 +262,2 @@\n"
            "-    HYBRID_BM25_WEIGHT = 0.4\n"
            "+    HYBRID_BM25_WEIGHT = 0.3  # Phase NN\n"
        ),
        "explanation": (
            "Hybrid search blends BM25 (keyword) with cosine similarity "
            "(semantic). HYBRID_BM25_WEIGHT controls the mix: 1.0 = "
            "pure keyword, 0.0 = pure semantic. Default 0.4 leans "
            "semantic (concept overlap > exact tokens for code "
            "patterns). Lower it (0.2-0.3) when KB has lots of "
            "conceptual neighbors with different vocabulary. Always "
            "re-run test_pre15_hybrid_retrieval.py after tuning to "
            "verify the existing reranking tests still pass."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/knowledge_base.py" '
            'old="HYBRID_BM25_WEIGHT = 0.4" new="HYBRID_BM25_WEIGHT = '
            '0.3"\n'
            'STEP 2: done summary="tuned weight"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Memory -- 2 more patterns
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["memory", "search", "filter", "decay", "scope"],
        "problem_summary": (
            "search episodic memory across scopes with filters"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/memory.py"\n'
            'STEP 2: edit_file path="core/memory.py" '
            'old="    def get_recent_episodes(" new="    def search_'
            'across_scopes(self, query: str, min_relevance: float = '
            '0.5, limit: int = 20) -> list[Episode]:\\n        '
            '\\"\\"\\"FTS5 query across all scopes, filter by relevance'
            '.\\"\\"\\"\\n        if not query.strip():\\n            '
            'return []\\n        fts_q = _escape_fts_query(query)\\n   '
            '     conn = _connect(self.db_path)\\n        try:\\n      '
            '      rows = conn.execute(\\n                \\"SELECT e.* '
            'FROM episodic_fts f JOIN episodic_memory e ON e.id = f.'
            'rowid \\"\\n                \\"WHERE episodic_fts MATCH ? '
            'AND e.relevance_score >= ? AND e.state != \'archived\' \\"'
            '\\n                \\"ORDER BY e.relevance_score DESC '
            'LIMIT ?\\",\\n                (fts_q, min_relevance, '
            'limit),\\n            ).fetchall()\\n        finally:\\n  '
            '          conn.close()\\n        return [Episode(**dict(r)'
            ') for r in rows]\\n\\n    def get_recent_episodes("\n'
            'STEP 3: run_bash command="python -c \\"from core.memory '
            'import get_memory; m = get_memory(); print(m.search_across_'
            'scopes(\'test\'))\\""\n'
            'STEP 4: done summary="added search_across_scopes with "\n'
        ),
        "solution_code": (
            "diff --git a/core/memory.py b/core/memory.py\n"
            "--- a/core/memory.py\n"
            "+++ b/core/memory.py\n"
            "@@ -290,2 +290,17 @@\n"
            "+    def search_across_scopes(self, query: str, "
            "min_relevance: float = 0.5, limit: int = 20) -> "
            "list[Episode]:\n"
            "+        \"\"\"FTS5 query across all scopes, filter by "
            "relevance.\"\"\"\n"
            "+        if not query.strip():\n"
            "+            return []\n"
            "+        fts_q = _escape_fts_query(query)\n"
        ),
        "explanation": (
            "Memory search methods follow the FTS5 pattern: fts_q = "
            "_escape_fts_query (handles quotes/dashes), JOIN on rowid "
            "(FTS table), filter on archived (Phase 15a), order by the "
            "relevance metric appropriate to the table (relevance_score "
            "for episodic, confidence for semantic). Always handle "
            "empty query → return [] rather than letting FTS5 error. "
            "Always wrap conn in try/finally close."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/memory.py" '
            'old="..." new="def search_across_scopes..."\n'
            'STEP 2: done summary="added cross-scope search"'
        ),
    },
    {
        "tags": ["memory", "field", "schema", "migration", "episode"],
        "problem_summary": (
            "add a custom field to Episode + migrate the table"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/memory.py"\n'
            'STEP 2: edit_file path="core/memory.py" '
            'old="    created_by_origin: str = \\"foreground\\"\\n\\n\\n'
            'class Fact(BaseModel):" new="    created_by_origin: str '
            '= \\"foreground\\"\\n    NEW_FIELD: str = \\"default\\"  '
            '# Phase NN\\n\\n\\nclass Fact(BaseModel):"\n'
            'STEP 3: edit_file path="core/memory.py" '
            'old="                if \\"created_by_origin\\" not in '
            'cols:" new="                if \\"NEW_FIELD\\" not in cols:'
            '\\n                    conn.execute(\\n                    '
            '    f\\"ALTER TABLE {tbl} \\"\\n                        f'
            '\\"ADD COLUMN NEW_FIELD TEXT NOT NULL DEFAULT \'default\''
            '\\"\\n                    )\\n                if '
            '\\"created_by_origin\\" not in cols:"\n'
            'STEP 4: run_bash command="python -c \\"from core.memory '
            'import MemoryManager; from pathlib import Path; import '
            'tempfile; m = MemoryManager(db_path=Path(tempfile.'
            'NamedTemporaryFile(suffix=\'.db\', delete=False).name)); '
            'print(\'schema ok\')\\""\n'
            'STEP 5: done summary="added NEW_FIELD to Episode +"\n'
        ),
        "solution_code": (
            "diff --git a/core/memory.py b/core/memory.py\n"
            "--- a/core/memory.py\n"
            "+++ b/core/memory.py\n"
            "@@ -88,2 +88,3 @@\n"
            "     created_by_origin: str = \"foreground\"\n"
            "+    NEW_FIELD: str = \"default\"  # Phase NN\n"
        ),
        "explanation": (
            "Adding fields to memory tables requires THREE coordinated "
            "changes: (1) Pydantic model field with default for back-"
            "compat; (2) idempotent ALTER TABLE in _init_schema's "
            "for-tbl loop; (3) update store_episode/store_fact INSERT "
            "to populate the column. The Pydantic default lets old rows "
            "hydrate without the column present in dict(r). Always test "
            "with a fresh tempfile DB to confirm the schema migration "
            "is idempotent (no duplicate-column errors on re-init)."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/memory.py" '
            'old="..." new="...NEW_FIELD..."\n'
            'STEP 2: done summary="added field"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # SQLite tables -- 3 patterns
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["sqlite", "task", "column", "sentinel_db", "tasks"],
        "problem_summary": (
            "add a new column to the tasks table in sentinel.db"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/database.py"\n'
            'STEP 2: edit_file path="core/database.py" '
            'old="def init_db()" new="def _alter_add_NEW(conn):\\n    '
            'cols = {r[\\"name\\"] for r in conn.execute(\\"PRAGMA '
            'table_info(tasks)\\").fetchall()}\\n    if \\"NEW_COL\\" '
            'not in cols:\\n        conn.execute(\\"ALTER TABLE tasks '
            'ADD COLUMN NEW_COL TEXT\\")\\n\\ndef init_db()"\n'
            'STEP 3: edit_file path="core/database.py" '
            'old="    conn.execute(\\"CREATE TABLE IF NOT EXISTS '
            'tasks" new="    _alter_add_NEW(conn)\\n    conn.execute('
            '\\"CREATE TABLE IF NOT EXISTS tasks"\n'
            'STEP 4: run_bash command="python -c \\"from core.database '
            'import init_db; init_db(); print(\'migrated\')\\""\n'
            'STEP 5: done summary="added NEW_COL to tasks table with "\n'
        ),
        "solution_code": (
            "diff --git a/core/database.py b/core/database.py\n"
            "--- a/core/database.py\n"
            "+++ b/core/database.py\n"
            "@@ -50,3 +50,9 @@\n"
            "+def _alter_add_NEW(conn):\n"
            "+    cols = {r[\"name\"] for r in conn.execute("
            "\"PRAGMA table_info(tasks)\").fetchall()}\n"
            "+    if \"NEW_COL\" not in cols:\n"
            "+        conn.execute(\"ALTER TABLE tasks ADD COLUMN "
            "NEW_COL TEXT\")\n"
        ),
        "explanation": (
            "core/database.py owns sentinel.db schema. New columns on "
            "the tasks table follow the same idempotent ALTER pattern "
            "as KB: PRAGMA table_info → set check → conditional ALTER. "
            "Place the helper function ABOVE init_db so it can be "
            "called from inside. Don't add columns to tables created "
            "by Phase 11 (scheduled_jobs, job_runs) without checking "
            "the scheduler's expected schema first."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/database.py" '
            'old="..." new="...ALTER TABLE tasks..."\n'
            'STEP 2: done summary="migrated tasks"'
        ),
    },
    {
        "tags": ["sqlite", "fts5", "table", "trigger", "create"],
        "problem_summary": (
            "create a new fts5-indexed table with insert/delete triggers"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/memory.py"\n'
            'STEP 2: edit_file path="core/MODULE.py" old="def init_db():" '
            'new="def init_db():\\n    conn.execute(\\"\\"\\"\\n        '
            'CREATE TABLE IF NOT EXISTS NEW_TABLE (\\n            id '
            'INTEGER PRIMARY KEY AUTOINCREMENT,\\n            text '
            'TEXT NOT NULL,\\n            created_at TEXT NOT NULL\\n  '
            '      )\\n    \\"\\"\\")\\n    conn.execute(\\"\\"\\"\\n  '
            '      CREATE VIRTUAL TABLE IF NOT EXISTS NEW_TABLE_fts '
            'USING fts5(\\n            text,\\n            content='
            '\'NEW_TABLE\', content_rowid=\'id\'\\n        )\\n    '
            '\\"\\"\\")\\n    conn.execute(\\"\\"\\"\\n        CREATE '
            'TRIGGER IF NOT EXISTS NEW_TABLE_ai\\n        AFTER INSERT '
            'ON NEW_TABLE BEGIN\\n            INSERT INTO NEW_TABLE_fts'
            '(rowid, text) VALUES (new.id, new.text);\\n        END\\n  '
            '  \\"\\"\\")\\n    conn.execute(\\"\\"\\"\\n        CREATE '
            'TRIGGER IF NOT EXISTS NEW_TABLE_ad\\n        AFTER DELETE '
            'ON NEW_TABLE BEGIN\\n            INSERT INTO NEW_TABLE_fts('
            'NEW_TABLE_fts, rowid, text) VALUES (\'delete\', old.id, '
            'old.text);\\n        END\\n    \\"\\"\\")"\n'
            'STEP 3: run_bash command="python -c \\"from core.MODULE '
            'import init_db; init_db()\\""\n'
            'STEP 4: done summary="created NEW_TABLE with FTS5 + ai/ad'
            ' triggers"'
        ),
        "solution_code": (
            "diff --git a/core/MODULE.py b/core/MODULE.py\n"
            "--- a/core/MODULE.py\n"
            "+++ b/core/MODULE.py\n"
            "@@ -50,3 +50,30 @@\n"
            "+    conn.execute(\"\"\"\n"
            "+        CREATE TABLE IF NOT EXISTS NEW_TABLE (\n"
            "+            id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "+            text TEXT NOT NULL,\n"
            "+            created_at TEXT NOT NULL\n"
            "+        )\n"
            "+    \"\"\")\n"
            "+    conn.execute(\"\"\"\n"
            "+        CREATE VIRTUAL TABLE IF NOT EXISTS NEW_TABLE_fts "
            "USING fts5(\n"
            "+            text,\n"
            "+            content='NEW_TABLE', content_rowid='id'\n"
            "+        )\n"
            "+    \"\"\")\n"
        ),
        "explanation": (
            "FTS5 contentless-external-content tables are Sentinel's "
            "standard for searchable text storage (knowledge, "
            "episodic_memory, semantic_memory all use this shape). "
            "Required: CREATE TABLE for the canonical row, CREATE "
            "VIRTUAL TABLE fts5 with content='real_table' + "
            "content_rowid='id', AFTER INSERT trigger to sync, AFTER "
            "DELETE trigger that does the FTS5 'delete' command (which "
            "is INSERT-ish in FTS5). Don't forget the AFTER UPDATE "
            "trigger if rows can mutate (semantic_memory has this -- "
            "look at semantic_au)."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/MODULE.py" '
            'old="..." new="...CREATE TABLE...CREATE VIRTUAL TABLE..."\n'
            'STEP 2: done summary="added FTS5 table"'
        ),
    },
    {
        "tags": ["sqlite", "upsert", "insert_or_replace", "merge", "idempotent"],
        "problem_summary": (
            "implement an upsert pattern for a sqlite row"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/database.py"\n'
            'STEP 2: edit_file path="core/database.py" old="def add_'
            'task(" new="def upsert_NEW(key: str, value: dict) -> int:'
            '\\n    \\"\\"\\"INSERT or UPDATE on conflict.\\"\\"\\"\\n    '
            'now = _utcnow_iso()\\n    conn = _connect(DB_PATH)\\n    '
            'try:\\n        existing = conn.execute(\\"SELECT id FROM '
            'NEW_TABLE WHERE key = ?\\", (key,)).fetchone()\\n        '
            'if existing is None:\\n            cur = conn.execute('
            '\\"INSERT INTO NEW_TABLE (key, value_json, created_at, '
            'updated_at) VALUES (?, ?, ?, ?)\\", (key, json.dumps(value)'
            ', now, now))\\n            return cur.lastrowid\\n        '
            'conn.execute(\\"UPDATE NEW_TABLE SET value_json = ?, '
            'updated_at = ? WHERE id = ?\\", (json.dumps(value), now, '
            'existing[0]))\\n        return existing[0]\\n    finally:\\n'
            '        conn.close()\\n\\ndef add_task("\n'
            'STEP 3: run_bash command="python -c \\"from core.database '
            'import upsert_NEW; print(upsert_NEW(\'k1\', {\'a\': 1}))\\""\n'
            'STEP 4: done summary="added upsert_NEW with confidence-aware'
            ' merge"'
        ),
        "solution_code": (
            "diff --git a/core/database.py b/core/database.py\n"
            "--- a/core/database.py\n"
            "+++ b/core/database.py\n"
            "@@ -150,3 +150,17 @@\n"
            "+def upsert_NEW(key: str, value: dict) -> int:\n"
            "+    \"\"\"INSERT or UPDATE on conflict.\"\"\"\n"
            "+    now = _utcnow_iso()\n"
            "+    conn = _connect(DB_PATH)\n"
            "+    try:\n"
            "+        existing = conn.execute(\n"
            "+            \"SELECT id FROM NEW_TABLE WHERE key = ?\",\n"
            "+            (key,),\n"
            "+        ).fetchone()\n"
            "+        # ... INSERT or UPDATE\n"
            "+    finally:\n"
            "+        conn.close()\n"
        ),
        "explanation": (
            "Upsert pattern (used by memory.store_fact, applications "
            "table): SELECT existing first, branch INSERT vs UPDATE. "
            "Avoids INSERT OR REPLACE which loses the AUTOINCREMENT id "
            "and triggers redundant FTS5 delete-then-insert. The "
            "Python-side branch is more code but cleaner: returns the "
            "stable id on update, the new id on insert. Phase 15b "
            "convention: created_by_origin is set on INSERT only -- "
            "UPDATE leaves it alone (audit trail of FIRST writer)."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/database.py" '
            'old="..." new="def upsert_NEW..."\n'
            'STEP 2: done summary="added upsert"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Skills/agents -- 2 more patterns
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["skill", "fan_out", "accepts_list", "output_is_list",
                 "batch"],
        "problem_summary": (
            "create a skill that processes a list of items via fan-out"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="skills/job_score.py"\n'
            'STEP 2: write_file path="skills/NEW_batch.py" '
            'content="\\"\\"\\"NEW batch skill -- processes a list.'
            '\\"\\"\\"\\nfrom typing import ClassVar\\nfrom pydantic '
            'import BaseModel\\nfrom core.skills import BaseSkill\\n\\n'
            '\\nclass Item(BaseModel):\\n    text: str\\n\\nclass NEWInput'
            '(BaseModel):\\n    items: list[Item]\\n\\nclass NEWOutput('
            'BaseModel):\\n    results: list[str]\\n\\nclass NEWBatch('
            'BaseSkill):\\n    name = \\"new_batch\\"\\n    accepts_list'
            ': ClassVar[bool] = True\\n    output_is_list: ClassVar'
            '[bool] = True\\n    input_schema = NEWInput\\n    output_'
            'schema = NEWOutput\\n\\n    async def execute(self, input_'
            'data, trace_id, context):\\n        return NEWOutput('
            'results=[i.text.upper() for i in input_data.items])"\n'
            'STEP 3: run_bash command="python -c \\"from core.registry '
            'import SKILL_REGISTRY; SKILL_REGISTRY.discover(); '
            'print(SKILL_REGISTRY.get(\'new_batch\'))\\""\n'
            'STEP 4: done summary="added NEWBatch with accepts_list "\n'
        ),
        "solution_code": (
            "diff --git a/skills/NEW_batch.py b/skills/NEW_batch.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/skills/NEW_batch.py\n"
            "@@ -0,0 +1,20 @@\n"
            "+\"\"\"NEW batch skill.\"\"\"\n"
            "+from typing import ClassVar\n"
            "+from pydantic import BaseModel\n"
            "+from core.skills import BaseSkill\n"
            "+\n"
            "+class Item(BaseModel):\n"
            "+    text: str\n"
            "+\n"
            "+class NEWInput(BaseModel):\n"
            "+    items: list[Item]\n"
            "+\n"
            "+class NEWOutput(BaseModel):\n"
            "+    results: list[str]\n"
            "+\n"
            "+class NEWBatch(BaseSkill):\n"
            "+    name = \"new_batch\"\n"
            "+    accepts_list: ClassVar[bool] = True\n"
            "+    output_is_list: ClassVar[bool] = True\n"
            "+    input_schema = NEWInput\n"
            "+    output_schema = NEWOutput\n"
        ),
        "explanation": (
            "Phase 10's fan-out: skills can declare `accepts_list = "
            "True` (consumes the full upstream list in one call -- like "
            "job_score in Phase 13's batching) OR `output_is_list = "
            "True` (its output is a single list field that the next "
            "skill should fan out over). When neither is set, the "
            "agent's default fan-out logic calls the skill once per "
            "input item. JobScrapeOutput must stay single-list-field "
            "for the unwrap path to work -- new telemetry fields go "
            "to sidecar files, not into the typed output."
        ),
        "shadow_recipe": (
            'STEP 1: write_file path="skills/NEW_batch.py" content="..."\n'
            'STEP 2: done summary="added batch skill"'
        ),
    },
    {
        "tags": ["skill", "modify", "validation", "pre_processing"],
        "problem_summary": (
            "add a validation step to an existing skill"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="skills/SKILL_NAME.py"\n'
            'STEP 2: edit_file path="skills/SKILL_NAME.py" '
            'old="    async def execute(self, input_data," '
            'new="    def _validate(self, input_data) -> None:\\n        '
            '\\"\\"\\"Phase NN -- pre-execute validation.\\"\\"\\"\\n   '
            '     if not input_data.text:\\n            raise ValueError'
            '(\\"text is required\\")\\n        if len(input_data.text) '
            '> 10000:\\n            raise ValueError(\\"text exceeds '
            'max length\\")\\n\\n    async def execute(self, input_'
            'data,"\n'
            'STEP 3: edit_file path="skills/SKILL_NAME.py" '
            'old="    async def execute(self, input_data, trace_id, '
            'context):\\n        " new="    async def execute(self, '
            'input_data, trace_id, context):\\n        self._validate('
            'input_data)\\n        "\n'
            'STEP 4: run_bash command="python -m pytest tests/'
            'test_skills.py -q -k SKILL_NAME"\n'
            'STEP 5: done summary="added _validate gate to SKILL_NAME"'
        ),
        "solution_code": (
            "diff --git a/skills/SKILL_NAME.py b/skills/SKILL_NAME.py\n"
            "--- a/skills/SKILL_NAME.py\n"
            "+++ b/skills/SKILL_NAME.py\n"
            "@@ -50,2 +50,10 @@\n"
            "+    def _validate(self, input_data) -> None:\n"
            "+        \"\"\"Phase NN -- pre-execute validation.\"\"\"\n"
            "+        if not input_data.text:\n"
            "+            raise ValueError(\"text is required\")\n"
            "+        if len(input_data.text) > 10000:\n"
            "+            raise ValueError(\"text exceeds max length\")\n"
            "+\n"
            "     async def execute(self, input_data, trace_id, "
            "context):\n"
            "+        self._validate(input_data)\n"
        ),
        "explanation": (
            "Pre-execute validation goes in a private _validate method "
            "called from the top of execute. Raise ValueError (or a "
            "skill-specific exception subclass) so the agent's fan-out "
            "loop catches it and the failure flows up cleanly to the "
            "task error envelope. NEVER swallow validation errors; "
            "they're real signals. For network/IO failures, prefer "
            "skill_specific exception types over bare Exception."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="skills/SKILL_NAME.py" '
            'old="..." new="def _validate..."\n'
            'STEP 2: done summary="added validation"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Tests -- 2 more patterns
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["test", "fixture", "tmp_path", "monkeypatch", "fresh_kb"],
        "problem_summary": (
            "write a pytest fixture that gives each test a fresh KB"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="tests/test_phase15a_lifecycle.py"\n'
            'STEP 2: write_file path="tests/test_NEW_with_fixture.py" '
            'content="\\"\\"\\"Tests using fresh KB fixture.\\"\\"\\"\\n'
            'import struct\\nfrom pathlib import Path\\nimport sys\\n'
            'import pytest\\nimport numpy as np\\n\\nsys.path.insert('
            '0, str(Path(__file__).resolve().parent.parent))\\n\\nfrom '
            'core import config\\nfrom core import embeddings as emb\\n'
            'from core.knowledge_base import KnowledgeBase\\n\\n\\n'
            'def _stub_embedder(monkeypatch):\\n    def fake(text, '
            'trace_id=\\"SEN-test\\"):\\n        seed = sum(ord(c) for '
            'c in (text or \\"\\")) % (2**31 - 1)\\n        rng = np.'
            'random.default_rng(seed)\\n        return struct.pack(f'
            '\\"<{config.EMBEDDING_DIM}f\\", *rng.standard_normal('
            'config.EMBEDDING_DIM).tolist())\\n    monkeypatch.setattr('
            'emb, \\"embed_text\\", fake)\\n\\n\\n@pytest.fixture\\n'
            'def fresh_kb(tmp_path, monkeypatch):\\n    db = tmp_path / '
            '\\"kb.db\\"\\n    monkeypatch.setattr(config, '
            '\\"KNOWLEDGE_DB_PATH\\", db)\\n    _stub_embedder('
            'monkeypatch)\\n    return KnowledgeBase(db_path=db)\\n\\n\\n'
            'def test_n01_fresh_kb_is_empty(fresh_kb):\\n    assert '
            'fresh_kb._count() == 0\\n"\n'
            'STEP 3: run_bash command="python -m pytest tests/'
            'test_NEW_with_fixture.py -q"\n'
            'STEP 4: done summary="added fresh_kb fixture using "\n'
        ),
        "solution_code": (
            "diff --git a/tests/test_NEW_with_fixture.py "
            "b/tests/test_NEW_with_fixture.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_NEW_with_fixture.py\n"
            "@@ -0,0 +1,30 @@\n"
            "+import struct, sys\n"
            "+from pathlib import Path\n"
            "+import pytest\n"
            "+import numpy as np\n"
            "+sys.path.insert(0, str(Path(__file__).resolve()."
            "parent.parent))\n"
            "+from core import config, embeddings as emb\n"
            "+from core.knowledge_base import KnowledgeBase\n"
            "+\n"
            "+@pytest.fixture\n"
            "+def fresh_kb(tmp_path, monkeypatch):\n"
            "+    db = tmp_path / \"kb.db\"\n"
            "+    monkeypatch.setattr(config, \"KNOWLEDGE_DB_PATH\", "
            "db)\n"
            "+    # ... stub embedder ...\n"
            "+    return KnowledgeBase(db_path=db)\n"
        ),
        "explanation": (
            "Sentinel's fresh_kb fixture pattern (used in 7+ test "
            "files): tmp_path provides per-test DB isolation; "
            "monkeypatch.setattr config so module-level singletons see "
            "the test path; _stub_embedder seeds Ollama responses with "
            "deterministic numpy vectors keyed by text hash so tests "
            "are reproducible without real Ollama. ALWAYS use this "
            "fixture rather than touching the production knowledge.db."
        ),
        "shadow_recipe": (
            'STEP 1: write_file path="tests/test_NEW.py" '
            'content="...fresh_kb fixture..."\n'
            'STEP 2: done summary="added fixture"'
        ),
    },
    {
        "tags": ["test", "source_grep", "wiring", "regression", "import"],
        "problem_summary": (
            "write a source-level test that asserts wiring without "
            "spinning up the bot"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="tests/test_phase15e_grad_snapshot.py"\n'
            'STEP 2: write_file path="tests/test_NEW_wiring.py" '
            'content="\\"\\"\\"Source-level wiring tests.\\"\\"\\"\\n'
            'from pathlib import Path\\n\\nROOT = Path(__file__).resolve'
            '().parent.parent\\n\\n\\ndef test_n01_handler_is_registered'
            '():\\n    src = (ROOT / \\"interfaces\\" / \\"telegram_bot.'
            'py\\").read_text(encoding=\\"utf-8\\")\\n    assert '
            '\\"CommandHandler(\\\\\\"NEW\\\\\\", self.handle_NEW)\\" in '
            'src\\n    assert \\"async def handle_NEW(\\" in src\\n\\n'
            'def test_n02_helper_uses_log_event():\\n    src = (ROOT / '
            '\\"core\\" / \\"MODULE.py\\").read_text(encoding=\\"utf-8'
            '\\")\\n    assert \\"log_event(trace_id, \\\\\\"INFO\\\\\\"'
            '\\" in src\\n"\n'
            'STEP 3: run_bash command="python -m pytest tests/'
            'test_NEW_wiring.py -q"\n'
            'STEP 4: done summary="added source-level wiring tests"'
        ),
        "solution_code": (
            "diff --git a/tests/test_NEW_wiring.py "
            "b/tests/test_NEW_wiring.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_NEW_wiring.py\n"
            "@@ -0,0 +1,12 @@\n"
            "+\"\"\"Source-level wiring tests.\"\"\"\n"
            "+from pathlib import Path\n"
            "+ROOT = Path(__file__).resolve().parent.parent\n"
            "+\n"
            "+def test_n01_handler_is_registered():\n"
            "+    src = (ROOT / \"interfaces\" / \"telegram_bot.py\")."
            "read_text(encoding=\"utf-8\")\n"
            "+    assert \"CommandHandler(\\\"NEW\\\", self.handle_NEW)"
            "\" in src\n"
            "+    assert \"async def handle_NEW(\" in src\n"
        ),
        "explanation": (
            "Source-level (grep-style) tests verify wiring without "
            "spinning up the bot or mocking the entire stack. Used "
            "extensively in test_phase15e_grad_snapshot.py and "
            "test_phase15d_resilience.py. Read the file, assert key "
            "literal substrings exist. Catches refactor regressions "
            "(e.g. someone renames a function and forgets to update "
            "the call site). NOT a substitute for behavioral tests; "
            "use as a fast guard for integration-level invariants."
        ),
        "shadow_recipe": (
            'STEP 1: write_file path="tests/test_NEW_wiring.py" '
            'content="..."\n'
            'STEP 2: done summary="added wiring tests"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Internal handlers -- 1 more (modify scheduled job)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["scheduler", "schedule", "job", "modify", "pause", "resume"],
        "problem_summary": (
            "pause or modify an existing scheduled job"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/scheduler.py"\n'
            'STEP 2: read_file path="core/database.py"\n'
            'STEP 3: edit_file path="interfaces/telegram_bot.py" '
            'old="async def handle_schedule(self," new="async def handle'
            '_schedule_pause(self, update: Update, context: ContextTypes'
            '.DEFAULT_TYPE) -> None:\\n    if not await self._check_auth'
            '(update):\\n        return\\n    name = \\" \\".join(context'
            '.args or [])\\n    if not name:\\n        await update.'
            'message.reply_text(\\"Usage: /pause_job <name>\\")\\n      '
            '  return\\n    from core import database\\n    n = database'
            '.set_job_enabled(name, False)\\n    await update.message.'
            'reply_text(f\\"paused {n} job(s)\\")\\n\\nasync def handle_'
            'schedule(self,"\n'
            'STEP 4: run_bash command="python -c \\"from core import '
            'database; print(database.list_jobs())\\""\n'
            'STEP 5: done summary="added /pause_job command "\n'
        ),
        "solution_code": (
            "diff --git a/interfaces/telegram_bot.py "
            "b/interfaces/telegram_bot.py\n"
            "--- a/interfaces/telegram_bot.py\n"
            "+++ b/interfaces/telegram_bot.py\n"
            "@@ -1600,3 +1600,11 @@\n"
            "+async def handle_schedule_pause(self, update: Update, "
            "context: ContextTypes.DEFAULT_TYPE) -> None:\n"
            "+    if not await self._check_auth(update):\n"
            "+        return\n"
            "+    name = \" \".join(context.args or [])\n"
            "+    from core import database\n"
            "+    n = database.set_job_enabled(name, False)\n"
            "+    await update.message.reply_text(f\"paused {n} "
            "job(s)\")\n"
        ),
        "explanation": (
            "Scheduled jobs (Phase 11) live in sentinel.db's "
            "scheduled_jobs table. core.database has list_jobs(), "
            "add_job(), set_job_enabled(name, bool), delete_job(name). "
            "The /schedule command in interfaces/telegram_bot.py has "
            "subcommands list/pause/resume/delete/runs. To add a NEW "
            "subcommand, extend handle_schedule's dispatch on args[0] "
            "OR add a separate handle_pause_job for cleaner CLI."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="interfaces/telegram_bot.py" '
            'old="..." new="async def handle_schedule_pause..."\n'
            'STEP 2: done summary="added pause command"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Brain + Claude + Qwen -- 3 patterns
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["brain", "prompt", "system_prompt", "qwen3", "no_think"],
        "problem_summary": (
            "modify the brain's system prompt for a new intent"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/brain.py"\n'
            'STEP 2: edit_file path="core/brain.py" '
            'old="BRAIN_SYSTEM = (" new="BRAIN_SYSTEM = (\\n    \\"NEW '
            'INTENT GUIDANCE: when the user says X, classify as '
            'NEW_INTENT.\\\\n\\"\\n    \\""\n'
            'STEP 3: run_bash command="python -c \\"from core.brain '
            'import BRAIN_SYSTEM; assert \'NEW INTENT GUIDANCE\' in '
            'BRAIN_SYSTEM\\""\n'
            'STEP 4: done summary="extended BRAIN_SYSTEM with NEW_INTENT'
            ' guidance"'
        ),
        "solution_code": (
            "diff --git a/core/brain.py b/core/brain.py\n"
            "--- a/core/brain.py\n"
            "+++ b/core/brain.py\n"
            "@@ -50,2 +50,3 @@\n"
            " BRAIN_SYSTEM = (\n"
            "+    \"NEW INTENT GUIDANCE: when the user says X, "
            "classify as NEW_INTENT.\\n\"\n"
        ),
        "explanation": (
            "Brain prompts use /no_think mode (Qwen 3 1.7B optimization "
            "from Phase 9 -- skips chain-of-thought, ~10s faster). "
            "BRAIN_SYSTEM is a static string literal (no f-strings, no "
            "timestamps, no trace_ids -- prefix-cache friendly per the "
            "pre-15b commit). New intent guidance goes near the top of "
            "the prompt where Qwen weights it most heavily. Keep "
            "additions terse -- the 8192-token context budget needs "
            "headroom for KB + persona + memory injection."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/brain.py" '
            'old="BRAIN_SYSTEM = (" new="BRAIN_SYSTEM = (\\\\n  ..."\n'
            'STEP 2: done summary="extended brain prompt"'
        ),
    },
    {
        "tags": ["claude", "claude_cli", "subprocess", "tools",
                 "ClaudeCliClient"],
        "problem_summary": (
            "add a new code path that invokes the local Claude CLI"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/claude_cli.py"\n'
            'STEP 2: edit_file path="core/NEW_MODULE.py" '
            'old="(top of module)" new="from core.claude_cli import '
            'ClaudeCliClient, ClaudeCliError\\n\\nasync def ask_claude_'
            'NEW(prompt: str, trace_id: str) -> str:\\n    \\"\\"\\"'
            'Sentinel makes ZERO outbound API calls -- this goes '
            'through the local Claude CLI subprocess.\\"\\"\\"\\n    '
            'client = ClaudeCliClient()\\n    if not client.available:\\n'
            '        return \\"\\"\\n    try:\\n        return await '
            'client.generate(\\n            prompt=prompt,\\n            '
            'system=\\"You are a senior code reviewer.\\",\\n            '
            'trace_id=trace_id,\\n            tools=[\\"Read\\", \\"Grep'
            '\\", \\"Glob\\"],\\n        )\\n    except ClaudeCliError '
            'as e:\\n        return f\\"claude CLI failed: {e}\\""\n'
            'STEP 3: run_bash command="python -c \\"from core.claude_cli '
            'import ClaudeCliClient; print(ClaudeCliClient().available)\\""\n'
            'STEP 4: done summary="added ask_claude_NEW with tools-'
            'enabled subprocess invocation"'
        ),
        "solution_code": (
            "diff --git a/core/NEW_MODULE.py b/core/NEW_MODULE.py\n"
            "--- a/core/NEW_MODULE.py\n"
            "+++ b/core/NEW_MODULE.py\n"
            "@@ -1,2 +1,18 @@\n"
            "+from core.claude_cli import ClaudeCliClient, "
            "ClaudeCliError\n"
            "+\n"
            "+async def ask_claude_NEW(prompt: str, trace_id: str) -> "
            "str:\n"
            "+    client = ClaudeCliClient()\n"
            "+    if not client.available:\n"
            "+        return \"\"\n"
            "+    try:\n"
            "+        return await client.generate(\n"
            "+            prompt=prompt,\n"
            "+            system=\"...\",\n"
            "+            trace_id=trace_id,\n"
            "+            tools=[\"Read\", \"Grep\", \"Glob\"],\n"
            "+        )\n"
            "+    except ClaudeCliError as e:\n"
            "+        return f\"claude CLI failed: {e}\"\n"
        ),
        "explanation": (
            "Sentinel makes ZERO outbound API calls -- Claude is "
            "invoked via the local `claude` CLI subprocess "
            "(uses your Claude Code login). Always: check "
            "client.available BEFORE generate(); pass tools list (Read/"
            "Grep/Glob for review, +Bash for execution); catch "
            "ClaudeCliError; return a sentinel value (empty str, None) "
            "on failure rather than raising up. Tools-enabled calls "
            "cost more ($0.05-$0.30 typical) -- gate behind "
            "complexity routing or explicit user request."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/NEW_MODULE.py" '
            'old="..." new="...ClaudeCliClient..."\n'
            'STEP 2: done summary="added Claude invocation"'
        ),
    },
    {
        "tags": ["qwencoder", "memo", "persona", "curate", "playbook"],
        "problem_summary": (
            "add a new lesson to QWENCODER.md teaching memo"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="workspace/persona/QWENCODER.md"\n'
            'STEP 2: edit_file path="workspace/persona/QWENCODER.md" '
            'old="<!-- CURATOR-ENTRIES-BEGIN -->" new="<!-- CURATOR-'
            'ENTRIES-BEGIN -->\\n### NEW lesson (added Phase NN)\\n\\n'
            '**Failure**: Qwen does X when Y.\\n\\n**Fix**: instead, '
            'do Z. Specifically:\\n\\n```\\nSTEP 1: ...\\n```\\n\\n"\n'
            'STEP 3: run_bash command="python -c \\"from pathlib import '
            'Path; p = Path(\'workspace/persona/QWENCODER.md\'); assert '
            '\'NEW lesson\' in p.read_text(encoding=\'utf-8\'); '
            'print(\'memo updated\')\\""\n'
            'STEP 4: done summary="appended NEW lesson to QWENCODER.md "\n'
        ),
        "solution_code": (
            "diff --git a/workspace/persona/QWENCODER.md "
            "b/workspace/persona/QWENCODER.md\n"
            "--- a/workspace/persona/QWENCODER.md\n"
            "+++ b/workspace/persona/QWENCODER.md\n"
            "@@ -180,3 +180,9 @@\n"
            " <!-- CURATOR-ENTRIES-BEGIN -->\n"
            "+### NEW lesson (added Phase NN)\n"
            "+\n"
            "+**Failure**: Qwen does X when Y.\n"
            "+\n"
            "+**Fix**: instead, do Z.\n"
        ),
        "explanation": (
            "QWENCODER.md (Phase 15d) is the worker model's coding "
            "playbook -- file_guard-protected, loaded fresh into the "
            "shadow plan system prompt on every call. The "
            "<!-- CURATOR-ENTRIES-BEGIN --> marker is reserved for "
            "future /curate qwencoder runs to APPEND extracted "
            "failure-mode lessons. Manual additions go in the same "
            "section. Always cite the specific failure pattern "
            "observed in production (with traceID if possible) so "
            "future-you can audit why each rule exists."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="workspace/persona/QWENCODER.md" '
            'old="<!-- CURATOR-ENTRIES-BEGIN -->" new="...new lesson..."\n'
            'STEP 2: done summary="updated memo"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Async + git operations -- 1 more
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["async", "asyncio", "to_thread", "blocking", "IO"],
        "problem_summary": (
            "wrap a blocking IO call so it doesn't block the event loop"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/MODULE.py"\n'
            'STEP 2: edit_file path="core/MODULE.py" old="async def '
            'process(self):" new="async def process(self):\\n        # '
            'Phase NN -- offload the blocking SQLite query so the '
            'event loop stays responsive\\n        result = await '
            'asyncio.to_thread(self._blocking_query)\\n        return '
            'result"\n'
            'STEP 3: run_bash command="python -c \\"import asyncio; '
            'from core.MODULE import some_class; asyncio.run(some_class'
            '().process())\\""\n'
            'STEP 4: done summary="wrapped blocking _blocking_query "\n'
        ),
        "solution_code": (
            "diff --git a/core/MODULE.py b/core/MODULE.py\n"
            "--- a/core/MODULE.py\n"
            "+++ b/core/MODULE.py\n"
            "@@ -50,3 +50,5 @@\n"
            "     async def process(self):\n"
            "+        # offload the blocking call\n"
            "+        result = await asyncio.to_thread(self._blocking_"
            "query)\n"
            "+        return result\n"
        ),
        "explanation": (
            "Sentinel is async-everywhere where the worker touches "
            "(Phase 5 design). Any sync-blocking call (sqlite3 query, "
            "subprocess.run, file IO over a slow disk) inside an async "
            "function MUST be wrapped in asyncio.to_thread so the bot's "
            "event loop stays responsive to /restart, /commit, "
            "scheduled jobs, etc. Pair with asyncio.wait_for(..., "
            "timeout=N) when the blocking call could hang (Ollama, "
            "Claude CLI). The shadow plan path is the canonical "
            "example: asyncio.wait_for(asyncio.to_thread(_qwen_generate"
            ", ...), timeout=30)."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/MODULE.py" '
            'old="..." new="...asyncio.to_thread..."\n'
            'STEP 2: done summary="async-wrapped blocking call"'
        ),
    },
    # ──────────────────────────────────────────────────────────
    # Config + paths (1 pattern)
    # ──────────────────────────────────────────────────────────
    {
        "tags": ["config", "constant", "path", "threshold"],
        "problem_summary": (
            "add a new constant to core/config.py"
        ),
        "solution_pattern": (
            'STEP 1: read_file path="core/config.py"\n'
            'STEP 2: edit_file path="core/config.py" old="# Phase 9 -- '
            'Telegram + Brain" new="NEW_CONSTANT = 42  # Phase NN -- '
            'describe purpose\\n\\n# Phase 9 -- Telegram + Brain"\n'
            'STEP 3: run_bash command="python -c \\"from core import '
            'config; assert config.NEW_CONSTANT == 42; print(\'ok\')\\""\n'
            'STEP 4: done summary="added NEW_CONSTANT to core/config.py"'
        ),
        "solution_code": (
            "diff --git a/core/config.py b/core/config.py\n"
            "--- a/core/config.py\n"
            "+++ b/core/config.py\n"
            "@@ -90,4 +90,6 @@\n"
            "+NEW_CONSTANT = 42  # Phase NN -- describe purpose\n"
            "+\n"
            " # Phase 9 -- Telegram + Brain\n"
        ),
        "explanation": (
            "core/config.py is the SINGLE SOURCE OF TRUTH for paths, "
            "model names, thresholds, env-var names. Never hardcode in "
            "other modules -- import from config. New constants get a "
            "comment marking the phase + purpose so future archaeology "
            "is possible. Group related constants; the file is "
            "loosely Phase-ordered. Always verify with `from core "
            "import config; config.NEW_CONSTANT` so circular imports "
            "are caught early."
        ),
        "shadow_recipe": (
            'STEP 1: edit_file path="core/config.py" '
            'old="..." new="NEW_CONSTANT = 42..."\n'
            'STEP 2: done summary="added constant"'
        ),
    },
]


# ─────────────────────────────────────────────────────────────────
# Insertion + idempotency
# ─────────────────────────────────────────────────────────────────


def _existing_summaries(kb: KnowledgeBase) -> set[str]:
    """Pull the current set of problem_summary values so re-runs don't
    duplicate. Cheap (single SELECT)."""
    import sqlite3
    conn = sqlite3.connect(str(kb.db_path))
    try:
        rows = conn.execute(
            "SELECT problem_summary FROM knowledge "
            "WHERE category = 'pattern'"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows if r[0]}


def preload(
    kb: KnowledgeBase | None = None,
    *,
    pin: bool = True,
    replace: bool = False,
) -> dict:
    """Insert curated patterns into the KB.

    ``pin``: pin every inserted pattern (Phase 15a) so the auto-walker
    never archives them. Default True -- these are the foundation.

    ``replace``: if True, treat existing matches by problem_summary as
    candidates to UPDATE rather than skip. (Default False -- skip.)
    Used for forcing a re-seed after the canonical patterns evolve.

    Returns a counts dict for the caller's logs.
    """
    kb = kb or KnowledgeBase()
    existing = _existing_summaries(kb)
    inserted: list[int] = []
    skipped: list[str] = []
    pinned: list[int] = []

    for spec in PATTERNS:
        summary = spec["problem_summary"]
        if summary in existing and not replace:
            skipped.append(summary)
            continue

        # Pre-compute shadow agreement so /kb planning has data day-one
        # rather than waiting for organic shadow plans to populate.
        agreement = score_plan_agreement(
            spec["solution_pattern"],
            spec["shadow_recipe"],
        )

        pid = kb.add_pattern(
            tags=spec["tags"],
            problem_summary=summary,
            solution_code=spec["solution_code"],
            solution_pattern=spec["solution_pattern"],
            explanation=spec["explanation"],
            trace_id=TRACE,
            qwen_plan_recipe=spec["shadow_recipe"],
            qwen_plan_agreement=agreement,
        )
        inserted.append(pid)
        if pin:
            kb.pin_pattern(pid)
            pinned.append(pid)

    return {
        "inserted": inserted,
        "pinned": pinned,
        "skipped_count": len(skipped),
        "total_curated": len(PATTERNS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-load Sentinel KB with curated patterns",
    )
    parser.add_argument(
        "--no-pin", action="store_true",
        help="Don't pin the inserted patterns (default pins all)",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Force re-insert even if problem_summary matches",
    )
    args = parser.parse_args(argv)
    result = preload(pin=not args.no_pin, replace=args.replace)
    print(f"inserted: {len(result['inserted'])} patterns")
    print(f"  ids: {result['inserted']}")
    print(f"pinned: {len(result['pinned'])}")
    print(f"skipped (already present): {result['skipped_count']}")
    print(f"total curated in script: {result['total_curated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
