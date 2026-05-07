# User Profile (TEMPLATE)

> Loaded by the brain on every chat (Phase 10 persona system). Edit
> with your own info OR leave as a placeholder until you've used the
> bot enough that auto-extraction (`_maybe_auto_extract`) populates
> facts via /remember. file_guard tracks SHA-256 — edits outside
> `file_guard.authorize_update()` ping you on Telegram.

## Background
- Add a one-line description of yourself.
- Add your email if you want the bot to reference it.

## Working preferences
- How do you want the bot to communicate with you? (terse / verbose / structured)
- What kind of decisions can it make autonomously vs. ask you about?
- Any phrasings or signals you want it to remember?

## Communication
- Telegram bot username goes here (set on first BotFather chat).
- `/code <problem>` for Qwen + Claude-CLI ceiling.
- `/gwen <recipe>` to execute a literal recipe.
- `/gwenask <idea>` to have local Qwen author one for you.
- Free-text replies get classified by the brain.

## Job search preferences (optional — used by Phase 12+ pipeline)
- Industry / region / target roles → see `workspace/persona/PROFILE.yml`.

## Hardware reality
- One GPU model loaded at a time on a 4 GB constraint card.
- Worker keep_alive: 2 min. Brain is `sentinel-brain` (qwen3:1.7b custom).
- See `CLAUDE.md` for the full architecture decisions.
