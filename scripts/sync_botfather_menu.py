"""One-shot: push SentinelTelegramBot.BOT_COMMAND_MENU to BotFather.

Runs without touching the running bot. Use when you've added new
commands locally but don't want to /restart yet (or when an older bot
binary is still serving and you want the slash-autocomplete updated
right now).

  py -3.12 scripts/sync_botfather_menu.py

Reads SENTINEL_TELEGRAM_TOKEN from env (same source as main.py).
Idempotent on Telegram's side -- calling repeatedly with the same
list is a no-op. Prints what was pushed and what was previously set.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telegram import Bot, BotCommand

from core import config
from interfaces.telegram_bot import SentinelTelegramBot


async def _main() -> int:
    token = config.TELEGRAM_TOKEN
    if not token:
        print("ERROR: SENTINEL_TELEGRAM_TOKEN not set in env",
              file=sys.stderr)
        return 2
    menu = list(SentinelTelegramBot.BOT_COMMAND_MENU)
    bot = Bot(token=token)
    async with bot:
        before = await bot.get_my_commands()
        print(f"BEFORE: {len(before)} commands registered with BotFather")
        for c in before:
            print(f"  /{c.command:18s} {c.description}")
        commands = [BotCommand(c, d) for c, d in menu]
        await bot.set_my_commands(commands)
        after = await bot.get_my_commands()
        print(f"\nAFTER:  {len(after)} commands now registered:")
        for c in after:
            print(f"  /{c.command:18s} {c.description}")
        new_set = {c.command for c in after} - {c.command for c in before}
        if new_set:
            print(f"\nNEWLY ADDED: {sorted(new_set)}")
        else:
            print("\n(no new commands -- list was already in sync)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
