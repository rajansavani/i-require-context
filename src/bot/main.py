from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import discord

from src.bot.config import get_settings


def _setup_logging() -> None:
    # simple console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    # make it readable by dropping noisy logs
    logging.getLogger("discord").setLevel(logging.INFO)
    logging.getLogger("discord.http").setLevel(logging.WARNING)


def _build_intents() -> discord.Intents:
    # mainly just need voice status updates
    intents = discord.Intents.default()
    intents.guilds = True
    intents.voice_states = True
    return intents


def _load_cogs(bot: discord.Bot) -> None:
    # load all cogs from src/bot/cogs
    log = logging.getLogger("irc.cogs")
    cogs_path = Path(__file__).parent / "cogs"

    if not cogs_path.exists():
        log.warning("No cogs folder found at %s", cogs_path)
        return

    loaded = 0
    for file in sorted(cogs_path.glob("*.py")):
        if file.name.startswith("_"):
            continue

        module = f"src.bot.cogs.{file.stem}"
        try:
            bot.load_extension(module)
            loaded += 1
            log.info("Loaded cog: %s", module)
        except Exception:
            log.exception("Failed to load cog: %s", module)

    if loaded == 0:
        log.warning("No cogs were loaded. Add a cog under src/bot/cogs/ to register commands.")


async def _main() -> None:
    _setup_logging()

    settings = get_settings()
    log = logging.getLogger("irc.main")

    bot = discord.Bot(intents=_build_intents())

    _load_cogs(bot)

    @bot.event
    async def on_ready() -> None:
        if bot.user:
            log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)
        else:
            log.info("Bot is ready.")

    @bot.event
    async def on_connect() -> None:
        # syncing here helps ensure commands register after startup
        log.info("Syncing application commands (global)...")
        try:
            await bot.sync_commands()
            log.info("Slash commands synced.")
        except Exception:
            log.exception("Failed to sync slash commands.")

    await bot.start(settings.discord_bot_token)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Shutting down.")